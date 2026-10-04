#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class run_mode { tokenize, collect, patch_score, save_state, seed_score };

struct options {
    run_mode mode = run_mode::tokenize;
    std::string mode_name;
    fs::path model, input, output, capture_spec;
    int ctx = 5120, batch = 512, threads = 16, max_scored = 256;
    bool max_scored_set = false, full_decode = false;
    std::string cache_type = "f32", device = "cpu";
    bool flash_attn = false, swa_full = true, cache_options_set = false;
    bool gpu() const { return device != "cpu"; }
};

static const char * usage =
    "--mode {tokenize,collect,patch-score,save-state,seed-score} --model GGUF --input DOCS.jsonl --output NEW_DIR\n"
    "  [--ctx 5120] [--batch 512] [--threads 16]\n"
    "  [--capture-spec SPEC.json]   (collect; default {\"k\":\"all\",\"v\":\"all\"})\n"
    "  [--max-scored 256]           (patch-score, seed-score)\n"
    "  [--full-decode]              (patch-score, seed-score: decode whole documents, not only up to the last needed ubatch)\n"
    "  [--cache-type f32|f16] [--flash-attn off|on] [--swa-full on|off]\n"
    "                               (every mode but tokenize; defaults f32, off, on)\n"
    "  [--device cpu|vulkan]        (default cpu; vulkan: every layer and the KV cache on the first Vulkan GPU;\n"
    "                               tokenize loads the vocabulary only and never uses a device)\n";

static options parse_options(int argc, char ** argv) {
    static const std::set<std::string> path_options = {"--mode", "--model", "--input", "--output", "--capture-spec",
                                                       "--cache-type", "--flash-attn", "--swa-full", "--device"};
    static const std::set<std::string> integer_options = {"--ctx", "--batch", "--threads", "--max-scored"};
    options result;
    std::set<std::string> given;
    auto on_off = [](const std::string & key, const std::string & value) {
        if (value == "on") { return true; }
        if (value == "off") { return false; }
        throw std::runtime_error(key + " must be on or off");
    };
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { std::cout << usage; std::exit(0); }
        if (key != "--full-decode" && !path_options.count(key) && !integer_options.count(key)) {
            throw std::runtime_error("unknown option " + key);
        }
        if (!given.insert(key).second) { throw std::runtime_error("option " + key + " given more than once"); }
        if (key == "--full-decode") { result.full_decode = true; continue; }
        if (++i == argc) { throw std::runtime_error("missing argument for " + key); }
        const std::string value = argv[i];
        if      (key == "--mode")         { result.mode_name = value; }
        else if (key == "--model")        { result.model = value; }
        else if (key == "--input")        { result.input = value; }
        else if (key == "--output")       { result.output = value; }
        else if (key == "--capture-spec") { result.capture_spec = value; }
        else if (key == "--cache-type") {
            if (value != "f32" && value != "f16") { throw std::runtime_error("--cache-type must be f32 or f16"); }
            result.cache_type = value;
            result.cache_options_set = true;
        }
        else if (key == "--flash-attn")   { result.flash_attn = on_off(key, value); result.cache_options_set = true; }
        else if (key == "--swa-full")     { result.swa_full = on_off(key, value); result.cache_options_set = true; }
        else if (key == "--device") {
            if (value != "cpu" && value != "vulkan") { throw std::runtime_error("--device must be cpu or vulkan"); }
            result.device = value;
        }
        else {
            size_t used = 0;
            int number = 0;
            try { number = std::stoi(value, &used); } catch (const std::exception &) { used = 0; }
            if (used == 0 || used != value.size()) { throw std::runtime_error("invalid integer for " + key); }
            if      (key == "--ctx")        { result.ctx = number; }
            else if (key == "--batch")      { result.batch = number; }
            else if (key == "--threads")    { result.threads = number; }
            else                            { result.max_scored = number; result.max_scored_set = true; }
        }
    }
    if      (result.mode_name == "tokenize")    { result.mode = run_mode::tokenize; }
    else if (result.mode_name == "collect")     { result.mode = run_mode::collect; }
    else if (result.mode_name == "patch-score") { result.mode = run_mode::patch_score; }
    else if (result.mode_name == "save-state")  { result.mode = run_mode::save_state; }
    else if (result.mode_name == "seed-score")  { result.mode = run_mode::seed_score; }
    else { throw std::runtime_error("--mode must be tokenize, collect, patch-score, save-state or seed-score"); }
    if (result.model.empty() || result.input.empty() || result.output.empty()) {
        throw std::runtime_error("--model, --input and a new --output directory are required");
    }
    if (result.ctx < 1 || result.ctx > 32768 || result.batch < 1 || result.batch > result.ctx
            || result.threads < 1 || result.threads > 512 || result.max_scored < 1) {
        throw std::runtime_error("sizes must be positive, batch <= ctx <= 32768");
    }
    if (!result.capture_spec.empty() && result.mode != run_mode::collect) {
        throw std::runtime_error("--capture-spec is only valid with --mode collect");
    }
    if ((result.max_scored_set || result.full_decode)
            && result.mode != run_mode::patch_score && result.mode != run_mode::seed_score) {
        throw std::runtime_error("--max-scored and --full-decode are only valid with --mode patch-score or seed-score");
    }
    if (result.cache_options_set && result.mode == run_mode::tokenize) {
        throw std::runtime_error("--cache-type, --flash-attn and --swa-full are not valid with --mode tokenize");
    }
    return result;
}

// JSON parsing that rejects repeated object keys instead of keeping the last one.
static json parse_json_strict(const std::string & text) {
    std::vector<std::set<std::string>> keys;
    const json::parser_callback_t check = [&keys](int, json::parse_event_t event, json & parsed) {
        if (event == json::parse_event_t::object_start) { keys.emplace_back(); }
        else if (event == json::parse_event_t::object_end) { keys.pop_back(); }
        else if (event == json::parse_event_t::key && !keys.back().insert(parsed.get<std::string>()).second) {
            throw std::runtime_error("duplicate JSON key \"" + parsed.get<std::string>() + "\"");
        }
        return true;
    };
    return json::parse(text, check);
}

// ---------------------------------------------------------------------------
// SHA-256 (token id digests for patch-score)

static std::string sha256_hex(const uint8_t * data, size_t size) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> message(data, data + size);
    message.push_back(0x80);
    while (message.size() % 64 != 56) { message.push_back(0); }
    const uint64_t bits = uint64_t(size) * 8;
    for (int i = 7; i >= 0; --i) { message.push_back(uint8_t(bits >> (8 * i))); }
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t chunk = 0; chunk < message.size(); chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t * p = &message[chunk + 4 * i];
            w[i] = uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char * digits = "0123456789abcdef";
    std::string out;
    for (uint32_t word : h) {
        for (int i = 28; i >= 0; i -= 4) { out.push_back(digits[(word >> i) & 0xf]); }
    }
    return out;
}

// ---------------------------------------------------------------------------
// NPY v1.0 writer and a strict reader for patch inputs

static std::string npy_header(const std::string & descr, const std::vector<int64_t> & shape) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]);
        if (shape.size() == 1) { dims += ","; } else if (i + 1 < shape.size()) { dims += ", "; }
    }
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const size_t unpadded = 10 + dict.size() + 1;
    dict.append((64 - unpadded % 64) % 64, ' ');
    dict.push_back('\n');
    if (dict.size() > 65535) { throw std::runtime_error("NPY header too long"); }
    std::string header("\x93NUMPY\x01\x00", 8);
    header.push_back(char(dict.size() & 0xff));
    header.push_back(char(dict.size() >> 8));
    return header + dict;
}

static void write_npy(const fs::path & path, const std::string & descr, const std::vector<int64_t> & shape,
                      const void * data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    const auto header = npy_header(descr, shape);
    out.write(header.data(), std::streamsize(header.size()));
    out.write(static_cast<const char *>(data), std::streamsize(bytes));
    out.close();
}

struct npy_info {
    std::string descr;
    std::vector<int64_t> shape;
    uint64_t data_offset = 0;
};

static npy_info read_npy_header(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0 || magic[6] < 1 || magic[6] > 3) {
        throw std::runtime_error("not a supported NPY file: " + path.string());
    }
    uint64_t header_len = 0;
    if (magic[6] == 1) {
        uint8_t len[2];
        in.read(reinterpret_cast<char *>(len), 2);
        header_len = uint64_t(len[0]) | uint64_t(len[1]) << 8;
    } else {
        uint8_t len[4];
        in.read(reinterpret_cast<char *>(len), 4);
        header_len = uint64_t(len[0]) | uint64_t(len[1]) << 8 | uint64_t(len[2]) << 16 | uint64_t(len[3]) << 24;
    }
    if (!in || header_len > (1u << 20)) { throw std::runtime_error("invalid NPY header: " + path.string()); }
    std::string text(header_len, '\0');
    in.read(text.data(), std::streamsize(header_len));
    if (!in) { throw std::runtime_error("truncated NPY header: " + path.string()); }
    npy_info info;
    info.data_offset = (magic[6] == 1 ? 10 : 12) + header_len;
    size_t pos = 0;
    auto fail = [&]() -> std::runtime_error { return std::runtime_error("unsupported NPY header in " + path.string() + ": " + text); };
    auto ws = [&]() { while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n')) { ++pos; } };
    auto quoted = [&]() {
        if (pos >= text.size() || (text[pos] != '\'' && text[pos] != '"')) { throw fail(); }
        const char q = text[pos++];
        const size_t end = text.find(q, pos);
        if (end == std::string::npos) { throw fail(); }
        std::string s = text.substr(pos, end - pos);
        pos = end + 1;
        return s;
    };
    std::set<std::string> keys;
    bool fortran = true;
    ws();
    if (pos >= text.size() || text[pos++] != '{') { throw fail(); }
    for (;;) {
        ws();
        if (pos < text.size() && text[pos] == '}') { ++pos; break; }
        const std::string key = quoted();
        ws();
        if (pos >= text.size() || text[pos++] != ':') { throw fail(); }
        ws();
        if (!keys.insert(key).second) { throw fail(); }
        if (key == "descr") {
            info.descr = quoted();
        } else if (key == "fortran_order") {
            if (text.compare(pos, 5, "False") == 0) { fortran = false; pos += 5; }
            else if (text.compare(pos, 4, "True") == 0) { fortran = true; pos += 4; }
            else { throw fail(); }
        } else if (key == "shape") {
            if (pos >= text.size() || text[pos++] != '(') { throw fail(); }
            for (;;) {
                ws();
                if (pos < text.size() && text[pos] == ')') { ++pos; break; }
                size_t used = 0;
                long long value = 0;
                try { value = std::stoll(text.substr(pos), &used); } catch (const std::exception &) { throw fail(); }
                if (used == 0 || value < 0) { throw fail(); }
                info.shape.push_back(value);
                pos += used;
                ws();
                if (pos < text.size() && text[pos] == ',') { ++pos; }
            }
        } else {
            throw fail();
        }
        ws();
        if (pos < text.size() && text[pos] == ',') { ++pos; }
    }
    if (keys.size() != 3 || fortran) { throw fail(); }
    uint64_t elements = 1;
    for (auto dim : info.shape) {
        if (dim > 0 && elements > (uint64_t(1) << 40) / uint64_t(dim)) { throw fail(); }
        elements *= uint64_t(dim);
    }
    const uint64_t item = info.descr == "<f4" || info.descr == "<i4" ? 4 : 0;
    if (item == 0) { throw std::runtime_error("NPY dtype must be <f4 or <i4: " + path.string()); }
    if (fs::file_size(path) != info.data_offset + elements * item) {
        throw std::runtime_error("NPY file size does not match its header: " + path.string());
    }
    return info;
}

template <typename T>
static std::vector<T> read_npy_data(const fs::path & path, const npy_info & info) {
    uint64_t elements = 1;
    for (auto dim : info.shape) { elements *= uint64_t(dim); }
    std::vector<T> values(elements);
    std::ifstream in(path, std::ios::binary);
    in.seekg(std::streamoff(info.data_offset));
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(elements * sizeof(T)));
    if (!in) { throw std::runtime_error("cannot read NPY data: " + path.string()); }
    return values;
}

// ---------------------------------------------------------------------------
// GGUF inspection

static int64_t integer(const gguf_context * meta, const std::string & key, int64_t fallback = -1, int layer = -1) {
    const int64_t id = gguf_find_key(meta, key.c_str());
    if (id < 0) {
        if (fallback < 0) { throw std::runtime_error("missing GGUF field: " + key); }
        return fallback;
    }
    const bool array = gguf_get_kv_type(meta, id) == GGUF_TYPE_ARRAY;
    if (array && (layer < 0 || size_t(layer) >= gguf_get_arr_n(meta, id))) {
        throw std::runtime_error("invalid layer array: " + key);
    }
    const gguf_type type = array ? gguf_get_arr_type(meta, id) : gguf_get_kv_type(meta, id);
    const void * data = array ? gguf_get_arr_data(meta, id) : nullptr;
    switch (type) {
        case GGUF_TYPE_UINT32: return array ? static_cast<const uint32_t *>(data)[layer] : gguf_get_val_u32(meta, id);
        case GGUF_TYPE_INT32:  return array ? static_cast<const int32_t *>(data)[layer] : gguf_get_val_i32(meta, id);
        case GGUF_TYPE_UINT16: return array ? static_cast<const uint16_t *>(data)[layer] : gguf_get_val_u16(meta, id);
        case GGUF_TYPE_UINT8:  return array ? static_cast<const uint8_t *>(data)[layer] : gguf_get_val_u8(meta, id);
        case GGUF_TYPE_BOOL:   return array ? static_cast<const bool *>(data)[layer] : gguf_get_val_bool(meta, id);
        default: throw std::runtime_error("unsupported integer GGUF field: " + key);
    }
}

// ---------------------------------------------------------------------------
// Capture keys and supported architectures
//
// A capture key names one graph tensor per attention layer (llm_graph_context::cb names it
// "<tensor>-<layer>") and the op that produces the wanted node, which tells it apart from
// other nodes of the same name (build_qkv names both the 2-D V projection and its 3-D
// reshape "Vcur"; deepseek2 names both the k_pe view and its RoPE "k_pe"). Values are read
// as float32 [tokens, heads, dim] (or [tokens, heads * dim]). The K row of an MLA cache with
// RoPE is no single node before RoPE: it is captured from two, the normed latent "kv_cmpr"
// (build_norm's MUL, [kv_lora, tokens]) and the pre-RoPE "k_pe" ([n_rot, 1, tokens]), each
// written into its columns of one row [c_kv | k_pe] (src/models/deepseek2.cpp:526-590).

// norm_output: whatever build_norm returns, its bias ADD, else its weight MUL, else the NORM itself
// (llm_graph_context::build_norm names only that last node after the caller)
enum class key_op { mul, mul_mat, reshape_or_view, view, concat, norm_output, sigmoid, any };

struct node_def {
    const char * tensor;
    key_op op;
    bool two_d;             // [heads * dim, tokens] instead of [dim, heads, tokens]
};

// One capture key of an architecture: its node, and for a composite row (the MLA K row, a DeepSeek
// V4 compressor row [a | s]) the node whose rows fill the columns after it.
struct key_def {
    const char * key;
    node_def node;
    node_def tail = {nullptr, key_op::any, false};
};

// every capture key, in the order captures are listed
static const std::array<const char *, 9> capture_keys = {{"k", "v", "q", "gate", "attn_out", "ik", "csa", "hca", "lid"}};
// the DeepSeek V4 compressor keys (pooled caches) and the compress ratio of the layers that have them
static bool is_compressor_key(const std::string & key) { return key == "csa" || key == "hca" || key == "lid"; }
static int64_t compressor_ratio(const std::string & key) { return key == "hca" ? 128 : 4; }

// pre-RoPE key/query after their q/k RMSNorm (build_norm's MUL), V as build_qkv leaves it
// ([head_dim, n_head_kv, n_tokens]; with a fused QKV, a view)
static const key_def k_normed{"k", {"Kcur_normed", key_op::mul, false}};
static const key_def v_cur{"v", {"Vcur", key_op::reshape_or_view, false}};
static const key_def q_normed{"q", {"Qcur_normed", key_op::mul, false}};
// build_qkv's 3-D K/Q before any RoPE (models without q/k norms; the RoPE output is renamed
// "Kcur" too, which the op filter skips)
static const key_def k_qkv{"k", {"Kcur", key_op::reshape_or_view, false}};
static const key_def q_qkv{"q", {"Qcur", key_op::reshape_or_view, false}};
// MLA: the cache row [RMSNorm(c_kv) * g | k_pe before RoPE], V the c_kv view; DSA indexer keys
// before RoPE and the Hadamard rotation (the k_norm LayerNorm output; glm-dsa's k_norm weight and
// bias are optional, glm-dsa.cpp:144-145, so the output is an ADD, a MUL or the NORM)
static const key_def k_mla{"k", {"kv_cmpr", key_op::mul, true}, {"k_pe", key_op::view, false}};
static const key_def ik_dsa{"ik", {"indexer_k", key_op::norm_output, true}};
// DeepSeek V4 (src/models/deepseek4.cpp:928-1040): K is the raw latent before RoPE (kv_norm, the
// reshape [head, 1, tokens]; the cache reads it as both K and V, so there is no V key), and per
// compressed layer the compressor inputs of its pools as one row [a | s] from two 2-D MUL_MATs
// ({csa,hca,lid}_state_kv and _state_score: the score before APE): CSA and LID on ratio-4 layers,
// HCA on ratio-128 layers
static const key_def k_dsv4{"k", {"kv_norm", key_op::reshape_or_view, false}};
static const key_def csa_dsv4{"csa", {"csa_state_kv", key_op::mul_mat, true}, {"csa_state_score", key_op::mul_mat, true}};
static const key_def hca_dsv4{"hca", {"hca_state_kv", key_op::mul_mat, true}, {"hca_state_score", key_op::mul_mat, true}};
static const key_def lid_dsv4{"lid", {"lid_state_kv", key_op::mul_mat, true}, {"lid_state_score", key_op::mul_mat, true}};

// How the trunk layers of an architecture are classified.
enum class layer_rule {
    hybrid_recurrent,  // qwen35/qwen3next/qwen35moe/minimax-01: attention.recurrent_layers flags, else every
                       // full_attention_interval-th layer (default arch_def::interval) is full attention
    sliding_pattern,   // muse-glimmer, dots3note: attention.sliding_window_pattern flags or period; the rest is full
    dense,             // every trunk layer is full attention (one plain llama_kv_cache)
    kv_heads,          // recurrent iff attention.head_count_kv is 0 for the layer (KDA, Mamba, short-conv
                       // hybrids); lfm2's attention layers all slide when attention.sliding_window > 0
    nemotron,          // feed_forward_length > 0: MLP only; else no KV heads: recurrent; else full attention
    parallel,          // falcon-h1: every layer runs attention and an SSM side by side
    all_sliding,       // deepseek4: every trunk layer is a sliding-window layer (src/models/deepseek4.cpp:55-61)
};

// One row per architecture whose graph names (src/models/<arch>.cpp), tensor shapes and
// memory layout were checked against the pinned runtime. To add an architecture, check that
// its graph names the pre-RoPE key/query and V as one of the key definitions above uses them
// (or give it its own), then add a row; kvstate.py needs the matching ModelSpec. A row
// without keys still serves tokenize, save-state and seed-score.
struct arch_def {
    const char * name;
    layer_rule rule;
    bool nextn_allowed;                     // nextn_predict_layers > 0: those blocks are not trunk layers
    std::vector<key_def> keys;              // capture keys of this architecture
    bool mla = false;                       // K-only latent rows: one KV head, K = key_length, V = kv_lora_rank
    int interval = 4;                       // hybrid_recurrent: default full_attention_interval
    const key_def * find(const std::string & key) const {
        for (const auto & def : keys) {
            if (key == def.key) { return &def; }
        }
        return nullptr;
    }
    bool has_key(const std::string & key) const { return find(key) != nullptr; }
};

static const std::vector<arch_def> arch_table = {
    {"qwen35", layer_rule::hybrid_recurrent, true, {k_normed, v_cur, q_normed}},
    {"muse-glimmer", layer_rule::sliding_pattern, true,
     {k_normed, v_cur, q_normed, {"gate", {"attn_gate_sig", key_op::sigmoid, true}},
      {"attn_out", {"attn_out", key_op::any, true}}}},
    // src/models/qwen3.cpp: Kcur_normed / Qcur_normed (q/k RMSNorm before NEOX RoPE), Vcur from build_qkv
    {"qwen3", layer_rule::dense, false, {k_normed, v_cur, q_normed}},
    // GatedDeltaNet hybrids named like qwen35 (qwen3next.cpp:263-290, qwen35moe.cpp:302-337,
    // qwen4exp.cpp:777-822), and plamo2's Mamba hybrid (plamo2.cpp:225-243: V a view of the fused QKV)
    {"qwen3next", layer_rule::hybrid_recurrent, true, {k_normed, v_cur, q_normed}},
    {"qwen35moe", layer_rule::hybrid_recurrent, true, {k_normed, v_cur, q_normed}},
    {"qwen4exp", layer_rule::hybrid_recurrent, true, {k_normed, v_cur, q_normed}},
    {"plamo2", layer_rule::kv_heads, false, {k_normed, v_cur, q_normed}},
    // build_qkv attention next to recurrent layers: K/Q before RoPE (minimax-01.cpp:254-271, jamba.cpp:132,
    // granite-hybrid.cpp:208-221, nemotron-h.cpp:261); NoPE in jamba and nemotron-h
    {"minimax-01", layer_rule::hybrid_recurrent, true, {k_qkv, v_cur, q_qkv}, false, 8},
    {"jamba", layer_rule::kv_heads, false, {k_qkv, v_cur, q_qkv}},
    {"granitehybrid", layer_rule::kv_heads, false, {k_qkv, v_cur, q_qkv}},
    {"nemotron_h", layer_rule::nemotron, false, {k_qkv, v_cur, q_qkv}},
    {"nemotron_h_moe", layer_rule::nemotron, false, {k_qkv, v_cur, q_qkv}},
    // falcon-h1.cpp:139-150: the V reshape is renamed "Vcur-post-rope"
    {"falcon-h1", layer_rule::parallel, false, {k_qkv, {"v", {"Vcur-post-rope", key_op::reshape_or_view, false}}, q_qkv}},
    // lfm2.cpp:136-142 (lfm2moe runs the same attention block): the stored key is the k_layernorm
    // output (build_norm's MUL)
    {"lfm2", layer_rule::kv_heads, false,
     {{"k", {"model.layers.{}.self_attn.k_layernorm", key_op::mul, false}}, v_cur,
      {"q", {"model.layers.{}.self_attn.q_layernorm", key_op::mul, false}}}},
    {"lfm2moe", layer_rule::kv_heads, false,
     {{"k", {"model.layers.{}.self_attn.k_layernorm", key_op::mul, false}}, v_cur,
      {"q", {"model.layers.{}.self_attn.q_layernorm", key_op::mul, false}}}},
    // MLA (deepseek2.cpp:526-590; mistral4 runs deepseek2's graph; deepseek32.cpp, glm-dsa.cpp: indexer
    // keys, glm-dsa.cpp:276-293); the deepseek2/mistral4 cache keeps nextn layers, which are not captured
    {"deepseek2", layer_rule::dense, true, {k_mla, v_cur}, true},
    {"mistral4", layer_rule::dense, true, {k_mla, v_cur}, true},
    {"deepseek32", layer_rule::dense, false, {k_mla, v_cur, ik_dsa}, true},
    {"glm-dsa", layer_rule::dense, false, {k_mla, v_cur, ik_dsa}, true},
    // dots3note.cpp:316-344: k_pe gets an RMSNorm (MUL, named k_pe) before RoPE
    {"dots3note", layer_rule::sliding_pattern, false,
     {{"k", {"kv_cmpr", key_op::mul, true}, {"k_pe", key_op::mul, false}}, v_cur, ik_dsa}, true},
    // NoPE MLA + KDA (kimi-linear.cpp:394-437): the concatenated cache row is the pre-RoPE row
    {"kimi-linear", layer_rule::kv_heads, true, {{"k", {"Kcur", key_op::concat, false}}, v_cur}, true},
    // kimi-k3.cpp:481-556, bailingmoe3.cpp:303-351 and hy-v4.cpp:296-321 name neither the latent row
    // nor k_pe before RoPE: no capture keys (tokenize, save-state and seed-score only)
    {"kimi-k3", layer_rule::kv_heads, true, {}, true},
    {"bailingmoe3", layer_rule::kv_heads, true, {}, true},
    {"hy_v4", layer_rule::dense, false, {}, true},
    // DeepSeek V4, V4-Pro, V4-Flash (deepseek4.cpp; kvstate's dsv4 memory); its nextn (MTP) blocks are not trunk layers
    {"deepseek4", layer_rule::all_sliding, true, {k_dsv4, csa_dsv4, hca_dsv4, lid_dsv4}},
};

static const arch_def * find_arch(const std::string & name) {
    for (const auto & def : arch_table) {
        if (name == def.name) { return &def; }
    }
    return nullptr;
}

static std::string arch_list(const std::function<bool(const arch_def &)> & keep) {
    std::vector<std::string> names;
    for (const auto & def : arch_table) {
        if (keep(def)) { names.emplace_back(def.name); }
    }
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        out += (i == 0 ? "" : i + 1 == names.size() ? " and " : ", ") + names[i];
    }
    return out;
}

struct layer_info {
    std::string attention;                  // full | sliding | recurrent | parallel | mlp
    int64_t n_head = 0, n_head_kv = 0, dim_k = 0, dim_v = 0;
    int64_t kv_lora = 0;                    // MLA: latent width (the first kv_lora elements of the K row)
    int64_t indexer = 0;                    // DSA: indexer key length when this layer stores indexer keys
    int64_t compress = 0;                   // DeepSeek V4: compress ratio (0 window only, 4 CSA + LID, 128 HCA)
    int64_t lid_dim = 0;                    // DeepSeek V4: lightning-indexer key length on ratio-4 layers
};

struct model_info {
    std::string architecture, file_type;
    const arch_def * arch = nullptr;
    int64_t n_layer = 0;
    std::vector<layer_info> layers;
    bool is_attention(int64_t il) const {
        return il >= 0 && il < n_layer && layers[il].attention != "recurrent" && layers[il].attention != "mlp";
    }
};

static bool has_key(const gguf_context * meta, const std::string & key) { return gguf_find_key(meta, key.c_str()) >= 0; }

static model_info inspect_model(const fs::path & path) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
        gguf_init_from_file(path.string().c_str(), {true, nullptr}), gguf_free);
    if (!meta) { throw std::runtime_error("cannot read GGUF metadata"); }
    const auto * m = meta.get();
    if (integer(m, "split.count", 1) != 1) { throw std::runtime_error("kv probe requires a single-file GGUF"); }
    const auto arch_id = gguf_find_key(m, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(m, arch_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF architecture is missing");
    }
    model_info info;
    info.architecture = gguf_get_val_str(m, arch_id);
    info.arch = find_arch(info.architecture);
    if (!info.arch) {
        throw std::runtime_error("kv probe supports only " + arch_list([](const arch_def &) { return true; }));
    }
    const std::string prefix = info.architecture + ".";
    const int64_t blocks = integer(m, prefix + "block_count");
    const int64_t nextn = integer(m, prefix + "nextn_predict_layers", 0);
    if (blocks <= 0 || nextn < 0 || nextn >= blocks) { throw std::runtime_error("invalid block count"); }
    if (nextn > 0 && !info.arch->nextn_allowed) {
        throw std::runtime_error(info.architecture + " with nextn_predict_layers > 0 is not supported");
    }
    info.n_layer = blocks - nextn;
    const int64_t embd = integer(m, prefix + "embedding_length");
    // per-layer arrays (head counts, feed-forward lengths, flags) or one value for every layer
    auto per_layer = [&](const std::string & key, int64_t fallback, int il) {
        const auto id = gguf_find_key(m, key.c_str());
        if (id >= 0 && gguf_get_kv_type(m, id) == GGUF_TYPE_ARRAY) { return integer(m, key, -1, il); }
        return integer(m, key, fallback);
    };
    // only lfm2 reads attention.sliding_window among the kv_heads architectures (lfm2.cpp:26; kvstate.py)
    const bool swa = info.architecture == "lfm2" && integer(m, prefix + "attention.sliding_window", 0) > 0;
    // DSA indexer layers: deepseek32 every trunk layer, the others attention.indexer.types (glm-dsa without
    // them: every layer while the context is below 1M, src/models/glm-dsa.cpp:59-66)
    const bool indexer = info.arch->has_key("ik");
    const std::string types = prefix + "attention.indexer.types";
    for (int il = 0; il < info.n_layer; ++il) {
        layer_info layer;
        if (info.arch->rule == layer_rule::hybrid_recurrent) {
            const auto interval = integer(m, prefix + "full_attention_interval", info.arch->interval);
            if (interval <= 0) { throw std::runtime_error("invalid full-attention interval"); }
            const bool recurrent = integer(m, prefix + "attention.recurrent_layers", (il + 1) % interval != 0, il) != 0;
            layer.attention = recurrent ? "recurrent" : "full";
        } else if (info.arch->rule == layer_rule::sliding_pattern) {
            const std::string pattern = prefix + "attention.sliding_window_pattern";
            const auto id = gguf_find_key(m, pattern.c_str());
            bool sliding;
            if (id >= 0 && gguf_get_kv_type(m, id) == GGUF_TYPE_ARRAY) {
                sliding = integer(m, pattern, -1, il) != 0;
            } else {
                const auto period = integer(m, pattern, 4);
                if (period <= 0) { throw std::runtime_error("invalid sliding-window period"); }
                sliding = il % period < period - 1;
            }
            layer.attention = sliding ? "sliding" : "full";
        } else if (info.arch->rule == layer_rule::kv_heads) {
            const int64_t heads = per_layer(prefix + "attention.head_count_kv",
                                            per_layer(prefix + "attention.head_count", 0, il), il);
            layer.attention = heads == 0 ? "recurrent" : swa ? "sliding" : "full";
        } else if (info.arch->rule == layer_rule::nemotron) {
            const int64_t heads = per_layer(prefix + "attention.head_count_kv",
                                            per_layer(prefix + "attention.head_count", 0, il), il);
            layer.attention = per_layer(prefix + "feed_forward_length", 0, il) != 0 ? "mlp"
                              : heads == 0 ? "recurrent" : "full";
        } else if (info.arch->rule == layer_rule::parallel) {
            layer.attention = "parallel";
        } else if (info.arch->rule == layer_rule::all_sliding) {
            layer.attention = "sliding";
            layer.compress = integer(m, prefix + "attention.compress_ratios", -1, il);
            if (layer.compress != 0 && layer.compress != 4 && layer.compress != 128) {
                throw std::runtime_error("DeepSeek V4 compress ratios must be 0, 4 or 128");
            }
            if (layer.compress == 4) { layer.lid_dim = integer(m, prefix + "attention.indexer.key_length"); }
        } else {
            layer.attention = "full";
        }
        if (layer.attention != "recurrent" && layer.attention != "mlp") {
            layer.n_head = per_layer(prefix + "attention.head_count", -1, il);
            if (layer.n_head <= 0) { throw std::runtime_error("invalid attention head count"); }
            layer.n_head_kv = per_layer(prefix + "attention.head_count_kv", layer.n_head, il);
            const bool sliding = layer.attention == "sliding" && info.arch->mla;
            layer.dim_k = integer(m, prefix + (sliding ? "attention.key_length_swa" : "attention.key_length"),
                                  embd / layer.n_head, il);
            layer.dim_v = integer(m, prefix + "attention.value_length", embd / layer.n_head, il);
            if (info.arch->mla) {
                // one latent "head" per cell; attention reads V as its first kv_lora elements, whatever
                // attention.value_length says (src/llama-kv-cache.cpp:229-233)
                layer.kv_lora = integer(m, prefix + (sliding ? "attention.kv_lora_rank_swa" : "attention.kv_lora_rank"));
                layer.dim_v = layer.kv_lora;
                if (layer.n_head_kv != 1 || layer.dim_k <= layer.kv_lora) {
                    throw std::runtime_error("layer " + std::to_string(il) + " is not an MLA row [c_kv | k_pe] of one KV head");
                }
            }
            if (layer.n_head_kv <= 0 || layer.n_head % layer.n_head_kv != 0 || layer.dim_k <= 0 || layer.dim_v <= 0
                    || layer.n_head * std::max(layer.dim_k, layer.dim_v) > (1 << 20)) {
                throw std::runtime_error("invalid attention dimensions at layer " + std::to_string(il));
            }
            if (indexer && layer.attention == "full") {
                const bool full = info.architecture == "deepseek32" ? true
                                  : has_key(m, types) ? per_layer(types, -1, il) != 0
                                  : info.architecture == "glm-dsa" && integer(m, prefix + "context_length") < 1048576;
                if (!has_key(m, types) && info.architecture != "deepseek32" && !full) {
                    throw std::runtime_error(info.architecture + " needs attention.indexer.types");
                }
                if (full) { layer.indexer = integer(m, prefix + "attention.indexer.key_length"); }
            }
        }
        info.layers.push_back(layer);
    }
    info.file_type = std::to_string(integer(m, "general.file_type", 0));
    return info;
}

// ---------------------------------------------------------------------------
// Capture targets

struct target {
    std::string key, name, filename;
    int layer = 0;
    int64_t heads = 0, dim = 0;
    int64_t row_dim = 0, column = 0;  // a composite row (MLA K): the row's width and this node's first column
    key_op op = key_op::any;
    bool two_d = false;
    bool observe_only = false;  // patch-score on a GPU: split the graph here without touching the values
    int seen = 0;
    std::string mismatch;
    // collect (the parts of a composite row share their file)
    std::shared_ptr<std::fstream> file;
    uint64_t header_bytes = 0;
    // patch-score
    std::vector<float> values;
    std::vector<uint8_t> applied;
    bool composite() const { return dim != row_dim; }
};

// The targets of one key and layer: one node, or the nodes of a composite row in column order.
static std::vector<std::unique_ptr<target>> make_targets(const model_info & info, const std::string & key,
                                                         int64_t layer) {
    if (std::find_if(capture_keys.begin(), capture_keys.end(), [&key](const char * k) { return key == k; })
            == capture_keys.end()) {
        throw std::runtime_error("unknown capture key: " + key);
    }
    const key_def * def = info.arch->find(key);
    if (!def) {
        throw std::runtime_error("capture key " + key + " is only defined for "
                                 + arch_list([&key](const arch_def & arch) { return arch.has_key(key); }));
    }
    if (!info.is_attention(layer)) {
        throw std::runtime_error("layer " + std::to_string(layer) + " is not an attention layer of this model");
    }
    const auto & l = info.layers[size_t(layer)];
    if (key == "ik" && l.indexer == 0) {
        throw std::runtime_error("layer " + std::to_string(layer) + " stores no indexer keys");
    }
    if (is_compressor_key(key) && l.compress != compressor_ratio(key)) {
        throw std::runtime_error("layer " + std::to_string(layer) + " has no " + key + " compressor");
    }
    int64_t heads, dim;
    if      (key == "k")   { heads = l.n_head_kv; dim = l.dim_k; }
    else if (key == "v")   { heads = l.n_head_kv; dim = l.dim_v; }
    else if (key == "q")   { heads = l.n_head;    dim = l.dim_k; }
    else if (key == "ik")  { heads = 1;           dim = l.indexer; }
    // [a | s]: an overlapping pool's a and s each span two blocks' halves (2 * head), HCA's one head
    else if (key == "csa") { heads = 1;           dim = 4 * l.dim_k; }
    else if (key == "hca") { heads = 1;           dim = 2 * l.dim_k; }
    else if (key == "lid") { heads = 1;           dim = 4 * l.lid_dim; }
    else                   { heads = l.n_head;    dim = l.dim_v; }
    // a composite row: the MLA latent [0, kv_lora) or the compressor's a [0, dim / 2) from the first node,
    // the rest from the tail
    const int64_t split = is_compressor_key(key) ? dim / 2 : l.kv_lora;
    std::vector<std::unique_ptr<target>> out;
    for (const node_def * node : {&def->node, &def->tail}) {
        if (!node->tensor) { continue; }
        auto t = std::make_unique<target>();
        t->key = key;
        t->layer = int(layer);
        t->name = std::string(node->tensor) + "-" + std::to_string(layer);
        t->filename = key + "-" + std::to_string(layer) + ".npy";
        t->op = node->op;
        t->two_d = node->two_d;
        t->heads = heads;
        t->row_dim = dim;
        t->dim = def->tail.tensor ? (node == &def->node ? split : dim - split) : dim;
        t->column = node == &def->node ? 0 : split;
        out.push_back(std::move(t));
    }
    return out;
}

static std::unique_ptr<target> make_target(const model_info & info, const std::string & key, int64_t layer) {
    auto targets = make_targets(info, key, layer);
    if (targets.size() != 1) {
        throw std::runtime_error("capture key " + key + " of " + info.architecture
                                 + " is assembled from two graph nodes and can only be collected");
    }
    return std::move(targets[0]);
}

static bool op_allowed(const target & t, const ggml_tensor * tensor) {
    switch (t.op) {
        case key_op::mul: return tensor->op == GGML_OP_MUL;
        case key_op::mul_mat: return tensor->op == GGML_OP_MUL_MAT;
        case key_op::reshape_or_view: return tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_VIEW;
        case key_op::view: return tensor->op == GGML_OP_VIEW;
        case key_op::concat: return tensor->op == GGML_OP_CONCAT;
        case key_op::norm_output:
            return tensor->op == GGML_OP_ADD || tensor->op == GGML_OP_MUL || tensor->op == GGML_OP_NORM;
        case key_op::sigmoid: return tensor->op == GGML_OP_UNARY && ggml_get_unary_op(tensor) == GGML_UNARY_OP_SIGMOID;
        case key_op::any: return true;
    }
    return false;
}

static bool shape_matches(const target & t, const ggml_tensor * tensor, int64_t n_tokens) {
    if (t.two_d) {
        return tensor->ne[0] == t.heads * t.dim && tensor->ne[1] == n_tokens && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }
    return tensor->ne[0] == t.dim && tensor->ne[1] == t.heads && tensor->ne[2] == n_tokens && tensor->ne[3] == 1;
}

static size_t element_offset(const target & t, const ggml_tensor * tensor, int64_t token, int64_t head, int64_t dim) {
    if (t.two_d) { return size_t(token * tensor->nb[1] + (head * t.dim + dim) * tensor->nb[0]); }
    return size_t(token * tensor->nb[2] + head * tensor->nb[1] + dim * tensor->nb[0]);
}

struct probe_state {
    std::vector<std::unique_ptr<target>> targets;
    std::unordered_map<std::string, target *> by_name;
    bool active = false, patching = false;
    int64_t n_tokens = 0, pos_start = 0;
    uint64_t row_base = 0;
    const std::vector<int32_t> * positions = nullptr;
    std::string error;
    std::vector<uint8_t> bytes;
    std::vector<float> rows;

    void index() {
        by_name.clear();
        for (auto & t : targets) {
            if (!by_name.emplace(t->name, t.get()).second) { throw std::runtime_error("duplicate capture " + t->name); }
        }
    }
};

static void collect_rows(probe_state & st, target & t, ggml_tensor * tensor) {
    const size_t size = ggml_nbytes(tensor);
    st.bytes.resize(size);
    ggml_backend_tensor_get(tensor, st.bytes.data(), 0, size);
    st.rows.resize(size_t(st.n_tokens * t.heads * t.dim));
    float * out = st.rows.data();
    for (int64_t token = 0; token < st.n_tokens; ++token) {
        for (int64_t head = 0; head < t.heads; ++head) {
            const size_t base = element_offset(t, tensor, token, head, 0);
            if (base + size_t((t.dim - 1) * tensor->nb[0]) + sizeof(float) > size) {
                throw std::runtime_error("invalid tensor strides for " + t.name);
            }
            if (tensor->nb[0] == sizeof(float)) {
                std::memcpy(out, st.bytes.data() + base, size_t(t.dim) * sizeof(float));
            } else {
                for (int64_t d = 0; d < t.dim; ++d) {
                    std::memcpy(out + d, st.bytes.data() + base + size_t(d * tensor->nb[0]), sizeof(float));
                }
            }
            for (int64_t d = 0; d < t.dim; ++d) {
                if (!std::isfinite(out[d])) { throw std::runtime_error("nonfinite captured value in " + t.name); }
            }
            out += t.dim;
        }
    }
    const uint64_t row_bytes = uint64_t(t.heads * t.row_dim) * sizeof(float);
    const uint64_t first = t.header_bytes + (st.row_base + uint64_t(st.pos_start)) * row_bytes;
    if (!t.composite()) {
        t.file->seekp(std::streamoff(first));
        t.file->write(reinterpret_cast<const char *>(st.rows.data()), std::streamsize(st.rows.size() * sizeof(float)));
        return;
    }
    // this node's columns of every row (one head: make_targets builds composites for MLA and compressor rows only)
    for (int64_t token = 0; token < st.n_tokens; ++token) {
        t.file->seekp(std::streamoff(first + uint64_t(token) * row_bytes + uint64_t(t.column) * sizeof(float)));
        t.file->write(reinterpret_cast<const char *>(st.rows.data() + size_t(token * t.dim)),
                      std::streamsize(size_t(t.dim) * sizeof(float)));
    }
}

static void patch_rows(probe_state & st, target & t, ggml_tensor * tensor) {
    if (t.observe_only) { return; }
    const auto & positions = *st.positions;
    const size_t size = ggml_nbytes(tensor);
    const auto first = std::lower_bound(positions.begin(), positions.end(), int32_t(st.pos_start));
    const auto last = std::lower_bound(positions.begin(), positions.end(), int32_t(st.pos_start + st.n_tokens));
    if (first == last) { return; }
    if (!ggml_backend_buffer_is_host(tensor->buffer)) {
        // Device memory: one read and one write of the byte span holding the patched rows (bytes of other
        // rows, or of fused-QKV neighbours inside a strided view, are written back unchanged).
        size_t lo = size, hi = 0;
        for (auto it = first; it != last; ++it) {
            for (int64_t head = 0; head < t.heads; ++head) {
                const size_t base = element_offset(t, tensor, *it - st.pos_start, head, 0);
                const size_t end = base + size_t((t.dim - 1) * tensor->nb[0]) + sizeof(float);
                if (end > size) { throw std::runtime_error("invalid tensor strides for " + t.name); }
                lo = std::min(lo, base);
                hi = std::max(hi, end);
            }
        }
        st.bytes.resize(hi - lo);
        ggml_backend_tensor_get(tensor, st.bytes.data(), lo, hi - lo);
        for (auto it = first; it != last; ++it) {
            const size_t index = size_t(it - positions.begin());
            if (t.applied[index]++ != 0) { throw std::runtime_error("patch row applied twice in " + t.name); }
            for (int64_t head = 0; head < t.heads; ++head) {
                const float * source = t.values.data() + (index * size_t(t.heads) + size_t(head)) * size_t(t.dim);
                const size_t base = element_offset(t, tensor, *it - st.pos_start, head, 0) - lo;
                for (int64_t d = 0; d < t.dim; ++d) {
                    std::memcpy(st.bytes.data() + base + size_t(d * tensor->nb[0]), source + d, sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(tensor, st.bytes.data(), lo, hi - lo);
        return;
    }
    for (auto it = first; it != last; ++it) {
        const size_t index = size_t(it - positions.begin());
        if (t.applied[index]++ != 0) { throw std::runtime_error("patch row applied twice in " + t.name); }
        const int64_t token = *it - st.pos_start;
        for (int64_t head = 0; head < t.heads; ++head) {
            const float * source = t.values.data() + (index * size_t(t.heads) + size_t(head)) * size_t(t.dim);
            const size_t base = element_offset(t, tensor, token, head, 0);
            if (base + size_t((t.dim - 1) * tensor->nb[0]) + sizeof(float) > size) {
                throw std::runtime_error("invalid tensor strides for " + t.name);
            }
            if (tensor->nb[0] == sizeof(float)) {
                ggml_backend_tensor_set(tensor, source, base, size_t(t.dim) * sizeof(float));
            } else {
                for (int64_t d = 0; d < t.dim; ++d) {
                    ggml_backend_tensor_set(tensor, source + d, base + size_t(d * tensor->nb[0]), sizeof(float));
                }
            }
        }
    }
}

static bool probe_callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & st = *static_cast<probe_state *>(opaque);
    if (!st.active) { return !ask; }
    const auto found = st.by_name.find(tensor->name);
    if (found == st.by_name.end()) { return !ask; }
    target & t = *found->second;
    const bool selected = tensor->type == GGML_TYPE_F32 && op_allowed(t, tensor) && shape_matches(t, tensor, st.n_tokens);
    if (ask) {
        if (!selected) {
            t.mismatch = std::string("op ") + ggml_op_desc(tensor) + " type " + ggml_type_name(tensor->type) + " ne ["
                + std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1]) + ","
                + std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) + "]";
        }
        return selected;
    }
    if (!selected) { return true; }
    try {
        if (t.seen++ != 0) { throw std::runtime_error("tensor " + t.name + " matched more than once in one ubatch"); }
        if (st.patching) { patch_rows(st, t, tensor); } else { collect_rows(st, t, tensor); }
        return true;
    } catch (const std::exception & exc) {
        if (st.error.empty()) { st.error = exc.what(); }
        return false;
    }
}

// ---------------------------------------------------------------------------
// Documents and tokenization

struct document {
    std::string id, text;
    json record;
    std::vector<llama_token> tokens;
    std::vector<int64_t> byte_end;
    std::vector<uint8_t> special;
    uint64_t first_row = 0;
};

static bool is_hex64(const json & value) {
    if (!value.is_string()) { return false; }
    const auto & s = value.get_ref<const std::string &>();
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static std::vector<document> read_documents(const fs::path & path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open input documents"); }
    std::vector<document> documents;
    std::set<std::string> seen;
    size_t line_number = 0;
    for (std::string line; std::getline(input, line);) {
        ++line_number;
        const std::string where = "input line " + std::to_string(line_number);
        json record;
        try { record = parse_json_strict(line); } catch (const std::exception & exc) {
            throw std::runtime_error(where + ": invalid JSON: " + exc.what());
        }
        if (!record.is_object() || !record.contains("document_id") || !is_hex64(record["document_id"])
                || !record.contains("text") || !record["text"].is_string()) {
            throw std::runtime_error(where + ": document_id (64 lowercase hex) and text are required");
        }
        document doc;
        doc.id = record["document_id"].get<std::string>();
        doc.text = record["text"].get<std::string>();
        if (doc.text.empty() || doc.text.size() > 4 * 1024 * 1024) {
            throw std::runtime_error(where + ": text must be nonempty and at most 4 MiB of UTF-8");
        }
        if (!seen.insert(doc.id).second) { throw std::runtime_error(where + ": duplicate document_id"); }
        doc.record = std::move(record);
        documents.push_back(std::move(doc));
    }
    if (input.bad() || documents.empty()) { throw std::runtime_error("empty or unreadable input documents"); }
    return documents;
}

static void tokenize(const llama_vocab * vocab, document & doc, int ctx) {
    const int32_t special_mask = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN;
    std::vector<llama_token> tokens(doc.text.size() + 1);
    int count = llama_tokenize(vocab, doc.text.data(), int(doc.text.size()), tokens.data(), int(tokens.size()), false, true);
    if (count < 0 && count != INT32_MIN) {
        tokens.resize(size_t(-int64_t(count)));
        count = llama_tokenize(vocab, doc.text.data(), int(doc.text.size()), tokens.data(), int(tokens.size()), false, true);
    }
    if (count < 0) { throw std::runtime_error("tokenization failed for document " + doc.id); }
    tokens.resize(size_t(count));
    if (tokens.empty()) { throw std::runtime_error("document produced no tokens: " + doc.id); }
    if (tokens.size() > size_t(ctx)) {
        throw std::runtime_error("document " + doc.id + " has " + std::to_string(tokens.size())
                                 + " tokens, exceeding --ctx; no truncation is performed");
    }
    std::string reconstructed;
    std::vector<char> piece(256);
    for (const auto token : tokens) {
        int length = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, true);
        if (length < 0) {
            piece.resize(size_t(-int64_t(length)));
            length = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, true);
        }
        if (length <= 0) { throw std::runtime_error("empty or invalid token piece in document " + doc.id); }
        reconstructed.append(piece.data(), size_t(length));
        doc.byte_end.push_back(int64_t(reconstructed.size()));
        doc.special.push_back((llama_vocab_get_attr(vocab, token) & special_mask) != 0 ? 1 : 0);
    }
    if (reconstructed != doc.text) {
        throw std::runtime_error("token pieces do not reproduce the exact UTF-8 input of document " + doc.id);
    }
    doc.tokens = std::move(tokens);
}

static std::string token_sha256(const std::vector<llama_token> & tokens) {
    std::vector<uint8_t> bytes;
    for (const int32_t token : tokens) {
        const uint32_t v = uint32_t(token);
        for (int i = 0; i < 4; ++i) { bytes.push_back(uint8_t(v >> (8 * i))); }
    }
    return sha256_hex(bytes.data(), bytes.size());
}

// ---------------------------------------------------------------------------
// Runtime

// llama.cpp log lines kept for the report of a GPU run; every line is still printed to stderr.
struct runtime_log {
    std::string offloaded, graph_splits;
};

static void log_callback(ggml_log_level, const char * text, void * opaque) {
    std::fputs(text, stderr);
    std::fflush(stderr);
    auto & log = *static_cast<runtime_log *>(opaque);
    const std::string line(text);
    std::smatch match;
    static const std::regex offloaded("offloaded ([0-9]+/[0-9]+) layers to GPU");
    static const std::regex splits("graph splits = ([^\n]*)");
    if (std::regex_search(line, match, offloaded)) { log.offloaded = match[1].str(); }
    if (std::regex_search(line, match, splits)) { log.graph_splits = match[1].str(); }
}

static ggml_backend_dev_t vulkan_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg && std::string(ggml_backend_reg_name(reg)) == "Vulkan") { return dev; }
    }
    throw std::runtime_error("no Vulkan device; use a probe built against a runtime with GGML_VULKAN=ON and GGML_BACKEND_DL=OFF");
}

struct runtime {
    runtime_log log;
    ggml_backend_dev_t device = nullptr;  // the GPU of a --device vulkan run
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    llama_batch batch{};
    bool batch_ready = false;
    ~runtime() {
        if (batch_ready) { llama_batch_free(batch); }
        ctx.reset();
        model.reset();
        if (device) { llama_log_set(nullptr, nullptr); }
    }
};

static void load_runtime(runtime & rt, const options & args, const model_info & info, probe_state * state) {
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    if (!devices[0]) { throw std::runtime_error("CPU backend unavailable; use a runtime with GGML_BACKEND_DL=OFF"); }
    // tokenize (state == nullptr) loads the vocabulary only and never needs a device
    const bool gpu = args.gpu() && state != nullptr;
    if (gpu) {
        rt.device = devices[0] = vulkan_device();
        llama_log_set(log_callback, &rt.log);
    }
    auto model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = gpu ? -1 : 0;  // -1: every layer and the output layer
    if (gpu) {
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
    }
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.use_extra_bufts = false;
    model_params.load_mtp = false;
    model_params.vocab_only = state == nullptr;
    rt.model.reset(llama_model_load_from_file(args.model.string().c_str(), model_params));
    if (!rt.model) { throw std::runtime_error("model load failed"); }
    if (state == nullptr) { return; }
    if (llama_model_n_layer(rt.model.get()) != info.n_layer) {
        throw std::runtime_error("runtime layer count disagrees with GGUF metadata");
    }
    if (gpu) {
        const auto slash = rt.log.offloaded.find('/');
        if (slash == std::string::npos || rt.log.offloaded.substr(0, slash) != rt.log.offloaded.substr(slash + 1)) {
            throw std::runtime_error("not every layer was offloaded to the GPU (llama.cpp reported \""
                                     + rt.log.offloaded + "\")");
        }
    }
    auto params = llama_context_default_params();
    params.n_ctx = uint32_t(args.ctx);
    params.n_batch = params.n_ubatch = uint32_t(args.batch);
    params.n_seq_max = 1;
    params.n_outputs_max = 0;
    params.n_outputs_max_per_seq = 0;
    params.n_threads = params.n_threads_batch = args.threads;
    params.type_k = params.type_v = args.cache_type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
    params.flash_attn_type = args.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = params.op_offload = gpu;
    params.swa_full = args.swa_full;
    params.cb_eval = probe_callback;
    params.cb_eval_user_data = state;
    rt.ctx.reset(llama_init_from_model(rt.model.get(), params));
    if (!rt.ctx) { throw std::runtime_error("context creation failed"); }
    rt.batch = llama_batch_init(args.batch, 0, 1);
    rt.batch_ready = true;
}

// Decode one document from position 0; `want` marks positions whose logits are needed.
// Only the ubatches covering positions [0, n_needed) run. Ubatch boundaries stay those of a
// whole-document decode, so every executed ubatch computes exactly what it would have;
// later tokens cannot influence earlier positions. Returns the number of decoded tokens.
static size_t decode_document(runtime & rt, probe_state & st, const options & args, const document & doc,
                              const std::vector<uint8_t> & want,
                              const std::function<void(int, int64_t)> & on_logits, size_t n_needed) {
    llama_synchronize(rt.ctx.get());
    llama_memory_clear(llama_get_memory(rt.ctx.get()), true);
    const size_t n = doc.tokens.size();
    if (n_needed == 0 || n_needed > n) { throw std::runtime_error("invalid decode length"); }
    size_t decoded = 0;
    for (size_t start = 0; start < n_needed; start += size_t(args.batch)) {
        auto & batch = rt.batch;
        batch.n_tokens = int(std::min(size_t(args.batch), n - start));
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = doc.tokens[start + i];
            batch.pos[i] = llama_pos(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want.empty() ? i == batch.n_tokens - 1 : want[start + i] != 0;
        }
        for (auto & t : st.targets) { t->seen = 0; t->mismatch.clear(); }
        st.n_tokens = batch.n_tokens;
        st.pos_start = int64_t(start);
        st.row_base = doc.first_row;
        st.error.clear();
        st.active = !st.targets.empty();
        const int status = llama_decode(rt.ctx.get(), batch);
        llama_synchronize(rt.ctx.get());
        st.active = false;
        if (!st.error.empty()) { throw std::runtime_error(st.error); }
        if (status != 0) { throw std::runtime_error("llama_decode failed with status " + std::to_string(status)); }
        for (const auto & t : st.targets) {
            if (t->seen != 1) {
                throw std::runtime_error("tensor " + t->name + " was captured " + std::to_string(t->seen)
                                         + " times in a ubatch of " + std::to_string(batch.n_tokens) + " tokens"
                                         + (t->mismatch.empty() ? "" : "; nearest candidate: " + t->mismatch));
            }
        }
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] && !want.empty()) { on_logits(i, int64_t(start) + i); }
        }
        decoded += size_t(batch.n_tokens);
    }
    return decoded;
}

// ---------------------------------------------------------------------------
// Output helpers

struct staging {
    fs::path temp, final_path;
    bool created = false, published = false;
    explicit staging(const fs::path & output) : final_path(fs::absolute(output)) {
        if (fs::exists(final_path) || fs::is_symlink(final_path)) { throw std::runtime_error("output already exists"); }
        const auto parent = final_path.parent_path();
        if (!fs::is_directory(parent)) { throw std::runtime_error("output parent directory does not exist"); }
        const std::string prefix = "." + final_path.filename().string() + ".partial-";
        for (const auto & entry : fs::directory_iterator(parent)) {
            if (entry.path().filename().string().rfind(prefix, 0) == 0) {
                std::cerr << "kv-probe: warning: leftover staging directory from an interrupted run: "
                          << entry.path().string() << '\n';
            }
        }
        std::string pattern = (parent / (prefix + "XXXXXX")).string();
        if (!mkdtemp(pattern.data())) { throw std::runtime_error("cannot create staging directory"); }
        temp = pattern;
        created = true;
        const mode_t mask = umask(0);
        umask(mask);
        fs::permissions(temp, static_cast<fs::perms>(0777 & ~mask));
    }
    void publish() {
        if (fs::exists(final_path) || fs::is_symlink(final_path)) { throw std::runtime_error("output appeared while running"); }
        fs::rename(temp, final_path);
        published = true;
    }
    ~staging() {
        if (created && !published) {
            std::error_code ignored;
            fs::remove_all(temp, ignored);
        }
    }
};

static void write_json(const fs::path & path, const json & value) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << value.dump(2) << '\n';
    out.close();
}

static json layers_json(const model_info & info) {
    json layers = json::array();
    for (int64_t il = 0; il < info.n_layer; ++il) {
        const auto & layer = info.layers[size_t(il)];
        layers.push_back({{"layer", il}, {"attention", layer.attention}});
        if (layer.indexer) { layers.back()["indexer"] = layer.indexer; }  // DSA: the layer stores indexer keys
        if (layer.compress) { layers.back()["compress"] = layer.compress; }  // DeepSeek V4: pooled caches
    }
    return layers;
}

static json runtime_settings(const options & args) {
    json settings = {{"ctx", args.ctx}, {"batch", args.batch}, {"threads", args.threads}, {"cache_type", args.cache_type},
                     {"flash_attention", args.flash_attn}, {"swa_full", args.swa_full}, {"add_special", false},
                     {"parse_special", true}};
    if (args.gpu()) { settings["device"] = args.device; }  // absent: the CPU (reports of CPU runs are unchanged)
    return settings;
}

// GPU runs only: which device ran the model and what llama.cpp reported about the placement.
static void add_device_report(json & report, const runtime & rt) {
    if (!rt.device) { return; }
    report["device"] = {{"backend", "vulkan"}, {"name", ggml_backend_dev_name(rt.device)},
                        {"description", ggml_backend_dev_description(rt.device)},
                        {"offloaded_layers", rt.log.offloaded}, {"graph_splits", rt.log.graph_splits},
                        {"offload_kqv", true}, {"op_offload", true}};
}

static void write_token_arrays(const fs::path & dir, const std::vector<document> & documents, uint64_t rows) {
    std::vector<int32_t> ids;
    std::vector<int64_t> ends;
    std::vector<uint8_t> special;
    for (const auto & doc : documents) {
        ids.insert(ids.end(), doc.tokens.begin(), doc.tokens.end());
        ends.insert(ends.end(), doc.byte_end.begin(), doc.byte_end.end());
        special.insert(special.end(), doc.special.begin(), doc.special.end());
    }
    const std::vector<int64_t> shape = {int64_t(rows)};
    write_npy(dir / "token_id.npy", "<i4", shape, ids.data(), ids.size() * sizeof(int32_t));
    write_npy(dir / "byte_end.npy", "<i8", shape, ends.data(), ends.size() * sizeof(int64_t));
    write_npy(dir / "special.npy", "|u1", shape, special.data(), special.size());
}

static json documents_json(const std::vector<document> & documents) {
    json out = json::array();
    for (const auto & doc : documents) {
        out.push_back({{"document_id", doc.id}, {"n_tokens", doc.tokens.size()}, {"first_row", doc.first_row}});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Modes

static std::vector<std::unique_ptr<target>> parse_capture_spec(const options & args, const model_info & info) {
    // default: K, and V and the DeepSeek V4 compressor rows where the architecture has them
    json spec = {{"k", "all"}};
    for (const char * key : {"v", "csa", "hca", "lid"}) {
        if (info.arch->has_key(key)) { spec[key] = "all"; }
    }
    if (!args.capture_spec.empty()) {
        std::ifstream in(args.capture_spec);
        if (!in) { throw std::runtime_error("cannot open capture spec"); }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (in.bad()) { throw std::runtime_error("cannot read capture spec"); }
        try { spec = parse_json_strict(text); } catch (const std::exception & exc) {
            throw std::runtime_error(std::string("invalid capture spec JSON: ") + exc.what());
        }
    }
    if (!spec.is_object() || spec.empty()) { throw std::runtime_error("capture spec must be a nonempty JSON object"); }
    std::vector<std::unique_ptr<target>> targets;
    for (const char * key : capture_keys) {
        if (!spec.contains(key)) { continue; }
        if (!info.arch->has_key(key)) {
            throw std::runtime_error(std::string("capture key ") + key + " is only defined for "
                                     + arch_list([&key](const arch_def & arch) { return arch.has_key(key); }));
        }
        const auto & value = spec[key];
        std::vector<int64_t> layers;
        if (value.is_string() && value.get<std::string>() == "all") {
            // every attention layer; the indexer keys and compressor rows of the layers that have them
            for (int64_t il = 0; il < info.n_layer; ++il) {
                const auto & l = info.layers[size_t(il)];
                if (info.is_attention(il) && (std::string(key) != "ik" || l.indexer)
                        && (!is_compressor_key(key) || l.compress == compressor_ratio(key))) {
                    layers.push_back(il);
                }
            }
        } else if (value.is_array() && !value.empty()) {
            std::set<int64_t> unique;
            for (const auto & item : value) {
                if (!item.is_number_integer()) { throw std::runtime_error(std::string("capture layers for ") + key + " must be integers"); }
                const int64_t il = item.get<int64_t>();
                if (!unique.insert(il).second) { throw std::runtime_error(std::string("duplicate capture layer for ") + key); }
            }
            layers.assign(unique.begin(), unique.end());
        } else {
            throw std::runtime_error(std::string("capture spec value for ") + key + " must be \"all\" or a nonempty layer list");
        }
        for (const auto il : layers) {
            for (auto & t : make_targets(info, key, il)) { targets.push_back(std::move(t)); }
        }
    }
    for (const auto & item : spec.items()) {
        if (std::find_if(capture_keys.begin(), capture_keys.end(), [&item](const char * k) { return item.key() == k; })
                == capture_keys.end()) {
            throw std::runtime_error("unknown capture key: " + item.key());
        }
    }
    return targets;
}

static int run_tokenize(const options & args, const model_info & info, std::vector<document> & documents) {
    staging stage(args.output);
    runtime rt;
    load_runtime(rt, args, info, nullptr);
    uint64_t rows = 0;
    for (auto & doc : documents) {
        tokenize(llama_model_get_vocab(rt.model.get()), doc, args.ctx);
        doc.first_row = rows;
        rows += doc.tokens.size();
    }
    write_token_arrays(stage.temp, documents, rows);
    json report = {{"format_version", 2}, {"mode", "tokenize"}, {"architecture", info.architecture},
                   {"file_type", info.file_type}, {"n_layer", info.n_layer}, {"layers", layers_json(info)},
                   {"settings", {{"ctx", args.ctx}, {"add_special", false}, {"parse_special", true}}},
                   {"documents", documents_json(documents)}, {"rows", rows}};
    write_json(stage.temp / "report.json", report);
    stage.publish();
    std::cerr << "tokenized " << documents.size() << " documents, " << rows << " tokens\n";
    return 0;
}

static int run_collect(const options & args, const model_info & info, std::vector<document> & documents) {
    probe_state st;
    st.targets = parse_capture_spec(args, info);
    st.index();
    staging stage(args.output);
    runtime rt;
    load_runtime(rt, args, info, &st);
    uint64_t rows = 0;
    for (auto & doc : documents) {
        tokenize(llama_model_get_vocab(rt.model.get()), doc, args.ctx);
        doc.first_row = rows;
        rows += doc.tokens.size();
    }
    write_token_arrays(stage.temp, documents, rows);
    json captures = json::array();
    std::map<std::string, target *> first_part;     // file name -> the target that created it
    for (auto & t : st.targets) {
        const auto found = first_part.find(t->filename);
        if (found != first_part.end()) {            // the tail of a composite row: same file, its own columns
            t->file = found->second->file;
            t->header_bytes = found->second->header_bytes;
            captures.back()["tensor"] = captures.back()["tensor"].get<std::string>() + "+" + t->name;
            continue;
        }
        first_part[t->filename] = t.get();
        const auto path = stage.temp / t->filename;
        const auto header = npy_header("<f4", {int64_t(rows), t->heads, t->row_dim});
        {
            std::ofstream create(path, std::ios::binary);
            create.exceptions(std::ios::badbit | std::ios::failbit);
            create.write(header.data(), std::streamsize(header.size()));
        }
        fs::resize_file(path, header.size() + rows * uint64_t(t->heads * t->row_dim) * sizeof(float));
        t->header_bytes = header.size();
        t->file = std::make_shared<std::fstream>(path, std::ios::binary | std::ios::in | std::ios::out);
        t->file->exceptions(std::ios::badbit | std::ios::failbit);
        captures.push_back({{"key", t->key}, {"layer", t->layer}, {"tensor", t->name}, {"heads", t->heads},
                            {"dim", t->row_dim}, {"file", t->filename}, {"dtype", "<f4"}});
    }
    const std::vector<uint8_t> no_logits;
    for (const auto & doc : documents) {
        decode_document(rt, st, args, doc, no_logits, [](int, int64_t) {}, doc.tokens.size());
        std::cerr << "captured document " << doc.id.substr(0, 12) << ": " << doc.tokens.size() << " tokens\n";
    }
    for (auto & t : st.targets) {
        if (t->file->is_open()) { t->file->close(); }
    }
    json report = {{"format_version", 2}, {"mode", "collect"}, {"architecture", info.architecture},
                   {"file_type", info.file_type}, {"n_layer", info.n_layer}, {"layers", layers_json(info)},
                   {"settings", runtime_settings(args)}, {"captures", captures},
                   {"documents", documents_json(documents)}, {"rows", rows}};
    add_device_report(report, rt);
    write_json(stage.temp / "report.json", report);
    stage.publish();
    return 0;
}

struct condition {
    std::string name;
    fs::path patch_dir;
    std::vector<int32_t> positions;
    std::vector<std::pair<std::string, int64_t>> arrays;  // key, layer
};

struct score_plan {
    std::vector<int64_t> positions;  // scored token positions p (logits taken at p - 1)
    int64_t n_eligible = 0;          // positions inside score_ranges before the --max-scored cap
    std::vector<condition> conditions;
};

// Scored positions from expected_token_sha256 and score_ranges (shared by patch-score and seed-score).
static score_plan plan_scored_positions(const document & doc, const options & args) {
    const auto & r = doc.record;
    if (!r.contains("expected_token_sha256") || !is_hex64(r["expected_token_sha256"])) {
        throw std::runtime_error("document " + doc.id + ": expected_token_sha256 (64 lowercase hex) is required");
    }
    if (token_sha256(doc.tokens) != r["expected_token_sha256"].get<std::string>()) {
        throw std::runtime_error("document " + doc.id + ": token ids do not match expected_token_sha256");
    }
    if (!r.contains("score_ranges") || !r["score_ranges"].is_array() || r["score_ranges"].empty()) {
        throw std::runtime_error("document " + doc.id + ": score_ranges must be a nonempty list");
    }
    std::vector<std::pair<int64_t, int64_t>> ranges;
    for (const auto & range : r["score_ranges"]) {
        if (!range.is_array() || range.size() != 2 || !range[0].is_number_integer() || !range[1].is_number_integer()) {
            throw std::runtime_error("document " + doc.id + ": score range must be [start_byte, end_byte]");
        }
        const int64_t start = range[0].get<int64_t>(), end = range[1].get<int64_t>();
        if (start < 0 || start >= end || end > int64_t(doc.text.size())) {
            throw std::runtime_error("document " + doc.id + ": score range outside the text");
        }
        ranges.emplace_back(start, end);
    }
    score_plan plan;
    for (size_t p = 1; p < doc.tokens.size(); ++p) {
        const int64_t start = doc.byte_end[p - 1], end = doc.byte_end[p];
        for (const auto & range : ranges) {
            if (range.first <= start && end <= range.second) {
                if (plan.positions.size() < size_t(args.max_scored)) { plan.positions.push_back(int64_t(p)); }
                ++plan.n_eligible;
                break;
            }
        }
    }
    if (plan.positions.empty()) { throw std::runtime_error("document " + doc.id + " has no scored tokens"); }
    return plan;
}

static score_plan plan_document(const model_info & info, const document & doc, const options & args) {
    const auto & r = doc.record;
    score_plan plan = plan_scored_positions(doc, args);
    if (!r.contains("conditions") || !r["conditions"].is_array() || r["conditions"].empty()) {
        throw std::runtime_error("document " + doc.id + ": conditions must be a nonempty list");
    }
    static const std::regex name_pattern("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}");
    static const std::regex array_pattern("(k|v)-(0|[1-9][0-9]{0,5})\\.npy");
    std::set<std::string> names;
    for (const auto & item : r["conditions"]) {
        if (!item.is_object() || !item.contains("name") || !item["name"].is_string() || !item.contains("patch_dir")) {
            throw std::runtime_error("document " + doc.id + ": condition needs name and patch_dir");
        }
        condition c;
        c.name = item["name"].get<std::string>();
        if (!std::regex_match(c.name, name_pattern) || !names.insert(c.name).second) {
            throw std::runtime_error("document " + doc.id + ": invalid or duplicate condition name " + c.name);
        }
        const bool first = plan.conditions.empty();
        if (first != (c.name == "native") || (first && !item["patch_dir"].is_null())) {
            throw std::runtime_error("document " + doc.id + ": the first condition must be native with a null patch_dir");
        }
        if (!item["patch_dir"].is_null()) {
            if (!item["patch_dir"].is_string()) { throw std::runtime_error("patch_dir must be null or a path"); }
            c.patch_dir = item["patch_dir"].get<std::string>();
            if (!c.patch_dir.is_absolute() || !fs::is_directory(c.patch_dir)) {
                throw std::runtime_error("patch_dir must be an existing absolute directory: " + c.patch_dir.string());
            }
            const auto pos_info = read_npy_header(c.patch_dir / "positions.npy");
            if (pos_info.descr != "<i4" || pos_info.shape.size() != 1) {
                throw std::runtime_error("positions.npy must be int32 [P] in " + c.patch_dir.string());
            }
            c.positions = read_npy_data<int32_t>(c.patch_dir / "positions.npy", pos_info);
            for (size_t i = 0; i < c.positions.size(); ++i) {
                if (c.positions[i] < 0 || size_t(c.positions[i]) >= doc.tokens.size()
                        || (i > 0 && c.positions[i] <= c.positions[i - 1])) {
                    throw std::runtime_error("patch positions must be strictly increasing token positions of the document: "
                                             + c.patch_dir.string());
                }
            }
            for (const auto & entry : fs::directory_iterator(c.patch_dir)) {
                const auto file = entry.path().filename().string();
                if (file == "positions.npy" || entry.path().extension() != ".npy") { continue; }
                std::smatch match;
                if (!std::regex_match(file, match, array_pattern)) {
                    throw std::runtime_error("unexpected patch array " + file + " in " + c.patch_dir.string());
                }
                const int64_t layer = std::stoll(match[2].str());
                const auto probe = make_target(info, match[1].str(), layer);
                const auto array_info = read_npy_header(entry.path());
                if (array_info.descr != "<f4" || array_info.shape != std::vector<int64_t>{int64_t(c.positions.size()), probe->heads, probe->dim}) {
                    throw std::runtime_error("patch array " + file + " must be float32 [P, " + std::to_string(probe->heads)
                                             + ", " + std::to_string(probe->dim) + "] in " + c.patch_dir.string());
                }
                c.arrays.emplace_back(match[1].str(), layer);
            }
            std::sort(c.arrays.begin(), c.arrays.end());
        }
        plan.conditions.push_back(std::move(c));
    }
    return plan;
}

static void load_patch_targets(probe_state & st, const model_info & info, const condition & c) {
    st.targets.clear();
    for (const auto & [key, layer] : c.arrays) {
        auto t = make_target(info, key, layer);
        const auto path = c.patch_dir / t->filename;
        const auto header = read_npy_header(path);
        if (header.descr != "<f4" || header.shape != std::vector<int64_t>{int64_t(c.positions.size()), t->heads, t->dim}) {
            throw std::runtime_error("patch array changed after validation: " + path.string());
        }
        t->values = read_npy_data<float>(path, header);
        for (const float value : t->values) {
            if (!std::isfinite(value)) { throw std::runtime_error("nonfinite patch value in " + path.string()); }
        }
        t->applied.assign(c.positions.size(), 0);
        st.targets.push_back(std::move(t));
    }
    st.index();
}

// GPU backends fuse neighbouring ops differently when the eval callback splits the graph, which can change
// low-order bits. On a GPU every patch-score condition therefore splits at the k and v nodes of every
// attention layer, as a default collect does, and only the patched ones are rewritten.
static void add_split_points(probe_state & st, const model_info & info) {
    for (const char * key : {"k", "v"}) {
        if (!info.arch->has_key(key)) { continue; }
        for (int64_t il = 0; il < info.n_layer; ++il) {
            if (!info.is_attention(il)) { continue; }
            for (auto & t : make_targets(info, key, il)) {      // both nodes of a composite row
                if (st.by_name.count(t->name)) { continue; }
                t->observe_only = true;
                st.targets.push_back(std::move(t));
            }
        }
    }
    st.index();
}

struct logit_stats {
    double lse = 0;
    int64_t argmax = 0;
};

static logit_stats summarize_logits(const float * logits, int64_t n_vocab) {
    logit_stats s;
    double best = -INFINITY;
    for (int64_t v = 0; v < n_vocab; ++v) {
        if (!std::isfinite(logits[v])) { throw std::runtime_error("nonfinite logit"); }
        if (logits[v] > best) { best = logits[v]; s.argmax = v; }
    }
    double sum = 0;
    for (int64_t v = 0; v < n_vocab; ++v) { sum += std::exp(double(logits[v]) - best); }
    s.lse = best + std::log(sum);
    return s;
}

struct native_reference {
    std::vector<float> logits;  // [n_scored, n_vocab]
    std::vector<logit_stats> stats;
};

struct condition_scores {
    std::vector<float> scores;  // [n_scored, 3]: nll, kl(native || condition), top-1 agreement
    std::vector<uint8_t> filled;
    double nll_sum = 0, kl_sum = 0, agree_sum = 0;
    explicit condition_scores(size_t n_scored) : scores(n_scored * 3, 0.0f), filled(n_scored, 0) {}
};

// Score the logits row of batch_index, which predicts the token at position + 1.
static void score_logits(llama_context * ctx, const document & doc, const std::vector<int64_t> & slot, int64_t n_vocab,
                         bool native, native_reference & reference_logits, condition_scores & cs,
                         int batch_index, int64_t position) {
    const int64_t index = slot[size_t(position)];
    const float * logits = llama_get_logits_ith(ctx, batch_index);
    if (index < 0 || !logits || cs.filled[size_t(index)]++) { throw std::runtime_error("unexpected logits row"); }
    const auto target_token = doc.tokens[size_t(position + 1)];
    const auto stats = summarize_logits(logits, n_vocab);
    double nll = stats.lse - double(logits[target_token]);
    double kl = 0, agree = 1;
    if (native) {
        std::memcpy(reference_logits.logits.data() + size_t(index) * size_t(n_vocab), logits, size_t(n_vocab) * sizeof(float));
        reference_logits.stats[size_t(index)] = stats;
    } else {
        const float * reference = reference_logits.logits.data() + size_t(index) * size_t(n_vocab);
        const auto & ref = reference_logits.stats[size_t(index)];
        for (int64_t v = 0; v < n_vocab; ++v) {
            const double lp_ref = double(reference[v]) - ref.lse;
            const double lp = double(logits[v]) - stats.lse;
            kl += std::exp(lp_ref) * (lp_ref - lp);
        }
        agree = stats.argmax == ref.argmax ? 1 : 0;
    }
    if (!std::isfinite(nll) || !std::isfinite(kl)) { throw std::runtime_error("nonfinite score"); }
    cs.scores[size_t(index) * 3 + 0] = float(nll);
    cs.scores[size_t(index) * 3 + 1] = float(kl);
    cs.scores[size_t(index) * 3 + 2] = float(agree);
    cs.nll_sum += nll;
    cs.kl_sum += kl;
    cs.agree_sum += agree;
}

static json condition_json(const std::string & name, const condition_scores & cs, size_t n_scored) {
    return {{"name", name}, {"nll_mean", cs.nll_sum / double(n_scored)},
            {"kl_mean", cs.kl_sum / double(n_scored)}, {"top1_agree", cs.agree_sum / double(n_scored)}};
}

static json scored_document_json(const document & doc, const score_plan & plan, json conditions) {
    return {{"document_id", doc.id}, {"n_tokens", doc.tokens.size()}, {"n_scored", plan.positions.size()},
            {"n_eligible", plan.n_eligible}, {"scored_truncated", plan.n_eligible > int64_t(plan.positions.size())},
            {"scored_positions", plan.positions}, {"conditions", conditions}};
}

static void finish_condition(const condition_scores & cs) {
    if (std::find(cs.filled.begin(), cs.filled.end(), 0) != cs.filled.end()) { throw std::runtime_error("missing logits for a scored token"); }
}

static int run_patch_score(const options & args, const model_info & info, std::vector<document> & documents) {
    probe_state st;
    st.patching = true;
    staging stage(args.output);
    runtime rt;
    load_runtime(rt, args, info, &st);
    const auto * vocab = llama_model_get_vocab(rt.model.get());
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<score_plan> plans;
    for (auto & doc : documents) {
        tokenize(vocab, doc, args.ctx);
        plans.push_back(plan_document(info, doc, args));
    }
    fs::create_directory(stage.temp / "scores");
    json results_docs = json::array();
    for (size_t di = 0; di < documents.size(); ++di) {
        const auto & doc = documents[di];
        const auto & plan = plans[di];
        const size_t n_scored = plan.positions.size();
        std::vector<uint8_t> want(doc.tokens.size(), 0);
        std::vector<int64_t> slot(doc.tokens.size(), -1);
        for (size_t i = 0; i < n_scored; ++i) {
            want[size_t(plan.positions[i] - 1)] = 1;
            slot[size_t(plan.positions[i] - 1)] = int64_t(i);
        }
        native_reference reference;
        reference.logits.resize(n_scored * size_t(n_vocab));
        reference.stats.resize(n_scored);
        const auto doc_dir = stage.temp / "scores" / doc.id;
        fs::create_directory(doc_dir);
        json conditions = json::array();
        for (const auto & c : plan.conditions) {
            const bool native = c.name == "native";
            if (c.patch_dir.empty()) { st.targets.clear(); st.index(); } else { load_patch_targets(st, info, c); }
            if (args.gpu()) { add_split_points(st, info); }
            st.positions = &c.positions;
            condition_scores cs(n_scored);
            // Logits at p - 1 for the last scored p, and every patch row, must be reached.
            size_t needed = size_t(plan.positions.back());
            if (!c.positions.empty()) { needed = std::max(needed, size_t(c.positions.back()) + 1); }
            if (args.full_decode) { needed = doc.tokens.size(); }
            const size_t decoded = decode_document(rt, st, args, doc, want, [&](int batch_index, int64_t position) {
                score_logits(rt.ctx.get(), doc, slot, n_vocab, native, reference, cs, batch_index, position);
            }, needed);
            json applied = json::object();
            for (const auto & t : st.targets) {
                if (t->observe_only) { continue; }
                for (const auto count : t->applied) {
                    if (count != 1) { throw std::runtime_error("patch row of " + t->filename + " was applied " + std::to_string(count) + " times"); }
                }
                applied[t->key + "-" + std::to_string(t->layer)] = t->applied.size();
            }
            finish_condition(cs);
            write_npy(doc_dir / (c.name + ".npy"), "<f4", {int64_t(n_scored), 3}, cs.scores.data(), cs.scores.size() * sizeof(float));
            auto entry = condition_json(c.name, cs, n_scored);
            entry["applied_rows"] = applied;
            entry["decoded_tokens"] = decoded;
            conditions.push_back(entry);
            std::cerr << "scored " << doc.id.substr(0, 12) << " " << c.name << ": nll " << cs.nll_sum / double(n_scored)
                      << " kl " << cs.kl_sum / double(n_scored) << " decoded " << decoded << "/" << doc.tokens.size()
                      << " tokens\n";
        }
        st.targets.clear();
        st.index();
        st.positions = nullptr;
        results_docs.push_back(scored_document_json(doc, plan, conditions));
    }
    auto settings = runtime_settings(args);
    settings["max_scored"] = args.max_scored;
    settings["decode"] = args.full_decode ? "full" : "through_last_needed_ubatch";
    if (args.gpu()) { settings["callback_split_points"] = "k,v of every attention layer"; }
    json results = {{"format_version", 2}, {"mode", "patch-score"}, {"architecture", info.architecture},
                    {"file_type", info.file_type}, {"n_layer", info.n_layer}, {"n_vocab", n_vocab},
                    {"documents", results_docs}, {"settings", settings}};
    add_device_report(results, rt);
    write_json(stage.temp / "results.json", results);
    stage.publish();
    return 0;
}

// ---------------------------------------------------------------------------
// Sequence states (llama_state_seq_save_file / llama_state_seq_load_file)

// Continue the current memory with document tokens [begin, ...). Ubatches end at multiples of
// --batch counted from position 0, at the document end and at every cut, so a span that starts
// on that grid repeats the ubatches of a whole-document decode. Stops once `needed` tokens are in.
static size_t decode_span(runtime & rt, const options & args, const document & doc, size_t begin, size_t needed,
                          const std::vector<size_t> & cuts, const std::vector<uint8_t> & want,
                          const std::function<void(int, int64_t)> & on_logits,
                          const std::function<void(size_t)> & on_stop) {
    const size_t n = doc.tokens.size();
    const size_t step = size_t(args.batch);
    if (begin >= needed || needed > n) { throw std::runtime_error("invalid decode span"); }
    size_t decoded = 0;
    for (size_t start = begin; start < needed;) {
        size_t stop = std::min(n, (start / step + 1) * step);
        for (const size_t cut : cuts) {
            if (cut > start && cut < stop) { stop = cut; }
        }
        auto & batch = rt.batch;
        batch.n_tokens = int(stop - start);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = doc.tokens[start + size_t(i)];
            batch.pos[i] = llama_pos(start + size_t(i));
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want.empty() ? i == batch.n_tokens - 1 : want[start + size_t(i)] != 0;
        }
        const int status = llama_decode(rt.ctx.get(), batch);
        llama_synchronize(rt.ctx.get());
        if (status != 0) { throw std::runtime_error("llama_decode failed with status " + std::to_string(status)); }
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] && !want.empty()) { on_logits(i, int64_t(start) + i); }
        }
        decoded += stop - start;
        start = stop;
        if (on_stop) { on_stop(stop); }
    }
    return decoded;
}

static uint32_t read_u32(std::ifstream & in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char *>(b), 4);
    if (!in) { throw std::runtime_error("truncated state file header"); }
    return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
}

struct state_header {
    std::vector<llama_token> words;   // as stored
    std::vector<llama_token> tokens;  // plain ids (server_tokens::deserialize semantics, text only)
    bool server_packed = false;
};

// Header of a llama_state_seq_save_file file. llama-server stores server_tokens::serialize()
// words (LLAMA_TOKEN_NULL, version 1, n, tokens, n_media); a plain id list is also accepted.
static state_header read_state_header(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open state file " + path.string()); }
    const uint32_t magic = read_u32(in), version = read_u32(in), count = read_u32(in);
    if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
        throw std::runtime_error("not a llama.cpp sequence state file (magic/version): " + path.string());
    }
    if (12 + uint64_t(count) * 4 > fs::file_size(path)) {
        throw std::runtime_error("state file token count exceeds the file size: " + path.string());
    }
    state_header header;
    header.words.resize(count);
    in.read(reinterpret_cast<char *>(header.words.data()), std::streamsize(size_t(count) * sizeof(llama_token)));
    if (!in) { throw std::runtime_error("cannot read state file tokens: " + path.string()); }
    const auto & w = header.words;
    if (!w.empty() && w[0] == LLAMA_TOKEN_NULL) {
        if (w.size() < 4 || w[1] != 1) { throw std::runtime_error("unsupported server tokens state in " + path.string()); }
        const uint64_t n = uint32_t(w[2]);
        if (4 + n != w.size() || w[3 + n] != 0) {
            throw std::runtime_error("server tokens state must hold only text tokens and no media: " + path.string());
        }
        header.tokens.assign(w.begin() + 3, w.begin() + 3 + std::ptrdiff_t(n));
        header.server_packed = true;
    } else {
        header.tokens = w;
    }
    for (const auto token : header.tokens) {
        if (token < 0) { throw std::runtime_error("negative token id in state file " + path.string()); }
    }
    return header;
}

static std::vector<size_t> plan_snapshots(const document & doc) {
    const auto & r = doc.record;
    if (r.contains("expected_token_sha256")) {
        if (!is_hex64(r["expected_token_sha256"]) || token_sha256(doc.tokens) != r["expected_token_sha256"].get<std::string>()) {
            throw std::runtime_error("document " + doc.id + ": token ids do not match expected_token_sha256");
        }
    }
    std::vector<size_t> snapshots;
    if (!r.contains("save_at")) {
        snapshots.push_back(doc.tokens.size());
        return snapshots;
    }
    if (!r["save_at"].is_array() || r["save_at"].empty()) {
        throw std::runtime_error("document " + doc.id + ": save_at must be a nonempty list of token counts");
    }
    std::set<size_t> unique;
    for (const auto & item : r["save_at"]) {
        if (!item.is_number_integer() || item.get<int64_t>() < 1 || item.get<int64_t>() > int64_t(doc.tokens.size())) {
            throw std::runtime_error("document " + doc.id + ": save_at entries must be token counts in [1, "
                                     + std::to_string(doc.tokens.size()) + "]");
        }
        if (!unique.insert(size_t(item.get<int64_t>())).second) {
            throw std::runtime_error("document " + doc.id + ": duplicate save_at entry");
        }
    }
    snapshots.assign(unique.begin(), unique.end());
    return snapshots;
}

static int run_save_state(const options & args, const model_info & info, std::vector<document> & documents) {
    probe_state st;
    staging stage(args.output);
    runtime rt;
    load_runtime(rt, args, info, &st);
    uint64_t rows = 0;
    std::vector<std::vector<size_t>> snapshots;
    for (auto & doc : documents) {
        tokenize(llama_model_get_vocab(rt.model.get()), doc, args.ctx);
        doc.first_row = rows;
        rows += doc.tokens.size();
        snapshots.push_back(plan_snapshots(doc));
    }
    write_token_arrays(stage.temp, documents, rows);
    fs::create_directory(stage.temp / "states");
    auto * memory = llama_get_memory(rt.ctx.get());
    json docs = json::array();
    for (size_t di = 0; di < documents.size(); ++di) {
        const auto & doc = documents[di];
        const auto & cuts = snapshots[di];
        fs::create_directory(stage.temp / "states" / doc.id);
        llama_synchronize(rt.ctx.get());
        llama_memory_clear(memory, true);
        json states = json::array();
        std::vector<size_t> stops;
        const std::vector<uint8_t> no_logits;
        decode_span(rt, args, doc, 0, cuts.back(), cuts, no_logits, [](int, int64_t) {}, [&](size_t stop) {
            stops.push_back(stop);
            if (!std::binary_search(cuts.begin(), cuts.end(), stop)) { return; }
            const std::string relative = "states/" + doc.id + "/" + std::to_string(stop) + ".state";
            const auto path = stage.temp / relative;
            const size_t written = llama_state_seq_save_file(rt.ctx.get(), path.string().c_str(), 0, doc.tokens.data(), stop);
            if (written == 0 || written != fs::file_size(path)) {
                throw std::runtime_error("llama_state_seq_save_file failed for document " + doc.id);
            }
            states.push_back({{"n_tokens", stop}, {"file", relative}, {"bytes", written},
                              {"pos_min", llama_memory_seq_pos_min(memory, 0)},
                              {"pos_max", llama_memory_seq_pos_max(memory, 0)}});
        });
        if (states.size() != cuts.size()) { throw std::runtime_error("not every requested state was saved"); }
        std::cerr << "saved " << states.size() << " states of " << doc.id.substr(0, 12) << " after "
                  << cuts.back() << "/" << doc.tokens.size() << " tokens\n";
        docs.push_back({{"document_id", doc.id}, {"n_tokens", doc.tokens.size()}, {"first_row", doc.first_row},
                        {"prefill_tokens", cuts.back()}, {"ubatch_ends", stops}, {"states", states}});
    }
    json report = {{"format_version", 2}, {"mode", "save-state"}, {"architecture", info.architecture},
                   {"file_type", info.file_type}, {"n_layer", info.n_layer}, {"layers", layers_json(info)},
                   {"settings", runtime_settings(args)},
                   {"state_format", {{"writer", "llama_state_seq_save_file"}, {"magic", LLAMA_STATE_SEQ_MAGIC},
                                     {"version", LLAMA_STATE_SEQ_VERSION}, {"seq_id", 0},
                                     {"token_words", "plain"}}},
                   {"documents", docs}, {"rows", rows}};
    add_device_report(report, rt);
    write_json(stage.temp / "report.json", report);
    stage.publish();
    return 0;
}

struct seed_condition {
    std::string name;
    fs::path state_file;
    state_header header;
};

static std::vector<seed_condition> plan_seed_conditions(const document & doc, const score_plan & plan) {
    const auto & r = doc.record;
    if (!r.contains("conditions") || !r["conditions"].is_array() || r["conditions"].empty()) {
        throw std::runtime_error("document " + doc.id + ": conditions must be a nonempty list");
    }
    static const std::regex name_pattern("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}");
    std::set<std::string> names;
    std::vector<seed_condition> out;
    for (const auto & item : r["conditions"]) {
        if (!item.is_object() || !item.contains("name") || !item["name"].is_string() || !item.contains("state_file")) {
            throw std::runtime_error("document " + doc.id + ": condition needs name and state_file");
        }
        seed_condition c;
        c.name = item["name"].get<std::string>();
        if (!std::regex_match(c.name, name_pattern) || !names.insert(c.name).second) {
            throw std::runtime_error("document " + doc.id + ": invalid or duplicate condition name " + c.name);
        }
        const bool first = out.empty();
        if (first != (c.name == "native") || (first && !item["state_file"].is_null())) {
            throw std::runtime_error("document " + doc.id + ": the first condition must be native with a null state_file");
        }
        if (!item["state_file"].is_null()) {
            if (!item["state_file"].is_string()) { throw std::runtime_error("state_file must be null or a path"); }
            c.state_file = item["state_file"].get<std::string>();
            if (!c.state_file.is_absolute() || !fs::is_regular_file(c.state_file)) {
                throw std::runtime_error("state_file must be an existing absolute file: " + c.state_file.string());
            }
            c.header = read_state_header(c.state_file);
            const auto & tokens = c.header.tokens;
            if (tokens.empty() || tokens.size() > doc.tokens.size()
                    || !std::equal(tokens.begin(), tokens.end(), doc.tokens.begin())) {
                throw std::runtime_error("document " + doc.id + ": tokens of state file " + c.state_file.string()
                                         + " are not a prefix of the document tokens");
            }
            if (int64_t(tokens.size()) >= plan.positions.front()) {
                throw std::runtime_error("document " + doc.id + ": state file " + c.state_file.string() + " holds "
                                         + std::to_string(tokens.size()) + " tokens; scored position "
                                         + std::to_string(plan.positions.front())
                                         + " needs logits from a decoded token after the seeded prefix");
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

static int run_seed_score(const options & args, const model_info & info, std::vector<document> & documents) {
    probe_state st;
    staging stage(args.output);
    runtime rt;
    load_runtime(rt, args, info, &st);
    const auto * vocab = llama_model_get_vocab(rt.model.get());
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    auto * memory = llama_get_memory(rt.ctx.get());
    std::vector<score_plan> plans;
    std::vector<std::vector<seed_condition>> seeds;
    for (auto & doc : documents) {
        tokenize(vocab, doc, args.ctx);
        plans.push_back(plan_scored_positions(doc, args));
        seeds.push_back(plan_seed_conditions(doc, plans.back()));
    }
    fs::create_directory(stage.temp / "scores");
    json results_docs = json::array();
    for (size_t di = 0; di < documents.size(); ++di) {
        const auto & doc = documents[di];
        const auto & plan = plans[di];
        const size_t n_scored = plan.positions.size();
        std::vector<uint8_t> want(doc.tokens.size(), 0);
        std::vector<int64_t> slot(doc.tokens.size(), -1);
        for (size_t i = 0; i < n_scored; ++i) {
            want[size_t(plan.positions[i] - 1)] = 1;
            slot[size_t(plan.positions[i] - 1)] = int64_t(i);
        }
        native_reference reference;
        reference.logits.resize(n_scored * size_t(n_vocab));
        reference.stats.resize(n_scored);
        const auto doc_dir = stage.temp / "scores" / doc.id;
        fs::create_directory(doc_dir);
        json conditions = json::array();
        const size_t needed = args.full_decode ? doc.tokens.size() : size_t(plan.positions.back());
        for (const auto & c : seeds[di]) {
            const bool native = c.state_file.empty();
            condition_scores cs(n_scored);
            const auto on_logits = [&](int batch_index, int64_t position) {
                score_logits(rt.ctx.get(), doc, slot, n_vocab, native, reference, cs, batch_index, position);
            };
            json entry;
            size_t decoded = 0;
            if (native) {
                decoded = decode_document(rt, st, args, doc, want, on_logits, needed);
                entry = condition_json(c.name, cs, n_scored);
                entry["seed_tokens"] = 0;
                entry["state_file"] = nullptr;
            } else {
                const size_t seed = c.header.tokens.size();
                llama_synchronize(rt.ctx.get());
                llama_memory_clear(memory, true);
                std::vector<llama_token> words(std::max<size_t>(1, c.header.words.size()));
                size_t n_words = 0;
                const size_t read = llama_state_seq_load_file(rt.ctx.get(), c.state_file.string().c_str(), 0,
                                                              words.data(), words.size(), &n_words);
                if (read == 0) {
                    throw std::runtime_error("llama.cpp could not restore " + c.state_file.string()
                                             + " (layout, cache type, flash attention, or cache size differ; see the log)");
                }
                words.resize(n_words);
                if (words != c.header.words) { throw std::runtime_error("state file changed after validation: " + c.state_file.string()); }
                if (read != fs::file_size(c.state_file)) {
                    throw std::runtime_error("state file has " + std::to_string(fs::file_size(c.state_file) - read)
                                             + " trailing bytes: " + c.state_file.string());
                }
                const llama_pos pos_min = llama_memory_seq_pos_min(memory, 0), pos_max = llama_memory_seq_pos_max(memory, 0);
                if (pos_max != llama_pos(seed) - 1) {
                    throw std::runtime_error("restored state of " + c.state_file.string() + " ends at position "
                                             + std::to_string(pos_max) + "; its " + std::to_string(seed)
                                             + " tokens require " + std::to_string(seed - 1));
                }
                decoded = decode_span(rt, args, doc, seed, needed, {}, want, on_logits, nullptr);
                entry = condition_json(c.name, cs, n_scored);
                entry["seed_tokens"] = seed;
                entry["state_file"] = c.state_file.string();
                entry["state_bytes"] = read;
                entry["server_packed_tokens"] = c.header.server_packed;
                entry["restored_pos_min"] = pos_min;
                entry["restored_pos_max"] = pos_max;
            }
            finish_condition(cs);
            write_npy(doc_dir / (c.name + ".npy"), "<f4", {int64_t(n_scored), 3}, cs.scores.data(), cs.scores.size() * sizeof(float));
            entry["applied_rows"] = json::object();
            entry["decoded_tokens"] = decoded;
            conditions.push_back(entry);
            std::cerr << "scored " << doc.id.substr(0, 12) << " " << c.name << ": nll " << cs.nll_sum / double(n_scored)
                      << " kl " << cs.kl_sum / double(n_scored) << " decoded " << decoded << "/" << doc.tokens.size()
                      << " tokens\n";
        }
        results_docs.push_back(scored_document_json(doc, plan, conditions));
    }
    auto settings = runtime_settings(args);
    settings["max_scored"] = args.max_scored;
    settings["decode"] = args.full_decode ? "full" : "through_last_needed_ubatch";
    json results = {{"format_version", 2}, {"mode", "seed-score"}, {"architecture", info.architecture},
                    {"file_type", info.file_type}, {"n_layer", info.n_layer}, {"n_vocab", n_vocab},
                    {"documents", results_docs}, {"settings", settings}};
    add_device_report(results, rt);
    write_json(stage.temp / "results.json", results);
    stage.publish();
    return 0;
}

static int run(const options & args) {
    const uint32_t one = 1;
    if (*reinterpret_cast<const uint8_t *>(&one) != 1) { throw std::runtime_error("little-endian host required"); }
    if (fs::exists(args.output) || fs::is_symlink(args.output)) { throw std::runtime_error("output already exists"); }
    const auto info = inspect_model(args.model);
    auto documents = read_documents(args.input);
    llama_backend_init();
    struct backend_guard { ~backend_guard() { llama_backend_free(); } } guard;
    switch (args.mode) {
        case run_mode::tokenize:    return run_tokenize(args, info, documents);
        case run_mode::collect:     return run_collect(args, info, documents);
        case run_mode::patch_score: return run_patch_score(args, info, documents);
        case run_mode::save_state:  return run_save_state(args, info, documents);
        case run_mode::seed_score:  return run_seed_score(args, info, documents);
    }
    return 1;
}

int main(int argc, char ** argv) {
    try { return run(parse_options(argc, argv)); }
    catch (const std::exception & exc) { std::cerr << "kv-probe: " << exc.what() << '\n'; return 1; }
}
