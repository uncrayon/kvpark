// kvpark-state-driver: load any GGUF on the CPU and drive llama.cpp's sequence-state API with
// token ids, for layout and restore tests of architectures kv_probe has no table for.
//
//   save            decode --tokens (ubatches of --batch from position 0), write the sequence
//                   state with llama_state_seq_save_file (sequence --seq), optionally its
//                   PARTIAL_ONLY blob (llama_state_seq_get_data_ext: the 8-byte io prefix and the
//                   payload a server checkpoint keeps) with --partial, then optionally decode --next
//                   natively as one more ubatch and dump its logits (the reference for a restore)
//   restore-decode  load --state into a fresh context (llama_state_seq_load_file, sequence --seq),
//                   decode --next from llama_memory_seq_pos_max + 1 as one ubatch and dump the logits
//   capture         decode --tokens and dump the graph tensors named in --capture through the
//                   eval callback (every node with that name: views, ropes, norms, ...); with
//                   --state also save the sequence state of that same run (its cache rows are
//                   the captured graph's, bit for bit)
//
// --parallel N gives the context N sequences (llama-server's -np), --seq picks the one every mode
// uses, --rs-seq the rollback snapshots per sequence (n_rs_seq: what MTP/DSpark drafting asks for),
// and --cache-type also takes q8_0 (type_k = type_v, as llama-server's -ctk/-ctv).
//
// There is no architecture table: whatever llama.cpp loads, this drives. A JSON report goes to
// stdout. Logits are float32 .npy [n_next, n_vocab]; captures are float32 .npy files with the
// ggml shape reversed (numpy order) plus manifest.json naming each node's op, type and shape.
//
// It builds unchanged against both pinned runtimes, 85d5703a3 and a7b94df2c (where llama_batch_init
// and llama_decode are a compatibility path over llama_batch_ext / llama_process), so the layouts of
// either pin's states can be checked; build_collector.py's manifest names the runtime, and
// tiny_fixture.driver_revision reads it (else from a save). A restore that llama.cpp refuses fails
// restore-decode on both; a7b94df2c also zeroes the cells it had written before failing.

#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct options {
    std::string mode;
    fs::path model, state, logits, output, partial;
    std::vector<llama_token> tokens, next;
    std::vector<std::string> capture;
    bool token_list = true;
    int ctx = 256, batch = 64, threads = 4, parallel = 1, seq = 0, rs_seq = 0;
    std::string cache_type = "f16";
    bool flash_attn = false, swa_full = true;
    // runtime RoPE overrides, as llama-server's --rope-* / --yarn-* flags (unset: from the model)
    std::string rope_scaling;
    float rope_freq_base = 0.0f, rope_freq_scale = 0.0f;
    float yarn_ext_factor = -1.0f, yarn_attn_factor = -1.0f, yarn_beta_fast = -1.0f, yarn_beta_slow = -1.0f;
    int yarn_orig_ctx = 0;
};

static const char * usage =
    "--mode {save,restore-decode,capture} --model GGUF\n"
    "  save:           --tokens IDS --state NEW_FILE [--partial NEW_BLOB] [--next IDS --logits NEW.npy]\n"
    "                  [--no-token-list]\n"
    "  restore-decode: --state FILE --next IDS --logits NEW.npy\n"
    "  capture:        --tokens IDS --capture NAME[,NAME...] --output NEW_DIR [--state NEW_FILE]\n"
    "                  (NAME matches NAME and NAME-<layer>; a name with a layer suffix matches exactly)\n"
    "  IDS: comma-separated token ids, or @FILE holding ids separated by whitespace or commas\n"
    "  [--ctx 256] [--batch 64] [--threads 4] [--cache-type f16|f32|q8_0] [--flash-attn off|on] [--swa-full on|off]\n"
    "  [--parallel 1] [--seq 0] [--rs-seq 0]\n"
    "  [--rope-scaling none|linear|yarn] [--rope-freq-base F] [--rope-freq-scale F] [--yarn-orig-ctx N]\n"
    "  [--yarn-ext-factor F] [--yarn-attn-factor F] [--yarn-beta-fast F] [--yarn-beta-slow F]\n";

static std::vector<llama_token> parse_ids(const std::string & value, const std::string & key) {
    std::string text = value;
    if (!text.empty() && text[0] == '@') {
        std::ifstream in(text.substr(1));
        if (!in) { throw std::runtime_error("cannot open " + text.substr(1) + " for " + key); }
        std::stringstream buffer;
        buffer << in.rdbuf();
        text = buffer.str();
    }
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream in(text);
    std::vector<llama_token> ids;
    for (std::string word; in >> word;) {
        size_t used = 0;
        long long id = -1;
        try { id = std::stoll(word, &used); } catch (const std::exception &) { used = 0; }
        if (used != word.size() || id < 0 || id > INT32_MAX) { throw std::runtime_error("invalid token id in " + key); }
        ids.push_back(llama_token(id));
    }
    if (ids.empty()) { throw std::runtime_error(key + " holds no token ids"); }
    return ids;
}

static float parse_float(const std::string & value, const std::string & key) {
    size_t used = 0;
    float number = 0.0f;
    try { number = std::stof(value, &used); } catch (const std::exception &) { used = 0; }
    if (used == 0 || used != value.size()) { throw std::runtime_error("invalid number for " + key); }
    return number;
}

static int parse_int(const std::string & value, const std::string & key) {
    size_t used = 0;
    int number = 0;
    try { number = std::stoi(value, &used); } catch (const std::exception &) { used = 0; }
    if (used == 0 || used != value.size()) { throw std::runtime_error("invalid integer for " + key); }
    return number;
}

static options parse_options(int argc, char ** argv) {
    options o;
    std::set<std::string> given;
    auto on_off = [](const std::string & key, const std::string & value) {
        if (value == "on") { return true; }
        if (value == "off") { return false; }
        throw std::runtime_error(key + " must be on or off");
    };
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { std::cout << usage; std::exit(0); }
        if (!given.insert(key).second) { throw std::runtime_error("option " + key + " given more than once"); }
        if (key == "--no-token-list") { o.token_list = false; continue; }
        if (++i == argc) { throw std::runtime_error("missing argument for " + key); }
        const std::string value = argv[i];
        if      (key == "--mode")    { o.mode = value; }
        else if (key == "--model")   { o.model = value; }
        else if (key == "--state")   { o.state = value; }
        else if (key == "--logits")  { o.logits = value; }
        else if (key == "--output")  { o.output = value; }
        else if (key == "--partial") { o.partial = value; }
        else if (key == "--tokens")  { o.tokens = parse_ids(value, key); }
        else if (key == "--next")    { o.next = parse_ids(value, key); }
        else if (key == "--capture") {
            std::string item;
            std::istringstream in(value);
            while (std::getline(in, item, ',')) {
                if (item.empty()) { throw std::runtime_error("empty name in --capture"); }
                o.capture.push_back(item);
            }
        }
        else if (key == "--ctx")     { o.ctx = parse_int(value, key); }
        else if (key == "--batch")   { o.batch = parse_int(value, key); }
        else if (key == "--threads") { o.threads = parse_int(value, key); }
        else if (key == "--parallel") { o.parallel = parse_int(value, key); }
        else if (key == "--seq")     { o.seq = parse_int(value, key); }
        else if (key == "--rs-seq")  { o.rs_seq = parse_int(value, key); }
        else if (key == "--cache-type") {
            if (value != "f16" && value != "f32" && value != "q8_0") {
                throw std::runtime_error("--cache-type must be f16, f32 or q8_0");
            }
            o.cache_type = value;
        }
        else if (key == "--flash-attn") { o.flash_attn = on_off(key, value); }
        else if (key == "--swa-full")   { o.swa_full = on_off(key, value); }
        else if (key == "--rope-scaling") {
            if (value != "none" && value != "linear" && value != "yarn") {
                throw std::runtime_error("--rope-scaling must be none, linear or yarn");
            }
            o.rope_scaling = value;
        }
        else if (key == "--rope-freq-base")   { o.rope_freq_base = parse_float(value, key); }
        else if (key == "--rope-freq-scale")  { o.rope_freq_scale = parse_float(value, key); }
        else if (key == "--yarn-ext-factor")  { o.yarn_ext_factor = parse_float(value, key); }
        else if (key == "--yarn-attn-factor") { o.yarn_attn_factor = parse_float(value, key); }
        else if (key == "--yarn-beta-fast")   { o.yarn_beta_fast = parse_float(value, key); }
        else if (key == "--yarn-beta-slow")   { o.yarn_beta_slow = parse_float(value, key); }
        else if (key == "--yarn-orig-ctx")    { o.yarn_orig_ctx = parse_int(value, key); }
        else { throw std::runtime_error("unknown option " + key); }
    }
    if (o.model.empty()) { throw std::runtime_error("--model is required"); }
    if (o.ctx < 1 || o.batch < 1 || o.batch > o.ctx || o.threads < 1 || o.yarn_orig_ctx < 0) {
        throw std::runtime_error("sizes must be positive, batch <= ctx");
    }
    if (o.parallel < 1 || o.seq < 0 || o.seq >= o.parallel || o.rs_seq < 0) {
        throw std::runtime_error("--parallel must be positive, --seq in [0, parallel), --rs-seq nonnegative");
    }
    auto require = [&](bool ok, const std::string & what) {
        if (!ok) { throw std::runtime_error("--mode " + o.mode + " " + what); }
    };
    if (o.mode == "save") {
        require(!o.tokens.empty() && !o.state.empty(), "needs --tokens and --state");
        require(o.next.empty() == o.logits.empty(), "takes --next and --logits together");
        require(o.capture.empty() && o.output.empty(), "takes no --capture/--output");
    } else if (o.mode == "restore-decode") {
        require(!o.state.empty() && !o.next.empty() && !o.logits.empty(), "needs --state, --next and --logits");
        require(o.partial.empty(), "takes no --partial");
        require(o.tokens.empty() && o.capture.empty() && o.output.empty() && o.token_list,
                "takes no --tokens/--capture/--output/--no-token-list");
    } else if (o.mode == "capture") {
        require(!o.tokens.empty() && !o.capture.empty() && !o.output.empty(), "needs --tokens, --capture and --output");
        require(o.next.empty() && o.logits.empty() && o.token_list, "takes no --next/--logits/--no-token-list");
        require(o.partial.empty(), "takes no --partial");
    } else {
        throw std::runtime_error("--mode must be save, restore-decode or capture");
    }
    for (const auto & path : {o.state, o.logits, o.output, o.partial}) {
        if (path.empty() || (o.mode == "restore-decode" && path == o.state)) { continue; }
        if (fs::exists(path) || fs::is_symlink(path)) { throw std::runtime_error("output exists: " + path.string()); }
    }
    if (o.mode == "restore-decode" && !fs::is_regular_file(o.state)) {
        throw std::runtime_error("no state file " + o.state.string());
    }
    const size_t total = o.tokens.size() + o.next.size();
    if (total > size_t(o.ctx)) { throw std::runtime_error("--tokens plus --next exceed --ctx"); }
    // one ubatch for the whole continuation, so native and restored runs compute the same graph
    if (o.next.size() > size_t(o.batch)) { throw std::runtime_error("--next must fit in one ubatch (--batch)"); }
    return o;
}

// ---------------------------------------------------------------------------
// New output files

// A new output file written through "<path>.partial": publish() hard-links it to `path`, which
// fails rather than replace a file that appeared meanwhile (as kvstate.write_new), and every
// exit removes the .partial name. A .partial that already exists (another run's, or a symlink)
// is refused instead of being written through.
class staged_output {
  public:
    explicit staged_output(fs::path path) : path_(std::move(path)), temp_(path_.string() + ".partial") {
        if (fs::exists(fs::symlink_status(temp_))) { throw std::runtime_error("temporary file exists: " + temp_.string()); }
    }
    staged_output(const staged_output &) = delete;
    staged_output & operator=(const staged_output &) = delete;
    ~staged_output() {
        std::error_code ignored;
        fs::remove(temp_, ignored);
    }
    const fs::path & temp() const { return temp_; }
    void publish() const { fs::create_hard_link(temp_, path_); }

  private:
    fs::path path_, temp_;
};

// ---------------------------------------------------------------------------
// NPY v1.0 writer (float32)

static void write_npy_f32(const fs::path & path, const std::vector<int64_t> & shape, const std::vector<float> & data) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]);
        if (shape.size() == 1) { dims += ","; } else if (i + 1 < shape.size()) { dims += ", "; }
    }
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    const size_t unpadded = 10 + dict.size() + 1;
    dict.append((64 - unpadded % 64) % 64, ' ');
    dict.push_back('\n');
    std::string header("\x93NUMPY\x01\x00", 8);
    header.push_back(char(dict.size() & 0xff));
    header.push_back(char(dict.size() >> 8));
    const staged_output staged(path);
    {
        std::ofstream out(staged.temp(), std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write(header.data(), std::streamsize(header.size()));
        out.write(dict.data(), std::streamsize(dict.size()));
        out.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * sizeof(float)));
    }
    staged.publish();
}

// ---------------------------------------------------------------------------
// Eval-callback capture

struct capture_state {
    std::vector<std::string> names;
    fs::path dir;
    json manifest = json::array();
    int ubatch = 0;
    int64_t pos_start = 0;
    std::string error;
};

static bool name_selected(const capture_state & st, const char * name) {
    const std::string full(name);
    for (const auto & want : st.names) {
        if (full == want) { return true; }
        if (full.size() > want.size() + 1 && full.compare(0, want.size(), want) == 0 && full[want.size()] == '-'
                && full.find_first_not_of("0123456789", want.size() + 1) == std::string::npos) {
            return true;
        }
    }
    return false;
}

// Float32 copy of any F32/F16 tensor (views and permutes included), in ggml element order.
static std::vector<float> tensor_values(const ggml_tensor * t) {
    const size_t span = ggml_nbytes(t);
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, span);
    std::vector<float> out;
    out.reserve(size_t(ggml_nelements(t)));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t offset = size_t(i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    if (t->type == GGML_TYPE_F32) {
                        float value;
                        std::memcpy(&value, raw.data() + offset, sizeof(float));
                        out.push_back(value);
                    } else {
                        ggml_fp16_t value;
                        std::memcpy(&value, raw.data() + offset, sizeof(value));
                        out.push_back(ggml_fp16_to_fp32(value));
                    }
                }
            }
        }
    }
    return out;
}

static bool capture_callback(ggml_tensor * t, bool ask, void * opaque) {
    auto & st = *static_cast<capture_state *>(opaque);
    const bool selected = name_selected(st, t->name) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
    if (ask) { return selected; }
    if (!selected) { return true; }
    try {
        const std::string file = "capture-" + std::to_string(st.manifest.size()) + ".npy";
        std::vector<int64_t> shape;
        for (int d = ggml_n_dims(t) - 1; d >= 0; --d) { shape.push_back(t->ne[d]); }
        write_npy_f32(st.dir / file, shape, tensor_values(t));
        st.manifest.push_back({{"file", file}, {"name", t->name}, {"op", ggml_op_desc(t)},
                               {"type", ggml_type_name(t->type)}, {"ne", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
                               {"ubatch", st.ubatch}, {"pos_start", st.pos_start}});
        return true;
    } catch (const std::exception & exc) {
        if (st.error.empty()) { st.error = exc.what(); }
        return false;
    }
}

// ---------------------------------------------------------------------------
// Runtime

struct runtime {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    llama_batch batch{};
    bool batch_ready = false;
    int32_t n_vocab = 0;
    ~runtime() {
        if (batch_ready) { llama_batch_free(batch); }
        ctx.reset();
        model.reset();
    }
};

static void load_runtime(runtime & rt, const options & o, capture_state * capture) {
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    if (!devices[0]) { throw std::runtime_error("CPU backend unavailable; use a runtime with GGML_BACKEND_DL=OFF"); }
    auto model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.use_extra_bufts = false;
    model_params.load_mtp = false;
    rt.model.reset(llama_model_load_from_file(o.model.string().c_str(), model_params));
    if (!rt.model) { throw std::runtime_error("model load failed"); }
    rt.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(rt.model.get()));
    for (const auto & ids : {o.tokens, o.next}) {
        for (const auto id : ids) {
            if (id >= rt.n_vocab) { throw std::runtime_error("token id " + std::to_string(id) + " outside the vocabulary"); }
        }
    }
    auto params = llama_context_default_params();
    params.n_ctx = uint32_t(o.ctx);
    params.n_batch = params.n_ubatch = uint32_t(o.batch);
    params.n_seq_max = uint32_t(o.parallel);
    params.n_rs_seq = uint32_t(o.rs_seq);
    params.n_outputs_max = 0;
    params.n_outputs_max_per_seq = 0;
    params.n_threads = params.n_threads_batch = o.threads;
    params.type_k = params.type_v = o.cache_type == "f16" ? GGML_TYPE_F16
                                  : o.cache_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
    params.flash_attn_type = o.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = params.op_offload = false;
    params.swa_full = o.swa_full;
    if (o.rope_scaling == "none")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
    if (o.rope_scaling == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
    if (o.rope_scaling == "yarn")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
    params.rope_freq_base = o.rope_freq_base;
    params.rope_freq_scale = o.rope_freq_scale;
    params.yarn_ext_factor = o.yarn_ext_factor;
    params.yarn_attn_factor = o.yarn_attn_factor;
    params.yarn_beta_fast = o.yarn_beta_fast;
    params.yarn_beta_slow = o.yarn_beta_slow;
    params.yarn_orig_ctx = uint32_t(o.yarn_orig_ctx);
    if (capture) {
        params.cb_eval = capture_callback;
        params.cb_eval_user_data = capture;
    }
    rt.ctx.reset(llama_init_from_model(rt.model.get(), params));
    if (!rt.ctx) { throw std::runtime_error("context creation failed"); }
    rt.batch = llama_batch_init(o.batch, 0, 1);
    rt.batch_ready = true;
}

// Decode `ids` at positions [pos0, pos0 + n) in ubatches of --batch. A span that starts at 0
// repeats the ubatches of llama-server's prompt processing with the same --batch/--ubatch.
// Logits of every token are appended to `logits` when it is given.
static void decode(runtime & rt, const options & o, const std::vector<llama_token> & ids, llama_pos pos0,
                   std::vector<float> * logits, capture_state * capture) {
    for (size_t start = 0; start < ids.size(); start += size_t(o.batch)) {
        auto & batch = rt.batch;
        batch.n_tokens = int(std::min(size_t(o.batch), ids.size() - start));
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = ids[start + size_t(i)];
            batch.pos[i] = pos0 + llama_pos(start + size_t(i));
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = o.seq;
            batch.logits[i] = logits != nullptr || i == batch.n_tokens - 1;
        }
        if (capture) { capture->pos_start = pos0 + llama_pos(start); }
        const int status = llama_decode(rt.ctx.get(), batch);
        llama_synchronize(rt.ctx.get());
        if (capture) {
            ++capture->ubatch;
            if (!capture->error.empty()) { throw std::runtime_error(capture->error); }
        }
        if (status != 0) { throw std::runtime_error("llama_decode failed with status " + std::to_string(status)); }
        if (logits) {
            for (int i = 0; i < batch.n_tokens; ++i) {
                const float * row = llama_get_logits_ith(rt.ctx.get(), i);
                if (!row) { throw std::runtime_error("no logits for a token"); }
                logits->insert(logits->end(), row, row + rt.n_vocab);
            }
        }
    }
}

static json settings_json(const options & o) {
    return {{"ctx", o.ctx}, {"batch", o.batch}, {"threads", o.threads}, {"cache_type", o.cache_type},
            {"parallel", o.parallel}, {"seq", o.seq}, {"rs_seq", o.rs_seq},
            {"flash_attention", o.flash_attn}, {"swa_full", o.swa_full},
            {"rope_scaling", o.rope_scaling.empty() ? "model" : o.rope_scaling},
            {"rope_freq_base", o.rope_freq_base}, {"rope_freq_scale", o.rope_freq_scale},
            {"yarn_ext_factor", o.yarn_ext_factor}, {"yarn_attn_factor", o.yarn_attn_factor},
            {"yarn_beta_fast", o.yarn_beta_fast}, {"yarn_beta_slow", o.yarn_beta_slow},
            {"yarn_orig_ctx", o.yarn_orig_ctx}};
}

static int run(const options & o) {
    runtime rt;
    capture_state capture;
    const bool capturing = o.mode == "capture";
    if (capturing) {
        capture.names = o.capture;
        capture.dir = o.output;
    }
    load_runtime(rt, o, capturing ? &capture : nullptr);
    if (capturing && !fs::create_directory(capture.dir)) {
        throw std::runtime_error("output exists: " + capture.dir.string());
    }
    auto * memory = llama_get_memory(rt.ctx.get());
    json report = {{"mode", o.mode}, {"model", o.model.string()}, {"n_vocab", rt.n_vocab}, {"settings", settings_json(o)}};
    if (o.mode == "save" || o.mode == "capture") {
        decode(rt, o, o.tokens, 0, nullptr, capturing ? &capture : nullptr);
        report["n_tokens"] = o.tokens.size();
        report["pos_max"] = llama_memory_seq_pos_max(memory, o.seq);
    }
    if (o.mode == "save" || (capturing && !o.state.empty())) {
        const staged_output staged(o.state);
        const size_t written = llama_state_seq_save_file(rt.ctx.get(), staged.temp().string().c_str(), o.seq,
                                                         o.token_list ? o.tokens.data() : nullptr,
                                                         o.token_list ? o.tokens.size() : 0);
        if (written == 0 || written != fs::file_size(staged.temp())) {
            throw std::runtime_error("llama_state_seq_save_file failed");
        }
        staged.publish();
        report["state_bytes"] = written;
        report["token_list"] = o.token_list;
    }
    if (!o.partial.empty()) {
        // what a server checkpoint keeps: io magic, seq id, then the PARTIAL_ONLY payload
        const size_t size = llama_state_seq_get_size_ext(rt.ctx.get(), o.seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> blob(size);
        const size_t written = llama_state_seq_get_data_ext(rt.ctx.get(), blob.data(), blob.size(), o.seq,
                                                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0 || written != size) { throw std::runtime_error("llama_state_seq_get_data_ext failed"); }
        const staged_output staged(o.partial);
        {
            std::ofstream out(staged.temp(), std::ios::binary);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.write(reinterpret_cast<const char *>(blob.data()), std::streamsize(blob.size()));
        }
        staged.publish();
        report["partial_bytes"] = written;
    }
    if (o.mode == "restore-decode") {
        size_t count = 0;
        llama_state_seq_load_file(rt.ctx.get(), o.state.string().c_str(), o.seq, nullptr, 0, &count);
        std::vector<llama_token> words(std::max<size_t>(count, 1));
        size_t read_count = 0;
        const size_t read = llama_state_seq_load_file(rt.ctx.get(), o.state.string().c_str(), o.seq, words.data(),
                                                      words.size(), &read_count);
        if (read == 0 || read_count != count) { throw std::runtime_error("llama_state_seq_load_file failed"); }
        report["state_bytes"] = read;
        report["token_words"] = count;
        report["pos_min"] = llama_memory_seq_pos_min(memory, o.seq);
        report["pos_max"] = llama_memory_seq_pos_max(memory, o.seq);
    }
    if (!o.next.empty()) {
        const llama_pos pos0 = llama_memory_seq_pos_max(memory, o.seq) + 1;
        if (o.mode == "save" && size_t(pos0) != o.tokens.size()) { throw std::runtime_error("unexpected memory end"); }
        if (size_t(pos0) + o.next.size() > size_t(o.ctx)) { throw std::runtime_error("--next does not fit in --ctx"); }
        std::vector<float> logits;
        decode(rt, o, o.next, pos0, &logits, nullptr);
        write_npy_f32(o.logits, {int64_t(o.next.size()), int64_t(rt.n_vocab)}, logits);
        report["next_pos"] = pos0;
        report["n_next"] = o.next.size();
    }
    if (capturing) {
        std::ofstream out(capture.dir / "manifest.json");
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << json{{"report", report}, {"captures", capture.manifest}}.dump(1) << '\n';
        report["captures"] = capture.manifest.size();
    }
    std::cout << report.dump() << std::endl;
    return 0;
}

int main(int argc, char ** argv) {
    try {
        const options o = parse_options(argc, argv);
        llama_backend_init();
        const int status = run(o);
        llama_backend_free();
        return status;
    } catch (const std::exception & exc) {
        std::cerr << "kvpark-state-driver: error: " << exc.what() << '\n';
        return 1;
    }
}
