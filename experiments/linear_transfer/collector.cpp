#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct options {
    fs::path model, input, output;
    int layer = -1, ctx = 1024, batch = 128, threads = 4;
    bool tokenize_only = false;
};

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--tokenize-only") { result.tokenize_only = true; continue; }
        if (key == "--help") {
            std::cout << "--model GGUF --input corpus.jsonl --output NEW_DIRECTORY --layer N "
                         "[--ctx 1024 --batch 128 --threads 4 --tokenize-only]\n";
            std::exit(0);
        }
        if (++i == argc) { throw std::runtime_error("missing argument for " + key); }
        const std::string value = argv[i];
        if      (key == "--model")  { result.model = value; }
        else if (key == "--input")  { result.input = value; }
        else if (key == "--output") { result.output = value; }
        else {
            size_t used = 0;
            const int number = std::stoi(value, &used);
            if (used != value.size()) { throw std::runtime_error("invalid integer for " + key); }
            if      (key == "--layer")   { result.layer = number; }
            else if (key == "--ctx")     { result.ctx = number; }
            else if (key == "--batch")   { result.batch = number; }
            else if (key == "--threads") { result.threads = number; }
            else { throw std::runtime_error("unknown option " + key); }
        }
    }
    if (result.model.empty() || result.input.empty() || result.output.empty() || result.layer < 0
            || result.ctx < 1 || result.ctx > 32768 || result.batch < 1 || result.batch > result.ctx
            || result.threads < 1) {
        throw std::runtime_error("model, input, new output directory, valid layer and positive sizes required; ctx <= 32768");
    }
    return result;
}

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

struct layout {
    std::string architecture, attention_type, file_type;
    int64_t heads, dimension;
};

static layout inspect_model(const options & args) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
        gguf_init_from_file(args.model.string().c_str(), {true, nullptr}), gguf_free);
    if (!meta) { throw std::runtime_error("cannot read GGUF metadata"); }
    if (integer(meta.get(), "split.count", 1) != 1) {
        throw std::runtime_error("initial collector requires a single-file GGUF");
    }
    const auto arch_id = gguf_find_key(meta.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(meta.get(), arch_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF architecture is missing");
    }
    layout result;
    result.architecture = gguf_get_val_str(meta.get(), arch_id);
    const std::string prefix = result.architecture + ".";
    if (args.layer >= integer(meta.get(), prefix + "block_count")) {
        throw std::runtime_error("layer is outside the model");
    }
    if (result.architecture == "qwen35") {
        const auto interval = integer(meta.get(), prefix + "full_attention_interval", 4);
        if (interval <= 0) { throw std::runtime_error("invalid full-attention interval"); }
        if (integer(meta.get(), prefix + "attention.recurrent_layers", (args.layer + 1) % interval != 0, args.layer)) {
            throw std::runtime_error("selected Qwen layer is recurrent, not a value-cache layer");
        }
        result.attention_type = "full";
    } else if (result.architecture == "muse-glimmer") {
        const std::string pattern = prefix + "attention.sliding_window_pattern";
        const auto id = gguf_find_key(meta.get(), pattern.c_str());
        bool sliding;
        if (id >= 0 && gguf_get_kv_type(meta.get(), id) == GGUF_TYPE_ARRAY) {
            sliding = integer(meta.get(), pattern, -1, args.layer) != 0;
        } else {
            const auto period = integer(meta.get(), pattern, 4);
            if (period <= 0) { throw std::runtime_error("invalid sliding-window period"); }
            sliding = (args.layer + 1) % period != 0;
        }
        result.attention_type = sliding ? "sliding" : "full";
    } else {
        throw std::runtime_error("initial collector supports only qwen35 and muse-glimmer");
    }
    result.heads = integer(meta.get(), prefix + "attention.head_count_kv", -1, args.layer);
    const auto query_heads = integer(meta.get(), prefix + "attention.head_count", -1, args.layer);
    if (query_heads <= 0) { throw std::runtime_error("invalid attention head count"); }
    result.dimension = integer(meta.get(), prefix + "attention.value_length",
                               integer(meta.get(), prefix + "embedding_length") / query_heads, args.layer);
    if (result.heads <= 0 || result.dimension <= 0 || result.heads * result.dimension > 65536) {
        throw std::runtime_error("invalid value-cache dimensions");
    }
    result.file_type = std::to_string(integer(meta.get(), "general.file_type", 0));
    return result;
}

struct capture_state {
    std::string name, error;
    int64_t heads, dimension;
    bool active = false;
    size_t expected_rows = 0;
    std::vector<float> values;
};

static bool capture_callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<capture_state *>(opaque);
    const bool selected = state.active && state.name == tensor->name
        && (tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_VIEW)
        && tensor->ne[0] == state.dimension && tensor->ne[1] == state.heads && tensor->ne[3] == 1;
    if (ask) { return selected; }
    if (!selected) { return true; }
    try {
        const size_t width = size_t(state.heads * state.dimension);
        if (tensor->type != GGML_TYPE_F32 || tensor->ne[2] <= 0
                || state.values.size() / width + size_t(tensor->ne[2]) > state.expected_rows) {
            throw std::runtime_error("unexpected value tensor dtype or duplicate/excess rows");
        }
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        for (int64_t token = 0; token < tensor->ne[2]; ++token) {
            for (int64_t head = 0; head < state.heads; ++head) {
                for (int64_t dim = 0; dim < state.dimension; ++dim) {
                    const size_t offset = token * tensor->nb[2] + head * tensor->nb[1] + dim * tensor->nb[0];
                    if (offset + sizeof(float) > bytes.size()) { throw std::runtime_error("invalid tensor strides"); }
                    float value;
                    std::memcpy(&value, bytes.data() + offset, sizeof(value));
                    if (!std::isfinite(value)) { throw std::runtime_error("nonfinite captured value"); }
                    state.values.push_back(value);
                }
            }
        }
        return true;
    } catch (const std::exception & exc) {
        state.error = exc.what();
        return false;
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, std::vector<int64_t> & ends) {
    std::vector<llama_token> tokens(text.size() + 1);
    int count = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), false, false);
    if (count < 0) { throw std::runtime_error("tokenization failed"); }
    tokens.resize(count);
    if (tokens.empty()) { throw std::runtime_error("document produced no tokens"); }
    std::string reconstructed;
    for (const auto token : tokens) {
        std::vector<char> piece(256);
        int length = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
        if (length < 0) {
            piece.resize(size_t(-int64_t(length)));
            length = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
        }
        if (length <= 0) { throw std::runtime_error("empty or invalid token piece"); }
        reconstructed.append(piece.data(), length);
        ends.push_back(int64_t(reconstructed.size()));
    }
    if (reconstructed != text) { throw std::runtime_error("token pieces do not reproduce the exact UTF-8 input"); }
    return tokens;
}

static json read_documents(const fs::path & path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open corpus"); }
    json documents = json::array();
    std::set<std::string> seen;
    for (std::string line; std::getline(input, line);) {
        const auto doc = json::parse(line);
        const auto hash = doc.at("document_sha256").get<std::string>();
        const auto text = doc.at("text").get<std::string>();
        if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos
                || !seen.insert(hash).second || text.empty() || text.size() > 1024 * 1024) {
            throw std::runtime_error("invalid or duplicate document; limit is 1 MiB of UTF-8 per document");
        }
        documents.push_back(doc);
    }
    if (input.bad() || documents.empty()) { throw std::runtime_error("empty or unreadable corpus"); }
    return documents;
}

static int collect(const options & args) {
    const uint32_t one = 1;
    if (*reinterpret_cast<const uint8_t *>(&one) != 1) { throw std::runtime_error("little-endian host required"); }
    if (fs::exists(args.output)) { throw std::runtime_error("output directory already exists"); }
    const auto info = inspect_model(args);
    const auto documents = read_documents(args.input);
    llama_backend_init();
    struct backend_guard { ~backend_guard() { llama_backend_free(); } } guard;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    if (!devices[0]) { throw std::runtime_error("CPU backend unavailable; use a runtime with GGML_BACKEND_DL=OFF"); }
    auto model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.use_extra_bufts = false;
    model_params.load_mtp = false;
    model_params.vocab_only = args.tokenize_only;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(args.model.string().c_str(), model_params), llama_model_free);
    if (!model) { throw std::runtime_error("model load failed"); }
    capture_state capture;
    capture.name = "Vcur-" + std::to_string(args.layer);
    capture.heads = info.heads;
    capture.dimension = info.dimension;
    auto params = llama_context_default_params();
    params.n_ctx = args.ctx;
    params.n_batch = params.n_ubatch = args.batch;
    params.n_seq_max = 1;
    params.n_threads = params.n_threads_batch = args.threads;
    params.type_k = params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = params.op_offload = false;
    params.cb_eval = capture_callback;
    params.cb_eval_user_data = &capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        args.tokenize_only ? nullptr : llama_init_from_model(model.get(), params), llama_free);
    if (!args.tokenize_only && !ctx) { throw std::runtime_error("context creation failed"); }
    llama_batch batch = llama_batch_init(args.batch, 0, 1);
    struct batch_guard { llama_batch batch; ~batch_guard() { llama_batch_free(batch); } } batch_cleanup{batch};
    if (!fs::create_directories(args.output)) { throw std::runtime_error("cannot create output directory"); }
    json report = {{"format_version", 1}, {"architecture", info.architecture}, {"file_type", info.file_type},
                   {"layer", args.layer}, {"attention_type", info.attention_type}, {"cache_dtype", "f32"},
                   {"extraction", "Vcur_pre_cache_f32"}, {"tokenize_only", args.tokenize_only},
                   {"documents", json::array()}};
    for (const auto & doc : documents) {
        const auto hash = doc.at("document_sha256").get<std::string>();
        const auto text = doc.at("text").get<std::string>();
        std::vector<int64_t> ends;
        auto tokens = tokenize(llama_model_get_vocab(model.get()), text, ends);
        if (tokens.size() > size_t(args.ctx)) { throw std::runtime_error("document exceeds --ctx; no truncation is performed"); }
        json entry = {{"document_sha256", hash}, {"token_ids", tokens}, {"byte_end", ends},
                      {"shape", {tokens.size(), info.heads, info.dimension}}};
        if (!args.tokenize_only) {
            llama_synchronize(ctx.get());
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            const std::string filename = hash + ".f32";
            std::ofstream values(args.output / filename, std::ios::binary);
            values.exceptions(std::ios::badbit | std::ios::failbit);
            for (size_t start = 0; start < tokens.size(); start += args.batch) {
                batch.n_tokens = int(std::min(size_t(args.batch), tokens.size() - start));
                for (int i = 0; i < batch.n_tokens; ++i) {
                    batch.token[i] = tokens[start + i];
                    batch.pos[i] = int(start + i);
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = i == batch.n_tokens - 1;
                }
                capture.values.clear();
                capture.error.clear();
                capture.expected_rows = size_t(batch.n_tokens);
                capture.active = true;
                const int status = llama_decode(ctx.get(), batch);
                llama_synchronize(ctx.get());
                capture.active = false;
                if (!capture.error.empty()) { throw std::runtime_error(capture.error); }
                if (status != 0 || capture.values.size() != size_t(batch.n_tokens * info.heads * info.dimension)) {
                    throw std::runtime_error("decode failed or final Vcur callback row count mismatched");
                }
                values.write(reinterpret_cast<const char *>(capture.values.data()), std::streamsize(capture.values.size() * sizeof(float)));
            }
            values.close();
            entry["values_file"] = filename;
        }
        report["documents"].push_back(entry);
        std::cerr << "captured document " << report["documents"].size() << ": " << tokens.size() << " tokens\n";
    }
    std::ofstream output(args.output / "report.json");
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << report.dump(2) << '\n';
    output.close();
    return 0;
}

int main(int argc, char ** argv) {
    try { return collect(parse_options(argc, argv)); }
    catch (const std::exception & exc) { std::cerr << "collector: " << exc.what() << '\n'; return 1; }
}
