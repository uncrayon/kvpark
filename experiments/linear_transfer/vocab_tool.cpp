// kvpark-vocab: llama.cpp's own vocabulary, read with vocab_only (no weights are loaded).
//
//   kvpark-vocab MODEL info                     -> one JSON object: type, n_tokens, add_bos, bos, ...
//   kvpark-vocab MODEL pieces OUT.jsonl         -> per token: {"id", "attr", "piece": hex}
//                                                  (llama_token_to_piece, special = true, lstrip 0,
//                                                  as llama-server's /tokenize with_pieces)
//   kvpark-vocab MODEL tokenize IN.jsonl OUT.jsonl
//                                               -> per input line {"text", "add_special", "parse_special"}:
//                                                  {"ids": [...], "pieces": [hex, ...]}
//
// The reference experiments/universal_kv/test_engine.py RealVocabularyTest compares the engine's
// vocabularies with (KVPARK_VOCAB_TOOL). Built like the other tools, against either pinned runtime:
// build_collector.py --target kvpark-vocab (README, "Tests on both pinned runtimes").

#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::string hex(const std::string & bytes) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) {
        out.push_back(digits[c >> 4]);
        out.push_back(digits[c & 15]);
    }
    return out;
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(64);
    int n = llama_token_to_piece(vocab, token, buf.data(), (int) buf.size(), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int) buf.size(), 0, true);
    }
    return std::string(buf.data(), n);
}

static std::string meta(const llama_model * model, const char * key) {
    char buf[512];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf, std::min<int>(n, sizeof(buf) - 1));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL info|pieces OUT|tokenize IN OUT\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[2];
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) {
        fprintf(stderr, "failed to load the vocabulary of %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_tokens = llama_vocab_n_tokens(vocab);

    if (mode == "info") {
        json info = {
            {"type", llama_vocab_type(vocab)},
            {"n_tokens", n_tokens},
            {"add_bos", llama_vocab_get_add_bos(vocab)},
            {"add_eos", llama_vocab_get_add_eos(vocab)},
            {"bos", llama_vocab_bos(vocab)},
            {"eos", llama_vocab_eos(vocab)},
            {"tokenizer_model", meta(model, "tokenizer.ggml.model")},
            {"tokenizer_pre", meta(model, "tokenizer.ggml.pre")},
            {"add_space_prefix", meta(model, "tokenizer.ggml.add_space_prefix")},
        };
        std::cout << info.dump() << "\n";
    } else if (mode == "pieces" && argc == 4) {
        std::ofstream out(argv[3]);
        for (llama_token id = 0; id < n_tokens; ++id) {
            json line = {{"id", id}, {"attr", (int) llama_vocab_get_attr(vocab, id)}, {"piece", hex(piece(vocab, id))}};
            out << line.dump() << "\n";
        }
    } else if (mode == "tokenize" && argc == 5) {
        std::ifstream in(argv[3]);
        std::ofstream out(argv[4]);
        std::string text;
        while (std::getline(in, text)) {
            if (text.empty()) continue;
            json request = json::parse(text);
            const std::string content = request.at("text");
            const bool add_special = request.value("add_special", false);
            const bool parse_special = request.value("parse_special", true);
            std::vector<llama_token> ids(content.size() + 16);
            int n = llama_tokenize(vocab, content.data(), (int) content.size(), ids.data(), (int) ids.size(),
                                   add_special, parse_special);
            if (n < 0) {
                ids.resize(-n);
                n = llama_tokenize(vocab, content.data(), (int) content.size(), ids.data(), (int) ids.size(),
                                   add_special, parse_special);
            }
            ids.resize(n);
            json pieces = json::array();
            for (llama_token id : ids) pieces.push_back(hex(piece(vocab, id)));
            out << json{{"ids", ids}, {"pieces", pieces}}.dump() << "\n";
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
