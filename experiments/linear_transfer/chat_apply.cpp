// llama.cpp's own chat-template application, as a reference for chat.py renders.
//
// For every corpus conversation ({"messages": [...], "tools": [...] | null, ...} per line) this
// runs what llama-server does for a chat request with --jinja: common_chat_msgs_parse_oaicompat
// and common_chat_tools_parse_oaicompat on the OpenAI-style JSON, common_chat_templates_apply
// with the model's tokenizer.chat_template (add_generation_prompt off unless asked, the other
// inputs at their defaults, e.g. enable_thinking on), then llama_tokenize of the prompt with
// add_special and parse_special on (the server's tokenize_input_prompts). The model is loaded
// vocabulary-only. One output line per input line, in order:
//   {"line": n, "id": <input id or null>, "prompt": "...", "token_ids": [...]}
// A conversation the template refuses gets {"line": n, "id": ..., "error": "..."} instead.

#include "chat.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ojson = nlohmann::ordered_json;

static const char * usage =
    "--model GGUF --input CONVERSATIONS.jsonl --output NEW.jsonl [--add-generation-prompt]\n";

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 2);
    int count = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), true, true);
    if (count < 0 && count != INT32_MIN) {
        tokens.resize(size_t(-int64_t(count)));
        count = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), true, true);
    }
    if (count < 0) { throw std::runtime_error("tokenization failed"); }
    tokens.resize(size_t(count));
    return tokens;
}

static int run(int argc, char ** argv) {
    fs::path model, input, output;
    bool add_generation_prompt = false;
    std::set<std::string> given;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { std::cout << usage; return 0; }
        if (!given.insert(key).second) { throw std::runtime_error("option " + key + " given more than once"); }
        if (key == "--add-generation-prompt") { add_generation_prompt = true; continue; }
        if (key != "--model" && key != "--input" && key != "--output") { throw std::runtime_error("unknown option " + key); }
        if (++i == argc) { throw std::runtime_error("missing argument for " + key); }
        (key == "--model" ? model : key == "--input" ? input : output) = argv[i];
    }
    if (model.empty() || input.empty() || output.empty()) { throw std::runtime_error(usage); }
    if (fs::exists(output) || fs::is_symlink(output)) { throw std::runtime_error("output already exists"); }

    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> loaded(
        llama_model_load_from_file(model.string().c_str(), params), llama_model_free);
    if (!loaded) { throw std::runtime_error("model load failed"); }
    const auto * vocab = llama_model_get_vocab(loaded.get());
    auto templates = common_chat_templates_init(loaded.get(), "");

    std::ifstream in(input);
    if (!in) { throw std::runtime_error("cannot open input"); }
    const fs::path partial = output.string() + ".partial";
    std::ofstream out(partial, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    size_t number = 0, failed = 0;
    for (std::string line; std::getline(in, line);) {
        ++number;
        const ojson record = ojson::parse(line);
        ojson result = {{"line", number}, {"id", record.contains("id") ? record["id"] : ojson()}};
        try {
            common_chat_templates_inputs inputs;
            inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(record.at("messages").dump()));
            if (record.contains("tools") && !record["tools"].is_null()) {
                inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(record["tools"].dump()));
            }
            inputs.use_jinja = true;
            inputs.add_generation_prompt = add_generation_prompt;
            const auto applied = common_chat_templates_apply(templates.get(), inputs);
            result["prompt"] = applied.prompt;
            result["token_ids"] = tokenize(vocab, applied.prompt);
        } catch (const std::exception & exc) {
            result["error"] = exc.what();
            ++failed;
        }
        out << result.dump(-1, ' ', false, ojson::error_handler_t::strict) << '\n';
    }
    out.close();
    fs::rename(partial, output);
    std::cerr << "applied the chat template to " << number - failed << " of " << number << " conversations\n";
    templates.reset();
    loaded.reset();
    llama_backend_free();
    return failed == 0 ? 0 : 2;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & exc) { std::cerr << "kvpark-chat-apply: " << exc.what() << '\n'; return 1; }
}
