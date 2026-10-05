"""Compare chat.py renders with llama.cpp's own chat-template application, as text and as tokens.

For every conversation of a corpus v2 JSONL: chat.render with the model's GGUF chat template
(the text chat.py writes into rendered records) against kvpark-chat-apply (llama.cpp's
common_chat_templates_apply on the same messages and tools, as llama-server --jinja runs it,
then llama_tokenize with add_special). With --probe the rendered texts are also tokenized by
kvpark-kv-probe's tokenize mode (what collect/save-state/seed-score decode) and compared with
llama.cpp's token ids. Writes a JSON report (new file) and exits nonzero on any difference.

    template_check.py --model GGUF --server-binary llama-server --output REPORT.json
    template_check.py --model GGUF --url http://127.0.0.1:PORT --output REPORT.json
    (either with --conversations JSONL: those conversations instead of BATTERY)

The server comparison (compare_server) checks what the bridge engine relies on: for BATTERY
(chat with and without a system message, earlier reasoning, tool calls with their answers, two
calls in one message, arguments of every JSON type, non-ASCII text, and the shapes of real
traffic: whitespace at the edges of messages, NBSP and U+3000 there, string arguments with
newlines and quotes, float and null arguments, a Hermes agent transcript) under every
OPTION_SETS entry (reasoning_effort, enable_thinking, preserve_reasoning), llama-server's
/apply-template prompt against chat.render in each of chat.server_contexts, in the engine's order
(engine.render_target): parity when one of them gives the server's bytes; the report also counts
per conversation. --server-binary
starts a throwaway CPU llama-server on a tokenizer twin of the model (tokenizer_twin: a tiny
model carrying every tokenizer.* key of the GGUF, so its template, BOS/EOS text and vocabulary
are the real ones; no weights are loaded); --url compares with a running server of the model
itself.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request

import numpy as np

import chat
import collect_v2

# the throwaway servers of the tokenizer twins (KVPARK_TWIN_PORT moves the ten ports)
_TWIN_PORT = int(os.environ.get("KVPARK_TWIN_PORT", "18660"))
TWIN_PORTS = range(_TWIN_PORT, _TWIN_PORT + 10)
# the request shapes the bridge renders (validated by chat.validate_conversation, live)
TOOLS = [{"type": "function", "function": {
             "name": "get_weather", "description": "Current weather of a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                            "required": ["city"]}}},
         {"type": "function", "function": {
             "name": "search_notes", "description": "Search the user's notes",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                             "limit": {"type": "integer"}},
                            "required": ["query"]}}}]
HERMES_TOOLS = [{"type": "function", "function": {
                    "name": "terminal", "description": "Run a shell command and return its output.",
                    "parameters": {"type": "object", "properties": {
                        "command": {"type": "string", "description": "The command to run"},
                        "timeout": {"type": "integer", "description": "Seconds before it is stopped"}},
                        "required": ["command"]}}},
                {"type": "function", "function": {
                    "name": "write_file", "description": "Write a file, replacing what it held.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                    "content": {"type": "string"}},
                                   "required": ["path", "content"]}}}]
BATTERY = [
    {"id": "chat", "tools": None, "messages": [
        {"role": "system", "content": "You are a careful assistant. Answer briefly."},
        {"role": "user", "content": "What causes the tides along the coast?"},
        {"role": "assistant", "content": "Mostly the pull of the moon, and to a lesser degree the sun."},
        {"role": "user", "content": "And why are there two high tides a day?"}]},
    {"id": "no-system", "tools": None, "messages": [
        {"role": "user", "content": "Name a prime number between 10 and 20."},
        {"role": "assistant", "content": "13 is one."},
        {"role": "user", "content": "Another one, please."}]},
    {"id": "reasoning", "tools": None, "messages": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 17 times 3?"},
        {"role": "assistant", "content": "17 times 3 is 51.",
         "reasoning_content": "17 times 3: 10 times 3 is 30, 7 times 3 is 21, together 51."},
        {"role": "user", "content": "And 51 divided by 3?"},
        {"role": "assistant", "content": "It is 17.", "reasoning_content": "Division undoes the product: 17."},
        {"role": "user", "content": "Thanks. Now add the two answers."}]},
    {"id": "tools", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You can call tools. Use them when they help."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "The user wants the weather; call the tool.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Rain, 14 C"},
        {"role": "assistant", "content": "It is raining in Paris, 14 C."},
        {"role": "user", "content": "Find my notes about umbrellas, at most 3."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_2", "type": "function",
                         "function": {"name": "search_notes", "arguments": {"query": "umbrellas", "limit": 3}}}]},
        {"role": "tool", "tool_call_id": "call_2", "name": "search_notes", "content": "No notes found."},
        {"role": "user", "content": "Then remind me to buy one."}]},
    {"id": "tools-parallel", "tools": TOOLS, "messages": [
        {"role": "user", "content": "Weather in Rome and in Oslo?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
                        {"id": "call_b", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "tool_call_id": "call_a", "name": "get_weather", "content": "Sunny, 24 C"},
        {"role": "tool", "tool_call_id": "call_b", "name": "get_weather", "content": "Snow, -2 C"},
        {"role": "user", "content": "Which one is warmer?"}]},
    {"id": "tools-json", "tools": [{"type": "function", "function": {
        "name": "forecast", "description": "Forecast for a city: \"days\" ahead, <metric> & more (Zürich is fine)",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 14},
            "units": {"type": "string", "enum": ["metric", "imperial"]}, "detailed": {"type": "boolean"},
            "threshold": {"type": "number", "default": 0.5}, "tags": {"type": "array", "items": {"type": "string"}},
            "range": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}}},
            "required": ["city"]}}}], "messages": [
        {"role": "system", "content": "Use the forecast tool for weather questions."},
        {"role": "user", "content": "Detailed forecast for Zürich, next 3 days, rain and wind only."},
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "call_f", "type": "function", "function": {"name": "forecast", "arguments": {
             "city": "Zürich", "days": 3, "units": "metric", "detailed": True, "threshold": 0.25,
             "tags": ["rain", "wind"], "range": {"min": 1.0, "max": 10}}}}]},
        {"role": "tool", "tool_call_id": "call_f", "name": "forecast",
         "content": "{\"day1\": {\"rain_mm\": 4.5, \"wind_kmh\": 20}, \"day2\": null}"},
        {"role": "assistant", "content": "Day 1: 4.5 mm of rain, wind 20 km/h.\nDay 2: no data."},
        {"role": "user", "content": "Summarize in one line."}]},
    {"id": "unicode", "tools": None, "messages": [
        {"role": "system", "content": "Réponds en français, s'il te plaît."},
        {"role": "user", "content": "¿Qué significa 東京 y por qué 🚀?"},
        {"role": "assistant", "content": "東京 veut dire « capitale de l'Est ». La fusée 🚀 signifie le lancement."},
        {"role": "user", "content": "Und auf Deutsch? Grüße aus Zürich."}]},
    # ASCII whitespace at the edges, which many templates trim, and a whitespace-only tool answer
    {"id": "whitespace-edges", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a coding agent.\n\n"},
        {"role": "user", "content": "\n  Show me the notes about the deploy.  \n"},
        {"role": "assistant", "content": "Searching.\n", "reasoning_content": "\nI should search the notes.\n",
         "tool_calls": [{"id": "call_w", "type": "function",
                         "function": {"name": "search_notes", "arguments": {"query": " deploy ", "limit": 2}}}]},
        {"role": "tool", "tool_call_id": "call_w", "name": "search_notes",
         "content": "1. deploy on friday\n2. rollback plan\n"},
        {"role": "assistant", "content": "  Two notes: deploy on Friday, and a rollback plan.\n\n",
         "tool_calls": [{"id": "call_x", "type": "function",
                         "function": {"name": "search_notes", "arguments": {"query": "rollback"}}}]},
        {"role": "tool", "tool_call_id": "call_x", "name": "search_notes", "content": "\n"},
        {"role": "user", "content": "Thanks!\n"}]},
    # Unicode spaces at the edges: llama.cpp's trim keeps them
    {"id": "unicode-spaces", "tools": None, "messages": [
        {"role": "system", "content": "日本語で答えてください。"},
        {"role": "user", "content": "\u3000東京について教えて\u3000"},
        {"role": "assistant", "content": "東京は日本の首都です。\u00a0"},
        {"role": "user", "content": "Copied from a web page:\u00a0"}]},
    # string arguments tojson escapes (a template writing them JSON-escaped renders them without a span)
    {"id": "escaped-args", "tools": TOOLS, "messages": [
        {"role": "user", "content": "Note my two lines."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_e", "type": "function", "function": {"name": "search_notes", "arguments": {
             "query": "line one\nline 'two' \"quoted\"\ttab \\ back", "limit": 2}}}]},
        {"role": "tool", "tool_call_id": "call_e", "name": "search_notes", "content": "{\"hits\": []}"},
        {"role": "user", "content": "OK"}]},
    {"id": "float-null-args", "tools": TOOLS, "messages": [
        {"role": "user", "content": "Search notes about pi."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_p", "type": "function", "function": {
            "name": "search_notes", "arguments": {"query": "pi", "limit": 3.14159265, "eps": 1e-07, "cursor": None,
                                                  "opts": {"exact": False, "w": [0.1, 2.5e10]}}}}]},
        {"role": "tool", "tool_call_id": "call_p", "name": "search_notes", "content": "None."},
        {"role": "user", "content": "OK"}]},
    {"id": "hermes-agent", "tools": HERMES_TOOLS, "messages": [
        {"role": "system", "content": "# Role\nYou are Hermes, an agent working in the user's terminal.\n\n"
                                      "## Rules\n- Prefer small, safe commands.\n- Explain what you did.\n"},
        {"role": "user", "content": "Count the Python files under src/ and save the number to notes/count.txt"},
        {"role": "assistant", "content": "", "reasoning_content": "Use find and wc, then write the file.",
         "tool_calls": [{"id": "call_t1", "type": "function", "function": {"name": "terminal", "arguments": {
             "command": "find src -name '*.py' | wc -l", "timeout": 30}}}]},
        {"role": "tool", "tool_call_id": "call_t1", "name": "terminal",
         "content": "{\"output\": \"42\\n\", \"exit_code\": 0}\n"},
        {"role": "assistant", "content": "There are 42 files; saving the count.",
         "tool_calls": [{"id": "call_t2", "type": "function", "function": {"name": "write_file", "arguments": {
             "path": "notes/count.txt", "content": "Python files under src/: 42\n(counted with find | wc -l)\n"}}}]},
        {"role": "tool", "tool_call_id": "call_t2", "name": "write_file",
         "content": "Wrote 50 bytes to notes/count.txt\n"},
        {"role": "assistant", "content": "Done:\n\n```\nnotes/count.txt\n```\n\nIt holds **42**.\n"},
        {"role": "user", "content": "Thanks. Now show me the file."}]},
]
OPTION_SETS = (None, {"reasoning_effort": "low"}, {"reasoning_effort": "high"}, {"reasoning_effort": "none"},
               {"chat_template_kwargs": {"enable_thinking": False}},
               {"chat_template_kwargs": {"enable_thinking": True}},
               {"chat_template_kwargs": {"preserve_reasoning": False}},
               {"reasoning_effort": "medium", "chat_template_kwargs": {"preserve_reasoning": True}})


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(model, conversations, chat_apply, probe=None, *, workdir=None):
    """Report dict; report["identical"] is True when every text (and token list) agrees."""
    model = Path(model).resolve()
    info = chat.gguf_tokenizer_info(model)
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    template, bos = info["chat_template"], info["bos_token"]
    conversations = list(conversations)
    with tempfile.TemporaryDirectory(prefix=".kvpark-template-check-", dir=workdir) as temporary:
        source = Path(temporary) / "conversations.jsonl"
        source.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in conversations), encoding="utf-8")
        output = Path(temporary) / "applied.jsonl"
        result = subprocess.run([str(chat_apply), "--model", str(model), "--input", str(source), "--output", str(output)],
                                capture_output=True, text=True, timeout=600)
        if result.returncode not in (0, 2) or not output.is_file():
            raise RuntimeError(f"kvpark-chat-apply failed:\n{result.stderr[-3000:]}")
        applied = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        rendered = []
        for conversation in conversations:
            try:
                rendered.append(chat.render(conversation, template, bos_token=bos))
            except ValueError as exc:
                rendered.append(exc)
        tokens = {}
        if probe is not None:
            records = [{"document_id": chat.document_id(c), "text": r.text}
                       for c, r in zip(conversations, rendered) if not isinstance(r, Exception)]
            if records:
                tokens = collect_v2.tokenize(probe, model, records, ctx=32768, workdir=temporary)
    entries, counts = [], {"conversations": len(conversations), "with_tools": 0, "with_tool_calls": 0,
                           "text_identical": 0, "tokens_identical": 0, "tokens_checked": 0,
                           "render_errors": 0, "llama_cpp_errors": 0}
    for conversation, render, reference in zip(conversations, rendered, applied):
        has_tools = bool(conversation.get("tools"))
        has_calls = any(m.get("tool_calls") for m in conversation["messages"])
        counts["with_tools"] += has_tools
        counts["with_tool_calls"] += has_calls
        entry = {"id": conversation.get("id"), "tools": has_tools, "tool_calls": has_calls}
        if isinstance(render, Exception) or "error" in reference:
            counts["render_errors"] += isinstance(render, Exception)
            counts["llama_cpp_errors"] += "error" in reference
            entry.update(render_error=str(render) if isinstance(render, Exception) else None,
                         llama_cpp_error=reference.get("error"))
            entries.append(entry)
            continue
        text_equal = render.text == reference["prompt"]
        counts["text_identical"] += text_equal
        entry["text_identical"] = text_equal
        if not text_equal:
            entry["first_text_difference"] = first_difference(render.text, reference["prompt"])
        document = chat.document_id(conversation)
        if document in tokens:
            ids = tokens[document]["token_id"].tolist()
            counts["tokens_checked"] += 1
            token_equal = ids == reference["token_ids"]
            counts["tokens_identical"] += token_equal
            entry.update(tokens_identical=token_equal, n_tokens=len(reference["token_ids"]))
            if not token_equal:
                entry["first_token_difference"] = first_difference(ids, reference["token_ids"])
        entries.append(entry)
    identical = (counts["text_identical"] == counts["conversations"]
                 and (probe is None or counts["tokens_identical"] == counts["conversations"]))
    return {"model": str(model), "template_sha256": chat.template_sha256(template), "bos_token": bos,
            "add_bos": info["add_bos"], "chat_apply": str(chat_apply), "probe": None if probe is None else str(probe),
            "identical": identical, "counts": counts,
            "n_tokens": int(sum(e.get("n_tokens", 0) for e in entries)),
            "differences": [e for e in entries if not (e.get("text_identical") and e.get("tokens_identical", True))]}


def tokenizer_twin(model, path, *, seed=0):
    """Write a tiny qwen3 GGUF (tiny_fixture's four layers, 64 wide) carrying every tokenizer.*
    key of `model` (vocabulary, types, merges or scores, BOS/EOS, add_bos, pre-tokenizer, chat
    templates): a llama-server on it applies the chat template and tokenizes exactly as one on
    the real model, and loads in a second. The embedding and output rows are random F16, one per
    token of the real vocabulary."""
    import tiny_fixture
    gguf = tiny_fixture._gguf_module()
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    real = gguf.GGUFReader(str(model))
    tokens = real.get_field("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError(f"{model} has no vocabulary")
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        base_path = Path(directory) / "base.gguf"
        tiny_fixture.create_fixture(base_path, "qwen3", seed=seed)
        base = gguf.GGUFReader(str(base_path))
        writer = gguf.GGUFWriter(str(path), "qwen3")
        for source, keep in ((base, lambda name: not name.startswith("tokenizer.")),
                             (real, lambda name: name.startswith("tokenizer."))):
            for field in source.fields.values():
                if field.name == "general.architecture" or field.name.startswith("GGUF.") or not keep(field.name):
                    continue
                kind = field.types[0]
                if kind == gguf.GGUFValueType.ARRAY and not field.data:
                    continue                    # gguf-py cannot write empty arrays
                writer.add_key_value(field.name, field.contents(), kind,
                                     sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
        rng = np.random.default_rng(seed)
        for tensor in base.tensors:
            if tensor.name in ("token_embd.weight", "output.weight"):
                rows = rng.normal(0, 0.05, (len(tokens.data), tensor.data.shape[-1])).astype(np.float16)
                writer.add_tensor(tensor.name, rows)
            else:
                writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    return path


def server_request(url, method, path, body=None, timeout=300):
    """JSON answer of a llama-server endpoint; an HTTP error answer raises RuntimeError with its body."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        with error:
            text = error.read()[:600].decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {text}") from None


def served_prompt(url, conversation, options=None):
    """llama-server's /apply-template prompt for a conversation and request options."""
    body = {"messages": conversation["messages"], **(options or {})}
    if conversation.get("tools"):
        body["tools"] = conversation["tools"]
    return server_request(url, "POST", "/apply-template", body)["prompt"]


def _around(text, at, width=60):
    return text[max(0, at - width):at + width]


def compare_server(url, template, bos_token, eos_token, conversations=None, option_sets=OPTION_SETS, *,
                   add_bos=False, add_eos=False):
    """Report dict: per conversation and options, the server's /apply-template prompt (or its
    error) against chat.render in each chat.server_contexts candidate, in the engine's order;
    "context" names the first whose rendering is the server's bytes (None: no parity), and
    "escaped" the segments that rendering writes JSON-escaped (no span for the engine).
    report["parity"] is True when every case the server renders has a matching context;
    report["by_conversation"] counts each conversation's cases.

    add_bos / add_eos: the vocabulary adds BOS / EOS as tokens, so llama.cpp drops the template's
    BOS text from the start of the prompt and its EOS text from the end
    (common_chat_template_direct_apply_impl); the engine accepts that leading token
    (engine._spelling_shift), and the comparison drops the same text."""
    conversations = BATTERY if conversations is None else conversations

    def as_served(text):
        if add_bos and bos_token and text.startswith(bos_token):
            text = text[len(bos_token):]
        if add_eos and eos_token and text.endswith(eos_token):
            text = text[:-len(eos_token)]
        return text
    cases, by_conversation = [], {}
    counts = {"cases": 0, "matched": 0, "server_errors": 0, "both_errors": 0, "mismatch": 0, "chat_py_errors": 0}
    for conversation in conversations:
        conversation = copy.deepcopy(conversation)
        for options in option_sets:
            counts["cases"] += 1
            case = {"conversation": conversation.get("id"), "options": options}
            try:
                prompt = served_prompt(url, conversation, options)
            except RuntimeError as error:
                prompt, case["server_error"] = None, str(error)[:300]
            renderings, errors = {}, {}
            try:
                contexts = chat.server_contexts(options, eos_token=eos_token)
            except ValueError as error:
                contexts, errors["options"] = [], str(error)
            escaped = {}
            for name, context in contexts:
                try:
                    rendered = chat.render({"messages": conversation["messages"], "tools": conversation.get("tools")},
                                           template, bos_token=bos_token, add_generation_prompt=True,
                                           extra_context=context, live=True)
                    renderings[name] = as_served(rendered.text)
                    escaped[name] = [rendered.segment_keys[number] for number in rendered.escaped]
                except Exception as error:      # noqa: BLE001  (jinja2 errors, transformed content)
                    errors[name] = f"{type(error).__name__}: {error}"[:300]
            if errors:
                case["chat_py_errors"] = errors
            if prompt is None:
                outcome = "server_errors"
                counts["both_errors"] += not renderings
            else:
                case["context"] = next((name for name, text in renderings.items() if text == prompt), None)
                if case["context"] is not None:
                    outcome = "matched"
                    if escaped[case["context"]]:
                        case["escaped"] = escaped[case["context"]]
                elif renderings:
                    outcome = "mismatch"
                    # the candidate that agrees longest
                    name, text = max(renderings.items(), key=lambda item: first_difference(item[1], prompt) or 0)
                    at = first_difference(text, prompt)
                    case.update(compared=name, first_difference=at, server=_around(prompt, at),
                                chat_py=_around(text, at))
                else:
                    outcome = "chat_py_errors"
                    case["server"] = prompt[:300]
            counts[outcome] += 1
            mine = by_conversation.setdefault(conversation.get("id"), dict.fromkeys(
                ("cases", "matched", "server_errors", "mismatch", "chat_py_errors"), 0))
            mine["cases"] += 1
            mine[outcome] += 1
            cases.append(case)
    rendered = counts["cases"] - counts["server_errors"]
    return {"parity": counts["matched"] == rendered, "counts": counts, "by_conversation": by_conversation,
            "contexts_used": sorted({case["context"] for case in cases if case.get("context")}),
            "cases": cases}


def check_server(model, *, url=None, server_binary=None, workdir=None, ports=TWIN_PORTS, conversations=None):
    """compare_server for a GGUF (on `conversations`, default BATTERY): against a running server of
    it (url) or a throwaway CPU llama-server (server_binary) on its tokenizer twin. The report also
    names the server's template capabilities and whether its /props template is the GGUF's."""
    import fixtures_media
    model = Path(model).resolve()
    info = chat.gguf_tokenizer_info(model)
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    with tempfile.TemporaryDirectory(prefix=".kvpark-template-check-", dir=workdir) as temporary:
        server = None
        if url is None:
            twin = tokenizer_twin(model, Path(temporary) / "twin.gguf")
            slots = Path(temporary) / "slots"
            slots.mkdir()
            server = fixtures_media.Server(server_binary, twin, slots, Path(temporary) / "server.log",
                                           fixtures_media.free_port(ports), ctx=4096, extra=["--jinja"])
            url = server.url
        try:
            props = server_request(url, "GET", "/props")

            def tokens(text, **flags):
                return [t if isinstance(t, int) else t["id"]
                        for t in server_request(url, "POST", "/tokenize", {"content": text, **flags})["tokens"]]
            special = {}
            for key in ("bos_token", "eos_token"):
                ids = tokens(info[key], add_special=False, parse_special=True) if info[key] else []
                special[key] = ids[0] if len(ids) == 1 else None
            # what the vocabulary adds around a prompt (llama_vocab add_bos / add_eos): the tokens of
            # a text, not of an empty one, so that a BOS that is also the EOS is not taken for both
            plain, full = tokens("a", add_special=False), tokens("a", add_special=True)
            start = 1 if full[:1] == [special["bos_token"]] and full[1:1 + len(plain)] == plain else 0
            add_bos = start == 1
            add_eos = special["eos_token"] is not None and full[start + len(plain):] == [special["eos_token"]]
            report = compare_server(url, info["chat_template"], info["bos_token"], info["eos_token"],
                                    conversations, add_bos=add_bos, add_eos=add_eos)
        finally:
            if server is not None:
                server.close()
    report.update(model=str(model), server="twin" if server is not None else url,
                  template_sha256=chat.template_sha256(info["chat_template"]),
                  props_template_is_gguf=props.get("chat_template") == info["chat_template"],
                  tool_use_template=bool(props.get("chat_template_tool_use")),
                  caps=props.get("chat_template_caps"), bos_token=info["bos_token"], eos_token=info["eos_token"],
                  add_bos=add_bos, add_eos=add_eos, build_info=props.get("build_info"))
    return report


def _write(output, report):
    """Write the report as a new file (never overwritten)."""
    with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=".kvpark-template-check-",
                                     delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.link(stream.name, output)
    os.unlink(stream.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--conversations", type=Path, help="corpus v2 JSONL (needed with --chat-apply); with "
                        "--server-binary or --url, the conversations to compare instead of BATTERY")
    parser.add_argument("--chat-apply", type=Path, help="kvpark-chat-apply binary")
    parser.add_argument("--probe", type=Path, help="kvpark-kv-probe: also compare token ids")
    parser.add_argument("--server-binary", type=Path, help="llama-server: compare /apply-template on a tokenizer twin")
    parser.add_argument("--url", help="a running llama-server of the model: compare its /apply-template")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if sum(x is not None for x in (args.chat_apply, args.server_binary, args.url)) != 1:
        raise ValueError("give exactly one of --chat-apply, --server-binary and --url")
    if args.chat_apply is not None and args.conversations is None:
        raise ValueError("--chat-apply needs --conversations")
    if args.chat_apply is None:
        report = check_server(args.model, url=args.url,
                              server_binary=None if args.server_binary is None else args.server_binary.resolve(),
                              workdir=args.output.parent, conversations=None if args.conversations is None
                              else list(chat.read_conversations(args.conversations)))
        _write(args.output, report)
        print(json.dumps({"output": str(args.output), "parity": report["parity"], **report["counts"]}))
        return 0 if report["parity"] else 1
    report = compare(args.model, chat.read_conversations(args.conversations), args.chat_apply.resolve(),
                     None if args.probe is None else args.probe.resolve(), workdir=args.output.parent)
    _write(args.output, report)
    print(json.dumps({"output": str(args.output), "identical": report["identical"], **report["counts"]}))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
