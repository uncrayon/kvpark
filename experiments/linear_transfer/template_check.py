"""Compare chat.py renders with llama.cpp's own chat-template application, as text and as tokens.

For every conversation of a corpus v2 JSONL: chat.render with the model's GGUF chat template
(the text chat.py writes into rendered records) against kvpark-chat-apply (llama.cpp's
common_chat_templates_apply on the same messages and tools, as llama-server --jinja runs it,
then llama_tokenize with add_special). With --probe the rendered texts are also tokenized by
kvpark-kv-probe's tokenize mode (what collect/save-state/seed-score decode) and compared with
llama.cpp's token ids. Writes a JSON report (new file) and exits nonzero on any difference.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

import chat
import collect_v2


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--conversations", required=True, type=Path, help="corpus v2 JSONL")
    parser.add_argument("--chat-apply", required=True, type=Path, help="kvpark-chat-apply binary")
    parser.add_argument("--probe", type=Path, help="kvpark-kv-probe: also compare token ids")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    report = compare(args.model, chat.read_conversations(args.conversations), args.chat_apply.resolve(),
                     None if args.probe is None else args.probe.resolve(), workdir=args.output.parent)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, prefix=".kvpark-template-check-",
                                     delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.link(stream.name, args.output)
    os.unlink(stream.name)
    print(json.dumps({"output": str(args.output), "identical": report["identical"], **report["counts"]}))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
