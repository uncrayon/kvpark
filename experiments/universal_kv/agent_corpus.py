"""Coding-agent trajectories as a pair_eval corpus (chat.py format).

Reads samples.jsonl files of resolved agent trajectories (OpenAI messages with tool calls and a
tools.json beside each, the layout of the Gemma 4 Developer Agent experiments' data/sft/<set>/) and
writes one conversation per task: the trajectory cut at its first assistant turn whose start lies past
--cut of the trajectory's characters, ending with that turn (its reasoning and tool call are what
pair_eval scores). One trajectory per task, the first in file order, so the document bootstrap
resamples independent tasks.

    python agent_corpus.py --input SET_DIR [--input SET_DIR ...] --output DIR [--cut 0.5]

Writes DIR/conversations.jsonl and DIR/manifest.json (the inputs' sha256, the rule, what was left out
and why). The conversations hold competition data: keep DIR out of anything published.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import chat  # noqa: E402

KEEP = {"system": ("role", "content"), "user": ("role", "content"),
        "assistant": ("role", "content", "reasoning_content", "tool_calls"),
        "tool": ("role", "tool_call_id", "name", "content")}


def clean(messages):
    """chat.py's message rules: known keys only, texts stripped, tool names from their calls, empty
    reasoning dropped."""
    names, out = {}, []
    for message in messages:
        role = message["role"]
        item = {key: message[key] for key in KEEP[role] if key in message}
        content = item.get("content") or ""
        if isinstance(content, list):  # OpenAI content parts
            content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        item["content"] = content.strip()
        if role == "assistant":
            item["reasoning_content"] = (item.get("reasoning_content") or "").strip()
            if not item["reasoning_content"]:
                item.pop("reasoning_content")
            if item.get("tool_calls"):
                item["tool_calls"] = [{"id": call["id"], "type": "function",
                                       "function": {"name": call["function"]["name"],
                                                    "arguments": call["function"]["arguments"]}}
                                      for call in item["tool_calls"]]
                names.update((call["id"], call["function"]["name"]) for call in item["tool_calls"])
            else:
                item.pop("tool_calls", None)
        elif role == "tool":
            item["name"] = message.get("name") or names.get(message["tool_call_id"], "")
        out.append(item)
    return out


def cut(messages, fraction):
    """The prefix ending with the first assistant turn that starts past `fraction` of the characters."""
    sizes = [len(json.dumps(message, ensure_ascii=False)) for message in messages]
    total, seen = sum(sizes), 0
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and seen >= fraction * total and index >= 3:
            return messages[:index + 1]
        seen += sizes[index]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cut", type=float, default=0.5)
    args = parser.parse_args(argv)
    rows, skipped, seen_tasks, inputs = [], [], set(), []
    for directory in args.input:
        samples, tools_file = directory / "samples.jsonl", directory / "tools.json"
        tools = json.loads(tools_file.read_text())
        inputs.append({"set": directory.name, **{name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for name, path in (("samples", samples), ("tools", tools_file))}})
        for line in samples.read_text().splitlines():
            sample = json.loads(line)
            task = sample["task_id"]
            if task in seen_tasks:
                skipped.append({"set": directory.name, "task": task, "why": "task already taken"})
                continue
            if not sample.get("resolved"):
                skipped.append({"set": directory.name, "task": task, "why": "unresolved"})
                continue
            messages = cut(clean(sample["messages"]), args.cut)
            if messages is None:
                skipped.append({"set": directory.name, "task": task, "why": "no assistant turn past the cut"})
                continue
            conversation = {"source": directory.name, "task_id": task, "tools": tools, "messages": messages}
            try:
                chat.validate_conversation(conversation)
            except ValueError as exc:
                skipped.append({"set": directory.name, "task": task, "why": f"invalid: {exc}"})
                continue
            seen_tasks.add(task)
            rows.append({"id": chat.document_id(conversation), **conversation})
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "conversations.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(
        {"inputs": inputs, "cut": args.cut, "conversations": len(rows), "skipped": skipped}, indent=1) + "\n")
    print(f"{len(rows)} conversations, {len(skipped)} skipped -> {args.output}")


if __name__ == "__main__":
    main()
