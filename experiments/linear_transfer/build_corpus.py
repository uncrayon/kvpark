"""Build chat corpora from pinned SmolTalk2 subsets, sized by exact token counts.

Rows are converted to structured messages (system, user, assistant with
reasoning_content and tool_calls, tool results), rendered with each model's own
GGUF chat template, and tokenized vocab-only with kvpark-kv-probe. By default
(corpus v2) selection fills per-subset token quotas in a seeded order, then splits
whole conversations into train/dev/test. With --prefix-buckets it instead selects a
long-context evaluation set: whole conversations binned by the number of Muse tokens
before the final message, spread over equal-width strata of each bucket, optionally
limited to Latin-script text. Conversation text comes from the dataset; only whitespace
is stripped, over-long tool outputs and reasoning are cut at a line boundary,
conversations are shortened at message boundaries, and (long mode) the final message
is cut at a whitespace boundary after the scored window. No model weights are loaded.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

import chat


REVISION = "HuggingFaceTB/smoltalk2@fc6cc2103c066455aade5d7fbb346039ae36ca5e"
MIX = {
    "multi_turn_reasoning_if_think": .26,
    "smolagents_toolcalling_traces_think": .20,
    "smoltalk_systemchats_Qwen3_32B_think": .14,
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": .14,
    "hermes_function_calling_v1_no_think": .13,
    "aya_dataset_Qwen3_32B_think": .13,
}
TOOL_SUBSETS = {"smolagents_toolcalling_traces_think", "hermes_function_calling_v1_no_think"}
DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_TOOL_SYSTEM = "You are a helpful assistant. Use the available tools when they help you complete the task."
THINK = re.compile(r"\A<think>(.*?)</think>", re.S)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
TOOLS_WRAPPER = re.compile(r"<tools>(.*?)</tools>", re.S)
WHITESPACE_CUT = re.compile(r"(?<=\S)\s")
TOKENIZE_CTX = 32768  # kvpark-kv-probe maximum
CUT_MARGIN_BYTES = 64  # kept past the last verified token when cutting the final message
REFERENCE_MARGIN_BYTES = 4096  # uncut text kept past the verified tokens in the reference tokenization
FIRST_TURN_MIN_CHARS = 64
OVER_CTX = re.compile(r"document ([0-9a-f]{64}) has [0-9]+ tokens, exceeding --ctx")
LETTER = re.compile(r"[^\W\d_]")
LATIN_LETTER = re.compile("[A-Za-zªºÀ-ÖØ-öø-ɏḀ-ỿ"
                          "Ⱡ-Ɀ꜠-ꟿＡ-Ｚａ-ｚ]")
CJK_LETTER = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
LANGUAGE_PROBE_CHARS = 2048  # final-message characters checked for script before tokenizing


class Reject(ValueError):
    """A row that cannot become a valid corpus v2 conversation; the message is the reason."""


def cut(text, limit):
    """Strip; if limit is nonzero and the text is longer, cut at the last newline before the limit."""
    text = text.strip()
    if not limit or len(text) <= limit:
        return text
    head = text[:limit]
    boundary = head.rfind("\n")
    return (head[:boundary] if boundary > limit // 2 else head).strip()


def literal(payload):
    """Parse a JSON or Python-literal payload (the datasets use both)."""
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
            raise Reject("unparseable tool payload") from exc


def normalize_tool(tool):
    if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
    elif isinstance(tool, dict) and "name" in tool:
        function = tool
    else:
        raise Reject("unrecognized tool definition")
    if not isinstance(function.get("name"), str) or not function["name"].strip():
        raise Reject("tool without a name")
    return {"type": "function", "function": {
        "name": function["name"].strip(),
        "description": str(function.get("description") or "").strip(),
        "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict)
        else {"type": "object", "properties": {}}}}


def parse_tools(xml_tools):
    """Return (tools or None, instruction text from entries that hold no tool definitions)."""
    tools, notes = [], []
    for entry in xml_tools or ():
        # Hermes rows embed "<tools> [...] </tools>" in instructions that also mention "<tools></tools>".
        wrapped = [body for body in TOOLS_WRAPPER.findall(entry) if body.strip()]
        if len(wrapped) > 1:
            raise Reject("several tool lists in one entry")
        payload = wrapped[0] if wrapped else entry
        if not wrapped and not payload.lstrip().startswith(("{", "[")):
            notes.append(payload.strip())
            continue
        try:
            values = [literal(payload)]
        except Reject:
            # One JSON object per line (smolagents traces).
            values = [literal(line) for line in payload.splitlines() if line.strip()]
        for value in values:
            tools.extend(value if isinstance(value, list) else [value])
    tools = [normalize_tool(tool) for tool in tools]
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise Reject("duplicate tool names")
    return tools or None, "\n\n".join(note for note in notes if note)


def parse_assistant(text, index, reasoning_limit):
    """Split <think> reasoning and <tool_call> blocks out of an assistant message."""
    message = {"role": "assistant"}
    text = text.strip()
    think = THINK.match(text)
    if think:
        reasoning = cut(think.group(1), reasoning_limit)
        text = text[think.end():]
        if reasoning:
            message["reasoning_content"] = reasoning
    if "<think>" in text or "</think>" in text:
        raise Reject("unexpected think markup")
    calls = []
    for number, match in enumerate(TOOL_CALL.finditer(text)):
        payload = literal(match.group(1))
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise Reject("tool call without a name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            arguments = literal(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict) or any(not isinstance(key, str) or not key for key in arguments):
            raise Reject("tool call arguments are not an object")
        arguments = {key: value.strip() if isinstance(value, str) else value
                     for key, value in arguments.items()}
        calls.append({"id": f"call_{index}_{number}", "type": "function",
                      "function": {"name": payload["name"].strip(), "arguments": arguments}})
    content = TOOL_CALL.sub("", text).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise Reject("unbalanced tool call markup")
    message["content"] = content
    if calls:
        message["tool_calls"] = calls
    elif not content:
        raise Reject("empty assistant message")
    return message


def tool_messages(text, calls, tool_limit):
    """Answer the previous assistant's calls in order; tagged responses or one plain observation."""
    blocks = [block.strip() for block in TOOL_RESPONSE.findall(text)]
    if not blocks:
        blocks = [text.strip()]
    if len(blocks) != len(calls):
        raise Reject("tool responses do not match the tool calls")
    messages = []
    for call, block in zip(calls, blocks):
        content = cut(block, tool_limit)
        if not content:
            raise Reject("empty tool response")
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "name": call["function"]["name"], "content": content})
    return messages


def final_answer_message(message):
    """A trailing unanswered `final_answer` call (smolagents traces) becomes the answer text."""
    calls = message.get("tool_calls", [])
    arguments = calls[0]["function"]["arguments"] if len(calls) == 1 else {}
    answer = next(iter(arguments.values()), None) if len(arguments) == 1 else None
    if calls[0]["function"]["name"] != "final_answer" or not isinstance(answer, str) or not answer.strip():
        raise Reject("tool calls without responses")
    result = {key: value for key, value in message.items() if key != "tool_calls"}
    result["content"] = answer.strip()
    return result


def convert_row(subset, row_index, row, *, tool_limit=4000, reasoning_limit=6000):
    """Return a corpus v2 conversation (without split) or raise Reject."""
    kwargs = row.get("chat_template_kwargs") or {}
    if kwargs.get("python_tools"):
        raise Reject("python tools are not representable as chat-template tools")
    tools, notes = parse_tools(kwargs.get("xml_tools"))
    system = (kwargs.get("custom_instructions") or "").strip() or notes
    system = system or (DEFAULT_TOOL_SYSTEM if tools else DEFAULT_SYSTEM)
    messages = [{"role": "system", "content": system}]
    pending_calls = []
    for message in row["messages"]:
        role, text = message.get("role"), message.get("content") or ""
        if role == "system":
            raise Reject("system message inside the conversation")
        if pending_calls and role != "tool":
            raise Reject("tool calls without responses")
        if role == "user":
            content = text.strip()
            if not content:
                raise Reject("empty user message")
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            parsed = parse_assistant(text, len(messages), reasoning_limit)
            messages.append(parsed)
            pending_calls = parsed.get("tool_calls", [])
        elif role == "tool":
            if not pending_calls:
                raise Reject("tool response without a call")
            messages.extend(tool_messages(text, pending_calls, tool_limit))
            pending_calls = []
        else:
            raise Reject(f"unknown role {role!r}")
    if pending_calls:
        try:
            messages[-1] = final_answer_message(messages[-1])
        except Reject:
            messages.pop()  # an unanswered trailing call ends the usable conversation
    if tools is None and any("tool_calls" in message for message in messages):
        raise Reject("tool calls without tool definitions")
    conversation = {"id": f"{subset}:{row_index}", "source": subset, "tools": tools,
                    "messages": messages}
    conversation = end_with_answer(conversation)
    chat.validate_conversation(conversation)
    return conversation


def end_with_answer(conversation, limit=None):
    """Keep at most `limit` messages, ending on an assistant message with content."""
    messages = conversation["messages"][:limit]
    while messages and not (messages[-1]["role"] == "assistant" and messages[-1]["content"]):
        messages.pop()
    if len(messages) < 3:
        raise Reject("no user turn followed by an assistant answer")
    return {**conversation, "messages": messages}


def shorten(conversation, render_bytes, max_bytes):
    """Drop trailing turns until the rendered text fits the byte estimate."""
    while render_bytes(conversation) > max_bytes:
        ends = [count for count, message in enumerate(conversation["messages"][:-1], 1)
                if message["role"] == "assistant" and message["content"]]
        if not ends:
            raise Reject("first exchange exceeds the length limit")
        conversation = end_with_answer(conversation, ends[-1])
    return conversation


def special_inside_segments(tokens, segments):
    """True if a special token overlaps any content segment span."""
    starts = np.concatenate(([0], tokens["byte_end"][:-1]))
    special = tokens["special"].astype(bool)
    for _, begin, end in segments:
        if np.any(special & (starts < end) & (tokens["byte_end"] > begin)):
            return True
    return False


def assign_splits(selected, fractions, seed):
    """Stratified by subset: each subset's conversations split by seeded order."""
    rng = np.random.default_rng(seed + 1)
    by_subset = {}
    for item in selected:
        by_subset.setdefault(item["conversation"]["source"], []).append(item)
    for items in by_subset.values():
        order = rng.permutation(len(items))
        total = sum(item["tokens"]["muse"] for item in items)
        used = {"test": 0, "dev": 0}
        for position in order:
            item = items[position]
            for split in ("test", "dev"):
                if used[split] < fractions[split] * total:
                    item["conversation"]["split"] = split
                    used[split] += item["tokens"]["muse"]
                    break
            else:
                item["conversation"]["split"] = "train"
    for item in selected:
        chat.validate_conversation(item["conversation"])


def subset_files(dataset_dir, subset):
    paths = sorted(dataset_dir.glob(f"{subset}-*.parquet"))
    if not paths:
        raise ValueError(f"missing parquet file for {subset}")
    return paths


def read_subset(paths):
    """One table over the subset's parquet shards in name order (row numbers run across shards)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if isinstance(paths, (str, Path)):
        paths = [paths]
    tables = [pq.read_table(path, columns=["messages", "chat_template_kwargs", "source"]) for path in paths]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def user_turns_key(conversation):
    users = [m["content"] for m in conversation["messages"] if m["role"] == "user"]
    return hashlib.sha256("\x00".join(users).encode("utf-8")).hexdigest()


def first_user_key(conversation):
    """Digest of a substantive first user turn (>= FIRST_TURN_MIN_CHARS), else None (greetings recur)."""
    first = next(m["content"] for m in conversation["messages"] if m["role"] == "user")
    return hashlib.sha256(first.encode("utf-8")).hexdigest() if len(first) >= FIRST_TURN_MIN_CHARS else None


def read_exclusions(paths):
    """Ids and user-turn digests (whole sequence, substantive first turn) of conversations to keep out."""
    exclusions = {"ids": set(), "user_turns": set(), "first_user": set(), "files": {}}
    for path in paths:
        conversations = 0
        for conversation in chat.read_conversations(path):
            chat.validate_conversation(conversation)
            if not isinstance(conversation.get("id"), str):
                raise ValueError(f"excluded conversation without an id in {path}")
            exclusions["ids"].add(conversation["id"])
            exclusions["user_turns"].add(user_turns_key(conversation))
            exclusions["first_user"].add(first_user_key(conversation))
            conversations += 1
        exclusions["first_user"].discard(None)
        exclusions["files"][str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "conversations": conversations}
    return exclusions


def exclusion_reason(conversation, exclusions):
    if exclusions is None:
        return None
    if conversation["id"] in exclusions["ids"]:
        return "excluded id"
    if (user_turns_key(conversation) in exclusions["user_turns"]
            or first_user_key(conversation) in exclusions["first_user"]):
        return "excluded user turns"
    return None


def tokenize_records(probe, model, records, *, ctx=TOKENIZE_CTX, workdir=None):
    """collect_v2.tokenize, except that documents over ctx tokens map to None instead of aborting.

    The probe stops at the first over-long document and names it on stderr (captured here and passed
    on); that document is dropped and the rest are tokenized again. Any other failure is raised.
    Documents go to the probe longest first, so a failing run stops early."""
    import collect_v2
    records, over = sorted(records, key=lambda record: -len(record["text"].encode("utf-8"))), {}
    while records:
        with tempfile.TemporaryFile() as log:
            sys.stdout.flush()
            sys.stderr.flush()
            saved = os.dup(2)
            os.dup2(log.fileno(), 2)
            try:
                result, failure = collect_v2.tokenize(probe, model, records, ctx=ctx, batch=min(512, ctx),
                                                     workdir=workdir), None
            except subprocess.CalledProcessError as exc:
                failure = exc
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
                os.close(saved)
            log.seek(0)
            output = log.read().decode("utf-8", "replace")
        sys.stderr.write(output)
        sys.stderr.flush()
        if failure is None:
            return {**result, **over}
        match = OVER_CTX.search(output)
        if match is None or match.group(1) not in {record["document_id"] for record in records}:
            raise failure
        over[match.group(1)] = None
        records = [record for record in records if record["document_id"] != match.group(1)]
    return over


def cut_at_whitespace(text, max_bytes, keep_bytes=0):
    """Longest prefix of at most max_bytes UTF-8 bytes, but at least keep_bytes, that ends right
    before whitespace; else the shortest such prefix past both. Unchanged if it already fits or no
    boundary qualifies. The prefix never ends in whitespace."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    best, offset, last = None, 0, 0
    for match in WHITESPACE_CUT.finditer(text):
        offset += len(text[last:match.start()].encode("utf-8"))
        last = match.start()
        if offset < keep_bytes:
            continue
        if offset <= max_bytes:
            best = match.start()
            continue
        if best is None:
            best = match.start()
        break
    return text if best is None else text[:best]


def final_answer_problem(conversation):
    message = conversation["messages"][-1]
    if message["role"] != "assistant" or not message["content"]:
        return "final message is not an assistant answer"
    if "tool_calls" in message:
        return "final message has tool calls"
    return None


def cut_final_message(conversation, max_bytes, keep=None):
    """Cut the final assistant message to about max_bytes of reasoning plus content, at whitespace.

    Both templates render reasoning before content. Reasoning is cut first, to no less than half
    the budget, so the answer keeps text (at least its first word). keep maps a field to bytes that
    must survive. Returns (conversation, {field: [old_bytes, new_bytes]} for changed fields)."""
    problem = final_answer_problem(conversation)
    if problem:
        raise Reject(problem)
    keep = keep or {}
    size = lambda text: len(text.encode("utf-8"))
    message = dict(conversation["messages"][-1])
    reasoning, content = message.get("reasoning_content", ""), message["content"]
    if size(reasoning) + size(content) <= max_bytes:
        return conversation, {}
    if reasoning:
        reasoning = cut_at_whitespace(reasoning, max(max_bytes - size(content), max_bytes // 2),
                                      keep.get("reasoning_content", 0))
        message["reasoning_content"] = reasoning
    message["content"] = cut_at_whitespace(content, max(max_bytes - size(reasoning), 0), keep.get("content", 0))
    original = conversation["messages"][-1]
    changes = {field: [size(original[field]), size(message[field])]
               for field in ("reasoning_content", "content") if field in message and message[field] != original[field]}
    result = {**conversation, "messages": conversation["messages"][:-1] + [message]}
    chat.validate_conversation(result)
    return result, changes


def script_shares(text):
    """Shares of the letters in text that are outside the Latin script, and that are CJK (0 without letters)."""
    letters = len(LETTER.findall(text))
    if not letters:
        return {"non_latin": 0.0, "cjk": 0.0}
    return {"non_latin": 1 - len(LATIN_LETTER.findall(text)) / letters,
            "cjk": len(CJK_LETTER.findall(text)) / letters}


def user_text(conversation):
    return "\n".join(m["content"] for m in conversation["messages"] if m["role"] == "user")


def final_text_head(conversation, chars=LANGUAGE_PROBE_CHARS):
    """Opening characters of the final message as rendered (reasoning first), where the scored window starts."""
    message = conversation["messages"][-1]
    return (message.get("reasoning_content", "") + "\n" + message["content"]).strip()[:chars]


def scored_positions(tokens, rendered):
    """Token positions >= 1 that lie fully inside a segment of the final message."""
    byte_end = np.asarray(tokens["byte_end"])
    starts = np.concatenate(([0], byte_end[:-1]))
    final = [(start, end) for index, start, end in rendered.segments if index in rendered.final_message_segments]
    if not final:
        raise Reject("final message has no rendered segment")
    inside = np.zeros(len(byte_end), dtype=bool)
    for start, end in final:
        inside |= (starts >= start) & (byte_end <= end)
    inside[0] = False
    return np.flatnonzero(inside), min(start for start, _ in final)


def scored_text(tokens, rendered, window):
    """Text of the first `window` scored tokens (joined token by token)."""
    positions, _ = scored_positions(tokens, rendered)
    byte_end = np.asarray(tokens["byte_end"])
    data = rendered.text.encode("utf-8")
    return b"".join(data[(int(byte_end[p - 1]) if p else 0):int(byte_end[p])]
                    for p in positions[:window]).decode("utf-8", "replace")


def final_window(tokens, rendered, window):
    """Muse-style handoff point and scored window of a rendered, tokenized conversation.

    prefix_tokens P: tokens ending at or before the first rendered segment of the final message
    (as attention_eval/inject); scored: tokens at positions >= 1 fully inside a final-message
    segment, the first `window` of which are scored; check_tokens: max(P + window, last scored
    position + 1), capped at the document length."""
    byte_end = np.asarray(tokens["byte_end"])
    scored, final_start = scored_positions(tokens, rendered)
    prefix = int(np.searchsorted(byte_end, final_start, side="right"))
    last = int(scored[:window][-1]) + 1 if len(scored) else 0
    return {"n_tokens": int(len(byte_end)), "prefix_tokens": prefix, "final_tokens": int(len(byte_end)) - prefix,
            "scored_tokens": int(len(scored)), "check_tokens": min(int(len(byte_end)), max(prefix + window, last))}


def keep_bytes(tokens, rendered, check_tokens, margin=CUT_MARGIN_BYTES):
    """Bytes of each final-message field that must survive so the first check_tokens stay intact.

    Not clamped to the field: a value past the field's end means the field may not be cut at all."""
    end = int(tokens["byte_end"][check_tokens - 1]) + margin
    keep = {}
    for index, start, _ in rendered.segments:
        if index in rendered.final_message_segments and start < end:
            field = "reasoning_content" if rendered.segment_keys[index].endswith(":reasoning") else "content"
            keep[field] = end - start
    return keep


def parse_buckets(text):
    try:
        bounds = [int(value) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError("--prefix-buckets must be comma-separated integers") from exc
    if len(bounds) < 2 or bounds[0] < 1 or any(high <= low for low, high in zip(bounds, bounds[1:])):
        raise ValueError("--prefix-buckets needs at least two strictly increasing positive bounds")
    return list(zip(bounds, bounds[1:]))


def parse_bucket_subsets(text, buckets, subsets):
    """JSON {"<bucket lower bound>": "a,b" | ["a", "b"]}; unlisted buckets draw from every subset."""
    result = {bucket: list(subsets) for bucket in buckets}
    if text is None:
        return result
    value = json.loads(text)
    lows = {str(low): (low, high) for low, high in buckets}
    if not isinstance(value, dict) or set(value) - set(lows):
        raise ValueError(f"--bucket-subsets keys must be bucket lower bounds {sorted(lows)}")
    for key, names in value.items():
        names = names.split(",") if isinstance(names, str) else names
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(name not in subsets for name in names)):
            raise ValueError(f"--bucket-subsets entry {key} must list distinct subsets from --subsets")
        result[lows[key]] = list(names)
    return result


def bucket_of(value, buckets):
    return next(((low, high) for low, high in buckets if low <= value < high), None)


def split_bucket(bucket, parts):
    """Equal-width half-open sub-bands (strata) of a bucket."""
    low, high = bucket
    if parts < 1 or high - low < parts:
        raise ValueError(f"cannot split [{low}, {high}) into {parts} strata")
    bounds = [low + round((high - low) * number / parts) for number in range(parts + 1)]
    return list(zip(bounds, bounds[1:]))


def nearest_band(value, bands):
    """The band holding value, else the closest one (bands are sorted and disjoint)."""
    return min(bands, key=lambda band: band[0] - value if value < band[0]
               else 0 if value < band[1] else value - band[1] + 1)


def count(rejected, reason):
    rejected[reason] = rejected.get(reason, 0) + 1


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {"min": int(values.min()), "p25": float(np.percentile(values, 25)), "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)), "max": int(values.max()), "mean": float(values.mean())}


def build_quota(args, templates, shares, exclusions, rejected, files):
    """Corpus v2 selection: per-subset Muse token quotas filled in seeded order."""
    compiled_bytes = lambda conversation: len(chat.render(
        conversation, templates["muse"][0], bos_token=templates["muse"][1]).text.encode("utf-8"))
    rng = np.random.default_rng(args.seed)
    candidates = []
    for subset, share in shares.items():
        paths = subset_files(args.dataset_dir, subset)
        files.update({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        table = read_subset(paths)
        quota_bytes = share * args.target_tokens * 3.6 * args.oversample
        gathered = 0
        for row_index in rng.permutation(table.num_rows):
            if gathered >= quota_bytes:
                break
            row = table.slice(int(row_index), 1).to_pylist()[0]
            limit = (int(rng.integers(args.short_tokens[0], args.short_tokens[1] + 1))
                     if rng.random() < args.short_fraction else args.max_doc_tokens)
            try:
                conversation = convert_row(subset, int(row_index), row, tool_limit=args.tool_limit,
                                           reasoning_limit=args.reasoning_limit)
                try:
                    conversation = shorten(conversation, compiled_bytes, int(limit * 3.2))
                except Reject:
                    conversation = shorten(conversation, compiled_bytes, int(args.max_doc_tokens * 3.2))
                size = compiled_bytes(conversation)
                reason = exclusion_reason(conversation, exclusions) or (
                    final_answer_problem(conversation) if args.split == "test" else None)
                if reason:
                    raise Reject(reason)
            except ValueError as exc:
                count(rejected, f"{subset}: {exc}")
                continue
            candidates.append({"conversation": conversation, "estimate": size})
            gathered += size
        print(json.dumps({"stage": "converted", "subset": subset,
                          "candidates": sum(c["conversation"]["source"] == subset for c in candidates)}),
              flush=True)

    seen_prompts, unique = set(), []
    for item in candidates:
        key = user_turns_key(item["conversation"])
        if key in seen_prompts:
            count(rejected, "duplicate user turns")
            continue
        seen_prompts.add(key)
        unique.append(item)

    rendered = {}
    for name, (template, bos) in templates.items():
        rendered[name] = [chat.render(item["conversation"], template, bos_token=bos) for item in unique]
    tokens = {}
    for name, path in (("qwen", args.qwen_model), ("muse", args.muse_model)):
        records = [{"document_id": chat.document_id(item["conversation"]), "text": result.text}
                   for item, result in zip(unique, rendered[name])]
        tokens[name] = tokenize_records(args.probe, path, records, workdir=args.output.parent)
    selected, totals = [], {subset: 0 for subset in shares}
    for position, item in enumerate(unique):
        subset, doc = item["conversation"]["source"], chat.document_id(item["conversation"])
        reason = None
        if any(tokens[name][doc] is None for name in tokens):
            reason = "over tokenizer ctx"
        else:
            counts = {name: len(tokens[name][doc]["token_id"]) for name in tokens}
            if max(counts.values()) > args.max_doc_tokens:
                reason = "over max tokens"
            elif counts["muse"] < args.min_doc_tokens:
                reason = "under min tokens"
            elif any(special_inside_segments(tokens[name][doc], rendered[name][position].segments)
                     for name in tokens):
                reason = "special token text inside content"
            elif totals[subset] >= shares[subset] * args.target_tokens:
                reason = "quota filled"
        if reason:
            count(rejected, f"{subset}: {reason}")
            continue
        totals[subset] += counts["muse"]
        selected.append({"conversation": item["conversation"], "tokens": counts})
    if args.split is None:
        assign_splits(selected, {"dev": args.dev_fraction, "test": args.test_fraction}, args.seed)
    else:
        for item in selected:
            item["conversation"]["split"] = args.split
            chat.validate_conversation(item["conversation"])
    return selected, None


def render_pair(conversation, templates):
    return {name: chat.render(conversation, template, bos_token=bos) for name, (template, bos) in templates.items()}


def build_long(args, templates, exclusions, rejected, files):
    """Long-context evaluation selection: whole conversations binned by Muse prefix tokens.

    Every bucket is split into --strata-per-bucket equal-width prefix strata. Per subset (seeded
    order) conversations are converted without shortening, the final message is cut to
    --final-message-max-bytes, and a byte estimate of the Muse prefix places the candidate in the
    nearest stratum the subset serves; a stratum stops taking candidates once it holds its share, so
    long strata are not crowded out by the common lengths. With --max-non-latin-share, candidates
    whose user text or final-message opening is mostly outside the Latin script are skipped. The
    pool is tokenized with both tokenizers; exact prefix counts bin it, and each bucket takes docs +
    reserve conversations round-robin over its strata (low to high), each stratum round-robin over
    its subsets, so primaries spread over the bucket's prefix range. Every cut is checked against a
    tokenization of the uncut text (Muse and Qwen, first check_tokens)."""
    buckets = parse_buckets(args.prefix_buckets)
    bucket_subsets = parse_bucket_subsets(args.bucket_subsets, buckets, args.subset_list)
    strata = {bucket: split_bucket(bucket, args.strata_per_bucket) for bucket in buckets}
    per_bucket = args.docs_per_bucket + args.reserve_per_bucket
    wanted = math.ceil(args.oversample * per_bucket / args.strata_per_bucket)
    rng = np.random.default_rng(args.seed)
    too_foreign = lambda text: (args.max_non_latin_share is not None
                                and script_shares(text)["non_latin"] > args.max_non_latin_share)
    pool, seen = [], set()
    for subset in args.subset_list:
        serving = [bucket for bucket in buckets if subset in bucket_subsets[bucket]]
        paths = subset_files(args.dataset_dir, subset)
        files.update({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        if not serving:
            continue
        table = read_subset(paths)
        bands = [band for bucket in serving for band in strata[bucket]]
        low_limit, high_limit = serving[0][0] * .75, serving[-1][1] * 1.25
        gathered = {band: 0 for band in bands}
        scanned = 0
        for row_index in rng.permutation(table.num_rows):
            if min(gathered.values()) >= wanted:
                break
            scanned += 1
            row = table.slice(int(row_index), 1).to_pylist()[0]
            if sum(len((m.get("content") or "").encode("utf-8")) for m in row["messages"]) < buckets[0][0] * 1.5:
                count(rejected, f"{subset}: too short for the buckets (raw bytes)")
                continue
            try:
                conversation = convert_row(subset, int(row_index), row, tool_limit=args.tool_limit,
                                           reasoning_limit=args.reasoning_limit)
                reason = exclusion_reason(conversation, exclusions) or final_answer_problem(conversation)
                if reason:
                    raise Reject(reason)
                cut_conversation, changes = (cut_final_message(conversation, args.final_message_max_bytes)
                                             if args.final_message_max_bytes else (conversation, {}))
                muse = chat.render(cut_conversation, templates["muse"][0], bos_token=templates["muse"][1])
                final_start = min(start for index, start, _ in muse.segments if index in muse.final_message_segments)
                estimate = final_start / args.bytes_per_token
                if not low_limit <= estimate < high_limit:
                    raise Reject("prefix estimate outside the buckets")
                band = nearest_band(estimate, bands)
                if gathered[band] >= wanted:
                    raise Reject("prefix stratum already gathered (estimate)")
                if too_foreign(user_text(conversation)) or too_foreign(final_text_head(cut_conversation)):
                    raise Reject("non-Latin script share over --max-non-latin-share")
                rendered = {"muse": muse, "qwen": chat.render(cut_conversation, templates["qwen"][0],
                                                              bos_token=templates["qwen"][1])}
                if any(len(r.text.encode("utf-8")) > args.max_doc_tokens * args.prefilter_bytes_per_token
                       for r in rendered.values()):
                    raise Reject("over byte pre-filter")
                key = user_turns_key(conversation)
                if key in seen:
                    raise Reject("duplicate user turns")
            except ValueError as exc:
                count(rejected, f"{subset}: {exc}")
                continue
            seen.add(key)
            pool.append({"original": conversation, "conversation": cut_conversation, "changes": changes,
                         "rendered": rendered, "document_id": chat.document_id(cut_conversation)})
            gathered[band] += 1
        print(json.dumps({"stage": "gathered", "subset": subset, "rows_scanned": scanned,
                          "by_estimated_stratum": {f"{low}-{high}": n for (low, high), n in gathered.items()}}),
              flush=True)

    def tokenize_pool(items):
        result = {}
        for name, path in (("qwen", args.qwen_model), ("muse", args.muse_model)):
            records = [{"document_id": item["document_id"], "text": item["rendered"][name].text} for item in items]
            result[name] = tokenize_records(args.probe, path, records, workdir=args.output.parent)
        return result

    tokens = tokenize_pool(pool)
    accepted, recut = [], []
    for attempt in range(2):
        for item in pool:
            subset, doc = item["conversation"]["source"], item["document_id"]
            item_tokens = {name: tokens[name][doc] for name in tokens}
            reason = None
            if any(value is None for value in item_tokens.values()):
                reason = "over tokenizer ctx"
            elif max(len(value["token_id"]) for value in item_tokens.values()) > args.max_doc_tokens:
                reason = "over max tokens"
            elif any(special_inside_segments(item_tokens[name], item["rendered"][name].segments) for name in tokens):
                reason = "special token text inside content"
            if reason is None:
                windows = {name: final_window(item_tokens[name], item["rendered"][name], args.scored_window)
                           for name in tokens}
                bucket = bucket_of(windows["muse"]["prefix_tokens"], buckets)
                if bucket is None:
                    reason = "prefix outside the buckets"
                elif subset not in bucket_subsets[bucket]:
                    reason = "subset not drawn for this prefix bucket"
                elif windows["muse"]["scored_tokens"] < args.min_scored_tokens:
                    reason = "too few scored tokens in the final message"
                elif too_foreign(scored_text(item_tokens["muse"], item["rendered"]["muse"], args.scored_window)):
                    reason = "scored window non-Latin share over --max-non-latin-share"
            if reason is None and item["changes"]:
                keep = {}
                for name in tokens:
                    need = keep_bytes(item_tokens[name], item["rendered"][name], windows[name]["check_tokens"])
                    keep.update({field: max(keep.get(field, 0), value) for field, value in need.items()})
                final = item["conversation"]["messages"][-1]
                if any(len(final[field].encode("utf-8")) < need for field, need in keep.items()
                       if field in item["changes"]):
                    if attempt == 0:
                        item["keep"] = keep
                        recut.append(item)
                        continue
                    reason = "final cut inside the scored window"
            if reason:
                count(rejected, f"{subset}: {reason}")
                continue
            accepted.append({**item, "tokens": item_tokens, "windows": windows, "bucket": bucket})
        if not recut:
            break
        pool, recut = recut, []
        for item in pool:
            item["conversation"], item["changes"] = cut_final_message(item["original"], args.final_message_max_bytes,
                                                                      item.pop("keep"))
            item["recut"] = True
            item["rendered"] = render_pair(item["conversation"], templates)
            item["document_id"] = chat.document_id(item["conversation"])
        tokens = tokenize_pool(pool)

    cut_items = [item for item in accepted if item["changes"]]
    references = {}
    for name, path in (("qwen", args.qwen_model), ("muse", args.muse_model)):
        records = []
        for item in cut_items:
            full = chat.render(item["original"], templates[name][0], bos_token=templates[name][1]).text
            end = int(item["tokens"][name]["byte_end"][item["windows"][name]["check_tokens"] - 1])
            text = full.encode("utf-8")
            limit = end + REFERENCE_MARGIN_BYTES
            reference = cut_at_whitespace(full, limit, limit) if len(text) > limit else full
            records.append({"document_id": item["document_id"], "text": reference})
        references[name] = tokenize_records(args.probe, path, records, workdir=args.output.parent)
    verified = []
    for item in accepted:
        if item["changes"]:
            item["verified_tokens"] = {}
            for name in references:
                reference, check = references[name][item["document_id"]], item["windows"][name]["check_tokens"]
                ours = item["tokens"][name]
                if (reference is None or len(reference["token_id"]) < check
                        or not np.array_equal(reference["token_id"][:check], ours["token_id"][:check])
                        or not np.array_equal(reference["byte_end"][:check], ours["byte_end"][:check])):
                    count(rejected, f"{item['conversation']['source']}: final cut changes the {name} token prefix")
                    break
                item["verified_tokens"][name] = check
            else:
                verified.append(item)
        else:
            verified.append(item)

    selected, shortfalls = [], []
    for bucket in buckets:
        lanes = []
        for band in strata[bucket]:
            queues = [[item for item in verified if item["bucket"] == bucket
                       and band[0] <= item["windows"]["muse"]["prefix_tokens"] < band[1]
                       and item["conversation"]["source"] == subset] for subset in bucket_subsets[bucket]]
            lanes.append({"band": band, "queues": queues, "next": 0})
        chosen = []
        while len(chosen) < per_bucket and any(queue for lane in lanes for queue in lane["queues"]):
            for lane in lanes:
                queues = lane["queues"]
                for step in range(len(queues)):
                    position = (lane["next"] + step) % len(queues)
                    if queues[position] and len(chosen) < per_bucket:
                        item = queues[position].pop(0)
                        item["stratum"] = lane["band"]
                        chosen.append(item)
                        lane["next"] = position + 1
                        break
        if len(chosen) < per_bucket:
            shortfalls.append(f"[{bucket[0]}, {bucket[1]}): {len(chosen)} of {per_bucket}")
        for number, item in enumerate(chosen):
            item["role"] = "primary" if number < args.docs_per_bucket else "reserve"
            item["conversation"]["split"] = args.split
            chat.validate_conversation(item["conversation"])
            selected.append(item)
    if shortfalls:
        raise ValueError("not enough conversations for prefix buckets " + "; ".join(shortfalls)
                         + " (raise --oversample or widen --bucket-subsets)")
    for item in selected:
        user = script_shares(user_text(item["conversation"]))
        scored = script_shares(scored_text(item["tokens"]["muse"], item["rendered"]["muse"], args.scored_window))
        item["script"] = {"user_non_latin": round(user["non_latin"], 4), "user_cjk": round(user["cjk"], 4),
                          "scored_non_latin": round(scored["non_latin"], 4), "scored_cjk": round(scored["cjk"], 4)}
        item["tokens"] = {name: len(item["tokens"][name]["token_id"]) for name in item["tokens"]}

    documents = [{
        "id": item["conversation"]["id"], "document_id": item["document_id"], "source": item["conversation"]["source"],
        "bucket": list(item["bucket"]), "stratum": list(item["stratum"]), "role": item["role"],
        "muse_tokens": item["tokens"]["muse"], "qwen_tokens": item["tokens"]["qwen"],
        "muse_prefix_tokens": item["windows"]["muse"]["prefix_tokens"],
        "qwen_prefix_tokens": item["windows"]["qwen"]["prefix_tokens"],
        "muse_final_tokens": item["windows"]["muse"]["final_tokens"],
        "muse_scored_tokens": item["windows"]["muse"]["scored_tokens"],
        "muse_check_tokens": item["windows"]["muse"]["check_tokens"],
        "final_cut": item["changes"], "final_cut_widened": item.get("recut", False),
        "verified_tokens": item.get("verified_tokens"), "script": item["script"],
    } for item in selected]
    by_bucket = []
    for bucket in buckets:
        rows = [document for document in documents if tuple(document["bucket"]) == bucket]
        by_bucket.append({
            "bucket": list(bucket),
            "documents": {role: sum(d["role"] == role for d in rows) for role in ("primary", "reserve")},
            "by_subset": {subset: sum(d["source"] == subset for d in rows) for subset in bucket_subsets[bucket]},
            "subsets": bucket_subsets[bucket],
            "by_stratum": [{"stratum": list(band), **{role: sum(tuple(d["stratum"]) == band and d["role"] == role
                                                             for d in rows) for role in ("primary", "reserve")}}
                           for band in strata[bucket]],
            "max_script_share": {key: max((d["script"][key] for d in rows), default=0.0)
                                 for key in ("user_non_latin", "user_cjk", "scored_non_latin", "scored_cjk")},
            "muse_prefix_tokens": distribution([d["muse_prefix_tokens"] for d in rows]),
            "primary_muse_prefix_tokens": distribution([d["muse_prefix_tokens"] for d in rows
                                                        if d["role"] == "primary"]),
            "muse_scored_window_tokens": distribution([min(d["muse_scored_tokens"], args.scored_window)
                                                       for d in rows]),
            "qwen_prefix_tokens": distribution([d["qwen_prefix_tokens"] for d in rows]),
            "muse_final_tokens": distribution([d["muse_final_tokens"] for d in rows]),
            "muse_tokens": distribution([d["muse_tokens"] for d in rows]),
            "qwen_tokens": distribution([d["qwen_tokens"] for d in rows]),
            "final_cuts": sum(bool(d["final_cut"]) for d in rows),
        })
    long_eval = {
        "prefix_definition": "Muse tokens ending at or before the first rendered segment of the final message "
                             "(attention_eval/inject convention; the final message's template header counts as prefix)",
        "buckets": [list(bucket) for bucket in buckets], "docs_per_bucket": args.docs_per_bucket,
        "reserve_per_bucket": args.reserve_per_bucket, "strata_per_bucket": args.strata_per_bucket,
        "scored_window": args.scored_window,
        "script_definition": "share of letters outside the Latin script (non_latin) and in CJK/kana/hangul "
                             "blocks (cjk), over all user turns (user_*) and the Muse scored window (scored_*)",
        "max_non_latin_share": args.max_non_latin_share,
        "final_message_max_bytes": args.final_message_max_bytes,
        "check_tokens_definition": "max(P + scored_window, last of the first scored_window scored positions + 1); "
                                   "for cut conversations these first tokens equal (ids and byte ends) a "
                                   "tokenization of the uncut text, for Muse and for Qwen",
        "by_bucket": by_bucket, "documents": documents,
    }
    return selected, long_eval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="directory with the pinned parquet files")
    parser.add_argument("--probe", required=True, type=Path, help="kvpark-kv-probe binary (tokenize mode)")
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--muse-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new corpus directory")
    parser.add_argument("--subsets", help="comma-separated subsets (default: those of --mix, or the v2 mix)")
    parser.add_argument("--mix", help='JSON {"subset": share} of --target-tokens (default: the v2 mix; with '
                        "--subsets alone, the v2 shares of those subsets renormalized)")
    parser.add_argument("--target-tokens", type=int, default=150_000, help="total Muse tokens")
    parser.add_argument("--max-doc-tokens", type=int, default=4608,
                        help=f"exact limit for both models (at most {TOKENIZE_CTX})")
    parser.add_argument("--min-doc-tokens", type=int, default=160)
    parser.add_argument("--tool-limit", type=int, default=4000, help="tool output characters (0: no cut)")
    parser.add_argument("--reasoning-limit", type=int, default=6000, help="reasoning characters (0: no cut)")
    parser.add_argument("--split", choices=sorted(chat.SPLITS), help="put every conversation in this split")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="conversations JSONL whose ids and user turns (whole sequence, or a first turn "
                             f"of >= {FIRST_TURN_MIN_CHARS} characters) are kept out; repeatable")
    parser.add_argument("--dev-fraction", type=float, default=.13)
    parser.add_argument("--test-fraction", type=float, default=.13)
    parser.add_argument("--oversample", type=float, default=2.0)
    parser.add_argument("--short-fraction", type=float, default=.5,
                        help="share of conversations shortened to a random length in --short-tokens")
    parser.add_argument("--short-tokens", type=int, nargs=2, default=(600, 1800))
    parser.add_argument("--seed", type=int, default=20260928)
    long_mode = parser.add_argument_group("long-context evaluation selection")
    long_mode.add_argument("--prefix-buckets", help="bucket bounds on Muse prefix tokens, e.g. 4096,8192,16384,29500")
    long_mode.add_argument("--docs-per-bucket", type=int)
    long_mode.add_argument("--reserve-per-bucket", type=int, default=0)
    long_mode.add_argument("--bucket-subsets", help='JSON {"<bucket lower bound>": "a,b"}: subsets drawn '
                           "round-robin for that bucket (default: all --subsets)")
    long_mode.add_argument("--strata-per-bucket", type=int, default=1,
                           help="equal-width prefix strata per bucket; gathering fills each and selection "
                                "rotates over them, spreading the chosen prefixes over the bucket")
    long_mode.add_argument("--max-non-latin-share", type=float,
                           help="skip conversations whose user text or scored window has more than this share "
                                "of letters outside the Latin script (default: no filter)")
    long_mode.add_argument("--final-message-max-bytes", type=int, default=0,
                           help="cut the final message at whitespace past the scored window (0: no cut)")
    long_mode.add_argument("--scored-window", type=int, default=256)
    long_mode.add_argument("--min-scored-tokens", type=int, default=0)
    long_mode.add_argument("--bytes-per-token", type=float, default=3.6,
                           help="prefix estimate used only to gather candidates before exact tokenization")
    long_mode.add_argument("--prefilter-bytes-per-token", type=float, default=6.0,
                           help="skip documents over max-doc-tokens times this many bytes before tokenizing")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if not 0 < args.dev_fraction + args.test_fraction < .5 or min(args.dev_fraction, args.test_fraction) <= 0:
        raise ValueError("dev and test fractions must be positive and sum below 0.5")
    if not 0 <= args.short_fraction <= 1 or not 0 < args.short_tokens[0] <= args.short_tokens[1] <= args.max_doc_tokens:
        raise ValueError("invalid short-conversation settings")
    if not 0 < args.max_doc_tokens <= TOKENIZE_CTX:
        raise ValueError(f"--max-doc-tokens must be in 1..{TOKENIZE_CTX}")
    if min(args.tool_limit, args.reasoning_limit) < 0:
        raise ValueError("--tool-limit and --reasoning-limit must be >= 0")
    shares = MIX
    if args.mix is not None:
        shares = json.loads(args.mix)
        if (not isinstance(shares, dict) or not shares
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0 for v in shares.values())):
            raise ValueError("--mix must be a JSON object of positive shares")
    if args.subsets is not None:
        args.subset_list = args.subsets.split(",")
        if not all(args.subset_list) or len(set(args.subset_list)) != len(args.subset_list):
            raise ValueError("--subsets must be distinct nonempty names")
    else:
        args.subset_list = list(shares)
    exclusions = read_exclusions(args.exclude) if args.exclude else None
    if args.prefix_buckets is not None:
        if args.docs_per_bucket is None or args.docs_per_bucket < 1 or args.reserve_per_bucket < 0:
            raise ValueError("--prefix-buckets needs --docs-per-bucket >= 1 and --reserve-per-bucket >= 0")
        if args.split is None:
            raise ValueError("--prefix-buckets needs --split")
        if (args.final_message_max_bytes < 0 or args.scored_window < 1 or args.min_scored_tokens < 0
                or args.bytes_per_token <= 0 or args.prefilter_bytes_per_token <= 0 or args.oversample < 1
                or args.strata_per_bucket < 1
                or (args.max_non_latin_share is not None and not 0 <= args.max_non_latin_share < 1)):
            raise ValueError("invalid long-context selection settings")
        for bucket in parse_buckets(args.prefix_buckets):
            split_bucket(bucket, args.strata_per_bucket)
    else:
        if args.subsets is not None:
            if args.mix is None:
                missing = [name for name in args.subset_list if name not in MIX]
                if missing:
                    raise ValueError(f"subsets without a v2 share need --mix: {missing}")
                total = sum(MIX[name] for name in args.subset_list)
                shares = {name: MIX[name] / total for name in args.subset_list}
            else:
                missing = [name for name in args.subset_list if name not in shares]
                if missing:
                    raise ValueError(f"subsets without a --mix share: {missing}")
                shares = {name: shares[name] for name in args.subset_list}

    templates = {name: chat.gguf_chat_context(path)
                 for name, path in (("qwen", args.qwen_model), ("muse", args.muse_model))}
    rejected, files = {}, {}
    if args.prefix_buckets is None:
        selected, long_eval = build_quota(args, templates, shares, exclusions, rejected, files)
    else:
        selected, long_eval = build_long(args, templates, exclusions, rejected, files)

    summary = {}
    for item in selected:
        key = (item["conversation"]["source"], item["conversation"]["split"])
        entry = summary.setdefault(key, {"conversations": 0, "muse_tokens": 0, "qwen_tokens": 0,
                                         "over_2048": 0})
        entry["conversations"] += 1
        entry["muse_tokens"] += item["tokens"]["muse"]
        entry["qwen_tokens"] += item["tokens"]["qwen"]
        entry["over_2048"] += item["tokens"]["muse"] > 2048
    settings = ["target_tokens", "max_doc_tokens", "min_doc_tokens", "dev_fraction", "test_fraction",
                "oversample", "short_fraction", "short_tokens", "seed", "tool_limit", "reasoning_limit", "split",
                "subset_list"]
    if long_eval is not None:
        settings += ["prefix_buckets", "docs_per_bucket", "reserve_per_bucket", "bucket_subsets",
                     "strata_per_bucket", "max_non_latin_share", "final_message_max_bytes", "scored_window", "min_scored_tokens", "bytes_per_token",
                     "prefilter_bytes_per_token"]
    manifest = {
        "format_version": 2,
        "dataset": REVISION,
        "files_sha256": files,
        "mix": shares if long_eval is None else None,
        "settings": {name: getattr(args, name) for name in settings},
        "template_sha256": {name: hashlib.sha256(template.encode("utf-8")).hexdigest()
                            for name, (template, _) in templates.items()},
        "totals": {
            "conversations": len(selected),
            "muse_tokens": sum(item["tokens"]["muse"] for item in selected),
            "qwen_tokens": sum(item["tokens"]["qwen"] for item in selected),
            "muse_tokens_in_docs_over_2048": sum(item["tokens"]["muse"] for item in selected
                                                 if item["tokens"]["muse"] > 2048),
        },
        "by_subset_and_split": [{"subset": s, "split": p, **v} for (s, p), v in sorted(summary.items())],
        "rejected": dict(sorted(rejected.items())),
        "excluded_from": exclusions["files"] if exclusions else {},
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if long_eval is not None:
        manifest["long_eval"] = long_eval
    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=args.output.parent) as temporary:
        pending = Path(temporary) / "corpus"
        pending.mkdir()
        with (pending / "conversations.jsonl").open("x", encoding="utf-8") as stream:
            for item in selected:
                stream.write(json.dumps(item["conversation"], ensure_ascii=False, allow_nan=False) + "\n")
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        os.rename(pending, args.output)
    print(json.dumps(manifest["totals"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
