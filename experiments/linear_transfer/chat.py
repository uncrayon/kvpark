"""Render corpus v2 conversations with a model's own chat template and locate every text span."""

import argparse
import copy
from dataclasses import dataclass
import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import jinja2.ext
import jinja2.meta
from jinja2.sandbox import ImmutableSandboxedEnvironment


FIXED_DATE = datetime.datetime(2026, 9, 27)
SENTINEL = "kvseg{}"
SENTINEL_PATTERN = re.compile("kvseg([0-9]+)")
SENTINEL_CHARACTERS = ("", "")
RESERVED_CONTEXT = {"messages", "tools", "bos_token", "add_generation_prompt"}
SPLITS = {"train", "dev", "test"}
MESSAGE_KEYS = {
    "system": ({"role", "content"}, set()),
    "user": ({"role", "content"}, set()),
    "assistant": ({"role", "content"}, {"reasoning_content", "tool_calls"}),
    "tool": ({"role", "tool_call_id", "name", "content"}, set()),
}
# a chat request's fields that reach llama-server's chat template (server_context)
TEMPLATE_OPTIONS = ("chat_template_kwargs", "reasoning_effort")
# the template kwargs llama-server sets by default: --reasoning-preserve is on unless
# --no-reasoning-preserve (common/arg.cpp, end of common_params_parse_ex)
SERVER_DEFAULT_KWARGS = {"preserve_reasoning": True}


def raise_exception(message):
    raise ValueError(f"chat template error: {message}")


def strftime_now(format_string):
    return FIXED_DATE.strftime(format_string)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def make_environment():
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols])
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    return environment


@functools.lru_cache(maxsize=8)
def compile_template(template):
    if not isinstance(template, str) or not template:
        raise ValueError("chat template must be a nonempty string")
    return make_environment().from_string(template)


def template_sha256(template):
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def template_uses_bos(template):
    return "bos_token" in jinja2.meta.find_undeclared_variables(make_environment().parse(template))


def _stripped_text(value, where, *, allow_empty=False):
    if not isinstance(value, str) or value != value.strip() or (not value and not allow_empty):
        raise ValueError(f"{where} must be a stripped{'' if allow_empty else ' nonempty'} string")
    return value


def _live_text(value, where, *, allow_empty=False):
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def validate_conversation(conversation, *, live=False):
    """Check the corpus v2 conversation record rules that rendering depends on.

    live=True (a running chat, e.g. the bridge engine): text may carry surrounding whitespace
    or be empty, and the system message is optional (still only first); structure is checked
    as usual."""
    _stripped_text = _live_text if live else globals()["_stripped_text"]
    if not isinstance(conversation, dict):
        raise ValueError("conversation must be a JSON object")
    messages, tools = conversation.get("messages"), conversation.get("tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation messages must be a nonempty list")
    if tools is not None and (not isinstance(tools, list) or not tools
                              or any(not isinstance(tool, dict) for tool in tools)):
        raise ValueError("conversation tools must be null or a nonempty list of objects")
    call_ids = set()
    for index, message in enumerate(messages):
        where = f"message {index}"
        if not isinstance(message, dict) or message.get("role") not in MESSAGE_KEYS:
            raise ValueError(f"{where} must be an object with a known role")
        required, optional = MESSAGE_KEYS[message["role"]]
        keys = set(message)
        if not required <= keys or keys - required - optional:
            raise ValueError(f"{where} has missing or unexpected fields: {sorted(keys)}")
        if (index == 0) != (message["role"] == "system") and not (live and index == 0):
            raise ValueError("the first message, and only the first, must be the system message")
        calls = message.get("tool_calls")
        _stripped_text(message["content"], f"{where} content", allow_empty=calls is not None)
        if "reasoning_content" in message:
            _stripped_text(message["reasoning_content"], f"{where} reasoning_content")
        if calls is not None:
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"{where} tool_calls must be a nonempty list")
            for number, call in enumerate(calls):
                function = call.get("function") if isinstance(call, dict) else None
                if (not isinstance(call, dict) or set(call) != {"id", "type", "function"}
                        or call["type"] != "function" or not isinstance(function, dict)
                        or set(function) != {"name", "arguments"}
                        or not isinstance(function["arguments"], dict)):
                    raise ValueError(f"{where} tool call {number} is malformed")
                _stripped_text(call["id"], f"{where} tool call id")
                _stripped_text(function["name"], f"{where} tool call name")
                if call["id"] in call_ids:
                    raise ValueError(f"duplicate tool call id {call['id']!r}")
                call_ids.add(call["id"])
                if any(not isinstance(name, str) or not name for name in function["arguments"]):
                    raise ValueError(f"{where} tool call {number} has an invalid argument name")
        if message["role"] == "tool":
            _stripped_text(message["tool_call_id"], f"{where} tool_call_id")
            _stripped_text(message["name"], f"{where} name")
            if message["tool_call_id"] not in call_ids:
                raise ValueError(f"{where} answers an unknown tool call")
    if "split" in conversation and conversation["split"] not in SPLITS:
        raise ValueError("conversation split must be train, dev or test")
    if conversation.get("split") == "test":
        last = messages[-1]
        if last["role"] != "assistant" or not last["content"] or "tool_calls" in last:
            raise ValueError("test conversations must end with an assistant message with content "
                             "and no tool calls")
    return conversation


def document_id(conversation):
    canonical = json.dumps({"tools": conversation.get("tools"), "messages": conversation["messages"]},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def segment_texts(conversation):
    """Canonical segment list: per message, reasoning, content, then string tool-call arguments."""
    segments = []
    for index, message in enumerate(conversation["messages"]):
        if message.get("reasoning_content"):
            segments.append((f"{index}:reasoning", message["reasoning_content"]))
        if message["content"]:
            segments.append((f"{index}:content", message["content"]))
        for number, call in enumerate(message.get("tool_calls") or ()):
            for name, value in call["function"]["arguments"].items():
                if isinstance(value, str) and value:
                    segments.append((f"{index}:arg:{number}:{name}", value))
    return segments


def _substituted_messages(messages, replacement):
    """Copy messages, replacing each segment text by replacement(segment_number)."""
    counter = iter(range(1 << 62))
    result = []
    for message in messages:
        message = copy.deepcopy(message)
        if message.get("reasoning_content"):
            message["reasoning_content"] = replacement(next(counter))
        if message["content"]:
            message["content"] = replacement(next(counter))
        for call in message.get("tool_calls") or ():
            arguments = call["function"]["arguments"]
            for name, value in arguments.items():
                if isinstance(value, str) and value:
                    arguments[name] = replacement(next(counter))
        result.append(message)
    return result


def _contains_sentinel_characters(value):
    if isinstance(value, str):
        return any(character in value for character in SENTINEL_CHARACTERS)
    if isinstance(value, dict):
        return any(_contains_sentinel_characters(key) or _contains_sentinel_characters(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_sentinel_characters(item) for item in value)
    return False


@dataclass(frozen=True)
class Rendered:
    text: str
    segments: list  # (segment_index, start_byte, end_byte), rendered segments in text order
    segment_keys: list
    final_message_segments: list
    template_sha256: str = None
    bos_token: str = None


def render(conversation, template, *, bos_token="", add_generation_prompt=False, extra_context=None, live=False):
    validate_conversation(conversation, live=live)
    extra_context = dict(extra_context or {})
    if RESERVED_CONTEXT & set(extra_context):
        raise ValueError(f"extra_context may not set {sorted(RESERVED_CONTEXT & set(extra_context))}")
    if not isinstance(bos_token, str):
        raise ValueError("bos_token must be a string")
    if _contains_sentinel_characters([conversation["messages"], conversation.get("tools"),
                                      template, bos_token, extra_context]):
        raise ValueError("conversation, template or context contains reserved sentinel characters")
    compiled = compile_template(template)
    segments = segment_texts(conversation)
    keys = [key for key, _ in segments]

    def run(messages):
        return compiled.render(messages=messages, tools=conversation.get("tools"),
                               bos_token=bos_token, add_generation_prompt=add_generation_prompt,
                               **extra_context)

    real = run(copy.deepcopy(conversation["messages"]))
    marked = run(_substituted_messages(conversation["messages"], SENTINEL.format))
    pieces, spans, seen, offset, last = [], [], set(), 0, 0
    for match in SENTINEL_PATTERN.finditer(marked):
        number = int(match.group(1))
        if number >= len(segments) or match.group(1) != str(number):
            raise ValueError("template produced an unknown segment sentinel")
        if number in seen:
            raise ValueError(f"template renders segment {keys[number]} more than once")
        seen.add(number)
        literal = marked[last:match.start()]
        pieces.append(literal)
        offset += len(literal.encode("utf-8"))
        text = segments[number][1]
        pieces.append(text)
        size = len(text.encode("utf-8"))
        spans.append((number, offset, offset + size))
        offset += size
        last = match.end()
    pieces.append(marked[last:])
    rebuilt = "".join(pieces)
    if _contains_sentinel_characters(rebuilt):
        raise ValueError("template transforms content: a segment sentinel was altered")
    if rebuilt != real:
        raise ValueError("template transforms content: sentinel substitution differs from the real render")
    final_index = f"{len(conversation['messages']) - 1}:"
    final = [number for number, _, _ in spans if keys[number].startswith(final_index)]
    return Rendered(text=real, segments=spans, segment_keys=keys, final_message_segments=final,
                    template_sha256=template_sha256(template), bos_token=bos_token)


def server_context(options=None, *, server=False, enable_thinking=True, eos_token="", now=None, server_kwargs=None):
    """The chat template variables llama-server sets for a chat request beyond messages, tools,
    bos_token and add_generation_prompt: render's extra_context for that request.

    options: the request's TEMPLATE_OPTIONS, mapped as tools/server/server-common.cpp
    (oaicompat_chat_params_parse) and common/chat.cpp (common_chat_template_direct_apply_impl)
    map them. chat_template_kwargs are merged over the server's own, each key a template
    variable (an enable_thinking kwarg also sets the thinking switch; a string one is refused,
    as the server refuses it); reasoning_effort "none" turns thinking off and drops a
    reasoning_effort kwarg, another string becomes that kwarg. Then llama.cpp's jinja caps: a
    boolean preserve_reasoning sets preserve_thinking and its negations clear_thinking,
    truncate_history_thinking and drop_thinking, a nonempty string reasoning_effort also
    reasoning_strength (common/jinja/caps.cpp); a kwarg of the same name wins over those.

    server=False gives only what the options set (the template's own defaults for the rest: the
    rendering of earlier chat.py versions when there are none). server=True gives everything
    llama-server sets: eos_token, the thinking switch (enable_thinking, which the server turns
    on for a template that supports thinking), datetime and date_string, strftime_now at the
    current local time (`now`, a datetime, else the clock) and its default template kwargs:
    `server_kwargs` (the kwargs its --chat-template-kwargs, --reasoning on|off, --reasoning-effort
    and --(no-)reasoning-preserve flags set, kvpark.bridge.messages.server_template_kwargs) over
    SERVER_DEFAULT_KWARGS (preserve_reasoning on), which are its defaults without such flags."""
    options = dict(options or {})
    unknown = set(options) - set(TEMPLATE_OPTIONS)
    if unknown:
        raise ValueError(f"unknown template options {sorted(unknown)}")
    requested = options.get("chat_template_kwargs")
    if requested is not None and not isinstance(requested, dict):
        raise ValueError("chat_template_kwargs must be an object")
    if server_kwargs is not None and not isinstance(server_kwargs, dict):
        raise ValueError("the server's template kwargs must be an object")
    kwargs = {**SERVER_DEFAULT_KWARGS, **(server_kwargs or {})} if server else {}
    kwargs.update(requested or {})
    thinking = enable_thinking if server else None
    if "enable_thinking" in kwargs:
        if isinstance(kwargs["enable_thinking"], str):
            raise ValueError('invalid type for "enable_thinking" (expected boolean, got string)')
        if isinstance(kwargs["enable_thinking"], bool):
            thinking = kwargs["enable_thinking"]
    effort = options.get("reasoning_effort")
    if isinstance(effort, str) and effort == "none":
        thinking = False
        kwargs.pop("reasoning_effort", None)
    elif isinstance(effort, str) and effort:
        kwargs["reasoning_effort"] = effort
    context = {}
    if isinstance(kwargs.get("preserve_reasoning"), bool):
        keep = kwargs["preserve_reasoning"]
        context.update(preserve_thinking=keep, clear_thinking=not keep, truncate_history_thinking=not keep,
                       drop_thinking=not keep)
    if isinstance(kwargs.get("reasoning_effort"), str) and kwargs["reasoning_effort"]:
        context.update(reasoning_effort=kwargs["reasoning_effort"], reasoning_strength=kwargs["reasoning_effort"])
    if server:
        moment = now or datetime.datetime.now()
        context.update(eos_token=eos_token, datetime=moment.strftime("%b %d %Y"),
                       date_string=moment.strftime("%d %b %Y"),
                       strftime_now=lambda format_string: moment.strftime(format_string))
    if thinking is not None:
        context["enable_thinking"] = thinking
    context.update(kwargs)
    return context


def server_contexts(options=None, *, eos_token="", now=None, server_kwargs=None):
    """[(name, extra_context)]: the template contexts a llama-server may use for a request with
    these options, most likely first; the server's own settings are not visible over HTTP, so a
    caller keeps the one whose rendering matches the server's bytes. "template": only what the
    options set; "server": llama-server's context (server_context(server=True), with the
    server's own template kwargs `server_kwargs` when they are known); "server-no-thinking": the
    same with the thinking switch off (a template the server finds without thinking support, or
    --reasoning off). Contexts that cannot differ from an earlier one are left out."""
    out, seen = [], []
    for name, extra in (("template", {}), ("server", {"server": True, "server_kwargs": server_kwargs}),
                        ("server-no-thinking", {"server": True, "server_kwargs": server_kwargs,
                                                "enable_thinking": False})):
        context = server_context(options, eos_token=eos_token, now=now, **extra)
        comparable = {key: value for key, value in context.items() if key != "strftime_now"}
        if comparable not in seen:
            seen.append(comparable)
            out.append((name, context))
    return out


def rendered_record(conversation, model_name, rendered):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model name must be a nonempty string")
    if conversation.get("split") not in SPLITS:
        raise ValueError("conversation split must be train, dev or test")
    if not isinstance(conversation.get("source"), str) or not conversation["source"]:
        raise ValueError("conversation source must be a nonempty string")
    if rendered.segment_keys != [key for key, _ in segment_texts(conversation)]:
        raise ValueError("rendering does not belong to this conversation")
    if not isinstance(rendered.template_sha256, str) or not isinstance(rendered.bos_token, str):
        raise ValueError("rendering lacks its template and BOS provenance")
    if conversation["split"] == "test":
        content = rendered.segment_keys.index(f"{len(conversation['messages']) - 1}:content")
        if content not in rendered.final_message_segments:
            raise ValueError("the template does not render the final message content of a test conversation")
    return {
        "document_id": document_id(conversation),
        "conversation_id": conversation.get("id"),
        "model": model_name,
        "text": rendered.text,
        "segments": [list(span) for span in rendered.segments],
        "segment_keys": list(rendered.segment_keys),
        "split": conversation["split"],
        "source": conversation["source"],
        "final_message_segments": list(rendered.final_message_segments),
        "template_sha256": rendered.template_sha256,
        "bos_token": rendered.bos_token,
    }


def gguf_tokenizer_info(model):
    """Chat template (or None), BOS and EOS text as llama.cpp passes them to templates, and add_bos
    (or None)."""
    gguf_path = Path(__file__).resolve().parents[2] / "runtime/source/gguf-py"
    if str(gguf_path) not in sys.path:
        sys.path.insert(0, str(gguf_path))
    from gguf import GGUFReader

    reader = GGUFReader(model, "r")

    def value(name):
        field = reader.get_field(name)
        return None if field is None else field.contents()

    def token_text(key, what):
        token_id = value(f"tokenizer.ggml.{key}_token_id")
        if token_id is None:
            return ""
        tokens = reader.get_field("tokenizer.ggml.tokens")
        if tokens is None or not 0 <= token_id < len(tokens.data):
            raise ValueError(f"{what} token id is outside the token list")
        return tokens.contents(token_id)

    template = value("tokenizer.chat_template")
    add_bos = value("tokenizer.ggml.add_bos_token")
    if add_bos is not None and not isinstance(add_bos, bool):
        raise ValueError("tokenizer.ggml.add_bos_token must be a boolean")
    return {"chat_template": template if isinstance(template, str) and template else None,
            "bos_token": token_text("bos", "BOS"), "eos_token": token_text("eos", "EOS"), "add_bos": add_bos}


def gguf_chat_context(model):
    """Read the chat template and BOS text from GGUF metadata only."""
    info = gguf_tokenizer_info(model)
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    return info["chat_template"], info["bos_token"]


def read_conversations(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank line {number} in conversations")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid conversation JSON on line {number}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", required=True, type=Path, help="corpus v2 JSONL")
    parser.add_argument("--model-name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--template", type=Path, help="Jinja chat template file")
    source.add_argument("--gguf", type=Path, help="read template and BOS text from GGUF metadata")
    parser.add_argument("--bos-token", help="with --template (required): the model's BOS text as "
                        "llama.cpp passes it to templates, even if empty; --gguf reads it from metadata")
    parser.add_argument("--output", required=True, type=Path, help="new rendered JSONL; never overwritten")
    parser.add_argument("--template-output", type=Path, help="also save the template used (new file)")
    args = parser.parse_args(argv)
    for path in (args.output, args.template_output):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"output already exists: {path}")
    if (args.template is not None) != (args.bos_token is not None):
        raise ValueError("--bos-token is required with --template and not accepted with --gguf")
    if args.template is not None:
        template, bos = args.template.read_text(encoding="utf-8"), args.bos_token
    else:
        template, bos = gguf_chat_context(args.gguf)
    if not bos and template_uses_bos(template):
        raise ValueError("the chat template uses bos_token but the BOS text is empty")
    seen, count = set(), 0
    with tempfile.TemporaryDirectory(prefix=".kvpark-render-", dir=args.output.parent) as temporary:
        pending = Path(temporary) / "rendered.jsonl"
        with pending.open("x", encoding="utf-8") as stream:
            for number, conversation in enumerate(read_conversations(args.conversations), 1):
                try:
                    record = rendered_record(conversation, args.model_name,
                                             render(conversation, template, bos_token=bos))
                except ValueError as exc:
                    raise ValueError(f"conversation line {number}: {exc}") from exc
                if record["document_id"] in seen:
                    raise ValueError(f"duplicate conversation on line {number}")
                seen.add(record["document_id"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        if not count:
            raise ValueError("no conversations to render")
        if args.template_output is not None:
            saved = Path(temporary) / "template.jinja"
            saved.write_text(template, encoding="utf-8")
            os.link(saved, args.template_output)
        os.link(pending, args.output)
    print(json.dumps({"output": str(args.output), "documents": count,
                      "template_sha256": template_sha256(template),
                      "bos_token": bos}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
