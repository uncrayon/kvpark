"""OpenAI-style chat messages -> the bridge engine's conversation form."""

import base64
import binascii
import hashlib
import json

ROLES = {"system", "user", "assistant", "tool"}
# OpenAI content parts llama-server accepts, by the /props "modalities" key that enables them
# (tools/server/server-common.cpp:1213-1260; upstream also reads "video_url" as "input_video")
MEDIA_PARTS = {"image_url": "vision", "input_audio": "audio", "input_video": "video", "video_url": "video"}
# request fields llama-server passes to the chat template (server-common.cpp:1308-1331): e.g.
# Qwen3.8's template writes the reasoning effort into the prompt, and enable_thinking: false drops it
TEMPLATE_FIELDS = ("chat_template_kwargs", "reasoning_effort")
# llama-server's --reasoning values (common/arg.cpp common_arg_utils::is_truthy / is_falsey / is_autoy)
REASONING_VALUES = {"on": True, "enabled": True, "true": True, "1": True,
                    "off": False, "disabled": False, "false": False, "0": False, "auto": None, "-1": None}


def media_reference(part):
    """Where an OpenAI media part's data is, as llama-server reads it (image_url.url,
    input_audio/input_video data else url): a data URL, raw base64, an http(s) URL or a file://
    path; "" when the part names none."""
    kind = part.get("type")
    value = part.get(kind)
    if not isinstance(value, dict):
        return ""
    reference = value.get("url") if kind == "image_url" else value.get("data") or value.get("url")
    return reference if isinstance(reference, str) else ""


def media_bytes(reference):
    """The bytes of inline media (a base64 data URL or raw base64, line breaks allowed), or None
    for a URL or a file path (the server fetches those itself) and for text that is not base64
    (a data URL without its comma included)."""
    if not reference or reference.startswith(("http://", "https://", "file://")):
        return None
    if reference.startswith("data:"):
        _, comma, reference = reference.partition(",")
        if not comma:
            return None
    try:
        return base64.b64decode("".join(reference.split()), validate=True)
    except (binascii.Error, ValueError):
        return None


def media_digest(part):
    """sha256 of an OpenAI media part's bytes; for remote media (a URL or file path, which the
    bridge never fetches) of "url:" and the reference; for a part that names no media in a form
    llama-server reads (e.g. image_url given as a string) of "part:" and the part's JSON, so two
    such parts never share a surrogate."""
    reference = media_reference(part)
    if not reference:
        return hashlib.sha256(b"part:" + json.dumps(part, sort_keys=True).encode()).hexdigest()
    data = media_bytes(reference)
    return hashlib.sha256(b"url:" + reference.encode() if data is None else data).hexdigest()


def _content(content, kept):
    """A string, or a list of {"type": "text", "text"} and {"type": "media", "modality", "sha256"}
    parts.

    Media parts keep their modality and the sha256 of their bytes; the parts themselves (with
    their data) go to `kept` by digest, for the surrogate producers (surrogates.py). They stay
    in memory: no image, audio or video reaches the engine's job or the bridge's files (its
    surrogate cache keeps texts, and a video's keyframes for vision targets)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text":
                text = part.get("text")
                if text is not None and not isinstance(text, str):
                    raise ValueError("a text part must hold a string")
                parts.append({"type": "text", "text": text or ""})
            elif kind in MEDIA_PARTS:
                digest = media_digest(part)
                kept.setdefault(digest, part)
                parts.append({"type": "media", "modality": MEDIA_PARTS[kind], "sha256": digest})
            else:
                raise ValueError(f"content part type {kind!r} cannot be bridged")
        return parts
    raise ValueError("message content must be text or a list of content parts")


def conversation_from_openai(messages, tools=None):
    """{"messages": [...], "tools": [...] | None} with tool-call arguments as objects.

    Keeps role, content, reasoning_content, tool calls (id, name, arguments) and tool results
    (tool_call_id, name); drops other fields. Content given as parts stays a list of text and
    media placeholders until ``flatten`` renders it for a target server; a conversation with
    media also keeps the original media parts ("media_parts": {sha256: part}, in memory only)
    for the surrogates of a target without that modality. Raises ValueError for content the
    bridge cannot carry (unknown part types, malformed tool calls), so the caller can fall back
    to a native prefill.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("a conversation needs messages")
    names, out, kept = {}, [], {}
    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in ROLES:
            raise ValueError(f"message {index} has an unsupported role {role!r}")
        item = {"role": role, "content": _content(message.get("content"), kept)}
        if role == "assistant":
            if isinstance(message.get("reasoning_content"), str) and message["reasoning_content"]:
                item["reasoning_content"] = message["reasoning_content"]
            calls = []
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        raise ValueError(f"message {index} has tool-call arguments that are not JSON") from None
                if not isinstance(arguments, dict) or not function.get("name") or not call.get("id"):
                    raise ValueError(f"message {index} has a malformed tool call")
                names[call["id"]] = function["name"]
                calls.append({"id": call["id"], "type": "function",
                              "function": {"name": function["name"], "arguments": arguments}})
            if calls:
                item["tool_calls"] = calls
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise ValueError(f"message {index} is a tool result without tool_call_id")
            item["tool_call_id"] = call_id
            item["name"] = message.get("name") or names.get(call_id) or "tool"
        out.append(item)
    conversation = {"messages": out, "tools": tools or None}
    if kept:
        conversation["media_parts"] = kept
    return conversation


def template_options(request):
    """The fields of a chat request that change how llama-server renders it (TEMPLATE_FIELDS), at
    the top level or in extra_body, where the OpenAI client puts them; {} when there are none."""
    extra = request.get("extra_body") if isinstance(request.get("extra_body"), dict) else {}
    out = {}
    for key in TEMPLATE_FIELDS:
        value = request.get(key, extra.get(key))
        if value is not None:
            out[key] = value
    return out


def server_template_kwargs(*, chat_template_kwargs=None, reasoning=None, reasoning_effort=None,
                           reasoning_preserve=None):
    """The template kwargs a llama-server started with these flags passes with every request
    (common/arg.cpp: default_template_kwargs), as JSON values; None when no flag is given (the
    server's defaults). chat_template_kwargs: --chat-template-kwargs (a JSON object or its text);
    reasoning: --reasoning on|off|auto (enable_thinking; auto sets nothing); reasoning_effort:
    --reasoning-effort LEVEL ("default" sets nothing); reasoning_preserve: True for
    --reasoning-preserve, False for --no-reasoning-preserve (preserve_reasoning, on unless set).
    Applied in that order, which is the server's when it is given its flags in that order."""
    if chat_template_kwargs is None and reasoning is None and reasoning_effort is None and reasoning_preserve is None:
        return None
    kwargs = {}
    if chat_template_kwargs is not None:
        if isinstance(chat_template_kwargs, str):
            try:
                chat_template_kwargs = json.loads(chat_template_kwargs)
            except json.JSONDecodeError:
                raise ValueError("--chat-template-kwargs must be a JSON object") from None
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("--chat-template-kwargs must be a JSON object")
        kwargs.update(chat_template_kwargs)
    if reasoning is not None:
        if str(reasoning) not in REASONING_VALUES:
            raise ValueError(f"unknown value for --reasoning: {reasoning!r} (on, off or auto)")
        if REASONING_VALUES[str(reasoning)] is not None:
            kwargs["enable_thinking"] = REASONING_VALUES[str(reasoning)]
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or not reasoning_effort:
            raise ValueError("--reasoning-effort takes a level such as low, medium or high, or default")
        if reasoning_effort == "default":
            kwargs.pop("reasoning_effort", None)
        else:
            kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_preserve is not None:
        kwargs["preserve_reasoning"] = bool(reasoning_preserve)
    if isinstance(kwargs.get("enable_thinking"), str):
        # the server would refuse every chat request (server-common.cpp: "expected boolean")
        raise ValueError("enable_thinking must be a boolean")
    kwargs.setdefault("preserve_reasoning", True)
    return kwargs


def _parts(conversation):
    """(message index, part) for every content part; raises ValueError for parts that are not in
    the form conversation_from_openai gives (e.g. raw OpenAI image_url parts)."""
    for index, message in enumerate(conversation["messages"]):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind not in ("text", "media"):
                    raise ValueError(f"message {index} has a content part of type {kind!r}; convert OpenAI "
                                     "messages with conversation_from_openai first")
                yield index, part


def media(conversation):
    """{modality: index of the first message holding it} ({} for a text-only conversation)."""
    found = {}
    for index, part in _parts(conversation):
        if part["type"] == "media":
            found.setdefault(part["modality"], index)
    return found


def media_count(conversation):
    """Media parts in the conversation."""
    return sum(1 for _, part in _parts(conversation) if part["type"] == "media")


def has_text_parts(conversation):
    """True when some message gives its text as a list of parts without media."""
    return any(isinstance(m.get("content"), list) and not any(p.get("type") == "media" for p in m["content"])
               for m in conversation["messages"])


def _joined(parts, marker, text_join):
    if not any(part["type"] == "media" for part in parts):
        # text-only parts reach the template as parts; text_join is how it renders them
        return text_join.join(part["text"] for part in parts)
    # a message with media is flattened before the template: text parts joined with a newline,
    # none next to a media marker (common/chat.cpp:195-218, common_chat_msg::to_json_oaicompat)
    text, last_was_marker = "", False
    for part in parts:
        if part["type"] == "text":
            if not last_was_marker and text:
                text += "\n"
            text += part["text"]
            last_was_marker = False
        else:
            if marker is None:
                raise ValueError("media parts need the target server's media marker")
            text += marker
            last_was_marker = True
    return text


def flatten(conversation, *, marker=None, text_join=""):
    """The conversation with every content a string, rendered as llama-server renders the real
    request: media parts become the target server's media marker (/props media_marker, random
    per server process), so /apply-template of the result matches the server's own rendering
    of the request byte for byte."""
    list(_parts(conversation))          # every part in the bridge's form, or ValueError
    messages = []
    for message in conversation["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": _joined(content, marker, text_join)}
        messages.append(message)
    return {**conversation, "messages": messages}
