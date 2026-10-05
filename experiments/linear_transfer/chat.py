"""Render corpus v2 conversations with a model's own chat template and locate every text span."""

import argparse
import collections.abc
import copy
from dataclasses import dataclass
import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import string as string_module
import sys
import tempfile
import types

import jinja2
import jinja2.compiler
import jinja2.ext
import jinja2.filters
import jinja2.meta
import jinja2.nodes
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
# the whitespace llama.cpp's jinja strips (isspace in the C locale): trim/strip keep NBSP, U+3000
# and other Unicode spaces at the edges of a string
EDGE_SPACE = " \t\n\v\f\r"


def raise_exception(message):
    raise ValueError(f"chat template error: {message}")


def strftime_now(format_string):
    return FIXED_DATE.strftime(format_string)


_JSON_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _cpp_double(value):
    """A float as C++'s `ostream << double` writes it (six significant digits, %g): 1.0 -> 1,
    0.25 -> 0.25, 1234567.0 -> 1.23457e+06."""
    return "%g" % value


def _json_text(value, level, indent, item_sep, key_sep):
    """llama.cpp's value_to_json_internal (common/jinja/value.cpp)."""
    if value is None or isinstance(value, jinja2.Undefined):
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _cpp_double(value)
    if isinstance(value, str):
        return '"' + "".join(_JSON_ESCAPES.get(c) or (f"\\u{ord(c):04x}" if c < " " else c) for c in value) + '"'
    newline = "\n" if indent is not None and indent >= 0 else ""
    pad = " " * (level * indent) if indent is not None and indent > 0 else ""
    step = " " * indent if indent is not None and indent > 0 else ""
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = [pad + step + _json_text(item, level + 1, indent, item_sep, key_sep) for item in value]
        return "[" + newline + (item_sep + newline).join(items) + newline + pad + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [pad + step + _json_text(str(key), level + 1, indent, item_sep, key_sep) + key_sep
                 + _json_text(item, level + 1, indent, item_sep, key_sep) for key, item in value.items()]
        return "{" + newline + (item_sep + newline).join(items) + newline + pad + "}"
    return "null"


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """The tojson filter as llama.cpp's jinja runs it (common/jinja/value.cpp tojson): keys in
    their order, ", " and ": " (or "," and ": " with an indent), floats as C++ prints them (a
    tool argument 1.0 renders as 1), the C++ string escapes, non-ASCII kept unless ensure_ascii
    (then \\uXXXX, surrogate pairs above the BMP); sort_keys is refused as llama.cpp refuses it."""
    if sort_keys:
        raise ValueError("chat template error: tojson sort_keys=true not implemented")
    indent = int(indent) if isinstance(indent, int) and not isinstance(indent, bool) else None
    separators = list(separators or [])
    item_sep = separators[0] if separators else (", " if indent is None or indent < 0 else ",")
    key_sep = separators[1] if len(separators) > 1 else ": "
    text = _json_text(value, 0, indent, item_sep, key_sep)
    if ensure_ascii:
        # inside strings only: every non-ASCII character as \uXXXX (both escapes are ASCII-only)
        text = "".join(c if ord(c) < 0x80 else "".join(f"\\u{u:04x}" for u in
                                                         ([ord(c)] if ord(c) <= 0xFFFF else
                                                          [0xD800 + ((ord(c) - 0x10000) >> 10),
                                                           0xDC00 + ((ord(c) - 0x10000) & 0x3FF)]))
                       for c in text)
    return text


# what llama.cpp's jinja holds as an array: lists, tuples, and what jinja2 makes of them (a
# range, dict views, the generators of map, select or items)
_SEQUENCES = (list, tuple, range, types.GeneratorType, collections.abc.KeysView,
              collections.abc.ValuesView, collections.abc.ItemsView)


def _printed_float(value):
    """A float as llama.cpp's jinja writes it as a string (value_float_t::as_string): std::to_string
    with the trailing zeros dropped but one decimal kept (0.25 -> 0.25, 3.0 -> 3.0, 1e-07 -> 0.0)."""
    text = ("%f" % value).rstrip("0")
    return text + "0" if text.endswith(".") else text


def _repr(value):
    """An item inside an array or mapping written as a string (value_to_string_repr): a string
    quoted ('x', or as JSON when it holds a quote), anything else as _as_string writes it."""
    if isinstance(value, str):
        return _json_text(value, 0, None, ", ", ": ") if "'" in value else "'" + value + "'"
    if isinstance(value, jinja2.Undefined):
        return "Undefined"
    return _as_string(value)


def _as_string(value):
    """A value as llama.cpp's jinja turns it into a string (value_t::as_string in
    common/jinja/value.h), what `~`, `+` with a string, join and the string filter write: None
    as "None", a float as _printed_float, an array as "[1, 'a']" (a tuple "(1,)"), a mapping as
    "{'k': 0.5}"; an undefined value has none."""
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _printed_float(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(_repr(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, _SEQUENCES):
        return "[" + ", ".join(_repr(item) for item in value) + "]"
    if isinstance(value, collections.abc.Mapping):
        return "{" + ", ".join(f"{_repr(key)}: {_repr(item)}" for key, item in value.items()) + "}"
    if isinstance(value, jinja2.Undefined):
        raise ValueError("chat template error: an undefined value is not a string")
    return str(value)


def _text(value):
    """A string filter's input as llama.cpp coerces it: an undefined value is empty, any other
    non-string is written _as_string."""
    if isinstance(value, jinja2.Undefined):
        return ""
    return value if isinstance(value, str) else _as_string(value)


def finalize(value):
    """What {{ value }} prints, as llama.cpp's jinja prints it (gather_string_parts_recursive in
    common/jinja/runtime.h), where jinja2 differs: a float as _printed_float, a list (or a tuple,
    range, dict view or filter generator) as its items printed one after another, and None,
    undefined values and mappings as nothing."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _printed_float(value)
    if isinstance(value, _SEQUENCES):
        return "".join(finalize(item) for item in value)
    return ""


def default(value, default_value="", boolean=False):
    """The default filter as llama.cpp's jinja runs it: None counts as missing too; a mapping has
    no default filter there."""
    if isinstance(value, collections.abc.Mapping):
        raise ValueError("chat template error: a mapping has no default filter")
    missing = not value if boolean else (value is None or isinstance(value, jinja2.Undefined))
    return default_value if missing else value


def as_list(value):
    """The list filter as llama.cpp's jinja runs it: an array's items (a string or a mapping is
    refused), an undefined value none."""
    if isinstance(value, jinja2.Undefined):
        return []
    if isinstance(value, (str, collections.abc.Mapping)) or not isinstance(value, _SEQUENCES):
        raise ValueError("chat template error: list() needs an array")
    return list(value)


@jinja2.pass_context
def map_items(context, value, *args, **kwargs):
    """The map filter as llama.cpp's jinja runs it: by attribute only (map('upper') is refused)."""
    if args:
        raise ValueError("chat template error: map: filter-mapping not implemented")
    return jinja2.filters.FILTERS["map"](context, value, **kwargs)


def _selecting(name):
    """select / reject as llama.cpp's jinja runs them: a single argument names an attribute whose
    truth decides (selectattr), not a test; without one they are refused."""
    @jinja2.pass_context
    def select(context, value, *args, **kwargs):
        if not args and not kwargs:
            raise ValueError(f"chat template error: {name}() needs a test or an attribute")
        if len(args) == 1 and not kwargs:
            return jinja2.filters.FILTERS[name + "attr"](context, value, args[0])
        return jinja2.filters.FILTERS[name](context, value, *args, **kwargs)
    return select


def _unsupported(name):
    """A jinja2 filter llama.cpp's jinja does not have: an error where a template reaches it (an
    undefined value sums to 0 there)."""
    def refuse(value, *args, **kwargs):
        if name == "sum" and isinstance(value, jinja2.Undefined):
            return 0
        raise ValueError(f"chat template error: llama.cpp's jinja has no {name} filter")
    return refuse


# jinja2 filters llama.cpp's jinja (common/jinja/value.cpp) lacks
UNSUPPORTED_FILTERS = ("attr", "batch", "center", "e", "escape", "filesizeformat", "forceescape", "groupby", "pprint",
                       "random", "round", "striptags", "sum", "urlencode", "urlize", "wordwrap", "xmlattr")


def string(value):
    """The string filter as llama.cpp's jinja runs it: a float as tojson writes it (1.0 -> 1),
    anything else _as_string (None -> "None", arrays and mappings as llama.cpp writes them), an
    undefined value empty."""
    if isinstance(value, jinja2.Undefined):
        return ""
    return _cpp_double(value) if isinstance(value, float) else _as_string(value)


def format_text(value, *args, **kwargs):
    """The format filter, and str.format, as llama.cpp's jinja runs them: each '{}' takes the next
    argument _as_string; any other '{' is refused, as llama.cpp refuses it, and '%' is plain text."""
    if not isinstance(value, str):
        raise ValueError("chat template error: format() first argument must be a string")
    if kwargs:
        raise ValueError("chat template error: format() takes positional arguments only")
    pieces, arguments = value.split("{"), iter(args)
    out = [pieces[0]]
    for piece in pieces[1:]:
        if not piece.startswith("}"):
            raise ValueError("chat template error: format() only supports simple '{}' placeholders")
        try:
            out.append(_as_string(next(arguments)))
        except StopIteration:
            raise ValueError("chat template error: format() has more '{}' than arguments") from None
        out.append(piece[1:])
    return "".join(out)


def safe(value):
    """The safe filter as llama.cpp's jinja runs it: plain text (jinja2's Markup would HTML-escape
    what is added to it), a number as tojson writes it, an undefined value empty; an array or a
    mapping has no safe filter there."""
    if isinstance(value, jinja2.Undefined):
        return ""
    if isinstance(value, float):
        return _cpp_double(value)
    if isinstance(value, (str, int)) or value is None:
        return str(_as_string(value))
    raise ValueError("chat template error: safe() needs a string or a number")


def length(value):
    """The length (count) filter as llama.cpp's jinja counts: a string's UTF-8 bytes, an undefined
    value 0."""
    if isinstance(value, jinja2.Undefined):
        return 0
    return len(value.encode("utf-8")) if isinstance(value, str) else len(value)


def join(value, d="", attribute=None):
    """The join filter as llama.cpp's jinja runs it: the items (or their `attribute`) must be
    strings or numbers and are written _as_string (a float 1e-07 as 0.0); a string or a mapping
    is refused."""
    if isinstance(value, jinja2.Undefined):
        return ""
    if isinstance(value, (str, collections.abc.Mapping)) or not isinstance(value, _SEQUENCES):
        raise ValueError("chat template error: join() first argument must be an array")
    separator = "" if isinstance(d, jinja2.Undefined) else _as_string(d)
    parts = []
    for item in value:
        if attribute is not None and not isinstance(attribute, jinja2.Undefined):
            if isinstance(item, collections.abc.Mapping):
                item = item.get(attribute)
            elif isinstance(item, (list, tuple)) and str(attribute).isdigit() and int(attribute) < len(item):
                item = item[int(attribute)]
            else:
                item = None
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError("chat template error: join() can only join arrays of strings or numerics")
        parts.append(_as_string(item))
    return separator.join(parts)


def reverse(value):
    """The reverse filter as llama.cpp's jinja runs it: an array reversed (a list, which prints its
    items; a tuple stays a tuple); a string or a mapping is refused."""
    if isinstance(value, jinja2.Undefined):
        return []
    if isinstance(value, tuple):
        return value[::-1]
    if isinstance(value, (str, collections.abc.Mapping)) or not isinstance(value, _SEQUENCES):
        raise ValueError("chat template error: reverse() needs an array")
    return list(value)[::-1]


_UPPER = str.maketrans(string_module.ascii_lowercase, string_module.ascii_uppercase)
_LOWER = str.maketrans(string_module.ascii_uppercase, string_module.ascii_lowercase)


def upper(value):
    """upper, lower, capitalize and title as llama.cpp's jinja runs them (toupper/tolower in the C
    locale): ASCII letters only, title capitalizing after ASCII whitespace only."""
    return _text(value).translate(_UPPER)


def lower(value):
    return _text(value).translate(_LOWER)


def capitalize(value):
    text = _text(value)
    return text[:1].translate(_UPPER) + text[1:].translate(_LOWER)


def title(value):
    out, start = [], True
    for character in _text(value):
        out.append(character.translate(_UPPER if start else _LOWER))
        start = character in EDGE_SPACE
    return "".join(out)


def _stripped(value, chars=None, *, left=True, right=True):
    """str.strip as llama.cpp's jinja runs it (jinja::string::strip): EDGE_SPACE by default, so NBSP
    and other Unicode spaces stay; `chars` is a set of bytes, and a strip that would cut a UTF-8
    character apart is refused."""
    chars = EDGE_SPACE if chars is None or isinstance(chars, jinja2.Undefined) else _as_string(chars)
    if chars.isascii():
        value = value.lstrip(chars) if left else value
        return value.rstrip(chars) if right else value
    data, table = value.encode("utf-8"), set(chars.encode("utf-8"))
    start, end = 0, len(data)
    while left and start < end and data[start] in table:
        start += 1
    while right and end > start and data[end - 1] in table:
        end -= 1
    try:
        return data[start:end].decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("chat template error: strip() would cut a UTF-8 character apart") from None


def trim(value, chars=None):
    """The trim filter (llama.cpp's strip, see _stripped); a non-string is written _as_string first
    (without chars, as llama.cpp coerces it), an undefined value is empty."""
    if isinstance(value, jinja2.Undefined):
        return ""
    if not isinstance(value, str):
        if chars is not None:
            raise ValueError("chat template error: strip() first argument must be a string")
        value = _as_string(value)
    return _stripped(value, chars)


def split_text(value, sep=" ", maxsplit=-1, *, reverse=False):
    """str.split and rsplit as llama.cpp's jinja runs them: the separator is one space by default
    (not runs of whitespace) and is written _as_string."""
    sep = _as_string(sep)
    if not sep:
        raise ValueError("chat template error: empty separator")
    return value.rsplit(sep, int(maxsplit)) if reverse else value.split(sep, int(maxsplit))


# str methods whose llama.cpp builtin (common/jinja/value.cpp, value_string_t) differs from Python's
_STRING_METHODS = {
    "strip": lambda value, chars=None: _stripped(value, chars),
    "lstrip": lambda value, chars=None: _stripped(value, chars, right=False),
    "rstrip": lambda value, chars=None: _stripped(value, chars, left=False),
    "upper": upper, "lower": lower, "capitalize": capitalize, "title": title,
    "split": split_text, "rsplit": functools.partial(split_text, reverse=True), "format": format_text,
}


def _concatenated(left, right):
    """`left ~ right`, and `+` with a string, as llama.cpp's jinja runs them: both written
    _as_string; None or an undefined value next to a string counts as empty, and anything else
    without a string is an error (jinja2 writes str() of each)."""
    null_left = left is None or isinstance(left, jinja2.Undefined)
    null_right = right is None or isinstance(right, jinja2.Undefined)
    if null_left or null_right:
        if null_left and isinstance(right, str):
            return right
        if null_right and isinstance(left, str):
            return left
        raise ValueError("chat template error: cannot concatenate a null or undefined value")
    if not isinstance(left, str) and not isinstance(right, str):
        raise ValueError(f"chat template error: cannot concatenate {type(left).__name__} and "
                         f"{type(right).__name__}")
    return "".join((_as_string(left), _as_string(right)))       # plain text, never jinja2's Markup


class _CodeGenerator(jinja2.compiler.CodeGenerator):
    """Compiles `a ~ b ~ c` as llama.cpp evaluates it, ((a ~ b) ~ c) with _concatenated, also
    where jinja2 would fold it into printed text at compile time."""

    def _output_child_to_const(self, node, frame, finalize):
        if isinstance(node, jinja2.nodes.Concat) or any(True for _ in node.find_all(jinja2.nodes.Concat)):
            raise jinja2.nodes.Impossible()
        return super()._output_child_to_const(node, frame, finalize)

    def visit_Concat(self, node, frame):
        self.write("environment.concatenated((")
        for operand in node.nodes:
            self.visit(operand, frame)
            self.write(", ")
        self.write("))")


class _Environment(ImmutableSandboxedEnvironment):
    """The sandbox with llama.cpp's `~`, `+` with a string and str methods (_STRING_METHODS).
    make_environment turns constant folding off: it would apply jinja2's own `~` to literals."""

    code_generator_class = _CodeGenerator
    intercepted_binops = frozenset({"+"})

    @staticmethod
    def concatenated(operands):
        return functools.reduce(_concatenated, operands)

    def call_binop(self, context, operator, left, right):
        if operator == "+" and (isinstance(left, str) or isinstance(right, str)):
            return _concatenated(left, right)
        return super().call_binop(context, operator, left, right)

    def getattr(self, obj, attribute):
        if isinstance(obj, str) and attribute in _STRING_METHODS:
            return functools.partial(_STRING_METHODS[attribute], obj)
        return super().getattr(obj, attribute)


class GenerationBlocks(jinja2.ext.Extension):
    """transformers' {% generation %}...{% endgeneration %} (it marks the assistant's tokens for
    training masks, e.g. Laguna's template): llama.cpp's jinja ignores both tags and renders
    what they enclose (common/jinja/parser.cpp), and so does this."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def make_environment():
    # ChainableUndefined: an attribute of an undefined value is undefined too, as in llama.cpp
    environment = _Environment(trim_blocks=True, lstrip_blocks=True, finalize=finalize, optimized=False,
                               undefined=jinja2.ChainableUndefined,
                               extensions=[jinja2.ext.loopcontrols, GenerationBlocks])
    environment.filters["tojson"] = tojson
    environment.filters["default"] = environment.filters["d"] = default
    environment.filters["string"] = string
    environment.filters["format"] = format_text
    environment.filters["length"] = environment.filters["count"] = length
    environment.filters["join"] = join
    environment.filters["reverse"] = reverse
    environment.filters["trim"] = trim
    environment.filters["safe"] = safe
    environment.filters["list"] = as_list
    environment.filters["map"] = map_items
    environment.filters["select"], environment.filters["reject"] = _selecting("select"), _selecting("reject")
    for name in ("upper", "lower", "capitalize", "title"):
        environment.filters[name] = globals()[name]
    for name in UNSUPPORTED_FILTERS:
        environment.filters[name] = _unsupported(name)
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
    escaped: tuple = ()  # (live) segment indices the template writes JSON-escaped, which have no span


def _escaped(text):
    """text as tojson writes it between its quotes."""
    return _json_text(text, 0, None, ", ", ": ")[1:-1]


def render(conversation, template, *, bos_token="", add_generation_prompt=False, extra_context=None, live=False):
    """Rendered: the text, and the byte span of every segment it holds verbatim.

    live=True (a running chat, see validate_conversation): a segment's span is its core, the text
    between the EDGE_SPACE whitespace at its edges, which templates often trim and which is
    template text then (a whitespace-only segment has no span); a segment the template writes
    JSON-escaped (a string tool argument with a newline or a quote, under tojson) has no span.
    Otherwise every rendered segment must appear verbatim."""
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
    cores = [value.strip(EDGE_SPACE) if live else value for _, value in segments]

    def marked_segment(number):
        # the sentinel takes the core's place; the whitespace around it is text the template sees
        value, core = segments[number][1], cores[number]
        if not core:
            return value
        start = len(value) - len(value.lstrip(EDGE_SPACE))
        return value[:start] + SENTINEL.format(number) + value[start + len(core):]

    def run(messages):
        return compiled.render(messages=messages, tools=conversation.get("tools"),
                               bos_token=bos_token, add_generation_prompt=add_generation_prompt,
                               **extra_context)

    real = run(copy.deepcopy(conversation["messages"]))
    marked = run(_substituted_messages(conversation["messages"], marked_segment))
    matches = list(SENTINEL_PATTERN.finditer(marked))
    pieces, spans, escaped, seen, offset, last, position = [], [], [], set(), 0, 0, 0
    for at, match in enumerate(matches):
        number = int(match.group(1))
        if number >= len(segments) or match.group(1) != str(number):
            raise ValueError("template produced an unknown segment sentinel")
        if number in seen:
            raise ValueError(f"template renders segment {keys[number]} more than once")
        seen.add(number)
        literal = marked[last:match.start()]
        pieces.append(literal)
        offset += len(literal.encode("utf-8"))
        position += len(literal)
        # the core as the real render writes it: verbatim, or (live) JSON-escaped; the template
        # text up to the next sentinel tells them apart
        core = cores[number]
        following = marked[match.end():matches[at + 1].start() if at + 1 < len(matches) else len(marked)]
        candidates = [core] + ([_escaped(core)] if live and _escaped(core) != core else [])
        written = next((text for text in candidates if real.startswith(text + following, position)), core)
        pieces.append(written)
        size = len(written.encode("utf-8"))
        if written == core:
            spans.append((number, offset, offset + size))
        else:
            escaped.append(number)
        offset += size
        position += len(written)
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
                    template_sha256=template_sha256(template), bos_token=bos_token, escaped=tuple(escaped))


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
