"""Chat-template rendering and segment location checks (no model weights involved)."""

import contextlib
import copy
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from chat import (Rendered, compile_template, document_id, gguf_tokenizer_info, main,
                  make_environment, render, rendered_record, segment_texts, server_context, server_contexts,
                  template_sha256, template_uses_bos, tojson, validate_conversation)


GGUF_PY = Path(__file__).resolve().parents[2] / "runtime/source/gguf-py"
# Read-only copies of the real GGUF templates (see provenance.json there).
TEMPLATE_DIR = Path(os.environ.get(
    "KVPARK_CHAT_TEMPLATES",
    Path(__file__).resolve().parents[2] / "evidence/linear-transfer/chat-templates"))
QWEN_GGUF = Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf")
MUSE_GGUF = Path("/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf")
MUSE_BOS = "<|begin_of_text|>"
QWEN_BOS = "<|endoftext|>"
# Templates of the dense Qwen3-4B-Instruct-2507 (qwen3) and Ornith-1.5-9B (qwen35), saved by
# chat.py --template-output from their GGUFs (evidence/linear-transfer/model-support-u1).
U1_DIR = Path(os.environ.get(
    "KVPARK_MODEL_SUPPORT_U1",
    Path(__file__).resolve().parents[2] / "evidence/linear-transfer/model-support-u1"))
QWEN3_GGUF = Path("/srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf")
ORNITH_GGUF = Path("/srv/models/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf")
CORPUS_V2 = Path(__file__).resolve().parents[2] / "evidence/linear-transfer/corpus-v2/conversations.jsonl"
# kvpark-chat-apply (llama.cpp's common_chat_templates_apply) and kvpark-kv-probe binaries
CHAT_APPLY = os.environ.get("KVPARK_CHAT_APPLY")
PROBE = os.environ.get("KVPARK_KV_PROBE")
# llama-server (CPU is enough): template_check's /apply-template comparison on tokenizer twins
SERVER = os.environ.get("KVPARK_LLAMA_SERVER")

# Miniature templates reproducing the behaviours of the real ones that matter here.
QWEN_LIKE = """\
{%- if tools %}<|im_start|>system
<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>

{{ messages[0].content | trim }}<|im_end|>
{% else %}<|im_start|>system
{{ messages[0].content | trim }}<|im_end|>
{% endif %}
{%- for message in messages[1:] %}
{%- set content = message.content | trim %}
{%- if message.role == 'assistant' %}
<|im_start|>assistant
<think>
{{ (message.reasoning_content or '') | trim }}
</think>

{{ content }}
{%- for call in message.tool_calls or [] %}
<tool_call><function={{ call.function.name }}>
{%- for name, value in call.function.arguments | items %}
<parameter={{ name }}>
{{ value if value is string else value | tojson }}
</parameter>
{%- endfor %}
</function></tool_call>
{%- endfor %}<|im_end|>
{% elif message.role == 'tool' %}<|im_start|>user
<tool_response>
{{ content }}
</tool_response><|im_end|>
{% else %}<|im_start|>{{ message.role }}
{{ content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

MUSE_LIKE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{%- if message.role == 'assistant' -%}"
    "{%- if message.get('reasoning_content') -%}<|start|>assistant to=self<|message|>"
    "{{- message.reasoning_content -}}<|eom|>{%- endif -%}"
    "{%- if message.get('tool_calls') -%}{%- for call in message.tool_calls -%}"
    "<|start|>assistant to={{ call.function.name }}<|message|>"
    "{%- for key, value in call.function.arguments.items() -%}"
    "<p name=\"{{ key }}\">{%- if value is mapping or (value is iterable and value is not string) -%}"
    "{{- value | tojson -}}{%- else -%}{{- value -}}{%- endif -%}</p>"
    "{%- endfor -%}<|eot|>{%- endfor -%}"
    "{%- else -%}<|start|>assistant<|message|>{{- message.content -}}<|eot|>{%- endif -%}"
    "{%- elif message.role == 'tool' -%}<|start|>tool {{ message.name }}<|message|>"
    "{{- message.content -}}<|eot|>"
    "{%- else -%}<|start|>{{ message.role }}<|message|>{{- message.content -}}<|eot|>{%- endif -%}"
    "{%- endfor -%}")


def conversation(split="test"):
    return {
        "id": "subset:7", "source": "subset", "split": split,
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Wetter für eine Stadt <b>",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                            "days": {"type": "integer"}}}}}],
        "messages": [
            {"role": "system", "content": "You are a careful assistant. Ünïcode ✓"},
            {"role": "user", "content": "Weather in Zürich for 3 days? 日本語で"},
            {"role": "assistant", "content": "Let me check that for you.",
             "reasoning_content": "The user wants a forecast; I should call the tool.",
             "tool_calls": [{"id": "call-1", "type": "function", "function": {
                 "name": "get_weather",
                 "arguments": {"city": "Zürich, CH", "days": 3, "detail": {"hourly": True}}}}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "get_weather",
             "content": '{"temp_c": 12, "sky": "cloudy ☁"}'},
            {"role": "assistant", "content": "Expect about 12 °C and clouds — bring a jacket.",
             "reasoning_content": "Summarize the tool output."},
        ],
    }


def segment_text_map(conv):
    return dict(segment_texts(conv))


class SyntheticTemplateTests(unittest.TestCase):
    def check_spans(self, conv, rendered):
        encoded = rendered.text.encode("utf-8")
        texts = segment_text_map(conv)
        previous = 0
        for index, start, end in rendered.segments:
            self.assertGreaterEqual(start, previous)
            previous = end
            self.assertEqual(encoded[start:end].decode("utf-8"), texts[rendered.segment_keys[index]])

    def test_canonical_segments_cover_reasoning_content_and_string_arguments_only(self):
        self.assertEqual([key for key, _ in segment_texts(conversation())],
                         ["0:content", "1:content", "2:reasoning", "2:content",
                          "2:arg:0:city", "3:content", "4:reasoning", "4:content"])

    def test_qwen_like_spans_are_exact_utf8_byte_ranges(self):
        conv = conversation()
        rendered = render(conv, QWEN_LIKE)
        self.check_spans(conv, rendered)
        self.assertEqual(len(rendered.segments), 8)
        self.assertIn("<parameter=days>\n3\n", rendered.text)
        self.assertIn('{"hourly": true}', rendered.text)
        self.assertIn("Wetter für eine Stadt <b>", rendered.text)  # no HTML escaping in tojson
        self.assertEqual(rendered.final_message_segments, [6, 7])

    def test_muse_like_drops_tool_call_content_and_emits_bos(self):
        conv = conversation()
        rendered = render(conv, MUSE_LIKE, bos_token=MUSE_BOS)
        self.check_spans(conv, rendered)
        self.assertTrue(rendered.text.startswith(MUSE_BOS + "<|start|>system"))
        present = {rendered.segment_keys[index] for index, _, _ in rendered.segments}
        self.assertNotIn("2:content", present)
        self.assertIn("2:content", rendered.segment_keys)
        self.assertIn('<p name="days">3</p>', rendered.text)
        self.assertNotIn(MUSE_BOS, render(conv, MUSE_LIKE).text)

    def test_duplicate_sentinel_is_rejected(self):
        template = MUSE_LIKE.replace("{{- message.content -}}<|eot|>{%- endif -%}",
                                     "{{- message.content -}}{{- message.content -}}<|eot|>{%- endif -%}")
        with self.assertRaisesRegex(ValueError, "more than once"):
            render(conversation(), template)

    def test_content_transforming_templates_are_detected(self):
        for old, new in (("{{- message.content -}}<|eot|>{%- endif -%}",
                          "{{- message.content | upper -}}<|eot|>{%- endif -%}"),
                         ("{{- message.content -}}<|eot|>{%- endif -%}",
                          "{{- message.content | replace('a', 'b') -}}<|eot|>{%- endif -%}"),
                         ("{{- message.content -}}<|eot|>{%- endif -%}",
                          "{{- message.content | tojson(ensure_ascii=True) -}}<|eot|>{%- endif -%}"),
                         ("{{- message.content -}}<|eot|>{%- endif -%}",
                          "{{- message.content[:5] -}}<|eot|>{%- endif -%}")):
            with self.subTest(new=new), self.assertRaises(ValueError):
                render(conversation(), MUSE_LIKE.replace(old, new))

    def test_trim_is_a_no_op_on_stripped_text_and_unstripped_text_is_rejected(self):
        conv = conversation()
        self.check_spans(conv, render(conv, QWEN_LIKE))
        conv["messages"][1]["content"] = " padded "
        with self.assertRaisesRegex(ValueError, "stripped"):
            render(conv, QWEN_LIKE)

    def test_live_spans_hold_the_text_inside_edge_whitespace(self):
        # a running chat keeps its whitespace: a template that trims it (QWEN_LIKE: ASCII only, as
        # llama.cpp trims) and one that writes it raw (MUSE_LIKE) both span the same core bytes,
        # and a whitespace-only segment has no span
        conv = conversation("train")
        conv["messages"][1]["content"] = "\n  Weather in Zürich?\u00a0 \n"
        conv["messages"][3]["content"] = "\n"
        conv["messages"][4]["content"] = "Bring a jacket.\n\n"
        with self.assertRaisesRegex(ValueError, "stripped"):
            render(conv, QWEN_LIKE)                         # corpus records stay stripped
        qwen, muse = render(conv, QWEN_LIKE, live=True), render(conv, MUSE_LIKE, bos_token=MUSE_BOS, live=True)
        self.assertIn("<|im_start|>user\nWeather in Zürich?\u00a0<|im_end|>", qwen.text)
        self.assertIn("<|start|>user<|message|>\n  Weather in Zürich?\u00a0 \n<|eot|>", muse.text)
        self.assertIn("<tool_response>\n\n</tool_response>", qwen.text)
        for rendered in (qwen, muse):
            found = {rendered.segment_keys[i]: rendered.text.encode()[a:b].decode() for i, a, b in rendered.segments}
            self.assertEqual(found["1:content"], "Weather in Zürich?\u00a0")
            self.assertEqual(found["4:content"], "Bring a jacket.")
            self.assertNotIn("3:content", found)
            self.assertEqual(rendered.escaped, ())

    def test_live_arguments_written_json_escaped_have_no_span(self):
        template = MUSE_LIKE.replace("{{- value -}}{%- endif -%}</p>", "{{- value | tojson -}}{%- endif -%}</p>")
        conv = conversation()
        rendered = render(conv, template, live=True)        # tojson writes a plain string as it is, quoted
        self.assertIn('<p name="city">"Zürich, CH"</p>', rendered.text)
        self.assertIn("2:arg:0:city", [rendered.segment_keys[i] for i, _, _ in rendered.segments])
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"]["city"] = 'Zürich "Altstadt"\nCH'
        with self.assertRaisesRegex(ValueError, "transforms content"):
            render(conv, template)                          # corpus records need every segment verbatim
        rendered = render(conv, template, live=True)
        self.assertIn('<p name="city">"Zürich \\"Altstadt\\"\\nCH"</p>', rendered.text)
        self.assertEqual([rendered.segment_keys[i] for i in rendered.escaped], ["2:arg:0:city"])
        self.assertNotIn("2:arg:0:city", [rendered.segment_keys[i] for i, _, _ in rendered.segments])
        self.check_spans(conv, rendered)
        # a template that transforms content otherwise is refused live too
        with self.assertRaisesRegex(ValueError, "transforms content"):
            render(conv, template.replace("{{- message.content -}}<|eot|>{%- endif -%}",
                                          "{{- message.content | upper -}}<|eot|>{%- endif -%}"), live=True)

    def test_conversation_validation(self):
        mutations = [
            lambda c: c["messages"].pop(0),
            lambda c: c["messages"][1].update(role="system"),
            lambda c: c["messages"][1].update(content=""),
            lambda c: c["messages"][1].update(extra="x"),
            lambda c: c["messages"][2].update(reasoning_content=""),
            lambda c: c["messages"][2]["tool_calls"][0]["function"].update(arguments='{"a": 1}'),
            lambda c: c["messages"][3].update(tool_call_id="unknown"),
            lambda c: c["messages"].pop(),
            lambda c: c.update(split="validation"),
            lambda c: c.update(tools=[]),
            lambda c: c["messages"][1].update(content="bad \ue000 sentinel"),
            lambda c: c["messages"][4].update(tool_calls=[{"id": "call-2", "type": "function",
                                                           "function": {"name": "f", "arguments": {}}}]),
        ]
        for number, mutate in enumerate(mutations):
            conv = conversation()
            mutate(conv)
            with self.subTest(number=number), self.assertRaises(ValueError):
                render(conv, QWEN_LIKE)
        conv = conversation(split="train")
        conv["messages"][2]["content"] = ""
        conv["messages"].pop()
        validate_conversation(conv)
        self.assertNotIn("2:content", segment_text_map(conv))
        self.check_spans(conv, render(conv, QWEN_LIKE))

    def test_document_id_follows_the_spec_and_ignores_corpus_metadata(self):
        conv = conversation()
        canonical = json.dumps({"tools": conv["tools"], "messages": conv["messages"]},
                               sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(document_id(conv), hashlib.sha256(canonical.encode()).hexdigest())
        other = copy.deepcopy(conv)
        other.update(id="x:1", split="dev", source="other")
        self.assertEqual(document_id(other), document_id(conv))
        other["messages"][1]["content"] += "!"
        self.assertNotEqual(document_id(other), document_id(conv))

    def test_rendered_record(self):
        conv = conversation()
        rendered = render(conv, MUSE_LIKE, bos_token=MUSE_BOS)
        record = rendered_record(conv, "muse", rendered)
        self.assertEqual(set(record), {"document_id", "conversation_id", "model", "text", "segments",
                                       "segment_keys", "split", "source", "final_message_segments",
                                       "template_sha256", "bos_token"})
        self.assertEqual(record["segments"], [list(span) for span in rendered.segments])
        self.assertEqual(record["split"], "test")
        self.assertEqual(record["template_sha256"], hashlib.sha256(MUSE_LIKE.encode()).hexdigest())
        self.assertEqual(record["bos_token"], MUSE_BOS)
        bare = Rendered(rendered.text, rendered.segments, rendered.segment_keys,
                        rendered.final_message_segments)
        with self.assertRaisesRegex(ValueError, "provenance"):
            rendered_record(conv, "muse", bare)
        dropping = "{% for m in messages[:-1] %}{{ m.content }}{% endfor %}"
        with self.assertRaisesRegex(ValueError, "final message content"):
            rendered_record(conv, "muse", render(conv, dropping))
        rendered_record(conversation("train"), "muse", render(conversation("train"), dropping))
        json.dumps(record, allow_nan=False)
        other = conversation("train")
        other["messages"].pop()
        with self.assertRaises(ValueError):
            rendered_record(other, "muse", rendered)
        with self.assertRaises(ValueError):
            rendered_record({**conv, "split": None}, "muse", rendered)

    def test_environment_globals_and_filters(self):
        environment = make_environment()
        self.assertEqual(environment.from_string("{{ strftime_now('%Y-%m-%d') }}").render(), "2026-09-27")
        self.assertEqual(environment.from_string("{{ x | tojson }}").render(x={"a": "<é>"}),
                         '{"a": "<é>"}')
        self.assertEqual(environment.from_string("{{ x | tojson(indent=2) }}").render(x=[1]),
                         "[\n  1\n]")
        with self.assertRaisesRegex(ValueError, "boom"):
            environment.from_string("{{ raise_exception('boom') }}").render()
        with self.assertRaises(Exception):
            environment.from_string("{{ x.append(1) }}").render(x=[])

    def test_tojson_and_printed_floats_follow_llama_cpp(self):
        # common/jinja/value.cpp: tojson writes floats as C++ streams do (%g), {{ x }} as
        # std::to_string without its trailing zeros; keys keep their order; C++ string escapes
        environment = make_environment()
        value = {"z": 1.0, "a": [0.25, 1e-07, 1234567.0, 3, True, None], "s": "tab\t\x1b \"q\" \\ é", "e": {}}
        self.assertEqual(tojson(value), '{"z": 1, "a": [0.25, 1e-07, 1.23457e+06, 3, true, null], '
                                        '"s": "tab\\t\\u001b \\"q\\" \\\\ é", "e": {}}')
        nested = {"a": [1, {"b": 1.5}], "c": []}
        self.assertEqual(environment.from_string("{{ x | tojson(indent=2) }}").render(x=nested),
                         json.dumps(nested, indent=2))
        self.assertEqual(tojson([1, {"a": 2}], separators=(",", ":")), '[1,{"a":2}]')
        self.assertEqual(tojson("é🚀", ensure_ascii=True), json.dumps("é🚀"))
        self.assertEqual(environment.from_string("{{ x | tojson }}").render(x=None), "null")
        with self.assertRaisesRegex(ValueError, "sort_keys"):
            tojson({"b": 1, "a": 2}, sort_keys=True)
        self.assertEqual(environment.from_string("{{ a }}|{{ b }}|{{ c }}|{{ 2.0 }}|{{ d }}").render(
            a=1.0, b=0.25, c=1e-07, d=7), "1.0|0.25|0.0|2.0|7")

    def test_printed_values_and_filters_follow_llama_cpp(self):
        # gather_string_parts_recursive: None, undefined and mappings print nothing, a list its
        # items one after another; default() takes None as missing; format() leaves '%' as it is
        environment = make_environment()
        cases = {"{{ none }}": "", "{{ undefined_name }}": "", "{{ {'a': 1} }}": "",
                 "{{ [1, 'a', none, [2.0]] }}": "1a2.0",
                 "{{ 'a,b'.split(',') }}": "ab", "{{ range(3) | list }}": "012", "{{ {'a': 1}.items() | list }}": "a1",
                 "{{ messages | map(attribute='role') }}": "useruser", "{{ none | default('d') }}": "d",
                 "{{ '' | default('d') }}": "", "{{ '' | default('d', true) }}": "d", "{{ x | default('d') }}": "d",
                 "{{ 1.0 | string }}": "1", "{{ [1, 2] | string }}": "[1, 2]", "{{ none | string }}": "None",
                 "{{ '%.2f' | format(3.14159) }}": "%.2f", "{{ true }}{{ 3 }}": "True3"}
        for template, expected in cases.items():
            with self.subTest(template=template):
                self.assertEqual(environment.from_string(template).render(
                    messages=[{"role": "user"}, {"role": "user"}]), expected)

    def test_string_filters_methods_and_operators_follow_llama_cpp(self):
        # common/jinja: strip/trim remove ASCII whitespace only (isspace in the C locale), the case
        # filters change ASCII letters only, split splits on one space, format fills '{}', `~` and
        # `+` with a string write values as_string (None and undefined next to a string as nothing),
        # join writes floats as printed, string writes containers as llama.cpp does, length counts
        # UTF-8 bytes, reverse gives an array, and an attribute of an undefined value is undefined
        environment = make_environment()
        cases = {"{{ '\u00a0a\u3000' | trim }}": "\u00a0a\u3000", "{{ ' \u00a0a\u3000\n'.strip() }}": "\u00a0a\u3000",
                 "{{ '\n x \t' | trim }}": "x", "{{ ' x '.lstrip() }}|{{ ' x '.rstrip() }}": "x | x",
                 "{{ 'xxaxx'.strip('x') }}": "a", "{{ 'ééa'.lstrip('é') }}": "a", "{{ 3 | trim }}": "3",
                 "{{ 'é ölçü' | upper }}|{{ 'ÉCOLE'.lower() }}": "é öLçü|École",
                 "{{ 'hello-world  foo' | title }}|{{ 'aBC dEF' | capitalize }}": "Hello-world  Foo|Abc def",
                 "{{ 'a b  c'.split() | length }}|{{ 'a,b,c'.rsplit(',', 1) | join('/') }}": "4|a,b/c",
                 "{{ '{} items, {}' | format(3, none) }}|{{ '{}{}'.format(1, 2.0) }}": "3 items, None|12.0",
                 "{{ '%d' | format(3) }}": "%d", "{{ 'a' ~ none ~ x ~ 1e-07 ~ [1, 'b'] ~ {'k': 0.5} }}":
                 "a0.0[1, 'b']{'k': 0.5}", "{{ 'a' + none }}|{{ 1 + 'a' }}|{{ 1 + 2 }}|{{ [1] + [2] }}": "a|1a|3|12",
                 "{{ [0.5, 1e-07, 2] | join(',') }}|{{ messages | join('-', attribute='role') }}": "0.5,0.0,2|user-user",
                 "{{ {'d': 'x\ny', 'q': \"it's\", 'n': none} | string }}": "{'d': 'x\ny', 'q': \"it's\", 'n': None}",
                 "{{ ('a', 1) | string }}|{{ ('a',) | string }}": "('a', 1)|('a',)", "{{ 'é' | length }}": "2",
                 "{{ [1, 2] | reverse }}|{{ x.y.z is defined }}|{{ x.y.z }}": "21|False|",
                 "{{ messages | select('role') | list | length }}": "2", "{{ x | sum }}": "0",
                 "{{ '<r>' ~ ('<a>' | safe) ~ '</r>' }}|{{ ('<b>' | safe) + '&' }}|{{ 1.0 | safe }}": "<r><a></r>|<b>&|1"}
        for template, expected in cases.items():
            with self.subTest(template=template):
                self.assertEqual(environment.from_string(template).render(
                    messages=[{"role": "user"}, {"role": "user"}]), expected)
        # what llama.cpp's jinja refuses where a template reaches it (jinja2 would render it)
        for template in ("{{ 1 ~ 2 }}", "{{ none ~ none }}", "{{ '{0}' | format(1) }}", "{{ '{} {}' | format(1) }}",
                         "{{ 'abc' | reverse }}", "{{ 'ab' | join }}", "{{ [true] | join }}", "{{ {'a': 1} | list }}",
                         "{{ {'a': 1} | default(2) }}", "{{ ['a'] | map('upper') | list }}", "{{ [1] | select | list }}",
                         "{{ [1.5] | sum }}", "{{ 1.5 | round }}", "{{ 'é'.strip('\u00a9') }}"):
            with self.subTest(template=template), self.assertRaises(ValueError):
                environment.from_string(template).render()
        # jinja2 would fold a constant `~` into the template at compile time: not here
        self.assertEqual(environment.from_string("{% set x = 'b' ~ none %}{{ x }}{{ 'c' ~ none }}").render(), "bc")

    def test_generation_blocks_render_their_body(self):
        # transformers' {% generation %} (Laguna's template): llama.cpp ignores the tags
        template = ("{%- for m in messages %}<{{ m.role }}>{% if m.role == 'assistant' %}{% generation %}"
                    "{{ m.content }}{% endgeneration %}{% else %}{{ m.content }}{% endif %}</{{ m.role }}>"
                    "{%- endfor %}")
        conv = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                "tools": None}
        rendered = render(conv, template, live=True)
        self.assertEqual(rendered.text, "<user>Hi</user><assistant>Hello</assistant>")
        self.assertEqual([rendered.text.encode()[a:b] for _, a, b in rendered.segments], [b"Hi", b"Hello"])

    def test_extra_context(self):
        template = "{{ messages[0].content }}{% if flag %}!{% endif %}"
        conv = conversation("train")
        self.assertTrue(render(conv, template, extra_context={"flag": True}).text.endswith("!"))
        with self.assertRaises(ValueError):
            render(conv, template, extra_context={"messages": []})

    def test_server_context_maps_request_options_as_llama_server(self):
        now = datetime.datetime(2026, 10, 3, 21, 5)
        # no options: nothing beyond the template's defaults, or llama-server's defaults
        self.assertEqual(server_context(), {})
        self.assertEqual({k: v for k, v in server_context(server=True, eos_token="</s>", now=now).items()
                          if k != "strftime_now"},
                         {"eos_token": "</s>", "datetime": "Oct 03 2026", "date_string": "03 Oct 2026",
                          "enable_thinking": True, "preserve_reasoning": True, "preserve_thinking": True,
                          "clear_thinking": False, "truncate_history_thinking": False, "drop_thinking": False})
        self.assertEqual(server_context(server=True, now=now)["strftime_now"]("%Y-%m-%d"), "2026-10-03")
        # reasoning_effort: a template variable (and reasoning_strength); "none" turns thinking off
        self.assertEqual(server_context({"reasoning_effort": "medium"}),
                         {"reasoning_effort": "medium", "reasoning_strength": "medium"})
        self.assertEqual(server_context({"reasoning_effort": "none",
                                         "chat_template_kwargs": {"reasoning_effort": "low"}}),
                         {"enable_thinking": False})
        self.assertEqual(server_context({"reasoning_effort": 3}), {})          # not a string: ignored
        # kwargs: template variables, the thinking switch, the preserve_reasoning caps
        self.assertEqual(server_context({"chat_template_kwargs": {"enable_thinking": False, "flag": [1]}}),
                         {"enable_thinking": False, "flag": [1]})
        self.assertEqual(server_context({"chat_template_kwargs": {"preserve_reasoning": False}}),
                         {"preserve_thinking": False, "clear_thinking": True, "truncate_history_thinking": True,
                          "drop_thinking": True, "preserve_reasoning": False})
        # a kwarg wins over what the caps derive
        self.assertEqual(server_context({"reasoning_effort": "low",
                                         "chat_template_kwargs": {"reasoning_strength": "max"}})["reasoning_strength"],
                         "max")
        with self.assertRaisesRegex(ValueError, "expected boolean, got string"):
            server_context({"chat_template_kwargs": {"enable_thinking": "false"}})
        for bad in ({"chat_template_kwargs": [1]}, {"temperature": 0}):
            with self.assertRaises(ValueError):
                server_context(bad)
        # candidates: the template's defaults first; contexts that cannot differ appear once
        names = [name for name, _ in server_contexts({"reasoning_effort": "medium"})]
        self.assertEqual(names, ["template", "server", "server-no-thinking"])
        names = [name for name, _ in server_contexts({"chat_template_kwargs": {"enable_thinking": True}})]
        self.assertEqual(names, ["template", "server"])
        self.assertEqual(server_contexts()[0], ("template", {}))
        # the server's own flags (--reasoning-effort low --no-reasoning-preserve --reasoning off):
        # under the request's kwargs, over the server defaults; the template context ignores them
        flags = {"reasoning_effort": "low", "preserve_reasoning": False, "enable_thinking": False}
        context = server_context(server=True, now=now, server_kwargs=flags)
        self.assertEqual({key: context[key] for key in ("reasoning_effort", "reasoning_strength", "enable_thinking",
                                                        "preserve_reasoning", "drop_thinking")},
                         {"reasoning_effort": "low", "reasoning_strength": "low", "enable_thinking": False,
                          "preserve_reasoning": False, "drop_thinking": True})
        context = server_context({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}},
                                 server=True, now=now, server_kwargs=flags)
        self.assertEqual((context["reasoning_effort"], context["enable_thinking"]), ("high", True))
        self.assertTrue(server_context(server=True, server_kwargs={})["preserve_reasoning"])
        contexts = dict(server_contexts(server_kwargs=flags))
        self.assertEqual((sorted(contexts), contexts["template"]), (["server", "template"], {}))
        self.assertEqual(contexts["server"]["reasoning_effort"], "low")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            server_context(server=True, server_kwargs=[("reasoning_effort", "low")])

    def test_cli_renders_new_file_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "conv.jsonl").write_text(json.dumps(conversation(), ensure_ascii=False) + "\n",
                                             encoding="utf-8")
            (root / "t.jinja").write_text(MUSE_LIKE, encoding="utf-8")
            arguments = ["--conversations", str(root / "conv.jsonl"), "--model-name", "muse",
                         "--template", str(root / "t.jinja"), "--bos-token", MUSE_BOS,
                         "--output", str(root / "out.jsonl")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            record = json.loads((root / "out.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["text"].startswith(MUSE_BOS))
            self.assertEqual((record["bos_token"], record["template_sha256"]),
                             (MUSE_BOS, template_sha256(MUSE_LIKE)))
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["conv.jsonl", "out.jsonl", "t.jinja"])

    def test_cli_requires_an_explicit_bos_for_template_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "conv.jsonl").write_text(json.dumps(conversation()) + "\n", encoding="utf-8")
            for name, template in (("muse.jinja", MUSE_LIKE), ("qwen.jinja", QWEN_LIKE)):
                (root / name).write_text(template, encoding="utf-8")

            def run(name, *bos):
                return main(["--conversations", str(root / "conv.jsonl"), "--model-name", "m",
                             "--template", str(root / name), *bos, "--output", str(root / "out.jsonl")])

            with self.assertRaisesRegex(ValueError, "--bos-token is required"):
                run("muse.jinja")
            with self.assertRaisesRegex(ValueError, "uses bos_token"):
                run("muse.jinja", "--bos-token", "")
            self.assertFalse((root / "out.jsonl").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run("qwen.jinja", "--bos-token", ""), 0)
            record = json.loads((root / "out.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["bos_token"], "")
        self.assertTrue(template_uses_bos(MUSE_LIKE))
        self.assertFalse(template_uses_bos(QWEN_LIKE))
        self.assertFalse(template_uses_bos("{% set bos_token = 'x' %}{{ bos_token }}"))

    @unittest.skipUnless(GGUF_PY.is_dir(), "pinned gguf-py not available")
    def test_cli_reads_template_and_bos_from_gguf_metadata(self):
        import sys
        if str(GGUF_PY) not in sys.path:
            sys.path.insert(0, str(GGUF_PY))
        import gguf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "tiny.gguf"
            writer = gguf.GGUFWriter(model, "muse-glimmer")
            writer.add_token_list(["a", "b", MUSE_BOS])
            writer.add_bos_token_id(2)
            writer.add_add_bos_token(True)
            writer.add_chat_template(MUSE_LIKE)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            self.assertEqual(gguf_tokenizer_info(model),
                             {"chat_template": MUSE_LIKE, "bos_token": MUSE_BOS, "eos_token": "", "add_bos": True})
            (root / "conv.jsonl").write_text(json.dumps(conversation()) + "\n", encoding="utf-8")
            arguments = ["--conversations", str(root / "conv.jsonl"), "--model-name", "m",
                         "--gguf", str(model), "--output", str(root / "out.jsonl")]
            with self.assertRaisesRegex(ValueError, "not accepted with --gguf"):
                main(arguments + ["--bos-token", ""])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            record = json.loads((root / "out.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["text"].startswith(MUSE_BOS))
            self.assertEqual(record["bos_token"], MUSE_BOS)


@unittest.skipUnless((TEMPLATE_DIR / "qwen.jinja").is_file() and (TEMPLATE_DIR / "muse.jinja").is_file(),
                     "real chat templates not available (set KVPARK_CHAT_TEMPLATES)")
class RealTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen = (TEMPLATE_DIR / "qwen.jinja").read_text(encoding="utf-8")
        cls.muse = (TEMPLATE_DIR / "muse.jinja").read_text(encoding="utf-8")

    def render_both(self, conv):
        return render(conv, self.qwen), render(conv, self.muse, bos_token=MUSE_BOS)

    def test_realistic_conversation_spans_match_in_both_templates(self):
        conv = conversation()
        texts = segment_text_map(conv)
        qwen, muse = self.render_both(conv)
        for rendered in (qwen, muse):
            encoded = rendered.text.encode("utf-8")
            for index, start, end in rendered.segments:
                self.assertEqual(encoded[start:end].decode("utf-8"), texts[rendered.segment_keys[index]])
            self.assertEqual(rendered.segment_keys, list(texts))
            self.assertEqual(rendered.final_message_segments, [6, 7])
        self.assertEqual(len(qwen.segments), 8)
        self.assertEqual([muse.segment_keys[index] for index, _, _ in muse.segments],
                         ["0:content", "1:content", "2:reasoning", "2:arg:0:city", "3:content",
                          "4:reasoning", "4:content"])
        self.assertTrue(muse.text.startswith(MUSE_BOS + "<|start|>system<|message|>"))
        self.assertTrue(qwen.text.startswith("<|im_start|>system\n"))
        self.assertNotIn(MUSE_BOS, qwen.text)
        self.assertIn("<parameter=days>\n3\n</parameter>", qwen.text)
        self.assertIn('<atem:parameter name="days">3</atem:parameter>', muse.text)
        self.assertIn('{"hourly": true}', qwen.text)
        self.assertIn('{"hourly": true}', muse.text)
        self.assertIn("Wetter für eine Stadt <b>", qwen.text)
        self.assertIn("Wetter für eine Stadt <b>", muse.text)

    def test_conversation_without_tools_and_generation_prompt(self):
        conv = conversation("train")
        conv["tools"] = None
        conv["messages"] = conv["messages"][:2]
        qwen = render(conv, self.qwen, add_generation_prompt=True)
        muse = render(conv, self.muse, bos_token=MUSE_BOS, add_generation_prompt=True)
        self.assertTrue(qwen.text.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(muse.text.endswith("<|start|>assistant"))
        self.assertNotIn("# Tools", qwen.text)
        self.assertEqual(len(qwen.segments), 2)
        self.assertEqual(len(muse.segments), 2)
        disabled = render(conv, self.qwen, extra_context={"enable_thinking": False})
        self.assertNotIn("Reasoning effort", disabled.text)

    def test_muse_without_system_uses_fixed_date(self):
        text = compile_template(self.muse).render(
            messages=[{"role": "user", "content": "hi"}], tools=None, bos_token=MUSE_BOS,
            add_generation_prompt=False)
        self.assertIn("Current date: 2026-09-27.", text)
        # llama-server's context renders the current date (strftime_now at the request's time)
        now = datetime.datetime(2026, 10, 3, 12)
        live = render({"messages": [{"role": "user", "content": "hi"}], "tools": None}, self.muse,
                      bos_token=MUSE_BOS, live=True, extra_context=server_context(server=True, now=now))
        self.assertIn("Current date: 2026-10-03.", live.text)

    def test_request_options_render_as_llama_server_does(self):
        conv = conversation("train")
        conv["tools"] = None
        default = render(conv, self.qwen, add_generation_prompt=True).text
        self.assertIn("Reasoning effort is set to xhigh.", default)

        def qwen(options, **context):
            return render(conv, self.qwen, add_generation_prompt=True,
                          extra_context=server_context(options, **context)).text
        # Hermes' reasoning_effort "medium": Qwen3.8 writes no effort instructions at all
        self.assertNotIn("Reasoning effort", qwen({"reasoning_effort": "medium"}))
        self.assertIn("Reasoning effort is set to low.", qwen({"reasoning_effort": "low"}))
        self.assertEqual(qwen({"reasoning_effort": "high"}), default)        # high maps to xhigh
        self.assertEqual(qwen(None, server=True), default)                   # the server's defaults
        off = qwen({"chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("Reasoning effort", off)
        self.assertTrue(off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(qwen({"reasoning_effort": "none"}), off)
        # preserve_reasoning false drops the reasoning of turns before the last user message
        conv["messages"] = conv["messages"][:2] + [{"role": "assistant", "content": "It will rain.",
                                                    "reasoning_content": "The user wants a forecast."},
                                                   {"role": "user", "content": "And tomorrow?"}]
        self.assertIn("The user wants a forecast", qwen({"chat_template_kwargs": {"preserve_reasoning": True}}))
        self.assertIn("The user wants a forecast", qwen(None, server=True))
        self.assertNotIn("The user wants a forecast", qwen({"chat_template_kwargs": {"preserve_reasoning": False}}))
        # Muse reads the effort as its reasoning strength
        muse = render(conv, self.muse, bos_token=MUSE_BOS, extra_context=server_context({"reasoning_effort": "medium"}))
        self.assertIn("Reasoning strength: medium.", muse.text)
        self.assertIn("Reasoning strength: high.", render(conv, self.muse, bos_token=MUSE_BOS).text)

    def test_qwen_trim_and_muse_raw_text_agree_on_segment_bytes(self):
        conv = conversation()
        conv["messages"][3]["content"] = "line one\n\n  indented line two\t tab"
        qwen, muse = self.render_both(conv)
        spans = [{index: rendered.text.encode()[start:end] for index, start, end in rendered.segments}
                 for rendered in (qwen, muse)]
        for index in spans[1]:
            self.assertEqual(spans[0][index], spans[1][index])

    def test_live_edge_whitespace_renders_with_the_same_core_spans(self):
        # Qwen3.8 trims message content (ASCII whitespace only, as llama.cpp does), Muse writes it raw
        conv = conversation("train")
        conv["messages"][1]["content"] = "\u3000Weather in Zürich?\u3000\n"
        conv["messages"][3]["content"] += "\n"
        conv["messages"][4]["content"] = "  " + conv["messages"][4]["content"] + "\n\n"
        qwen = render(conv, self.qwen, live=True)
        muse = render(conv, self.muse, bos_token=MUSE_BOS, live=True)
        self.assertIn("\u3000Weather in Zürich?\u3000<|im_end|>", qwen.text)
        self.assertIn("\u3000Weather in Zürich?\u3000\n", muse.text)
        spans = [{index: rendered.text.encode()[start:end] for index, start, end in rendered.segments}
                 for rendered in (qwen, muse)]
        for index in spans[1].keys() & spans[0].keys():
            self.assertEqual(spans[0][index], spans[1][index])
        texts = segment_text_map(conv)
        self.assertEqual(spans[0][qwen.segment_keys.index("4:content")].decode(), texts["4:content"].strip())

    def test_real_template_duplicates_and_transforms_are_rejected(self):
        doubled = self.muse.replace("{{- render_content(message['content']) -}}{{- '<|eot|>' -}}",
                                    "{{- render_content(message['content']) -}}"
                                    "{{- render_content(message['content']) -}}{{- '<|eot|>' -}}", 1)
        self.assertNotEqual(doubled, self.muse)
        with self.assertRaisesRegex(ValueError, "more than once"):
            render(conversation(), doubled, bos_token=MUSE_BOS)
        upper = self.qwen.replace("{{- content }}\n        {{- '\\n</tool_response>' }}",
                                  "{{- content|upper }}\n        {{- '\\n</tool_response>' }}")
        self.assertNotEqual(upper, self.qwen)
        with self.assertRaisesRegex(ValueError, "transforms content"):
            render(conversation(), upper)


@unittest.skipUnless(os.environ.get("KVPARK_GGUF_METADATA_TESTS") and QWEN_GGUF.is_file()
                     and MUSE_GGUF.is_file() and (TEMPLATE_DIR / "qwen.jinja").is_file(),
                     "set KVPARK_GGUF_METADATA_TESTS=1 to read templates from the real GGUF metadata")
class GgufTemplateTests(unittest.TestCase):
    def test_gguf_metadata_matches_saved_templates(self):
        # Qwen has no add_bos_token key (llama.cpp defaults it off); Muse sets it explicitly.
        for model, name, bos, add_bos in ((QWEN_GGUF, "qwen", QWEN_BOS, None),
                                          (MUSE_GGUF, "muse", MUSE_BOS, True)):
            info = gguf_tokenizer_info(model)
            self.assertEqual(info["chat_template"],
                             (TEMPLATE_DIR / f"{name}.jinja").read_text(encoding="utf-8"))
            self.assertEqual((info["bos_token"], info["add_bos"]), (bos, add_bos))


@unittest.skipUnless((U1_DIR / "qwen3-4b.jinja").is_file() and (U1_DIR / "ornith.jinja").is_file(),
                     "Qwen3-4B/Ornith templates not available (set KVPARK_MODEL_SUPPORT_U1)")
class NewModelTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen3 = (U1_DIR / "qwen3-4b.jinja").read_text(encoding="utf-8")
        cls.ornith = (U1_DIR / "ornith.jinja").read_text(encoding="utf-8")

    def test_realistic_conversation_spans_are_exact_in_both_templates(self):
        conv = conversation()
        texts = segment_text_map(conv)
        for template in (self.qwen3, self.ornith):
            self.assertFalse(template_uses_bos(template))
            rendered = render(conv, template)
            encoded = rendered.text.encode("utf-8")
            for index, start, end in rendered.segments:
                self.assertEqual(encoded[start:end].decode("utf-8"), texts[rendered.segment_keys[index]])
            self.assertEqual(rendered.final_message_segments, [6, 7])
            self.assertEqual(len(rendered.segments), 8)
            self.assertTrue(rendered.text.startswith("<|im_start|>system\n"))
        qwen3, ornith = render(conv, self.qwen3), render(conv, self.ornith)
        # Qwen3-4B: Hermes-style JSON tool calls (tojson arguments), tools listed as JSON lines
        self.assertIn('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Zürich, CH", "days": 3, '
                      '"detail": {"hourly": true}}}\n</tool_call>', qwen3.text)
        self.assertIn('{"type": "function", "function": {"name": "get_weather"', qwen3.text)
        self.assertIn("<tool_response>\n" + texts["3:content"] + "\n</tool_response>", qwen3.text)
        # Ornith: Qwen3.5-style XML parameters, like Qwen3.8
        self.assertIn("<parameter=city>\nZürich, CH\n</parameter>", ornith.text)
        self.assertIn("<parameter=days>\n3\n</parameter>", ornith.text)

    def test_qwen3_json_escaped_string_arguments_are_refused(self):
        # Qwen3-4B serializes arguments with tojson, so a string argument with a quote, backslash
        # or newline is rendered escaped: its bytes differ from the segment text, and chat.render
        # refuses rather than locating a wrong span. Ornith renders string arguments raw.
        conv = conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"]["city"] = 'Zürich "Altstadt"\nCH'
        with self.assertRaisesRegex(ValueError, "transforms content"):
            render(conv, self.qwen3)
        rendered = render(conv, self.ornith)
        spans = {rendered.segment_keys[i]: rendered.text.encode()[a:b].decode() for i, a, b in rendered.segments}
        self.assertEqual(spans["2:arg:0:city"], 'Zürich "Altstadt"\nCH')
        # a running chat (live, the bridge engine) renders it, without a span for that argument
        live = render(conv, self.qwen3, live=True)
        self.assertIn('"city": "Zürich \\"Altstadt\\"\\nCH"', live.text)
        self.assertEqual([live.segment_keys[i] for i in live.escaped], ["2:arg:0:city"])

    def test_qwen3_renders_reasoning_only_after_the_last_user_turn(self):
        conv = conversation("train")
        conv["tools"] = None
        conv["messages"] = [conv["messages"][0], conv["messages"][1],
                            {"role": "assistant", "content": "First answer.", "reasoning_content": "Old thoughts."},
                            {"role": "user", "content": "And tomorrow?"},
                            {"role": "assistant", "content": "Rain.", "reasoning_content": "New thoughts."}]
        rendered = render(conv, self.qwen3)
        keys = [rendered.segment_keys[index] for index, _, _ in rendered.segments]
        self.assertNotIn("2:reasoning", keys)
        self.assertIn("4:reasoning", keys)
        self.assertTrue(rendered.text.endswith("<|im_start|>assistant\n<think>\nNew thoughts.\n</think>\n\n"
                                               "Rain.<|im_end|>\n"))


@unittest.skipUnless(os.environ.get("KVPARK_GGUF_METADATA_TESTS") and QWEN3_GGUF.is_file() and ORNITH_GGUF.is_file()
                     and (U1_DIR / "qwen3-4b.jinja").is_file(),
                     "set KVPARK_GGUF_METADATA_TESTS=1 to read the Qwen3-4B/Ornith templates from their GGUFs")
class NewModelGgufTemplateTests(unittest.TestCase):
    def test_gguf_metadata_matches_saved_templates(self):
        # neither GGUF has a BOS token id; both set add_bos_token false
        for model, name in ((QWEN3_GGUF, "qwen3-4b"), (ORNITH_GGUF, "ornith")):
            info = gguf_tokenizer_info(model)
            self.assertEqual(info["chat_template"], (U1_DIR / f"{name}.jinja").read_text(encoding="utf-8"))
            self.assertEqual((info["bos_token"], info["add_bos"]), ("", False))


@unittest.skipUnless(CHAT_APPLY, "set KVPARK_CHAT_APPLY to compare with llama.cpp's own template application")
class LlamaCppTemplateTests(unittest.TestCase):
    """chat.render against kvpark-chat-apply (common_chat_templates_apply of the pinned runtime)."""

    def test_fixture_template_matches_llama_cpp_and_normalized_tools_are_detected(self):
        import template_check
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "qwen3.gguf"
            create_fixture(model, "qwen3", control_tokens=["<|im_start|>", "<|im_end|>"], chat_template=QWEN_LIKE)
            conv = conversation()
            plain = conversation("train")
            plain["tools"] = None
            plain["messages"] = plain["messages"][:2]
            report = template_check.compare(model, [conv, plain], CHAT_APPLY, PROBE, workdir=directory)
            self.assertTrue(report["identical"], report["differences"])
            self.assertEqual(report["counts"]["text_identical"], 2)
            self.assertEqual(report["counts"]["tokens_checked"], 2 if PROBE else 0)
            # llama.cpp re-serializes tools as {"type", "function": {"name", "description", "parameters"}}:
            # another key order renders differently there, and the comparison reports it
            reordered = copy.deepcopy(conv)
            reordered["tools"] = [{"function": tool["function"], "type": "function"} for tool in conv["tools"]]
            report = template_check.compare(model, [reordered], CHAT_APPLY, None, workdir=directory)
            self.assertFalse(report["identical"])
            self.assertIsNotNone(report["differences"][0]["first_text_difference"])

    @unittest.skipUnless(QWEN3_GGUF.is_file() and ORNITH_GGUF.is_file() and CORPUS_V2.is_file(),
                         "Qwen3-4B, Ornith and corpus v2 are needed")
    def test_real_templates_match_llama_cpp_on_corpus_v2(self):
        import template_check
        from chat import read_conversations
        conversations = list(read_conversations(CORPUS_V2))
        for model in (QWEN3_GGUF, ORNITH_GGUF):
            with self.subTest(model=model.name), tempfile.TemporaryDirectory() as directory:
                report = template_check.compare(model, conversations, CHAT_APPLY, PROBE, workdir=directory)
                self.assertTrue(report["identical"], report["differences"][:3])
                self.assertEqual(report["counts"]["text_identical"], len(conversations))
                self.assertGreater(report["counts"]["with_tool_calls"], 0)


class TokenizerTwinTests(unittest.TestCase):
    def test_twin_carries_the_tokenizer_and_templates_of_the_model(self):
        import template_check
        from tiny_fixture import _gguf_module, create_fixture
        gguf = _gguf_module()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            create_fixture(model, "muse-glimmer", control_tokens=["<|im_start|>", "<|im_end|>"],
                           chat_template=QWEN_LIKE, bos_token="<|begin|>", add_bos=True)
            twin = template_check.tokenizer_twin(model, Path(directory) / "twin.gguf")
            source, copy_ = gguf.GGUFReader(str(model)), gguf.GGUFReader(str(twin))
            fields = {name for name in source.fields if name.startswith("tokenizer.")}
            self.assertEqual(fields, {name for name in copy_.fields if name.startswith("tokenizer.")})
            for name in fields:
                self.assertEqual(copy_.fields[name].contents(), source.fields[name].contents(), name)
            self.assertEqual(copy_.fields["general.architecture"].contents(), "qwen3")
            rows = {t.name: t.data.shape for t in copy_.tensors}
            self.assertEqual(rows["token_embd.weight"][0], len(source.fields["tokenizer.ggml.tokens"].data))
            self.assertEqual(gguf_tokenizer_info(twin), gguf_tokenizer_info(model))
            with self.assertRaises(FileExistsError):
                template_check.tokenizer_twin(model, twin)


@unittest.skipUnless(SERVER, "set KVPARK_LLAMA_SERVER (a llama-server, CPU is enough) to compare with its "
                     "/apply-template")
class LlamaServerParityTests(unittest.TestCase):
    """chat.py against llama-server's own jinja engine: expressions where it differs from jinja2,
    the server comparison of template_check (BATTERY x OPTION_SETS) on tokenizer twins, and the
    real local GGUFs when present."""

    EXPRESSIONS = ("1.0 | string", "1.5", "4 / 2", "0.1 + 0.2", "'  x  ' | trim", "'a,b'.split(',')", "none",
                   "[1, 'a', none]", "{'a': 1}", "range(3) | list", "{'a': 1}.items() | list", "none | default('d')",
                   "'' | default('d', true)", "'%s-%d' | format('a', 1)", "{'a': [1, 2.0, 0.5]} | tojson",
                   "{'a': 1} | tojson(indent=4)", "'é🚀' | tojson(ensure_ascii=true)",
                   "[1, 2] | tojson(separators=(',', ':'))",
                   "messages | map(attribute='role') | join(',')", "'a\\nb' | indent(2, true)", "true | string",
                   "messages[0] | tojson", "3 is number", "'abc'[::-1]", "'a' ~ 1 ~ 1.0",
                   "'\u00a0a\u3000 ' | trim", "' \u00a0a\u3000\\n'.strip()", "'ééa'.lstrip('é')", "'é ölçü' | upper",
                   "'hello-world  foo' | title", "'aBC dEF' | capitalize", "'a b  c'.split() | length",
                   "'{} items, {}' | format(3, none)", "'a' ~ none ~ 0.0000001 ~ [1, 'b'] ~ {'k': 0.5}",
                   "'a' + none", "1 + 'a'", "[0.5, 0.0000001, 2] | join(',')", "{'d': 'x\\ny', 'q': \"it's\"} | string", "'é' | length",
                   "[1, 2] | reverse", "x.y.z is defined", "messages | select('role') | list | length",
                   "'<r>' ~ ('<a>' | safe) ~ '</r>'", "1.0 | safe")

    def test_expressions_render_as_llama_cpp_renders_them(self):
        import fixtures_media
        import template_check
        from tiny_fixture import create_fixture
        template = "{%- set e = messages[0].content -%}" + "".join(
            ("{%- if" if k == 0 else "{%- elif") + f" e == '{k}' -%}}{{{{ {expr} }}}}"
            for k, expr in enumerate(self.EXPRESSIONS)) + (
            "{%- elif e == 'ns' -%}{%- set ns = namespace(a=1) %}{%- for m in messages %}{%- set ns.a = ns.a + 1 %}"
            "{%- endfor %}{{- ns.a }}{%- elif e == 'ws' -%}  a  {%- if true %}\n  b\n{% endif %}  c\n"
            "{%- for x in [1,2] %}\n  {{ x }}\n{%- endfor %}\n{%- endif -%}")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "probe.gguf"
            create_fixture(model, "qwen3", chat_template=template)
            slots = Path(directory) / "slots"
            slots.mkdir()
            server = fixtures_media.Server(SERVER, model, slots, Path(directory) / "server.log",
                                           fixtures_media.free_port(template_check.TWIN_PORTS), extra=["--jinja"])
            try:
                for key in [str(k) for k in range(len(self.EXPRESSIONS))] + ["ns", "ws"]:
                    conv = {"messages": [{"role": "user", "content": key}, {"role": "assistant", "content": "A"},
                                         {"role": "user", "content": "B"}], "tools": None}
                    with self.subTest(expression=self.EXPRESSIONS[int(key)] if key.isdigit() else key):
                        mine = compile_template(template).render(messages=conv["messages"], tools=None, bos_token="",
                                                                 add_generation_prompt=True,
                                                                 **server_context(server=True))
                        self.assertEqual(mine, template_check.served_prompt(server.url, conv))
            finally:
                server.close()

    def test_battery_on_a_fixture_twin_and_mismatches_are_reported(self):
        import template_check
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            create_fixture(model, "qwen3", control_tokens=["<|im_start|>", "<|im_end|>"], chat_template=QWEN_LIKE)
            report = template_check.check_server(model, server_binary=SERVER, workdir=directory)
            self.assertTrue(report["parity"], [c for c in report["cases"] if not c.get("context")][:2])
            self.assertEqual(report["counts"]["cases"], len(template_check.BATTERY) * len(template_check.OPTION_SETS))
            self.assertEqual((report["add_bos"], report["props_template_is_gguf"], report["tool_use_template"]),
                             (False, True, False))
            # a template whose rendering differs from llama.cpp's (dictsort prints nothing there)
            odd = Path(directory) / "odd.gguf"
            create_fixture(odd, "qwen3", chat_template="{{ {'b': 1, 'a': 2} | dictsort }}" + QWEN_LIKE)
            report = template_check.check_server(odd, server_binary=SERVER, workdir=directory)
            self.assertFalse(report["parity"])
            self.assertEqual(report["counts"]["mismatch"], report["counts"]["cases"])
            self.assertEqual(report["cases"][0]["first_difference"], 0)

    def test_whole_conversations_compare_without_a_generation_prompt(self):
        """check_server(add_generation_prompt=False): conversations that end with the assistant's
        answer, as corpus records render them; the server neither adds a generation prompt nor
        continues the final message (--no-prefill-assistant)."""
        import template_check
        from tiny_fixture import create_fixture
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            create_fixture(model, "qwen3", control_tokens=["<|im_start|>", "<|im_end|>"], chat_template=QWEN_LIKE)
            whole = [{**c, "messages": c["messages"] + [{"role": "assistant", "content": "That is all."}]}
                     for c in template_check.BATTERY if c["messages"][-1]["role"] == "user"]
            report = template_check.check_server(model, server_binary=SERVER, workdir=directory, conversations=whole,
                                                 option_sets=[{}], add_generation_prompt=False)
            self.assertTrue(report["parity"], [c for c in report["cases"] if not c.get("context")][:2])
            self.assertEqual(report["counts"]["cases"], len(whole))
            self.assertGreater(len(whole), 3)
            # with a generation prompt the same conversations are no request shape the twin renders alike
            prompts = template_check.check_server(model, server_binary=SERVER, workdir=directory,
                                                  conversations=whole[:1], option_sets=[{}])
            self.assertFalse(prompts["parity"])

    def test_real_templates_have_parity(self):
        import template_check
        models = [m for m in (QWEN_GGUF, MUSE_GGUF, QWEN3_GGUF, ORNITH_GGUF) if m.is_file()]
        if not models:
            self.skipTest("no local GGUF")
        for model in models:
            with self.subTest(model=model.name), tempfile.TemporaryDirectory() as directory:
                report = template_check.check_server(model, server_binary=SERVER, workdir=directory)
                self.assertTrue(report["parity"], [c for c in report["cases"] if not c.get("context")][:2])


if __name__ == "__main__":
    unittest.main()
