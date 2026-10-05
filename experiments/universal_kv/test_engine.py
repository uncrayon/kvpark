"""engine.run on tiny GGUF fixtures (a dense qwen3 source, a Muse-like target, a random hub): the
target rendered with a request's template options as llama-server renders it (chat.server_contexts,
TemplateMismatch and its exit status), a source slot that holds media (read up to its first
medium), the features the engine advertises, and specs with granitehybrid's RoPE frequency
factors and the pin a server's build_info names. The real templates and servers are checked by the
live runs (live_bridge.py options).

The source record (LocatedRecordTest): segments located in order by the occurrences covering the
most bytes, each spanned inside the whitespace at its edges as chat.render spans it live; a
conversation whose messages keep edge whitespace switches into a template that trims it
(EdgeWhitespaceTest).

Vocabularies (TokenizerTest): GPT-2 byte-level BPE, Gemma 4's BPE and SentencePiece pieces
decoded as llama.cpp's token_to_piece decodes them, and SentencePiece's space prefix dropped where
llama.cpp's tokenizer adds it. With KVPARK_VOCAB_TOOL=/path/to/kvpark-vocab (llama.cpp's own
vocabulary, vocab_only: experiments/linear_transfer/vocab_tool.cpp, built against either pinned
runtime by build_collector.py --target kvpark-vocab) every local GGUF of LOCAL_MODELS is compared
whole, and real tokenizations of template_check.BATTERY renderings are spelled back to the rendered
text.

With KVPARK_LLAMA_SERVER=/path/to/runtime/build/bin/llama-server (ports 18574-18578, or from
KVPARK_TEMPLATE_PORT) a throwaway CPU llama-server checks chat.server_contexts against its own
/apply-template on tiny models carrying the real templates, and makes real slot saves holding an
image for the engine (a random tiny projector, NORMAL positions; KVPARK_QWEN_MMPROJ=/path/to/the
Qwen3.8 mmproj adds the tiny Qwen3.5 hybrid with M-RoPE positions, as test_media_state.py)."""

import datetime
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import backends
import engine
import hub
from test_translate import StubSpans, StubView, attention_widths, random_hub

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))
VOCAB_TOOL = os.environ.get("KVPARK_VOCAB_TOOL")
_WINDOWS = "/run/media/uncrayon/Windows/llm/models/benchmark"
# one GGUF per local tokenizer (evidence/models/catalog.json); split files: the first part
LOCAL_MODELS = [Path(p) for p in os.environ.get("KVPARK_VOCAB_MODELS", ":".join([
    "/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf", "/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf",
    "/srv/models/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf",
    "/srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf",
    "/srv/models/gte-qwen2-1.5b-instruct/gte-Qwen2-1.5B-instruct-Q8_0.gguf",
    "/srv/models/deepseek-v2-lite-chat-q8/DeepSeek-V2-Lite-Chat-Q8_0.gguf",
    "/srv/models/glm-4.7-flash-q5km/GLM-4.7-Flash-Q5_K_M.gguf",
    "/srv/models/kimi-linear-48b-a3b-instruct-q4km/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf",
    "/srv/models/ling-3.0-tiny-q8/Ling-3.0-tiny-Q8_0.gguf", "/srv/models/lfm2-2.6b-q8/LFM2-2.6B-Q8_0.gguf",
    "/srv/models/granite-4.0-h-tiny-q8/granite-4.0-h-tiny-Q8_0.gguf",
    "/srv/models/falcon-h1-3b-instruct-q8/Falcon-H1-3B-Instruct-Q8_0.gguf",
    "/srv/models/jamba-reasoning-3b-q8/ai21labs_AI21-Jamba-Reasoning-3B-Q8_0.gguf",
    f"{_WINDOWS}/deepseek-v4-flash-0731/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
    f"{_WINDOWS}/gemma4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    f"{_WINDOWS}/gemma4-31b/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
    f"{_WINDOWS}/laguna-s-2.1-q6k/Laguna-S-2.1-UD-Q6_K-00001-of-00003.gguf",
    f"{_WINDOWS}/nemotron-3.5-lightning-30b-a3b-q8/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
    f"{_WINDOWS}/qwen3.6-35b-a3b-q8/Qwen3.6-35B-A3B-Q8_0.gguf"])).split(":") if p]
import chat  # noqa: E402
import jinja2  # noqa: E402
import kvstate  # noqa: E402
import tiny_fixture  # noqa: E402
from test_kvstate import gqa_metadata, hybrid_metadata  # noqa: E402
from test_media_state import chunk_blob, server_words  # noqa: E402

# a template that reads the request's reasoning effort and thinking switch, as Qwen3.8's does
# (its system prompt carries the effort) and Muse's (reasoning_strength)
OPTIONS_TEMPLATE = ("{% set effort = reasoning_effort if reasoning_effort is defined else 'high' %}"
                    "{% if enable_thinking is defined and not enable_thinking %}{% set effort = 'none' %}{% endif %}"
                    "<|effort|>{{ effort }}<|end|>\n"
                    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>\n{% endfor %}"
                    "{% if add_generation_prompt %}<|assistant|>{% endif %}")
# one that needs what only the server context gives: the EOS text after each answer
EOS_TEMPLATE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}"
                "{% if m.role == 'assistant' %}{{ eos_token }}{% endif %}<|end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}")
# the same in expressions, as Mistral and Llama 2 style templates write it: without the server
# context `content + eos_token` renders without the EOS (llama.cpp's jinja takes an undefined
# operand next to a string as empty), and arithmetic on eos_token fails to render
EOS_EXPRESSION_TEMPLATE = ("<|year|>{{ strftime_now('%Y') }}<|end|>\n{% for m in messages %}<|{{ m.role }}|>"
                           "{% if m.role == 'assistant' %}{{ m.content + eos_token }}{% else %}{{ m.content }}"
                           "{% endif %}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")
EOS_FAILING_TEMPLATE = EOS_EXPRESSION_TEMPLATE.replace("m.content + eos_token", "m.content + eos_token * 1")
# one that trims message content, as Qwen3.8's and Gemma 4's do (ASCII whitespace, as llama.cpp)
TRIM_TEMPLATE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content | trim }}<|end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")
MESSAGES = [{"role": "system", "content": "You answer briefly and stay on topic."},
            {"role": "user", "content": "Tell me about the tides along the northern coast this week."},
            {"role": "assistant", "content": "The tides run high at dawn and low near noon, with a strong ebb."},
            {"role": "user", "content": "And what about next week, when the moon is full again?"}]
TAIL = 8


def render(messages, template, generation=True, **context):
    """The rendering a server with these template settings sends (chat.server_context)."""
    return chat.render({"messages": messages, "tools": None}, template, add_generation_prompt=generation, live=True,
                       extra_context=chat.server_context(**context) if context else None).text


class Fixture:
    """GGUFs with `template`, a hub bundle, source slot files and a job builder under one directory.

    The source is a tiny qwen3 (or `source`, with tiny_fixture.create_fixture `source_options`;
    only a qwen3 source gets the default text-only slot of the history), the target a tiny Muse;
    `context` (render's keywords) is the server context the source's history was rendered with."""

    def __init__(self, template, *, source="qwen3", source_options=None, context=None):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp" if os.path.isdir("/tmp") else None)
        self.root = Path(self.tmp.name)
        self.template = template
        self.gguf = {"src": self.root / "src.gguf", "tgt": self.root / "tgt.gguf"}
        tiny_fixture.create_fixture(self.gguf["src"], source, chat_template=template, **(source_options or {}))
        tiny_fixture.create_fixture(self.gguf["tgt"], "muse-glimmer", chat_template=template, seed=5)
        self.spec = {name: kvstate.ModelSpec.from_gguf(path) for name, path in self.gguf.items()}
        self.rng = np.random.default_rng(0)
        spec = self.spec["tgt"]
        layers = spec.layers_of("full", "sliding")
        k = {l: self.rng.standard_normal((2,) + spec.head_shape(l, "k")).astype(np.float32) for l in layers}
        v = {l: self.rng.standard_normal((2,) + spec.head_shape(l, "v")).astype(np.float32) for l in layers}
        self.template_state = self.root / "template.bin"
        kvstate.write_new(self.template_state, kvstate.build_muse_state(spec, [256, 72], np.arange(2), k, v,
                                                                        server_tokens=True).to_bytes())
        hubs = random_hub({"src": attention_widths(self.spec["src"]), "tgt": attention_widths(self.spec["tgt"])})
        views = [StubView(name, hubs, self.spec[name].architecture) for name in ("src", "tgt")]
        self.hub = self.root / "hub"
        hub.save_hub(self.hub, hubs, views, StubSpans(), {"note": "test"})
        self.source = None
        if source == "qwen3":
            self.source = self.source_state(render(MESSAGES[:-1], template, generation=False,
                                                   **(context or {})).encode())
        self.outputs = 0

    def rows(self, n):
        spec = self.spec["src"]
        layers = spec.layers_of("full")
        return ({l: self.rng.standard_normal((n,) + spec.head_shape(l, "k")).astype(np.float32) for l in layers},
                {l: self.rng.standard_normal((n,) + spec.head_shape(l, "v")).astype(np.float32) for l in layers})

    def source_state(self, text, *, rows=None, name=None):
        """A text-only source slot of these bytes (one token per byte)."""
        tokens = np.frombuffer(text, dtype=np.uint8).astype(np.int32)
        k, v = rows or self.rows(len(tokens))
        path = self.root / (name or f"source-{len(tokens)}-{self.rng.integers(1 << 30)}.bin")
        kvstate.write_new(path, kvstate.build_dense_state(self.spec["src"], tokens, np.arange(len(tokens)), k, v,
                                                          server_tokens=True).to_bytes())
        return path

    def job(self, text, *, messages=MESSAGES, options=None, source=None, **target):
        self.outputs += 1
        conversation = {"messages": messages, "tools": None}
        if options is not None:
            conversation["template_options"] = options
        return {"hub": str(self.hub), "tail": TAIL, "fill": "mean", "device": "cpu",
                "cache_dir": str(self.root / "profiles"), "conversation": conversation,
                "source": {"name": "src", "gguf": str(self.gguf["src"]), "state": str(source or self.source)},
                "target": {"name": "tgt", "gguf": str(self.gguf["tgt"]), "template_state": str(self.template_state),
                           "output": str(self.root / f"out-{self.outputs}.bin"),
                           "tokens": [[ord(c), c] for c in text], **target}}

    def run(self, job):
        return engine.run(job, backend=backends.select("cpu"))

    def close(self):
        self.tmp.cleanup()


class TemplateOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture(OPTIONS_TEMPLATE)

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_the_target_is_rendered_with_the_request_options(self):
        fx = self.fx
        for options, effort in (({"reasoning_effort": "low"}, "low"),
                                ({"chat_template_kwargs": {"enable_thinking": False}}, "none"),
                                ({"reasoning_effort": "none"}, "none")):
            with self.subTest(options=options):
                text = render(MESSAGES, OPTIONS_TEMPLATE, options=options)
                self.assertTrue(text.startswith(f"<|effort|>{effort}<|end|>"))
                job = fx.job(text, options=options)
                report = fx.run(job)
                self.assertEqual((report["rendering"], report["template_options"]), ("template", sorted(options)))
                last = text.index("<|user|>And what")
                self.assertEqual(report["t0"], min(len(text) - TAIL, last + len("<|user|>")))
                self.assertEqual(report["features"], list(engine.FEATURES))
                state = kvstate.read_file(job["target"]["output"], fx.spec["tgt"])
                self.assertEqual(bytes(state.tokens.astype(np.uint8)), text[:report["t0"]].encode())

    def test_options_the_server_did_not_render_with_are_a_template_mismatch(self):
        fx = self.fx
        default = render(MESSAGES, OPTIONS_TEMPLATE)
        report = fx.run(fx.job(default))                                        # no options: as before
        # the source slot holds the history as its own template renders it: every segment is known
        self.assertEqual((report["rendering"], report["source_segments_rendered"], report["source_segments_found"]),
                         ("template", 3, 3))
        with self.assertRaisesRegex(engine.TemplateMismatch, "with the request's template options"):
            fx.run(fx.job(default, options={"reasoning_effort": "low"}))
        # options the server refuses or that cannot be mirrored are a mismatch too (the bridge falls back)
        with self.assertRaisesRegex(engine.TemplateMismatch, "expected boolean"):
            fx.run(fx.job(default, options={"chat_template_kwargs": {"enable_thinking": "false"}}))
        with self.assertRaisesRegex(engine.TemplateMismatch, "cannot be rendered"):
            fx.run(fx.job(default, options={"chat_template_kwargs": ["x"]}))
        self.assertTrue(issubclass(engine.TemplateMismatch, ValueError))
        with self.assertRaisesRegex(ValueError, "template_options must be an object"):
            fx.run(fx.job(default, options=["x"]))

    def test_the_target_servers_own_template_flags(self):
        # a target server run with --reasoning-effort low: no candidate renders "low" unless the
        # job carries the server's template kwargs (kvpark bridge add --reasoning-effort low)
        fx = self.fx
        flags = {"reasoning_effort": "low", "preserve_reasoning": True}
        text = render(MESSAGES, OPTIONS_TEMPLATE, server=True, server_kwargs=flags)
        self.assertTrue(text.startswith("<|effort|>low<|end|>"))
        with self.assertRaises(engine.TemplateMismatch):
            fx.run(fx.job(text))
        report = fx.run(fx.job(text, template_kwargs=flags))
        self.assertEqual((report["rendering"], report["template_kwargs"]), ("server", sorted(flags)))
        # the request's own options win over the server's
        options = {"reasoning_effort": "high"}
        text = render(MESSAGES, OPTIONS_TEMPLATE, options=options, server=True, server_kwargs=flags)
        self.assertTrue(text.startswith("<|effort|>high<|end|>"))
        self.assertEqual(fx.run(fx.job(text, options=options, template_kwargs=flags))["rendering"], "template")
        with self.assertRaisesRegex(ValueError, "template_kwargs must be an object"):
            fx.run(fx.job(text, template_kwargs=["low"]))

    def test_one_shot_exit_status_for_a_template_mismatch(self):
        fx = self.fx
        path = fx.root / "mismatch.json"
        path.write_text(json.dumps(fx.job(render(MESSAGES, OPTIONS_TEMPLATE), options={"reasoning_effort": "low"})))
        result = subprocess.run([sys.executable, str(HERE / "engine.py"), str(path)], capture_output=True, text=True,
                                timeout=300, env={**os.environ, backends.ENV: "cpu"})
        self.assertEqual(result.returncode, engine.TEMPLATE_MISMATCH, result.stderr)
        self.assertTrue(result.stderr.strip().splitlines()[-1].startswith("TemplateMismatch: "))


class ServerContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture(EOS_TEMPLATE)

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_a_template_reading_server_variables_matches_the_server_context(self):
        fx = self.fx
        # llama-server passes eos_token (here "<eos>", the fixture's EOS) and the date; the
        # template's defaults leave eos_token empty, so only the server context spells these tokens
        profile = engine.profile(fx.gguf["tgt"])
        self.assertEqual((profile["eos_token"], profile["bos_token"]), ("<eos>", "<bos>"))
        text = render(MESSAGES, EOS_TEMPLATE, server=True, eos_token="<eos>", now=datetime.datetime(2026, 10, 3))
        self.assertIn("ebb.<eos><|end|>", text)
        report = fx.run(fx.job(text))
        self.assertEqual(report["rendering"], "server")
        plain = render(MESSAGES, EOS_TEMPLATE)
        self.assertEqual(fx.run(fx.job(plain))["rendering"], "template")
        with self.assertRaisesRegex(engine.TemplateMismatch, "do not spell the target rendering"):
            fx.run(fx.job(plain.replace("tides", "waves")))

    def test_a_template_that_fails_without_server_variables_tries_the_server_context(self):
        # `content + eos_token` renders without the EOS in the template context, which the server's
        # tokens do not spell; `eos_token * 1` raises jinja2's UndefinedError (not a ValueError)
        # there: the server context must still be tried
        server = {"server": True, "eos_token": "<eos>"}
        for template in (EOS_EXPRESSION_TEMPLATE, EOS_FAILING_TEMPLATE):
            fx = Fixture(template, context=server)
            try:
                if template == EOS_FAILING_TEMPLATE:
                    with self.assertRaises(jinja2.UndefinedError):
                        render(MESSAGES, template)
                else:
                    self.assertIn("ebb.<|end|>", render(MESSAGES, template))
                text = render(MESSAGES, template, **server)
                self.assertIn("ebb.<eos><|end|>", text)
                self.assertTrue(text.startswith(f"<|year|>{datetime.datetime.now():%Y}<|end|>"))
                report = fx.run(fx.job(text))
                self.assertEqual(report["rendering"], "server")
                state = kvstate.read_file(fx.root / f"out-{fx.outputs}.bin", fx.spec["tgt"])
                self.assertEqual(bytes(state.tokens.astype(np.uint8)), text[:report["t0"]].encode())
                # tokens no context spells: a mismatch naming the first failure, never a job error
                failure = "template context failed: UndefinedError" if template == EOS_FAILING_TEMPLATE else "bytes"
                with self.assertRaisesRegex(engine.TemplateMismatch, failure):
                    fx.run(fx.job(text.replace("tides", "waves")))
            finally:
                fx.close()


class LocatedRecordTest(unittest.TestCase):
    """engine.located_record: the source's segments in its own text, as chat.render spans them."""

    CONVERSATION = {"tools": None, "messages": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "weather", "content": "Rain, 14 C"},
        {"role": "assistant", "content": "It is raining in Paris, 14 C."}]}

    def located(self, text, conversation=None):
        record = engine.located_record(conversation or self.CONVERSATION, "doc", text.encode())
        return {record["segment_keys"][n]: text.encode()[a:b].decode() for n, a, b in record["segments"]}

    def test_a_segment_rendered_elsewhere_costs_only_itself(self):
        # the template writes no tool calls: the argument "Paris" is also in the last answer, and
        # taking it there would push out the tool answer and the last answer
        text = ("<user>What is the weather in Paris?</user><assistant>Let me check.</assistant>"
                "<tool>Rain, 14 C</tool><assistant>It is raining in Paris, 14 C.</assistant>")
        self.assertEqual(self.located(text), {"0:content": "What is the weather in Paris?",
                                              "1:content": "Let me check.", "2:content": "Rain, 14 C",
                                              "3:content": "It is raining in Paris, 14 C."})
        # content written after its tool calls (Gemma 4): the shorter one is left out
        text = ("<user>What is the weather in Paris?</user><call>weather(city=Paris)</call>"
                "<tool>Rain, 14 C</tool><assistant>Let me check.It is raining in Paris, 14 C.</assistant>")
        self.assertEqual(self.located(text), {"0:content": "What is the weather in Paris?",
                                              "1:arg:0:city": "Paris", "2:content": "Rain, 14 C",
                                              "3:content": "It is raining in Paris, 14 C."})
        # the earliest occurrences on ties, and nothing found in an unrelated text
        record = engine.located_record({"tools": None, "messages": [{"role": "user", "content": "ok"},
                                                                    {"role": "assistant", "content": "ok"}]},
                                       "doc", b"ok ok ok")
        self.assertEqual(record["segments"], [[0, 0, 2], [1, 3, 5]])
        self.assertEqual(engine.located_record(self.CONVERSATION, "doc", b"nothing here")["segments"], [])

    def test_the_source_templates_own_spans_come_first(self):
        # the tool answer "staging" is also in the call's arguments (template text there): searched
        # for, it is taken in the call; where the source's own template spells the slot's text,
        # its spans are kept, and only what follows them is searched
        template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% for c in m.tool_calls or [] %}"
                    "<call>{{ c.function.arguments | tojson }}</call>{% endfor %}<|end|>\n{% endfor %}"
                    "{% if add_generation_prompt %}<|assistant|>{% endif %}")
        conversation = {"tools": None, "messages": [
            {"role": "user", "content": "Deploy it."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "clarify", "arguments": {"choices": ["staging", "production"]}}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "clarify", "content": "staging"},
            {"role": "assistant", "content": "Deploying to staging now."},
            {"role": "user", "content": "Thanks"}]}
        # the slot: the previous request with its generation prompt, and the reply the source generated
        slot = (chat.render({"tools": None, "messages": conversation["messages"][:3]}, template,
                            add_generation_prompt=True, live=True).text + "Deploying to staging now.").encode()
        tool = slot.index(b"<|tool|>staging") + len(b"<|tool|>")
        searched = engine.located_record(conversation, "doc", slot)["segments"]
        self.assertLess(searched[1][1], tool)                   # "staging" taken inside the call
        profile = {"chat_template": template, "bos_token": "", "eos_token": ""}
        with tempfile.TemporaryDirectory() as directory:
            gguf = vocabulary_gguf(Path(directory) / "v.gguf", "gpt2", [("<s>", C), ("a", N)])
            pieces = [bytes([b]) for b in slot]
            known = engine.source_spans(conversation, profile, None, [1] * len(pieces), pieces, gguf)
            self.assertEqual(known, [[0, 8, 18], [1, tool, tool + 7], [2, len(slot) - 25, len(slot)]])
            record = engine.located_record(conversation, "doc", slot, known=known)
            self.assertEqual(record["segments"], known)         # "Thanks" is not in the slot
            # a slot that starts with a control token (BOS) the rendering lacks: the spans follow it
            shifted = engine.source_spans(conversation, profile, None, [0] + [1] * len(pieces), [b"<s>"] + pieces,
                                          gguf)
            self.assertEqual(shifted, [[n, a + 3, b + 3] for n, a, b in known])
            # no candidate renders (the template needs what the context lacks): nothing is known
            self.assertEqual(engine.source_spans(conversation, {**profile, "chat_template": "{{ x.y() }}"}, None,
                                                 [1] * len(pieces), pieces, gguf), [])

    def test_edge_whitespace_and_ambiguous_segments(self):
        conversation = {"tools": None, "messages": [{"role": "user", "content": "\n  Hello there.\u00a0\n"},
                                                    {"role": "assistant", "content": "\n"},
                                                    {"role": "user", "content": "x"}]}
        # the core, as chat.render spans it live, whether the template trimmed it or not
        for template in (TRIM_TEMPLATE, OPTIONS_TEMPLATE):
            text = render(conversation["messages"], template)
            rendered = chat.render(conversation, template, add_generation_prompt=True, live=True)
            record = engine.located_record(conversation, "doc", text.encode())
            self.assertEqual(record["segments"], [list(span) for span in rendered.segments])
            self.assertEqual(self.located(text, conversation)["0:content"], "Hello there.\u00a0")
        # a text found more than LOCATE_OCCURRENCES times is too ambiguous to place
        many = "x" * (engine.LOCATE_OCCURRENCES + 1)
        self.assertNotIn("2:content", self.located("Hello there.\u00a0" + many, conversation))
        self.assertIn("2:content", self.located("Hello there.\u00a0" + many[1:], conversation))


class EdgeWhitespaceTest(unittest.TestCase):
    def test_a_conversation_with_edge_whitespace_switches_into_a_trimming_template(self):
        # Hermes keeps a tool answer's final newline: chat.render spans the core, which both the
        # target (trimmed) and the source (written raw) records align on
        messages = [dict(m) for m in MESSAGES]
        messages[1]["content"] = "\n" + messages[1]["content"] + "\u00a0\n"
        messages[2]["content"] += "\n\n"
        fx = Fixture(TRIM_TEMPLATE)
        try:
            source = fx.source_state(render(messages[:-1], OPTIONS_TEMPLATE, generation=False).encode())
            text = render(messages, TRIM_TEMPLATE)
            self.assertIn("\u00a0<|end|>", text)
            report = fx.run(fx.job(text, messages=messages, source=source))
            self.assertEqual((report["rendering"], report["source_segments_found"], report["segments"]),
                             ("template", 3, 4))
            state = kvstate.read_file(fx.root / f"out-{fx.outputs}.bin", fx.spec["tgt"])
            self.assertEqual(state.tokens.tolist(), [ord(c) for c in text[:report["t0"]]])    # a token per character
        finally:
            fx.close()


class MediaSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture(OPTIONS_TEMPLATE)

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_a_source_slot_with_media_is_read_up_to_its_first_medium(self):
        fx = self.fx
        # the conversation as the target renders it (the image as the target's media marker, in
        # message 1) and the source's slot of its history: the same text with the image's cells
        # (NORMAL positions, one per cell) where the source's server put them
        messages = [MESSAGES[0], {"role": "user", "content": "Look at <__media__> and tell me about the tides."},
                    *MESSAGES[2:]]
        history = render(messages[:-1], OPTIONS_TEMPLATE, generation=False)
        before, after = history.split("<__media__>")
        head, tail = before.encode(), after.encode()
        cells = 6
        tokens = (list(np.frombuffer(head, dtype=np.uint8)) + [-1] * cells + list(np.frombuffer(tail, dtype=np.uint8)))
        n = len(tokens)
        k, v = fx.rows(n)
        words = server_words(tokens, [(len(head), chunk_blob(nx=cells, ny=1, pos_type=0))])
        state = kvstate.build_dense_state(fx.spec["src"], np.maximum(tokens, 0), np.arange(n), k, v, server_tokens=True)
        with_media = fx.root / "source-media.bin"
        kvstate.write_new(with_media, kvstate.StateFile(words, state.state).to_bytes())
        self.assertEqual(len(kvstate.read_file(with_media, fx.spec["src"]).media.media), 1)
        # the same rows up to the image as a text-only slot: what the engine must read
        cut = len(head)
        text_only = fx.source_state(head, rows=({l: a[:cut] for l, a in k.items()}, {l: a[:cut] for l, a in v.items()}))
        target = render(messages, OPTIONS_TEMPLATE)
        job = fx.job(target, messages=messages, source=with_media, max_t0_message=1)
        report = fx.run(job)
        self.assertEqual(report["source_media"], {"chunks": 1, "entries": n, "read": cut,
                                                  "t0_cap": target.index("<|end|>\n<|user|>Look")})
        self.assertEqual(report["source_tokens"], cut)
        reference = fx.job(target, messages=messages, source=text_only, max_t0_message=1)
        reference_report = fx.run(reference)
        self.assertNotIn("source_media", reference_report)
        self.assertEqual(report["t0"], reference_report["t0"])
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(reference["target"]["output"]).read_bytes())
        # T0 stops before message 1: its role header and everything after are prefilled natively
        self.assertEqual(report["t0"], target.index("<|end|>\n<|user|>Look"))
        # a request that no longer holds the image (a client replaced it by text) sets no cap: T0
        # still stops where the source's text does, after the last segment found in it
        plain = [MESSAGES[0], {"role": "user", "content": "Look at [a beach photo] and tell me about the tides."},
                 *MESSAGES[2:]]
        replaced = render(plain, OPTIONS_TEMPLATE)
        report = fx.run(fx.job(replaced, messages=plain, source=with_media))
        self.assertEqual((report["source_segments_found"], report["source_media"]["read"]), (1, cut))
        self.assertEqual(report["t0"], replaced.index("<|end|>\n<|user|>Look"))
        self.assertEqual(report["source_media"]["t0_cap"], report["t0"])
        self.assertLess(report["t0"], len(replaced) - TAIL)
        # a slot that starts with media holds no text to translate
        words = server_words([-1] * cells + tokens[cut + cells:], [(0, chunk_blob(nx=cells, ny=1, pos_type=0))])
        m = n - cut
        state = kvstate.build_dense_state(fx.spec["src"], np.zeros(m, dtype=np.int32), np.arange(m),
                                          {l: a[cut:] for l, a in k.items()}, {l: a[cut:] for l, a in v.items()},
                                          server_tokens=True)
        starts = fx.root / "source-starts-with-media.bin"
        kvstate.write_new(starts, kvstate.StateFile(words, state.state).to_bytes())
        with self.assertRaisesRegex(ValueError, "starts with media"):
            fx.run(fx.job(target, messages=messages, source=starts, max_t0_message=1))


SERVER = os.environ.get("KVPARK_LLAMA_SERVER")
QWEN_MMPROJ = os.environ.get("KVPARK_QWEN_MMPROJ")
FIRST_PORT = int(os.environ.get("KVPARK_TEMPLATE_PORT", "18574"))
PORTS = range(FIRST_PORT, FIRST_PORT + 5)


@unittest.skipUnless(SERVER, "set KVPARK_LLAMA_SERVER (runtime/build/bin/llama-server) to compare with llama-server's "
                     "/apply-template on tiny models carrying the real chat templates")
class LlamaServerTemplateTest(unittest.TestCase):
    """chat.server_contexts against llama-server's own rendering of requests with template options:
    a tiny model carrying each real template (Qwen3.8, Muse, Qwen3-4B), the server with its defaults
    and with the production Qwen3.8 flags (--reasoning-effort xhigh --reasoning-preserve), which the
    bridge does not need to know; and servers run with template flags that change the rendering,
    which it is told (kvpark bridge add --reasoning-effort ..., messages.server_template_kwargs)."""

    # (flags, server_template_kwargs keywords: what kvpark bridge add is given for them, None: nothing)
    FLAGS = {"qwen38": (((), None),
                        (("--reasoning-effort", "xhigh", "--reasoning-preserve"), None),
                        (("--reasoning-effort", "low"), {"reasoning_effort": "low"}),
                        (("--no-reasoning-preserve",), {"reasoning_preserve": False}),
                        (("--reasoning", "off"), {"reasoning": "off"}),
                        (("--chat-template-kwargs", '{"enable_thinking": false}'),
                         {"chat_template_kwargs": '{"enable_thinking": false}'})),
             "muse": (((), None), (("--reasoning-effort", "xhigh"), {"reasoning_effort": "xhigh"})),
             "qwen3-4b": (((), None),)}

    OPTIONS = (None, {"reasoning_effort": "medium"}, {"reasoning_effort": "low"}, {"reasoning_effort": "high"},
               {"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}},
               {"reasoning_effort": "medium", "chat_template_kwargs": {"preserve_reasoning": True}},
               {"chat_template_kwargs": {"preserve_reasoning": False}})

    def conversations(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                                   "parameters": {"type": "object",
                                                                  "properties": {"city": {"type": "string"}}}}}]
        chat_ = [{"role": "system", "content": "You are a careful assistant."},
                 {"role": "user", "content": "What is the tide like?"},
                 {"role": "assistant", "content": "High at dawn.", "reasoning_content": "Tides follow the moon."},
                 {"role": "user", "content": "And tomorrow?"}]
        agent = [{"role": "user", "content": "Weather in Paris?"},
                 {"role": "assistant", "content": "", "reasoning_content": "Use the tool.",
                  "tool_calls": [{"id": "c1", "type": "function",
                                  "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                 {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "rain"},
                 {"role": "user", "content": "Thanks, and in Rome?"}]
        return [{"messages": chat_, "tools": None}, {"messages": agent, "tools": tools}]

    def test_request_options_render_as_llama_server_renders_them(self):
        import fixtures_media
        sys.path.insert(0, str(HERE.parents[1] / "src"))
        from kvpark.bridge import messages
        templates = {"qwen38": "linear-transfer/chat-templates/qwen.jinja",
                     "muse": "linear-transfer/chat-templates/muse.jinja",
                     "qwen3-4b": "linear-transfer/model-support-u1/qwen3-4b.jinja"}
        evidence = HERE.parents[1] / "evidence"
        controls = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
        found = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, relative in templates.items():
                if not (evidence / relative).is_file():
                    self.skipTest(f"no saved {name} template under {evidence}")
                template = (evidence / relative).read_text(encoding="utf-8")
                model = root / f"{name}.gguf"
                tiny_fixture.create_fixture(model, "qwen3", chat_template=template, control_tokens=controls)
                info = chat.gguf_tokenizer_info(model)
                for variant, (flags, told) in enumerate(self.FLAGS[name]):
                    server_kwargs = messages.server_template_kwargs(**told) if told is not None else None
                    slots = root / f"slots-{name}-{variant}"
                    slots.mkdir()
                    server = fixtures_media.Server(SERVER, model, slots, root / f"{name}-{variant}.log",
                                                   fixtures_media.free_port(PORTS), extra=["--jinja", *flags])
                    try:
                        for number, conversation in enumerate(self.conversations()):
                            for options in self.OPTIONS:
                                body = {"messages": conversation["messages"], **(options or {})}
                                if conversation["tools"]:
                                    body["tools"] = conversation["tools"]
                                served = server.request("POST", "/apply-template", body)["prompt"]
                                matched = None
                                for context_name, context in chat.server_contexts(options, eos_token=info["eos_token"],
                                                                                  server_kwargs=server_kwargs):
                                    try:
                                        mine = chat.render(conversation, template, bos_token=info["bos_token"],
                                                           add_generation_prompt=True, extra_context=context,
                                                           live=True).text
                                    except (ValueError, jinja2.TemplateError):
                                        continue
                                    if mine == served:
                                        matched = context_name
                                        break
                                key = (name, " ".join(flags) or "defaults", number, json.dumps(options, sort_keys=True))
                                if told is None:
                                    found[key] = matched
                                with self.subTest(key=key):
                                    self.assertIsNotNone(matched, served[:400])
                    finally:
                        server.close()
        # without flags to tell, the template's defaults reproduce the server, except Muse's date line
        # (no system message: strftime_now at the request's time, which only the server context renders)
        expected = {key: "server" if key[0] == "muse" and key[2] == 1 else "template" for key in found}
        self.assertEqual(found, expected)


@unittest.skipUnless(SERVER, "set KVPARK_LLAMA_SERVER (runtime/build/bin/llama-server) to translate real slot saves "
                     "holding an image")
class LlamaServerMediaSourceTest(unittest.TestCase):
    """engine.run on a real llama-server slot save holding an image (as mtmd writes it, the
    <|vision_start|> it adds included) against a real text-only save of the same prefix: the
    translation of the media slot is the translation of that text-only slot, byte for byte."""

    def translate_both(self, fx, *, mmproj, flash_attn="on"):
        import fixtures_media
        png = fixtures_media.write_png(fx.root / "image.png", 128, 96, seed=1)
        slots = fx.root / "slots"
        slots.mkdir()
        server = fixtures_media.Server(SERVER, fx.gguf["src"], slots, fx.root / "server.log",
                                       fixtures_media.free_port(PORTS), mmproj=mmproj, ctx=1024,
                                       flash_attn=flash_attn, extra=["--image-max-tokens", "16"])
        try:
            marker = server.props["media_marker"]
            messages = [MESSAGES[0], {"role": "user", "content": f"Look at {marker} and tell me about the tides."},
                        *MESSAGES[2:]]
            server.complete(render(messages[:-1], OPTIONS_TEMPLATE, generation=False), images=[png])
            server.save("with-media.bin")
            media = kvstate.read_file(slots / "with-media.bin", fx.spec["src"]).media
            (chunk,) = media.media
            cut = kvstate.media_cut(media)            # as the engine cuts: no projector-specific ids
            self.assertEqual((cut, media.tokens[cut - 1]), (chunk.start, fixtures_media.VISION_START_ID))
            server.erase()
            server.complete([int(t) for t in media.tokens[:cut]])
            server.save("text-only.bin")
        finally:
            server.close()
        target = render(messages, OPTIONS_TEMPLATE)
        job = fx.job(target, messages=messages, source=slots / "with-media.bin", max_t0_message=1)
        report = fx.run(job)
        reference = fx.job(target, messages=messages, source=slots / "text-only.bin", max_t0_message=1)
        reference_report = fx.run(reference)
        self.assertEqual(report["source_media"], {"chunks": 1, "entries": len(media.tokens), "read": cut,
                                                  "t0_cap": target.index("<|end|>\n<|user|>Look")})
        self.assertEqual((report["t0"], report["source_tokens"], report["source_segments_found"]),
                         (reference_report["t0"], cut, reference_report["source_segments_found"]))
        self.assertGreater(report["mapped_cells"], 0)
        self.assertEqual(Path(job["target"]["output"]).read_bytes(), Path(reference["target"]["output"]).read_bytes())
        return chunk

    def test_normal_position_image_in_a_dense_source(self):
        import fixtures_media
        fx = Fixture(OPTIONS_TEMPLATE, source_options={"control_tokens": fixtures_media.VISION_TOKENS,
                                                       "ascii_output": True})
        try:
            mmproj = fx.root / "mmproj-tiny.gguf"
            fixtures_media.create_qwen3vl_mmproj(mmproj)
            self.assertEqual(self.translate_both(fx, mmproj=mmproj).pos_type, "normal")
        finally:
            fx.close()

    @unittest.skipUnless(QWEN_MMPROJ, "set KVPARK_QWEN_MMPROJ to a 5120-wide qwen3vl_merger mmproj (the Qwen3.8 one)")
    def test_mrope_image_in_a_hybrid_source(self):
        import fixtures_media
        fx = Fixture(OPTIONS_TEMPLATE, source="qwen35",
                     source_options={"hidden": 5120, "control_tokens": fixtures_media.VISION_TOKENS,
                                     "ascii_output": True})
        try:
            # the recurrent rows are reset (the translation never reads them); the attention rows
            # before the image are the text-only save's
            self.assertEqual(self.translate_both(fx, mmproj=QWEN_MMPROJ, flash_attn="off").pos_type, "mrope")
        finally:
            fx.close()


CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
N, U, C, D, X, B = (engine.NORMAL, engine.UNKNOWN, engine.CONTROL, engine.USER_DEFINED, engine.UNUSED, engine.BYTE)


def vocabulary_gguf(path, model, vocabulary, *, pre=None, add_space_prefix=None, template=CHATML):
    """A GGUF with only a tokenizer ([(token, type)]) and a chat template: what engine.profile reads."""
    gguf = tiny_fixture._gguf_module()
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tokenizer_model(model)
    if pre is not None:
        writer.add_tokenizer_pre(pre)
    writer.add_token_list([token for token, _ in vocabulary])
    writer.add_token_types([kind for _, kind in vocabulary])
    if model == "llama":
        writer.add_token_scores([0.0] * len(vocabulary))
    else:
        writer.add_token_merges(["a b"])
    if add_space_prefix is not None:
        writer.add_add_space_prefix(add_space_prefix)
    writer.add_chat_template(template)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TokenizerTest(unittest.TestCase):
    """engine.vocabulary / token_bytes / spelled_pieces against llama_vocab::impl::token_to_piece
    and tokenize (llama-vocab.cpp at a7b94df2c, the same at 85d5703a3)."""

    SPM = [("<unk>", U), ("<s>", C), ("</s>", C), ("<0x0A>", B), ("<0x20>", B), ("\u2581", N), ("\u2581Hello", N),
           ("\u2581world", N), ("ello", N), ("<|im_start|>", C), ("<|im_end|>", C), ("<think>", D), ("[PAD]", X),
           ("", N), ("H", N), ("\u2581user", N), ("\u2581assistant", N)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        engine._PROFILE_CACHE[0] = None

    def tearDown(self):
        self.tmp.cleanup()

    def ids(self, vocabulary, *tokens):
        index = {token: i for i, (token, _) in enumerate(vocabulary)}
        return [index[token] for token in tokens]

    def test_sentencepiece_pieces_as_llama_cpp_prints_them(self):
        gguf = vocabulary_gguf(self.root / "spm.gguf", "llama", self.SPM)
        vocab = engine.vocabulary(gguf)
        self.assertEqual((vocab.model, vocab.spelling, vocab.space_prefix), ("llama", "spm", True))
        pieces = engine.token_bytes(gguf, range(len(self.SPM)))
        self.assertEqual(pieces, [b"<unk>", b"<s>", b"</s>", b"\n", b" ", b" ", b" Hello", b" world", b"ello",
                                  b"<|im_start|>", b"<|im_end|>", b"<think>", b"", b"[EMPTY_13]", b"H", b" user",
                                  b" assistant"])
        # add_space_prefix=false (a converter's choice) keeps the pieces as printed
        plain = vocabulary_gguf(self.root / "spm-plain.gguf", "llama", self.SPM, add_space_prefix=False)
        ids = self.ids(self.SPM, "<s>", "\u2581Hello")
        self.assertFalse(engine.vocabulary(plain).space_prefix)
        self.assertEqual(engine.spelled_pieces(plain, ids, engine.token_bytes(plain, ids)), [b"<s>", b" Hello"])

    def test_the_space_sentencepiece_puts_after_special_tokens_is_dropped(self):
        gguf = vocabulary_gguf(self.root / "spm.gguf", "llama", self.SPM)
        # llama.cpp tokenizes " " + each text fragment that starts the prompt or follows a special
        # token: "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n"
        tokens = ["<|im_start|>", "\u2581user", "<0x0A>", "H", "ello", "\u2581world", "<|im_end|>", "\u2581", "<0x0A>",
                  "<|im_start|>", "\u2581assistant", "<0x0A>"]
        ids = self.ids(self.SPM, *tokens)
        raw = engine.token_bytes(gguf, ids)
        text = "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(b"".join(raw), b"<|im_start|> user\nHello world<|im_end|> \n<|im_start|> assistant\n")
        spelled = engine.token_bytes(gguf, ids, spelled=True)
        self.assertEqual(b"".join(spelled), text.encode())
        self.assertEqual(spelled[7], b"")                    # the "\u2581" that held only the added space
        # a text at the start (BOS first, or no special token at all) is prefixed as well
        self.assertEqual(engine.spelled_pieces(gguf, self.ids(self.SPM, "<s>", "\u2581Hello", "\u2581world"),
                                               [b"<s>", b" Hello", b" world"]), [b"<s>", b"Hello", b" world"])
        self.assertEqual(engine.spelled_pieces(gguf, self.ids(self.SPM, "\u2581Hello"), [b" Hello"]), [b"Hello"])
        # the target side: the server's /tokenize pieces spell the rendering once the space is dropped
        conversation = {"messages": [{"role": "user", "content": "Hello world"}], "tools": None}
        served = [[i, piece.decode()] for i, piece in zip(ids, raw)]
        target_ids, pieces = engine.target_pieces(served, gguf)
        self.assertEqual(pieces, spelled)
        profile = engine.profile(gguf)
        rendered, shift, name = engine.render_target(conversation, profile, None, target_ids, pieces, gguf, media=False)
        self.assertEqual((rendered.text, shift, name), (text, 0, "template"))
        with self.assertRaises(engine.TemplateMismatch):
            engine.render_target(conversation, profile, None, target_ids, raw, gguf, media=False)

    def test_gemma4_bpe_writes_spaces_as_u2581_and_has_byte_tokens(self):
        vocabulary = [("<pad>", C), ("<eos>", C), ("<bos>", C), ("<0x0A>", B), ("\u2581Hello", N), ("Hi", N),
                      ("  ", D), ("<|turn>", C)]
        gguf = vocabulary_gguf(self.root / "gemma4.gguf", "gemma4", vocabulary)
        vocab = engine.vocabulary(gguf)
        self.assertEqual((vocab.spelling, vocab.space_prefix), ("spm", False))
        ids = list(range(len(vocabulary)))
        self.assertEqual(engine.token_bytes(gguf, ids), [b"<pad>", b"<eos>", b"<bos>", b"\n", b" Hello", b"Hi", b"  ",
                                                         b"<|turn>"])
        self.assertEqual(engine.token_bytes(gguf, ids, spelled=True), engine.token_bytes(gguf, ids))
        # a byte-level BPE with an escaping pre-tokenizer spells the same way
        escaped = vocabulary_gguf(self.root / "escaped.gguf", "gpt2", vocabulary, pre="gemma4")
        self.assertEqual(engine.vocabulary(escaped).spelling, "spm")

    def test_byte_level_bpe_and_its_unknown_characters(self):
        vocabulary = [("\u0120hello", N), ("\u010a", N), ("<|im_end|>", C), ("x\u2192y", N), ("", N), ("[PAD5]", X),
                      ("\u0120tool", D)]
        gguf = vocabulary_gguf(self.root / "bpe.gguf", "gpt2", vocabulary, pre="qwen2")
        vocab = engine.vocabulary(gguf)
        self.assertEqual((vocab.spelling, vocab.space_prefix), ("bpe", False))
        # llama_decode_text: a character outside the byte map becomes [UNK_BYTE_0x<utf-8><token>]
        self.assertEqual(engine.token_bytes(gguf, range(len(vocabulary))),
                         [b" hello", b"\n", b"<|im_end|>", "x[UNK_BYTE_0xe28692x\u2192y]y".encode(), b"[EMPTY_4]", b"",
                          "\u0120tool".encode()])

    def test_other_tokenizers_are_refused(self):
        gguf = vocabulary_gguf(self.root / "t5.gguf", "t5", [("<pad>", C), ("\u2581a", N)])
        with self.assertRaisesRegex(ValueError, "tokenizer model 't5' is not supported yet"):
            engine.token_bytes(gguf, [1])

    def test_profiles_keep_the_pre_tokenizer_and_space_prefix(self):
        gguf = vocabulary_gguf(self.root / "spm.gguf", "llama", self.SPM, pre="default", add_space_prefix=False)
        cache = self.root / "profiles"
        data = engine.profile(gguf, str(cache))
        self.assertEqual((data["format"], data["tokenizer_pre"], data["add_space_prefix"]), (4, "default", False))
        # a profile cached by an older engine (format 2: no tokenizer keys) is built again
        cached = next(cache.iterdir())
        old = json.loads(cached.read_text())
        old.update(format=2)
        del old["tokenizer_pre"], old["add_space_prefix"]
        cached.write_text(json.dumps(old))
        engine._profile.cache_clear()
        self.assertEqual(engine.profile(gguf, str(cache))["tokenizer_pre"], "default")


@unittest.skipUnless(VOCAB_TOOL, "set KVPARK_VOCAB_TOOL (build_collector.py --target kvpark-vocab) to compare with "
                     "llama.cpp's own vocabularies")
class RealVocabularyTest(unittest.TestCase):
    """Every local vocabulary whole (engine.token_bytes == llama_token_to_piece, special=true), and
    llama.cpp's tokenization (add_special, parse_special: what /tokenize returns) of each
    template_check.BATTERY rendering spelled back to the rendering by engine.render_target, with
    engine.located_record placing no segment where chat.render does not span it."""

    def tool(self, model, *args):
        return subprocess.run([VOCAB_TOOL, str(model), *map(str, args)], capture_output=True, text=True, check=True,
                              timeout=600).stdout

    def test_local_vocabularies(self):
        import template_check
        models = [m for m in LOCAL_MODELS if m.is_file()]
        if not models:
            self.skipTest("no local GGUF")
        engine._PROFILE_CACHE[0] = None
        for model in models:
            with self.subTest(model=model.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                info = json.loads(self.tool(model, "info"))
                self.tool(model, "pieces", root / "pieces.jsonl")
                expected = [bytes.fromhex(json.loads(line)["piece"])
                            for line in (root / "pieces.jsonl").read_text().splitlines()]
                self.assertEqual(engine.token_bytes(model, range(len(expected))), expected)
                profile = engine.profile(model)
                cases = []
                for conversation in template_check.BATTERY:
                    context = chat.server_context(None, server=True, eos_token=profile["eos_token"])
                    text = chat.render({"messages": conversation["messages"], "tools": conversation["tools"]},
                                       profile["chat_template"], bos_token=profile["bos_token"],
                                       add_generation_prompt=True, extra_context=context, live=True).text
                    if info["add_bos"] and profile["bos_token"] and text.startswith(profile["bos_token"]):
                        text = text[len(profile["bos_token"]):]         # llama-server drops it, adds the token
                    cases.append((conversation, text))
                (root / "in.jsonl").write_text("".join(json.dumps({"text": text, "add_special": True}) + "\n"
                                                       for _, text in cases))
                self.tool(model, "tokenize", root / "in.jsonl", root / "out.jsonl")
                for (conversation, text), line in zip(cases, (root / "out.jsonl").read_text().splitlines()):
                    served = json.loads(line)
                    ids, pieces = served["ids"], [bytes.fromhex(p) for p in served["pieces"]]
                    self.assertEqual(engine.token_bytes(model, ids), pieces)
                    spelled = engine.spelled_pieces(model, ids, pieces)
                    if info["add_eos"]:
                        # an embedding model's vocabulary appends EOS (gte-Qwen2): not a bridge target
                        self.assertEqual(b"".join(spelled), text.encode() + spelled[-1])
                        continue
                    rendered, shift, _ = engine.render_target(conversation, profile, None, ids, spelled, model,
                                                              media=False)
                    self.assertEqual(b"".join(spelled)[shift:], rendered.text.encode())
                    # the source side on the same text: no segment located where chat.render does not span it
                    expected = {n: [n, a + shift, b + shift] for n, a, b in rendered.segments}
                    for span in engine.located_record(conversation, "doc", b"".join(spelled))["segments"]:
                        self.assertEqual(span, expected.get(span[0]))


def write_spec_gguf(path, meta, tensors):
    """A GGUF of synthetic metadata (integer arrays, floats, unsigned ints), a byte vocabulary, the
    options template and `tensors` ({name: array}): what engine.spec_of reads."""
    gguf = tiny_fixture._gguf_module()
    writer = gguf.GGUFWriter(path, meta["general.architecture"])
    for key, value in meta.items():
        if key == "general.architecture":
            continue
        if isinstance(value, list):
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
        elif isinstance(value, str):
            writer.add_key_value(key, value, gguf.GGUFValueType.STRING)
        else:
            writer.add_key_value(key, value, gguf.GGUFValueType.FLOAT32 if isinstance(value, float)
                                 else gguf.GGUFValueType.UINT32)
    tiny_fixture._byte_vocabulary(writer, gguf)
    writer.add_chat_template(OPTIONS_TEMPLATE)
    for tensor, value in tensors.items():
        writer.add_tensor(tensor, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class SpecTest(unittest.TestCase):
    def test_granitehybrid_specs_carry_the_rope_frequency_factors(self):
        meta = hybrid_metadata("granitehybrid", **{"attention.head_count_kv": [0, 1, 0, 1]})
        factors = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)          # one per rotated pair (n_rot 8)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, tensors in (("with", {"rope_freqs.weight": factors}), ("without", {})):
                paths[name] = write_spec_gguf(Path(directory) / f"granite-{name}.gguf", meta, tensors)
            engine._PROFILE_CACHE[0] = str(Path(directory) / "profiles")
            try:
                spec = engine.spec_of(paths["with"])
                self.assertEqual(spec, kvstate.ModelSpec.from_gguf(paths["with"]))
                self.assertEqual(spec.rope(1).freq_factors, tuple(float(x) for x in factors))
                # the cached profile gives the same spec (factors included) in a fresh process
                engine._profile.cache_clear()
                self.assertEqual(engine.spec_of(paths["with"]).rope(3).freq_factors, spec.rope(1).freq_factors)
                self.assertIsNone(engine.spec_of(paths["without"]).rope(1).freq_factors)
                self.assertEqual(engine.spec_of(paths["without"]), kvstate.ModelSpec.from_gguf(paths["without"]))
            finally:
                engine._PROFILE_CACHE[0] = None
                engine._profile.cache_clear()

    def test_gqa_family_specs_carry_the_rope_frequency_factors(self):
        # profile() keeps rope_freqs.weight of llama, mistral3 and gemma4 (PROFILE_FORMAT 4), and the refusal of a
        # LongRoPE GGUF (rope_factors_long/short, which llama.cpp picks by the context size), which spec_of raises
        factors = np.asarray([1.0, 2.0, 4.0, 8.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)   # n_rot 16: 8 pairs
        embedding = {"token_embd.weight": np.zeros((4, 4), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine._PROFILE_CACHE[0] = str(root / "profiles")
            try:
                for arch in ("llama", "mistral3"):
                    with self.subTest(arch=arch):
                        path = write_spec_gguf(root / f"{arch}.gguf", gqa_metadata(arch),
                                               {"rope_freqs.weight": factors, **embedding})
                        spec = engine.spec_of(path)
                        self.assertEqual(spec, kvstate.ModelSpec.from_gguf(path))
                        self.assertEqual(spec.rope(0).freq_factors, tuple(float(x) for x in factors))
                        plain = write_spec_gguf(root / f"{arch}-plain.gguf", gqa_metadata(arch), embedding)
                        self.assertIsNone(engine.spec_of(plain).rope(0).freq_factors)
                        self.assertEqual(engine.spec_of(plain), kvstate.ModelSpec.from_gguf(plain))
                # gemma4's proportional RoPE: its full layers' factors come from rope_freqs.weight
                tiny_fixture.create_gemma_fixture(root / "gemma4-raw.gguf", "gemma4")
                gemma = root / "gemma4.gguf"
                tiny_fixture.copy_with_metadata(root / "gemma4-raw.gguf", gemma,
                                                values={"tokenizer.chat_template": OPTIONS_TEMPLATE})
                spec = engine.spec_of(gemma)
                self.assertEqual(spec, kvstate.ModelSpec.from_gguf(gemma))
                self.assertTrue(spec.rope_freq_factors)
                # LongRoPE: phi3's factor tensors, a llama's, and a longrope scaling type
                long_short = {"rope_factors_long.weight": factors, "rope_factors_short.weight": factors, **embedding}
                for name, meta, tensors, pattern in (
                        ("phi3", gqa_metadata("phi3"), long_short, "rope_factors_long/short"),
                        ("llama-long", gqa_metadata("llama"), long_short, "rope_factors_long/short"),
                        ("llama-scaling", gqa_metadata("llama", **{"rope.scaling.type": "longrope"}), embedding,
                         "longrope")):
                    with self.subTest(model=name):
                        path = write_spec_gguf(root / f"{name}.gguf", meta, tensors)
                        with self.assertRaisesRegex(kvstate.StateFormatError, pattern):
                            engine.spec_of(path)
                data = engine.profile(root / "phi3.gguf", engine._PROFILE_CACHE[0])
                self.assertIn("rope_factors_long/short", data["rope_freq_factors"]["refused"])
                # a Phi-3 without factor tensors takes none
                phi = write_spec_gguf(root / "phi3-plain.gguf", gqa_metadata("phi3"), embedding)
                self.assertEqual(engine.spec_of(phi), kvstate.ModelSpec.from_gguf(phi))
            finally:
                engine._PROFILE_CACHE[0] = None
                engine._profile.cache_clear()

    def test_a_servers_build_info_names_the_pin_of_its_spec(self):
        # job["source"/"target"]["build_info"]: llama-server's /props build_info (the bridge adds it)
        self.assertIsNone(engine.job_revision({}))
        self.assertEqual(engine.job_revision({"build_info": "b11398-a7b94df2c"}), "a7b94df2c")
        self.assertEqual(engine.job_revision({"build_info": "b8929-85d5703a3"}), "85d5703a3")
        self.assertIsNone(engine.job_revision({"build_info": "b31-f90a6a0"}))   # a build naming no known pin
        with self.assertRaisesRegex(ValueError, "build_info"):
            engine.job_revision({"build_info": 11398})
        fx = Fixture(OPTIONS_TEMPLATE)
        try:
            engine._PROFILE_CACHE[0] = str(fx.root / "profiles")
            for revision in kvstate.LLAMA_REVISIONS:
                spec = engine.spec_of(fx.gguf["src"], llama_revision=revision)
                self.assertEqual(spec, kvstate.ModelSpec.from_gguf(fx.gguf["src"], llama_revision=revision))
                self.assertEqual(spec.llama_revision, revision)
            # a job naming the servers' builds translates as one without (qwen3 and Muse states are the
            # same on both pins)
            text = render(MESSAGES, OPTIONS_TEMPLATE)
            plain_job = fx.job(text)
            plain = fx.run(plain_job)
            job = fx.job(text, build_info="b11398-a7b94df2c")
            job["source"]["build_info"] = "b8929-85d5703a3"
            report = fx.run(job)
            self.assertEqual((report["t0"], report["mapped_cells"]), (plain["t0"], plain["mapped_cells"]))
            self.assertEqual(Path(job["target"]["output"]).read_bytes(),
                             Path(plain_job["target"]["output"]).read_bytes())
        finally:
            engine._PROFILE_CACHE[0] = None
            fx.close()


if __name__ == "__main__":
    unittest.main()
