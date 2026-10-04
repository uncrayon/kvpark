"""engine.run on tiny GGUF fixtures (a dense qwen3 source, a Muse-like target, a random hub): the
target rendered with a request's template options as llama-server renders it (chat.server_contexts,
TemplateMismatch and its exit status), a source slot that holds media (read up to its first
medium), the features the engine advertises, and specs with granitehybrid's RoPE frequency
factors. The real templates and servers are checked by the live runs (live_bridge.py options).

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
import chat  # noqa: E402
import jinja2  # noqa: E402
import kvstate  # noqa: E402
import tiny_fixture  # noqa: E402
from test_kvstate import hybrid_metadata  # noqa: E402
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
# context they fail to render instead of rendering an empty string
EOS_EXPRESSION_TEMPLATE = ("<|year|>{{ strftime_now('%Y') }}<|end|>\n{% for m in messages %}<|{{ m.role }}|>"
                           "{% if m.role == 'assistant' %}{{ m.content + eos_token }}{% else %}{{ m.content }}"
                           "{% endif %}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")
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
        self.assertEqual(fx.run(fx.job(default))["rendering"], "template")       # no options: as before
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
        # `content + eos_token` and strftime_now(...) raise jinja2's UndefinedError (not a
        # ValueError) in the template context: the server context must still be tried
        server = {"server": True, "eos_token": "<eos>"}
        fx = Fixture(EOS_EXPRESSION_TEMPLATE, context=server)
        try:
            with self.assertRaises(jinja2.UndefinedError):
                render(MESSAGES, EOS_EXPRESSION_TEMPLATE)
            text = render(MESSAGES, EOS_EXPRESSION_TEMPLATE, **server)
            self.assertIn("ebb.<eos><|end|>", text)
            self.assertTrue(text.startswith(f"<|year|>{datetime.datetime.now():%Y}<|end|>"))
            report = fx.run(fx.job(text))
            self.assertEqual(report["rendering"], "server")
            state = kvstate.read_file(fx.root / f"out-{fx.outputs}.bin", fx.spec["tgt"])
            self.assertEqual(bytes(state.tokens.astype(np.uint8)), text[:report["t0"]].encode())
            # tokens no context spells: a mismatch naming the first failure, never a job error
            with self.assertRaisesRegex(engine.TemplateMismatch, "template context failed: UndefinedError"):
                fx.run(fx.job(text.replace("tides", "waves")))
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


class SpecTest(unittest.TestCase):
    def test_granitehybrid_specs_carry_the_rope_frequency_factors(self):
        gguf = tiny_fixture._gguf_module()
        meta = hybrid_metadata("granitehybrid", **{"attention.head_count_kv": [0, 1, 0, 1]})
        factors = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)          # one per rotated pair (n_rot 8)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, tensors in (("with", {"rope_freqs.weight": factors}), ("without", {})):
                path = paths[name] = Path(directory) / f"granite-{name}.gguf"
                writer = gguf.GGUFWriter(path, "granitehybrid")
                for key, value in meta.items():
                    if key == "general.architecture":
                        continue
                    if isinstance(value, list):
                        writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
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


if __name__ == "__main__":
    unittest.main()
