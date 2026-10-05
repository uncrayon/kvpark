"""DeepSeek V4 (llama.cpp arch deepseek4) as a translation target, end to end on test-llama-archs'
random deepseek4 model (layers 0-1 window only, 2 CSA with its lightning indexer, 3 HCA; window 32,
head 64, n_rot 32) and its copy with an MTP layer (tiny_fixture.create_dsv4_mtp_fixture).

  PoolingTests     dsv4_pool's rows from captured per-token (a, s) against the rows llama.cpp
                   stores: float16 rounding apart (f16), block for block (q8_0), YaRN included;
  OracleTests      a state synthesized from the captures of a native run (dsv4_target.synthesize)
                   restored by the state driver, continued over the native tail [T0, T) and
                   compared with a restored native save at T0 (restored contexts agree bit for
                   bit, so this is the meaningful reference): f16, YaRN, q8_0, -np 2 with slot 1,
                   swa_full off (llama-server's setting; captures in other ubatches than the
                   template), the MTP layer, a one-block (128-token) split; with each component zeroed
                   (raw window, CSA/HCA/LID rows, CSA/LID rings) the logits move by orders of
                   magnitude more, and garbage in what a 128-aligned split never reads (the HCA
                   ring, the rest of the CSA/LID rings) changes nothing;
  TranslateTests   translate.translate into a DeepSeek V4 target: through an identity hub
                   (deepseek4 into itself, the source a capture: dsv4_target.CaptureRows) the
                   oracle state exactly, and a T0 of 200 rounded down to 128 (restored against a
                   native save at 128); from a qwen3 fixture through a random hub with head
                   decoders (shapes, restore, decoding);
  EngineBridgeTests  engine.py one-shot and engine_daemon.py jobs, then kvpark.bridge.switch (run by
                   the product's python3 against stand-in servers, one-shot and daemon engine),
                   from a qwen3 fixture slot into deepseek4 (a byte vocabulary and a chat template
                   added) through a format 2 bundle with head decoders, at T0 = P - tail = 332:
                   256 cells translated, and the report, the restore check, the .resume and the
                   reused cells all at 256; the daemon keeps DeepSeek V4 sides on numpy;
  HubFitTests      fit_hub.py --heads on kv_probe captures of a qwen3 fixture and the deepseek4
                   model (a byte vocabulary added), then qwen3 -> deepseek4 through the bundle
                   (random weights: plumbing, not quality);
  HeadFitTests     hub.fit_head (dev-picked alpha, held-out test R^2) and the head bundles;
  CapabilityTests  kvpark.bridge.capabilities: target only, every head kind, no draft.

Needs KVPARK_STATE_DRIVER (a kvpark-state-driver with --parallel/--seq/--partial and q8_0) and
KVPARK_LLAMA_ARCHS (test-llama-archs of either pin; tiny_fixture.llama_archs_model), HubFitTests KVPARK_KV_PROBE (a kvpark-kv-probe with the
deepseek4 row), and EngineBridgeTests' switch a python3 with psutil (kvpark.bridge's registry); tests
skip cleanly without them (CapabilityTests and HeadFitTests run anyway).
"""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import backends
import dsv4_pool
import dsv4_target
import fit_hub
import fixtures_families as ff
import hub as hub_module
import state_writer
import translate
from test_translate import StubSpans, StubView, random_hub

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import chat  # noqa: E402
import kvstate  # noqa: E402
from test_kv_probe_integration import copy_with_byte_vocabulary  # noqa: E402
from tiny_fixture import copy_with_metadata, create_dsv4_mtp_fixture, create_fixture, llama_archs_model  # noqa: E402
from kvpark.bridge import capabilities, resume  # noqa: E402

DRIVER = os.environ.get("KVPARK_STATE_DRIVER")
LLAMA_ARCHS = os.environ.get("KVPARK_LLAMA_ARCHS")
YARN = {"rope_scaling_type": "yarn", "rope_freq_scale": 0.25, "yarn_orig_ctx": 64}
YARN_FLAGS = ["--rope-scaling", "yarn", "--rope-freq-scale", "0.25", "--yarn-orig-ctx", "64"]
T = 300                 # native run; T0 = 256 (rounded down), the tail [256, 300) and NEXT more tokens native
NEXT = 3
# kvpark.bridge.switch in the product's Python (its registry needs psutil, which the engine venvs lack)
SWITCH = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from kvpark.bridge import engine_client, registry, switch
args = json.loads(sys.argv[2])
config = registry.empty()
for name, model in args["models"].items():
    registry.add_model(config, name, **model)
registry.set_engine(config, **args["engine"])
config.update(args["settings"])
path = Path(args["config"])
registry.save(config, path)
try:
    report = switch.switch(registry.load(path), args["conversation"], args["source"], args["target"], config_path=path)
finally:
    engine_client.stop(path.parent)
print(json.dumps(report))
"""


def bridge_python():
    """A Python that imports kvpark.bridge.registry (psutil), or None."""
    python = shutil.which("python3")
    if python is None or subprocess.run([python, "-c", "import psutil"], capture_output=True).returncode:
        return None
    return python


def driver_ready():
    if not (DRIVER and LLAMA_ARCHS):
        return False
    usage = subprocess.run([DRIVER, "--help"], capture_output=True, text=True, timeout=60).stdout
    return all(option in usage for option in ("--parallel", "--partial", "q8_0"))


READY = driver_ready()


class Fixtures:
    """The deepseek4 fixtures and state-driver runs under one directory."""

    def __init__(self, root):
        self.root = Path(root)
        # with the settings both pins' test-llama-archs share (weights N(0, 0.01), indexer top_k 8): the
        # tolerances below were measured on them
        self.models = {"dsv4": llama_archs_model(LLAMA_ARCHS, "deepseek4", self.root / "archs"),
                       "dsv4-mtp": self.root / "deepseek4-mtp.gguf"}
        create_dsv4_mtp_fixture(self.models["dsv4"], self.models["dsv4-mtp"])
        self.count = 0

    def fresh(self, stem):
        self.count += 1
        return self.root / f"{stem}-{self.count}"

    def run(self, mode, model, *args, flags=(), batch=64):
        command = [DRIVER, "--mode", mode, "--model", str(self.models[model]), "--threads", "4", "--ctx", "1024",
                   "--batch", str(batch), *map(str, args), *flags]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"state driver {mode}: {result.stderr[-1500:]}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    def save(self, model, tokens, flags=()):
        path = self.fresh("native").with_suffix(".state")
        self.run("save", model, "--tokens", ",".join(map(str, tokens)), "--state", path, flags=flags)
        return path

    def capture(self, model, tokens, spec, flags=(), batch=64):
        directory = self.fresh("capture")
        self.run("capture", model, "--tokens", ",".join(map(str, tokens)), "--capture",
                 ",".join(dsv4_target.capture_names(spec)), "--output", directory, flags=flags, batch=batch)
        return dsv4_target.driver_captures(directory, spec)

    def restore(self, model, state, next_ids, flags=()):
        """Logits [len(next_ids), vocab] after restoring `state` (a path or a StateFile) and decoding next_ids."""
        if not isinstance(state, (str, Path)):
            path = self.fresh("synth").with_suffix(".state")
            path.write_bytes(state.to_bytes())
            state = path
        logits = self.fresh("logits").with_suffix(".npy")
        # the whole continuation is one ubatch (restored and native runs alike)
        self.run("restore-decode", model, "--state", state, "--next", ",".join(map(str, next_ids)), "--logits", logits,
                 flags=flags, batch=256)
        return np.load(logits)


def inputs_of(captures, t0):
    """synthesize's inputs from driver_captures: the latents and (a, s) of [0, t0)."""
    out = {}
    for (kind, layer), rows in captures.items():
        if kind == dsv4_target.LATENT:
            out[kind, layer] = rows[:t0]
        else:
            half = rows.shape[1] // 2
            out[kind, layer] = (rows[:t0, :half], rows[:t0, half:])
    return out


@unittest.skipUnless(READY, "set KVPARK_STATE_DRIVER (with the DeepSeek V4 options) and KVPARK_LLAMA_ARCHS")
class Dsv4Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-dsv4-")
        cls.fx = Fixtures(cls.temp.name)
        cls.rng = np.random.default_rng(2029)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def spec(self, model="dsv4", runtime=None):
        return kvstate.ModelSpec.from_gguf(self.fx.models[model], runtime=runtime)

    def tokens(self, n=T + NEXT):
        return [int(t) for t in self.rng.integers(0, 128, n)]


class PoolingTests(Dsv4Case):
    def test_pools_reproduce_the_stored_rows(self):
        tokens = self.tokens(T)
        for runtime, flags, cache in ((None, [], "f16"), (YARN, YARN_FLAGS, "f16"), (None, [], "q8_0")):
            with self.subTest(runtime=runtime, cache=cache):
                flags = [*flags, "--cache-type", cache, "--flash-attn", "on" if cache == "q8_0" else "off"]
                spec = self.spec(runtime=runtime)
                captures = self.fx.capture("dsv4", tokens, spec, flags)
                native = kvstate.read_file(self.fx.save("dsv4", tokens, flags), spec)
                tensors = dsv4_pool.pool_tensors(self.fx.models["dsv4"], spec)
                for pool in spec.pools():
                    for layer in pool.layers:
                        rows = captures[pool.name, layer]
                        pooler = dsv4_pool.Pooler(spec, pool, layer, tensors, rotate=cache == "q8_0" or None)
                        pooler.feed(rows[:256, :pool.state_width], rows[:256, pool.state_width:])
                        pooler.feed(rows[256:, :pool.state_width], rows[256:, pool.state_width:])
                        got, (kv, score) = pooler.result()
                        stored = native.state.part(pool.name).rows[layer][0]
                        ring = native.state.part(pool.name + "_state")
                        # the ring holds the last state_size tokens' a and APE-added score exactly
                        np.testing.assert_array_equal(kv, ring.kv[layer][0])
                        np.testing.assert_array_equal(score, ring.score[layer][0])
                        self.assertEqual(len(got), T // pool.ratio)
                        if cache == "q8_0":
                            blocks = kvstate.quantize_q8_0(got)
                            self.assertGreaterEqual(np.mean(blocks == stored), 0.99, f"{pool.name} {layer}")
                            continue
                        mine = got.astype(np.float16)
                        # float16 rounding apart: llama.cpp sums in its own float32 order
                        differ = mine != stored
                        self.assertLessEqual(differ.mean(), 0.002, f"{pool.name} {layer}")
                        ulps = np.abs(mine.view(np.int16).astype(np.int32) - stored.view(np.int16).astype(np.int32))
                        self.assertLessEqual(int(ulps.max()), 1, f"{pool.name} {layer}")

    def test_window_rings_and_split(self):
        spec = kvstate.ModelSpec.from_gguf(self.fx.models["dsv4"])
        self.assertEqual([dsv4_target.split_t0(t) for t in (127, 128, 300, 384)], [0, 128, 256, 384])
        csa, hca = spec.pool("csa"), spec.pool("hca")
        a = np.arange(20 * csa.state_width, dtype=np.float32).reshape(20, -1)
        kv, score = dsv4_pool.ring_rows(csa, a, -a, np.arange(236, 256), minimal=True)
        # only the next block's previous half: the first half of the channels of the last 4 tokens
        np.testing.assert_array_equal(kv[4:, :csa.width], a[-4:, :csa.width])
        self.assertFalse(kv[:4].any() or kv[:, csa.width:].any() or score[4:, csa.width:].any())
        self.assertFalse(any(x.any() for x in dsv4_pool.ring_rows(hca, np.ones((128, hca.width)),
                                                                    np.ones((128, hca.width)), np.arange(128, 256),
                                                                    minimal=True)))
        self.assertEqual([dsv4_pool.rot_size(w) for w in (64, 128, 192, 512)], [64, 128, 64, 512])


class OracleTests(Dsv4Case):
    """dsv4_target.synthesize from a native run's captures against a restored native save at T0."""

    def oracle(self, model="dsv4", runtime=None, flags=(), t0=256, n=T, capture_batch=64):
        """(spec, native template, synthesized minimal state, reference logits, logits fn, tokens)."""
        tokens = self.tokens(n + NEXT)
        spec = self.spec(model, runtime)
        native = kvstate.read_file(self.fx.save(model, tokens[:t0], flags), spec)
        captures = self.fx.capture(model, tokens[:n], spec, flags, batch=capture_batch)
        tensors = dsv4_pool.pool_tensors(self.fx.models[model], spec)
        state = dsv4_target.synthesize(spec, native, tokens, t0, inputs_of(captures, t0), tensors)
        tail = tokens[t0:]

        def logits(state_):
            return self.fx.restore(model, state_, tail, flags)
        return spec, native, state, logits(native), logits, (tokens, captures, tensors)

    def check_close(self, got, reference, scale):
        error = float(np.abs(got - reference).max())
        self.assertLessEqual(error, 1e-6 * scale, "synthesized state continues away from the native one")
        return error

    def test_synthesized_states_restore_like_native(self):
        # llama-server runs DeepSeek V4 with swa_full off; captures need not share the template's ubatch
        configurations = [("dsv4", None, (), "f16", 64), ("dsv4", YARN, YARN_FLAGS, "f16 yarn", 64),
                          ("dsv4", None, ("--cache-type", "q8_0", "--flash-attn", "on"), "q8_0", 64),
                          ("dsv4", None, ("--parallel", "2", "--seq", "1", "--flash-attn", "on"), "np 2 slot 1", 64),
                          ("dsv4", None, ("--swa-full", "off"), "swa-full off, captures in ubatches of 37", 37),
                          ("dsv4-mtp", None, (), "mtp layer", 64)]
        for model, runtime, flags, label, capture_batch in configurations:
            with self.subTest(label):
                spec, native, state, reference, logits, (tokens, captures, tensors) = self.oracle(
                    model, runtime, flags, capture_batch=capture_batch)
                scale = float(np.abs(reference).max())
                self.check_close(logits(state), reference, scale)
                full = dsv4_target.synthesize(spec, native, tokens, 256, inputs_of(captures, 256), tensors, ring="full")
                np.testing.assert_array_equal(logits(full), logits(state))
                # byte for byte: the raw cache and the rings; pooled rows up to float16 rounding
                for mine, theirs in zip(full.state.parts, native.state.parts):
                    if isinstance(mine, kvstate.KVCache):
                        a, b = mine.stream(), theirs.stream()
                        np.testing.assert_array_equal(a.positions, b.positions)
                        self.assertTrue(all(np.array_equal(a.k[l], b.k[l]) for l in b.k), mine.name)
                    elif isinstance(mine, kvstate.CompressorRing):
                        self.assertTrue(all(np.array_equal(mine.kv[l], theirs.kv[l]) and
                                            np.array_equal(mine.score[l], theirs.score[l]) for l in mine.layers))
                    else:
                        self.assertEqual(mine.n_rows, theirs.n_rows)
                        for l in mine.layers:
                            self.assertLessEqual(np.mean(mine.rows[l] != theirs.rows[l]), 0.002, mine.name)
                # the native pooled rows in the synthesized state: bit for bit the native continuation
                exact = copy.deepcopy(state)
                for pool in spec.pools():
                    exact.state.parts[[p.name for p in exact.state.parts].index(pool.name)] = native.state.part(pool.name)
                np.testing.assert_array_equal(logits(exact), reference)

    def test_every_component_matters_and_unread_parts_do_not(self):
        spec, native, state, reference, logits, _ = self.oracle()
        synth = logits(state)
        deviation = max(float(np.abs(synth - reference).max()), 1e-12)

        def edited(edit):
            changed = copy.deepcopy(state)
            edit(changed.state)
            return logits(changed)

        def zero_window(st):
            stream = st.part("swa").stream()
            for layer in stream.k:
                stream.k[layer] = np.zeros_like(stream.k[layer])

        def zero_rows(name):
            def edit(st):
                part = st.part(name)
                for layer in part.rows:
                    part.rows[layer] = np.zeros_like(part.rows[layer])
            return edit

        def ring(name, garbage):
            def edit(st):
                part, rng = st.part(name), np.random.default_rng(5)
                for layer in part.layers:
                    for rows in (part.kv[layer], part.score[layer]):
                        if garbage:         # only what the minimal ring leaves unwritten
                            unread = rows == 0
                            rows[unread] = rng.normal(0, 5, int(unread.sum()))
                        else:
                            rows[:] = 0
            return edit
        effects = {"raw window": zero_window, "csa rows": zero_rows("csa"), "hca rows": zero_rows("hca"),
                   "lid rows": zero_rows("lid"), "csa ring": ring("csa_state", False),
                   "lid ring": ring("lid_state", False)}
        for name, edit in effects.items():
            with self.subTest(zeroed=name):
                effect = float(np.abs(edited(edit) - reference).max())
                self.assertGreater(effect, 100 * deviation, f"zeroing the {name} is not above the synthesis error")
        for name in ("csa_state", "lid_state", "hca_state"):
            with self.subTest(garbage=name):
                np.testing.assert_array_equal(edited(ring(name, True)), synth)

    def test_a_one_block_split(self):
        # a 200-token run synthesized up to split_t0(200) = 128: the tail [128, 200) is native
        t0 = dsv4_target.split_t0(200)
        self.assertEqual(t0, 128)
        spec, native, state, reference, logits, _ = self.oracle(t0=t0, n=200)
        self.assertEqual([state.state.part(p.name).n_rows for p in spec.pools()], [32, 1, 32])
        self.check_close(logits(state), reference, float(np.abs(reference).max()))


class TranslateTests(Dsv4Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.qwen3 = cls.fx.root / "qwen3.gguf"
        create_fixture(cls.qwen3, "qwen3", chat_template=ff.TEMPLATE)

    def identity_hub(self, name, spec):
        """An identity hub of one DeepSeek V4 model: K over its latents, heads over its own [a | s] rows."""
        hubs = ff.identity_hub(name, spec)
        heads = []
        for pool in spec.pools():
            layers = list(pool.layers)
            width = 2 * pool.state_width
            total = width * len(layers)
            eye = np.eye(total, dtype=np.float32)
            transform = hub_module.ViewTransform(layers, [np.zeros(width, np.float32) for _ in layers],
                                                 [np.eye(width, dtype=np.float32) for _ in layers], eye.copy(),
                                                 np.ones(total, np.float32))
            heads.append(hub_module.HeadKind(pool.name, "k", [name], {name: layers},
                                             {name: [np.zeros(width, np.float32) for _ in layers]},
                                             {name: np.zeros(total, np.float32)},
                                             {name: [np.ascontiguousarray(eye[:, i * width:(i + 1) * width])
                                                     for i in range(len(layers))]},
                                             sources={name: (transform, eye.copy())}))
        return hub_module.with_heads(hubs, heads)

    def test_identity_hub_reproduces_the_oracle_state(self):
        tokens = self.tokens()
        spec = self.spec()
        template = kvstate.read_file(self.fx.save("dsv4", tokens[:1]), spec)
        captures = self.fx.capture("dsv4", tokens[:T], spec)
        record, side_tokens = ff.byte_side(tokens[:T])
        side = translate.Side("dsv4", spec, record, side_tokens, gguf=self.fx.models["dsv4"])
        hubs = self.identity_hub("dsv4", spec)
        state, report = translate.translate(dsv4_target.CaptureRows(captures, tokens[:T]), side, side, hubs, t0=T,
                                            template=template)
        self.assertEqual((report["t0"], report["t0_requested"], report["mapped_cells"]), (256, T, 256))
        self.assertEqual(report["kinds"], ["k", "csa", "hca", "lid"])
        tensors = dsv4_pool.pool_tensors(self.fx.models["dsv4"], spec)
        expected = dsv4_target.synthesize(spec, template, tokens, 256, inputs_of(captures, 256), tensors)
        self.assert_same_state(state, expected)
        native = self.fx.save("dsv4", tokens[:256])
        reference = self.fx.restore("dsv4", native, tokens[256:])
        got = self.fx.restore("dsv4", state, tokens[256:])
        self.assertLessEqual(float(np.abs(got - reference).max()), 1e-6 * float(np.abs(reference).max()))
        self.assertEqual(translate.target_kinds(spec), ("k", "csa", "hca", "lid"))
        # the engine's streaming writer hands these sections to kvstate (header included)
        written = self.fx.fresh("written").with_suffix(".state")
        state_writer.write(written, state)
        self.assertEqual(written.read_bytes(), state.to_bytes())
        for options, message in (({"fill": "table"}, "fill 'mean'"), ({"native_prefix": 64}, "no native prefix")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                translate.translate(dsv4_target.CaptureRows(captures, tokens[:T]), side, side, hubs, t0=T,
                                    template=template, **options)

    def assert_same_state(self, state, expected):
        """Array for array: raw rows and positions, pooled rows, ring a and score."""
        for mine, theirs in zip(state.state.parts, expected.state.parts):
            if isinstance(mine, kvstate.KVCache):
                np.testing.assert_array_equal(mine.stream().positions, theirs.stream().positions)
                self.assertTrue(all(np.array_equal(mine.stream().k[l], theirs.stream().k[l]) for l in mine.layers))
            elif isinstance(mine, kvstate.PooledRows):
                self.assertEqual(mine.n_rows, theirs.n_rows, mine.name)
                self.assertTrue(all(np.array_equal(mine.rows[l], theirs.rows[l]) for l in mine.layers), mine.name)
            else:
                self.assertTrue(all(np.array_equal(mine.kv[l], theirs.kv[l]) and
                                    np.array_equal(mine.score[l], theirs.score[l]) for l in mine.layers), mine.name)

    def test_translate_rounds_t0_down_to_whole_hca_blocks(self):
        # a 200-token conversation handed over at T0 = 200: the target keeps [0, 128) translated and
        # prefills [128, 200) natively (through a numpy backend, whose resident sides translate reads)
        n = 200
        tokens = self.tokens(n + NEXT)
        spec = self.spec()
        template = kvstate.read_file(self.fx.save("dsv4", tokens[:1]), spec)
        captures = self.fx.capture("dsv4", tokens[:n], spec)
        record, side_tokens = ff.byte_side(tokens[:n])
        side = translate.Side("dsv4", spec, record, side_tokens, gguf=self.fx.models["dsv4"])
        hubs = self.identity_hub("dsv4", spec)
        backend = backends.select("cpu")
        state, report = translate.translate(dsv4_target.CaptureRows(captures, tokens[:n]), side, side, hubs, t0=n,
                                            template=template, backend=backend)
        self.assertEqual((report["t0"], report["t0_requested"], report["sections"]["base"]), (128, n, 128))
        self.assertEqual([state.state.part(p.name).n_rows for p in spec.pools()], [32, 1, 32])
        tensors = dsv4_pool.pool_tensors(self.fx.models["dsv4"], spec)
        self.assert_same_state(state, dsv4_target.synthesize(spec, template, tokens, 128, inputs_of(captures, 128),
                                                             tensors))
        reference = self.fx.restore("dsv4", self.fx.save("dsv4", tokens[:128]), tokens[128:])
        got = self.fx.restore("dsv4", state, tokens[128:])
        self.assertLessEqual(float(np.abs(got - reference).max()), 1e-6 * float(np.abs(reference).max()))

    def test_qwen3_into_deepseek4_through_head_decoders(self):
        ids = ff.ascii_ids(self.rng, T, 128)
        qwen_spec, spec = kvstate.ModelSpec.from_gguf(self.qwen3), self.spec()
        source_path = self.fresh_qwen_state(ids)
        source_state = kvstate.read_file(source_path, qwen_spec)
        record, side_tokens = ff.byte_side(ids)
        source = translate.Side("qwen3", qwen_spec, record, side_tokens)
        target = translate.Side("dsv4", spec, record, side_tokens, gguf=self.fx.models["dsv4"])
        widths = {"qwen3": {kind: {l: int(np.prod(qwen_spec.head_shape(l, kind))) for l in range(4)}
                            for kind in hub_module.KINDS},
                  "dsv4": {kind: {l: 64 for l in range(4)} for kind in hub_module.KINDS}}
        hubs = random_hub(widths, dim=12, seed=3)
        rng = np.random.default_rng(7)
        heads = []
        for pool in spec.pools():
            layers = list(pool.layers)
            heads.append(hub_module.HeadKind(
                pool.name, "k", ["dsv4"], {"dsv4": layers},
                {"dsv4": [rng.normal(0, 0.02, 2 * pool.state_width).astype(np.float32) for _ in layers]},
                {"dsv4": rng.normal(size=12).astype(np.float32)},
                {"dsv4": [rng.normal(0, 0.01, (12, 2 * pool.state_width)).astype(np.float32) for _ in layers]}))
        hubs = hub_module.with_heads(hubs, heads)
        template = kvstate.read_file(self.fx.save("dsv4", ids[:1]), spec)
        state, report = translate.translate(source_state, source, target, hubs, t0=290, template=template)
        self.assertEqual((report["t0"], report["mapped_cells"], report["fill_cells"]), (256, 256, 0))
        data = state.to_bytes()
        self.assertEqual(kvstate.parse_file(data, spec).to_bytes(), data)
        logits = self.fx.restore("dsv4", state, ids[256:])
        self.assertTrue(np.isfinite(logits).all())
        self.assertEqual(logits.shape, (T - 256, 128))
        # the engine's preload resolves head kinds to the source's K encoder
        self.assertEqual(hubs["csa"].encoder_side("qwen3").layers, hubs["k"].encoder_side("qwen3").layers)
        with self.assertRaisesRegex(ValueError, "at least 128 tokens"):
            translate.translate(source_state, source, target, hubs, t0=100, template=template)
        with self.assertRaisesRegex(ValueError, "no csa head decoders"):
            translate.translate(source_state, source, target, {k: v for k, v in hubs.items() if k != "csa"}, t0=256,
                                template=template)

    def fresh_qwen_state(self, ids):
        path = self.fx.fresh("qwen3").with_suffix(".state")
        result = subprocess.run([DRIVER, "--mode", "save", "--model", str(self.qwen3), "--tokens",
                                 ",".join(map(str, ids)), "--state", str(path), "--ctx", "512", "--threads", "4"],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-800:])
        return path


class SlotServer:
    """Just enough of a llama-server for kvpark.bridge.switch: chat.render as /apply-template, a byte
    /tokenize, slot saves that copy a prepared slot file (`saved`, with an empty .resume when
    `patched`, as a kvpark-patched server without a draft writes it), restores that read the file's
    token count (and keep a copy, `restored`), completions that report the restored cells reused."""

    def __init__(self, slot_dir, saved, *, template, patched=True):
        self.slot_dir, self.saved, self.patched = Path(slot_dir), Path(saved), patched
        self.resident, self.restored = 0, None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload, status=200):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props":
                    return self.reply({"build_info": "fixture", "total_slots": 1, "modalities": {}})
                return self.reply([{"id": 0, "speculative": False}] if self.path == "/slots" else {}, 200)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/apply-template":
                    conversation = {"messages": body["messages"], "tools": body.get("tools")}
                    return self.reply({"prompt": chat.render(conversation, template, add_generation_prompt=True,
                                                             live=True).text})
                if self.path == "/tokenize":
                    return self.reply({"tokens": [{"id": b, "piece": chr(b)} for b in body["content"].encode()]})
                if self.path == "/completion":
                    cached = min(server.resident, len(body["prompt"]) - 1)
                    server.resident = len(body["prompt"]) + body.get("n_predict", 0)
                    return self.reply({"tokens": [1], "timings": {"cache_n": cached,
                                                                  "prompt_n": len(body["prompt"]) - cached}})
                action = self.path.split("action=", 1)[-1]
                path = server.slot_dir / body.get("filename", "")
                if action == "erase":
                    server.resident = 0
                    return self.reply({"id_slot": 0})
                if action == "save":
                    shutil.copyfile(server.saved, path)
                    if server.patched:
                        Path(str(path) + ".resume").write_bytes(resume.Resume().to_bytes())
                    return self.reply({"n_saved": resume.slot_token_count(path), "n_written": path.stat().st_size})
                if action == "restore":
                    if server.patched and resume.parse_resume(Path(str(path) + ".resume").read_bytes()).draft:
                        return self.reply({"error": "Slot archive requires draft context"}, 400)
                    server.restored = path.read_bytes()
                    server.resident = resume.slot_token_count(path)
                    return self.reply({"n_restored": server.resident})
                return self.reply({"error": "not served"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class EngineBridgeTests(Dsv4Case):
    """A switch into a DeepSeek V4 target end to end: engine.py one-shot and engine_daemon.py jobs (a
    qwen3 fixture source slot, a format 2 hub bundle with head decoders) and kvpark.bridge.switch,
    with T0 = P - tail not a multiple of 128: the translated part, the restore check, the .resume
    and the native tail all follow the rounded T0."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import convert_hub
        root = cls.fx.root
        copy_with_byte_vocabulary(cls.fx.models["dsv4"], root / "deepseek4-bytes.gguf")
        cls.fx.models["dsv4-chat"] = root / "deepseek4-chat.gguf"
        # (test-llama-archs' BOS and EOS ids lie beyond the byte vocabulary)
        copy_with_metadata(root / "deepseek4-bytes.gguf", cls.fx.models["dsv4-chat"],
                           values={"tokenizer.chat_template": ff.TEMPLATE, "tokenizer.ggml.bos_token_id": 1,
                                   "tokenizer.ggml.eos_token_id": 2})
        cls.qwen3 = root / "qwen3.gguf"
        create_fixture(cls.qwen3, "qwen3", chat_template=ff.TEMPLATE)
        cls.specs = {"qwen3": kvstate.ModelSpec.from_gguf(cls.qwen3),
                    "dsv4": kvstate.ModelSpec.from_gguf(cls.fx.models["dsv4-chat"])}
        # a conversation whose newest message starts past 300 tokens: T0 = P - 40 lies between 256 and 384
        words = "the tide runs high along the northern coast at dawn and the boats wait in the harbor "
        cls.messages = [{"role": "system", "content": "You answer briefly."},
                        {"role": "user", "content": words * 2}, {"role": "assistant", "content": words * 1},
                        {"role": "user", "content": "And next week?"}]
        prompt = cls.render(cls.messages, generation=True)
        history = cls.render(cls.messages[:-1], generation=False)
        cls.P, cls.tail = len(prompt), 40
        cls.requested = cls.P - cls.tail
        # (the engine's T0 also stops at the newest message, which starts after this)
        assert 256 < cls.requested < 384 and cls.requested < len(history + "<|user|>"), (cls.P, len(history))
        rng = np.random.default_rng(12)
        spec = cls.specs["qwen3"]
        ids = np.frombuffer(history.encode(), dtype=np.uint8).astype(np.int32)
        layers = spec.layers_of("full")
        k = {l: rng.standard_normal((len(ids),) + spec.head_shape(l, "k")).astype(np.float32) for l in layers}
        v = {l: rng.standard_normal((len(ids),) + spec.head_shape(l, "v")).astype(np.float32) for l in layers}
        cls.source = root / "qwen3-source.bin"
        kvstate.write_new(cls.source, kvstate.build_dense_state(spec, ids, np.arange(len(ids)), k, v,
                                                                server_tokens=True).to_bytes())
        cls.template = cls.fx.save("dsv4-chat", [ord("H")])
        # a random hub (K and V of both, the K latent of DeepSeek V4) with its head decoders, in format 2
        spec = cls.specs["dsv4"]
        widths = {"qwen3": {kind: {l: int(np.prod(cls.specs["qwen3"].head_shape(l, kind))) for l in layers}
                            for kind in hub_module.KINDS},
                  "dsv4": {kind: {l: 64 for l in spec.layers_of("sliding")} for kind in hub_module.KINDS}}
        hubs = random_hub(widths, dim=12, seed=3)
        heads = []
        for pool in spec.pools():
            heads.append(hub_module.HeadKind(
                pool.name, "k", ["dsv4"], {"dsv4": list(pool.layers)},
                {"dsv4": [rng.normal(0, 0.02, 2 * pool.state_width).astype(np.float32) for _ in pool.layers]},
                {"dsv4": rng.normal(size=12).astype(np.float32)},
                {"dsv4": [rng.normal(0, 0.01, (12, 2 * pool.state_width)).astype(np.float32) for _ in pool.layers]}))
        hubs = hub_module.with_heads(hubs, heads)
        views = [StubView("qwen3", hubs, "qwen3"), StubView("dsv4", hubs, "deepseek4")]
        hub_module.save_hub(root / "hub-v1", hubs, views, StubSpans(), {"note": "test"})
        cls.hub = root / "hub-v2"
        convert_hub.convert(root / "hub-v1", cls.hub, log=lambda line: None)

    @staticmethod
    def render(messages, generation):
        return chat.render({"messages": messages, "tools": None}, ff.TEMPLATE, add_generation_prompt=generation,
                           live=True).text

    def job(self, name):
        prompt = self.render(self.messages, generation=True)
        return {"hub": str(self.hub), "tail": self.tail, "fill": "mean", "device": "cpu",
                "cache_dir": str(self.fx.root / "profiles"),
                "conversation": {"messages": self.messages, "tools": None},
                "source": {"name": "qwen3", "gguf": str(self.qwen3), "state": str(self.source)},
                "target": {"name": "dsv4", "gguf": str(self.fx.models["dsv4-chat"]),
                           "template_state": str(self.template), "output": str(self.fx.root / f"{name}.bin"),
                           "tokens": [[b, chr(b)] for b in prompt.encode()]}}

    def check_output(self, report, output):
        self.assertEqual((report["t0"], report["t0_requested"], report["tail"], report["P"]),
                         (256, self.requested, self.P - 256, self.P))
        state = kvstate.read_file(output, self.specs["dsv4"])
        self.assertEqual(len(state.tokens), 256)
        self.assertEqual([state.state.part(pool.name).n_rows for pool in self.specs["dsv4"].pools()], [64, 2, 64])
        # it restores and continues over the native tail
        prompt = list(self.render(self.messages, generation=True).encode())
        logits = self.fx.restore("dsv4-chat", Path(output), prompt[256:])
        self.assertTrue(np.isfinite(logits).all())
        return Path(output).read_bytes()

    def test_engine_one_shot_and_daemon_jobs_keep_the_rounded_t0(self):
        import engine_daemon
        job = self.job("one-shot")
        path = self.fx.root / "job.json"
        path.write_text(json.dumps(job))
        result = subprocess.run([sys.executable, str(HERE / "engine.py"), str(path)], capture_output=True, text=True,
                                timeout=600, env={**os.environ, backends.ENV: "auto"})
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((report["backend"], report["kinds"]), ("numpy", ["k", "csa", "hca", "lid"]))
        one_shot = self.check_output(report, job["target"]["output"])
        # the daemon, on whatever device it was asked for: the target's sides stay in the numpy backend
        directory = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="d"))
        daemon = engine_daemon.Daemon(directory, device="auto")
        self.assertTrue(daemon.bind())
        with mock.patch.object(engine_daemon, "log", lambda event: None):
            thread = threading.Thread(target=daemon.serve, daemon=True)
            thread.start()
            try:
                from test_engine_daemon import request
                loaded = request(directory, {"op": "load", "hub": str(self.hub), "source": "qwen3", "target": "dsv4",
                                             "source_gguf": str(self.qwen3),
                                             "target_gguf": str(self.fx.models["dsv4-chat"]),
                                             "cache_dir": str(self.fx.root / "profiles"), "device": "auto"})
                self.assertTrue(loaded["ok"], loaded)
                self.assertEqual(loaded["backend"], "numpy")
                self.assertEqual(sorted(map(tuple, loaded["loaded"])),
                                 sorted([("k", "qwen3", "encoder")] + [(kind, "dsv4", "decoder")
                                                                       for kind in ("k", "csa", "hca", "lid")]))
                self.assertIsInstance(loaded["warm_up_s"], float)
                job = self.job("daemon")
                reply = request(directory, {"op": "translate", "job": {**job, "device": "auto"}})
                self.assertTrue(reply["ok"], reply)
                self.assertEqual(reply["report"]["engine_timings"]["hub_load_s"] < 0.5, True)   # resident sides
                self.assertEqual(self.check_output(reply["report"], job["target"]["output"]), one_shot)
                # the memory guard counts the window, the pooled rows and the rings, not a row per cell:
                # the state's bytes but its headers and token words
                spec = self.specs["dsv4"]
                estimate = engine_daemon.state_bytes(spec, 256)
                self.assertTrue(0.9 * reply["report"]["output_bytes"] < estimate <= reply["report"]["output_bytes"])
                stats = request(directory, {"op": "stats"})
                resident = {tuple(item["key"][1:]) for item in stats["backends"]["numpy"]["resident"]}
                self.assertNotIn(("csa", "qwen3", "encoder"), resident)
                # nothing of this hub on the GPU (the backends are per process: other tests may have one)
                on_gpu = (stats["backends"].get("torch") or {}).get("resident") or []
                self.assertFalse([item for item in on_gpu if item["key"][0] == str(self.hub.resolve())])
            finally:
                daemon.stopping.set()
                thread.join(10)

    def test_a_switch_restores_and_resumes_the_rounded_t0(self):
        python = bridge_python()
        if python is None:
            self.skipTest("no python3 with psutil for kvpark.bridge")
        root = Path(tempfile.mkdtemp(dir=self.fx.root, prefix="s"))
        slots = {name: root / f"slots-{name}" for name in ("qwen3", "dsv4")}
        for directory in slots.values():
            directory.mkdir()
        servers = {"qwen3": SlotServer(slots["qwen3"], self.source, template=ff.TEMPLATE, patched=False),
                   "dsv4": SlotServer(slots["dsv4"], self.template, template=ff.TEMPLATE)}
        try:
            for mode in ("oneshot", "daemon"):
                with self.subTest(mode=mode):
                    models = (("qwen3", self.qwen3), ("dsv4", self.fx.models["dsv4-chat"]))
                    args = {"models": {name: {"url": servers[name].url, "gguf": str(gguf),
                                              "slot_dir": str(slots[name])} for name, gguf in models},
                            "engine": {"python": sys.executable, "script": str(HERE / "engine.py"),
                                       "hub": str(self.hub), "device": "cpu", "mode": mode},
                            "settings": {"tail": self.tail, "min_handover": 0},
                            "config": str(root / mode / "bridge.json"), "conversation": {"messages": self.messages},
                            "source": "qwen3", "target": "dsv4"}
                    result = subprocess.run([python, "-c", SWITCH, str(HERE.parents[1] / "src"), json.dumps(args)],
                                            capture_output=True, text=True, timeout=600)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    report = json.loads(result.stdout.strip().splitlines()[-1])
                    self.assertNotIn("skipped", report, report.get("note"))
                    self.assertEqual(report["engine_mode"], mode)
                    self.assertEqual((report["engine"]["t0"], report["engine"]["t0_requested"]),
                                     (256, self.requested))
                    self.assertEqual((report["restored_tokens"], report["reused_tokens"], report["reuse_ok"]),
                                     (256, 256, True))
                    # the .resume of the translated slot (no draft: an empty draft blob), written for 256 tokens
                    self.assertEqual((report["resume_sidecar"], report["draft"]["kind"],
                                      report["draft"]["blob_bytes"]), (True, "none", 0))
                    restored = kvstate.parse_file(servers["dsv4"].restored, self.specs["dsv4"])
                    self.assertEqual(len(restored.tokens), 256)
                    self.assertEqual(sorted(p.name for p in slots["dsv4"].iterdir()), [])     # nothing left
        finally:
            for server in servers.values():
                server.close()


@unittest.skipUnless(ff.PROBE, "set KVPARK_KV_PROBE (with the deepseek4 row) for the fitted DeepSeek V4 heads")
class HubFitTests(Dsv4Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        root = cls.fx.root
        cls.fx.models["dsv4-bytes"] = copy_with_byte_vocabulary(cls.fx.models["dsv4"], root / "deepseek4-bytes.gguf")
        if not ff.probe_supports(cls.fx.models["dsv4-bytes"], root):
            cls.temp.cleanup()
            raise unittest.SkipTest("this kvpark-kv-probe predates the deepseek4 row; rebuild it")
        cls.qwen3 = root / "qwen3.gguf"
        create_fixture(cls.qwen3, "qwen3", chat_template=ff.TEMPLATE)
        conversations = ff.conversations(16, seed=5, splits=["train"] * 12 + ["dev"] * 2 + ["test"] * 2)
        records = ff.rendered_records(conversations)
        (root / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        arguments = []
        for name, model in (("qwen3", cls.qwen3), ("dsv4", cls.fx.models["dsv4-bytes"])):
            capture = ff.probe_collect(model, records, root / f"{name}-capture")
            arguments += ["--model", f"{name}={capture}@{root / 'records.jsonl'}"]
        cls.bundle = root / "hub"
        with open(os.devnull, "w") as quiet:
            stdout, sys.stdout = sys.stdout, quiet
            try:
                fit_hub.main(arguments + ["--output", str(cls.bundle), "--hub-dim", "12", "--view-dim", "24",
                                          "--layer-rank", "16", "--heads", "dsv4"])
            finally:
                sys.stdout = stdout
        cls.record = next(r for r in records if r["split"] == "test")

    def test_fitted_heads_translate_qwen3_into_deepseek4(self):
        manifest = hub_module.read_manifest(self.bundle)
        self.assertEqual(sorted(manifest["heads"]), ["csa", "hca", "lid"])
        self.assertEqual(manifest["heads"]["hca"]["models"]["dsv4"]["layers"], [3])
        evaluation = json.loads((self.bundle / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(evaluation["heads"]), ["dsv4.csa", "dsv4.hca", "dsv4.lid"])
        self.assertEqual(sorted(evaluation["heads"]["dsv4.csa"]["test_r2"]), ["qwen3"])
        ids = list(self.record["text"].encode("ascii"))
        self.assertGreater(len(ids), 140)
        tokens = {"token_id": np.asarray(ids, dtype=np.int32), "byte_end": np.arange(1, len(ids) + 1),
                  "special": np.zeros(len(ids), dtype=np.uint8)}
        qwen_spec, spec = kvstate.ModelSpec.from_gguf(self.qwen3), self.spec("dsv4-bytes")
        source_path = self.fx.fresh("qwen3").with_suffix(".state")
        result = subprocess.run([DRIVER, "--mode", "save", "--model", str(self.qwen3), "--tokens",
                                 ",".join(map(str, ids)), "--state", str(source_path), "--ctx", "512"],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        source = translate.Side("qwen3", qwen_spec, self.record, tokens)
        target = translate.Side("dsv4", spec, self.record, tokens, gguf=self.fx.models["dsv4-bytes"])
        template = kvstate.read_file(self.fx.save("dsv4-bytes", ids[:1]), spec)
        for hub in (hub_module.open_hub(self.bundle), hub_module.load_hub(self.bundle)[0]):
            with self.subTest(hub=type(hub).__name__):
                state, report = translate.translate(kvstate.read_file(source_path, qwen_spec), source, target, hub,
                                                    t0=len(ids) - 8, template=template, server_tokens=False)
                # the template's tag text sits outside the message segments: those cells take the decoder means
                self.assertEqual((report["t0"], report["mapped_cells"] + report["fill_cells"]), (128, 128))
                self.assertGreater(report["mapped_cells"], 64)
                logits = self.fx.restore("dsv4-bytes", state, ids[128:])
                self.assertEqual(logits.shape, (len(ids) - 128, 128))
                self.assertTrue(np.isfinite(logits).all())


class HeadFitTests(unittest.TestCase):
    """hub.fit_head on synthetic spans and the head bundles (no llama.cpp)."""

    def test_fit_head_recovers_a_linear_head_and_round_trips_in_bundles(self):
        rng = np.random.default_rng(11)
        n, dim = 600, 6
        latent = rng.normal(size=(n, dim)).astype(np.float32)
        # a pivot whose K rows are the latent itself, so its encoder estimate is z
        transform = hub_module.ViewTransform([0], [np.zeros(dim, np.float32)], [np.eye(dim, dtype=np.float32)],
                                             np.eye(dim, dtype=np.float32), np.ones(dim, np.float32))
        base = hub_module.HubKind(["src"], {"src": transform}, {"src": np.eye(dim, dtype=np.float32)}, np.ones(dim),
                                  {"src": [np.zeros(dim, np.float32)]}, {"src": np.zeros(dim, np.float32)},
                                  {"src": [np.eye(dim, dtype=np.float32)]}, {"src": {"alpha": 0.0}})
        weight = rng.normal(size=(dim, 10)).astype(np.float32)

        class View:
            def __init__(self, name, data):
                self.name, self.data = name, data

            def layers(self, kind):
                return [layer for key, layer in self.data if key == kind]

            def width(self, kind, layer):
                return self.data[kind, layer].shape[1]

            def rows(self, kind, layer, rows):
                return self.data[kind, layer][rows]

        class Spans:
            doc = np.zeros(n, dtype=np.int64)
            rows = {"src": np.arange(n), "dsv4": np.arange(n)}

            def select(self, split):
                return np.arange(400) if split == "train" else np.arange(400, 500) if split == "dev" else np.arange(500, n)
        pivot = View("src", {("k", 0): latent})
        target = View("dsv4", {("csa", 2): latent @ weight + 0.5})
        head = hub_module.fit_head(base, [pivot], target, Spans(), "csa", log=lambda line: None)
        decoder = head.decoder_side("dsv4")
        test = latent[500:]
        np.testing.assert_allclose(decoder.decode(test, [2]), test @ weight + 0.5, atol=1e-2)
        # held out: the test spans, per pivot (dev R^2 picked the alpha)
        self.assertEqual(sorted(head.alphas["dsv4"]["test_r2"]), ["src"])
        self.assertGreater(head.alphas["dsv4"]["test_r2"]["src"], 0.999)
        # signal variance ~6 per channel, noise 9: R^2 ~0.4
        noisy = View("dsv4", {("csa", 2): latent @ weight + rng.normal(0, 3, (n, 10)).astype(np.float32)})
        r2 = hub_module.fit_head(base, [pivot], noisy, Spans(), "csa", log=lambda line: None).alphas["dsv4"]
        self.assertTrue(0.2 < r2["test_r2"]["src"] < 0.6, r2["test_r2"])
        hubs = hub_module.with_heads({"k": base}, [head])
        self.assertIs(hubs["csa"].base_kind, base)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "bundle"
            views = [StubView("src", {"k": base, "v": base}, "qwen3"), StubView("dsv4", {"k": base, "v": base}, "deepseek4")]
            base.names.append("dsv4")
            for table in (base.transforms, base.encoders, base.decoder_mean, base.decoder_z_mean, base.decoders,
                          base.alphas):
                table["dsv4"] = table["src"]
            hub_module.save_hub(directory, {"k": base, "v": base, "csa": head}, views, StubSpans(), {})
            manifest = hub_module.read_manifest(directory)
            self.assertEqual(manifest["heads"]["csa"]["models"]["dsv4"]["layers"], [2])
            loaded, _ = hub_module.load_hub(directory)
            self.assertIsInstance(loaded["csa"], hub_module.HeadKind)
            np.testing.assert_array_equal(loaded["csa"].decoders["dsv4"][0], head.decoders["dsv4"][0])
            lazy = hub_module.open_hub(directory)
            own, side = hub_module.head_sides(lazy, "csa", "src", "dsv4")
            self.assertIsNone(own)
            np.testing.assert_array_equal(side.decode(test, [2]), decoder.decode(test, [2]))
            # the engine's preload: a head kind's encoder side is the base kind's
            np.testing.assert_array_equal(lazy.side("csa", "src", "encoder").weight, lazy.encoder_side("k", "src").weight)
            np.testing.assert_array_equal(hub_module.sides(lazy, "csa", "src", "dsv4")[1].decoders[0],
                                          head.decoders["dsv4"][0])
            with self.assertRaisesRegex(ValueError, "no csa head decoders for src"):
                lazy.side("csa", "src", "decoder")
            # a format 2 bundle keeps the head files as they are (hash-checked, listed in "files")
            import convert_hub
            packed = Path(temp) / "bundle-v2"
            convert_hub.convert(directory, packed, log=lambda line: None)
            manifest = hub_module.read_manifest(packed)
            self.assertEqual((manifest["format_version"], sorted(manifest["files"])), (2, ["dsv4.csa"]))
            self.assertEqual(manifest["heads"]["csa"]["models"]["dsv4"]["layers"], [2])
            self.assertEqual((packed / "dsv4.csa.npz").read_bytes(), (directory / "dsv4.csa.npz").read_bytes())
            loaded, _ = hub_module.load_hub(packed)
            np.testing.assert_array_equal(loaded["csa"].decoders["dsv4"][0], head.decoders["dsv4"][0])
            v2 = hub_module.open_hub(packed)
            np.testing.assert_array_equal(hub_module.head_sides(v2, "csa", "src", "dsv4")[1].decode(test, [2]),
                                          decoder.decode(test, [2]))
            np.testing.assert_array_equal(v2.side("csa", "src", "encoder").weight, v2.encoder_side("k", "src").weight)
            (packed / "dsv4.csa.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "sha256|hash|changed"):
                hub_module.open_hub(packed).head_decoder_side("csa", "dsv4")
            # ... and reads no head file (only the target's decoders are read, when asked for)
            (directory / "dsv4.csa.npz").unlink()
            np.testing.assert_array_equal(lazy.side("csa", "src", "encoder").weight, lazy.encoder_side("k", "src").weight)


class CapabilityTests(unittest.TestCase):
    def test_deepseek4_is_a_target_only(self):
        self.assertIsNone(capabilities.check(("qwen38", "qwen35"), ("v4", "deepseek4")))
        self.assertEqual(capabilities.check(("v4", "deepseek4"), ("qwen38", "qwen35"))[0], "dsv4-source")
        manifest = {"heads": {kind: {"models": {"v4": {}}} for kind in hub_module.HEAD_KINDS}}
        self.assertIsNone(capabilities.heads("v4", "deepseek4", manifest))
        self.assertEqual(capabilities.heads("other", "deepseek4", manifest)[0], "no-head-decoders")
        self.assertIsNone(capabilities.heads("qwen38", "qwen35", {}))
        # every head kind: with one missing the engine would refuse the job
        code, reason = capabilities.heads("v4", "deepseek4", {"heads": {"csa": {"models": {"v4": {}}}}})
        self.assertEqual(code, "no-head-decoders")
        self.assertIn("no hca, lid head decoders", reason)
        self.assertEqual(capabilities.DSV4_HEADS, hub_module.HEAD_KINDS)

    def test_deepseek4_targets_keep_no_draft(self):
        def window(layers):
            # f16 K and V rows of head 64 (what an iSWA cache over non-MLA hparams writes)
            rows = tuple(resume.LayerRows(1, 128, 1, 128) for _ in range(layers))
            return resume.KVSection(1, [resume.KVStream((0,), None, ((0,),))], False, 0, rows)

        def blob(stages, n=205, n_swa=32):
            return (struct.pack("<Ii", resume.IO_MAGIC, 0) + resume._kv_bytes(window(0), range(n), 0)
                    + resume._kv_bytes(window(stages), range(n - n_swa, n), 0))
        # what the kvpark-patched llama-server saved for --spec-type draft-mtp on the MTP fixture after
        # 205 tokens: 11100 bytes, an empty base and one K/V window layer; a DSpark draft (the same iSWA
        # cache over the draft's own non-MLA hparams) has one such layer per stage
        self.assertEqual(len(blob(1)), 11100)
        mtp, dspark = (resume.classify_draft(blob(stages), n_tokens=205) for stages in (1, 3))
        self.assertEqual((mtp.kind, dspark.kind), ("dflash", "dflash"))
        other = resume.DraftLayout("other", "draft memory the bridge cannot synthesize")
        # an upstream server's draft, named after its speculative types (no .resume, so no blob)
        upstream = resume.DraftLayout("mtp", "speculative types draft-mtp; the server keeps no draft state")
        self.assertIsNone(capabilities.draft("v4", "deepseek4", None))
        self.assertIsNone(capabilities.draft("v4", "deepseek4", resume.DraftLayout("none", "no draft context")))
        for layout in (mtp, dspark, other, upstream):
            with self.subTest(layout=layout.note):
                code, reason = capabilities.draft("v4", "deepseek4", layout)
                self.assertEqual(code, "draft-unsupported")
        self.assertIn("a DSpark draft or the MTP head", capabilities.draft("v4", "deepseek4", dspark)[1])
        # other targets keep the bridge's general rules
        self.assertIsNone(capabilities.draft("qwen38", "qwen35", mtp))


if __name__ == "__main__":
    unittest.main()
