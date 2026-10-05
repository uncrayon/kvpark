"""Sequence-state layouts against llama.cpp itself: the MLA, DSA and K-only hybrid architectures
(StateLayoutTests) and every recurrent family, hybrid or recurrent-only (RecurrentFamilyTests).

Needs KVPARK_STATE_DRIVER=/path/to/kvpark-state-driver (built from state_driver.cpp against a
CPU build of either pinned runtime, 85d5703a3 or a7b94df2c: the specs take the driver's revision,
tiny_fixture.driver_revision, and the layouts that differ are checked per revision).
KVPARK_LLAMA_ARCHS=/path/to/test-llama-archs (the same build with LLAMA_BUILD_TESTS=ON; its models
are written as tiny_fixture.llama_archs_model does) adds its random kimi-k3, kimi-linear, bailingmoe3, mistral4,
hy_v4, deepseek2, deepseek32 and glm-dsa models (576/512 geometry, no vocabulary: token ids
only; bailingmoe3 also with rope sections past the w run, SECTIONS_PAST_W) and its qwen3next,
qwen35moe, qwen4exp, minimax-01, jamba, falcon-h1, granitehybrid, nemotron_h(_moe), plamo2,
lfm2, lfm2moe, mamba and mamba2 models; the RWKV models come from
tiny_fixture.create_rwkv_fixture (test-llama-archs skips them). Dsv4LayoutTests checks DeepSeek
V4 (deepseek4) saves of test-llama-archs' fixture and of tiny_fixture.create_dsv4_mtp_fixture's
copy with an MTP layer; it needs a driver with --parallel/--seq/--rs-seq/--partial and q8_0
caches. Without the driver every test here is skipped.
"""

import dataclasses
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import gguf_tensors
import kvstate
from tiny_fixture import RWKV_FIXTURE_ARCHITECTURES, YARN_V2, copy_with_metadata, create_dsv4_mtp_fixture, \
    create_gemma_fixture, create_mla_fixture, create_nemotron_mtp_fixture, create_rwkv_fixture, driver_revision, \
    llama_archs_model


DRIVER = os.environ.get("KVPARK_STATE_DRIVER")
LLAMA_ARCHS = os.environ.get("KVPARK_LLAMA_ARCHS")
# gguf-py fixtures: name -> (architecture, create_mla_fixture options)
TINY = {
    "deepseek2": ("deepseek2", {}),
    "deepseek2-q0": ("deepseek2", {"q_lora": 0}),
    "deepseek2-yarn": ("deepseek2", {"yarn": YARN_V2}),
    "deepseek2-nextn": ("deepseek2", {"nextn": 1}),
    "mistral4-yarn": ("mistral4", {"yarn": YARN_V2}),
    "mistral4-yarn-m1": ("mistral4", {"yarn": {**YARN_V2, "log_multiplier": 0.1}}),
    "deepseek32": ("deepseek32", {}),
    "deepseek32-yarn": ("deepseek32", {"yarn": YARN_V2}),
    "glm-dsa": ("glm-dsa", {"layers": 4, "indexer_types": [1, 0, 1, 0]}),
    "glm-dsa-yarn": ("glm-dsa", {"layers": 4, "indexer_types": [1, 0, 1, 0],
                                 "yarn": {"factor": 4.0, "original_context_length": 64}}),
    "dots3note": ("dots3note", {"layers": 4}),
}
ARCHS = ("deepseek2", "deepseek32", "glm-dsa", "mistral4", "hy_v4", "kimi-k3", "kimi-linear", "bailingmoe3")
# rope sections of archs-bailingmoe3-sections: of its 32 k_pe pairs, sectors 13-17 (pairs 13-17 and 31)
# take the fourth component, and pairs 18-31 wrap around to t
SECTIONS_PAST_W = (4, 6, 3, 5)
# recurrent families: model -> (recurrent_row_floats family, payload sections in order: k = a KV
# section, r = the recurrent section; walk_payload checks the row sizes without kvstate's shapes, and
# reads every KV section with and without V blocks: a7b94df2c's qwen4exp indexer stores none)
RECURRENT_ARCHS = {"mamba": ("ssm", "r"), "mamba2": ("ssm", "r"), "jamba": ("ssm", "kr"), "falcon-h1": ("ssm", "kr"),
                   "granitehybrid": ("ssm", "kr"), "nemotron_h": ("ssm", "kr"), "nemotron_h_moe": ("ssm", "kr"),
                   "plamo2": ("ssm", "kr"), "qwen3next": ("ssm", "kr"), "qwen35moe": ("ssm", "kr"),
                   "qwen4exp": ("ssm", "krk"), "glm5-next": ("kda", "krk"), "lfm2": ("shortconv", "kkr"),
                   "lfm2moe": ("shortconv", "kr"), "minimax-01": ("lightning", "kr"),
                   **{a: ("rwkv", "r") for a in RWKV_FIXTURE_ARCHITECTURES}}
# the sections kvstate reads from those fixtures (flags 0): (name, layers)
FAMILY_SECTIONS = {
    "falcon-h1": [("attn", (0, 1)), ("recurrent", (0, 1))],
    "lfm2": [("base", ()), ("swa", (0,)), ("recurrent", (1,))],
    "qwen4exp": [("attn", (1,)), ("recurrent", (0,)), ("indexer", (1,))],
    "glm5-next": [("attn", (0,)), ("recurrent", (1,)), ("indexer", (0,))],
    **{a: [("attn", (0,)), ("recurrent", (1,))] for a in ("granitehybrid", "jamba", "lfm2moe", "nemotron_h",
                                                          "nemotron_h_moe", "plamo2")},
    **{a: [("attn", (1,)), ("recurrent", (0,))] for a in ("minimax-01", "qwen35moe", "qwen3next")},
    **{a: [("recurrent", (0, 1))] for a in ("mamba", "mamba2") + RWKV_FIXTURE_ARCHITECTURES}}
# the graph node holding each hybrid's attention keys right before RoPE (or the cached keys, NoPE)
PRE_ROPE_KEYS = {"qwen3next": "Kcur_normed", "qwen35moe": "Kcur_normed", "qwen4exp": "Kcur_normed",
                 "plamo2": "Kcur_normed", "minimax-01": "Kcur", "falcon-h1": "Kcur", "granitehybrid": "Kcur",
                 "jamba": "Kcur", "nemotron_h": "Kcur", "nemotron_h_moe": "Kcur",
                 "lfm2": "model.layers.{}.self_attn.k_layernorm", "lfm2moe": "model.layers.{}.self_attn.k_layernorm"}
RUNTIME_YARN = {"rope_scaling_type": "yarn", "rope_freq_scale": 0.125, "yarn_orig_ctx": 32}
RUNTIME_YARN_FLAGS = ["--rope-scaling", "yarn", "--rope-freq-scale", "0.125", "--yarn-orig-ctx", "32"]


def walk_payload(data, order, ple_layers):
    """Recurrent (R, S, P) row-size sets and n_layer of a save, skipping KV sections by their own
    row sizes; tries with and without cell ext, V blocks in each KV section or none, and every
    recurrent layer count."""
    def walk(ext, n_recurrent, with_v):
        with_v = list(with_v)
        offset = 12 + 4 * struct.unpack_from("<I", data, 8)[0]

        def take(fmt):
            # read, then advance (callers never write `offset += ... take(...)`: the augmented
            # assignment would load offset before take advances it)
            nonlocal offset
            value = struct.unpack_from(fmt, data, offset)[0]
            offset += struct.calcsize(fmt)
            return value
        rows = {"r": set(), "s": set(), "p": set()}
        n_layer = None
        for section in order:
            if section == "k":
                n_stream = take("<I")
                if not 1 <= n_stream <= 4:
                    return None
                for _ in range(n_stream):
                    cells = take("<I")
                    if not cells:
                        continue
                    for _ in range(cells):
                        take("<i")
                        if take("<I") != 1:
                            return None
                        offset += 12 * ext + 4
                    v_trans, layers = take("<I"), take("<I")
                    if v_trans > 1 or layers > 64:
                        return None
                    for _ in range(layers):
                        if take("<i") not in (0, 1):
                            return None
                        row = take("<Q")
                        offset += cells * row
                    for _ in range(layers if with_v[0] else 0):
                        if take("<i") not in (0, 1):
                            return None
                        row = take("<I") * take("<I") if v_trans else take("<Q")
                        offset += cells * row
                with_v.pop(0)
                continue
            cells = take("<I")
            for _ in range(cells):
                take("<i")
                n_seq = take("<I")
                offset += 4 * n_seq
            take("<I")
            n_layer = take("<I")
            for key in ("r", "s"):
                for index in range(n_recurrent):
                    if take("<i") != 0:
                        return None
                    row = take("<Q")
                    offset += cells * row
                    rows[key].add(row)
                    if key == "r" and index in ple_layers:
                        row = take("<Q")
                        offset += cells * row
                        rows["p"].add(row)
        return (rows, n_layer) if offset == len(data) else None

    for ext in (False, True):
        for n_recurrent in range(1, 8):
            for with_v in itertools.product((True, False), repeat=order.count("k")):
                try:
                    found = walk(ext, n_recurrent, with_v)
                except struct.error:
                    continue
                if found:
                    return found
    return None


def bits_equal(a, b):
    a, b = np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def assert_zero_recurrent_rows(test, state_file, native):
    """Every recurrent section of state_file is R0: the R, S and PLE rows of `native`'s sections
    (same layers, types and row sizes) with every byte zero, 0-byte rows included."""
    pairs = [(mine, theirs) for mine, theirs in zip(state_file.state.parts, native.state.parts)
             if isinstance(mine, kvstate.RecurrentState)]
    test.assertTrue(pairs, "no recurrent section")
    for mine, theirs in pairs:
        for kind in ("r", "s", "p"):
            rows, expected = getattr(mine, kind), getattr(theirs, kind)
            test.assertEqual({il: (r.ggml_type, r.row_bytes) for il, r in rows.items()},
                             {il: (r.ggml_type, r.row_bytes) for il, r in expected.items()}, f"{mine.name} {kind}")
            test.assertFalse(any(any(r.data) for r in rows.values()), f"{mine.name} {kind} rows are not zero")


class DriverMixin:
    """kvpark-state-driver runs on the fixtures in cls.models (name -> GGUF) with cls.vocab (name -> size);
    prompts come from cls.rng, or from the fixture's own stream in cls.rngs (name -> Generator)."""

    def workdir(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def ids(self, name, n):
        rng = getattr(self, "rngs", {}).get(name, self.rng)
        return [int(t) for t in rng.integers(0, self.vocab[name], n)]

    def drive(self, mode, name, *args, cache="f32", fa="off", extra=()):
        command = [DRIVER, "--mode", mode, "--model", str(self.models[name]), "--cache-type", cache,
                   "--flash-attn", fa, "--threads", "4", *map(str, args), *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    def save(self, name, prompt, *, next_ids=None, cache="f32", fa="off", extra=()):
        directory = self.workdir()
        args = ["--tokens", ",".join(map(str, prompt)), "--state", directory / "native.state"]
        if next_ids is not None:
            args += ["--next", ",".join(map(str, next_ids)), "--logits", directory / "native.npy"]
        self.drive("save", name, *args, cache=cache, fa=fa, extra=extra)
        return directory

    def restore(self, name, state, next_ids, *, cache="f32", fa="off", extra=()):
        logits = state.with_suffix(".npy")
        report = self.drive("restore-decode", name, "--state", state, "--next", ",".join(map(str, next_ids)),
                            "--logits", logits, cache=cache, fa=fa, extra=extra)
        return np.load(logits), report

    def capture(self, name, prompt, names, *, extra=(), state=None, cache="f32", fa="off"):
        """{node name: [(op, array), ...]} of one capture run; with `state` the run also saves
        its sequence state there (the cache rows of the captured graph)."""
        directory = self.workdir() / "capture"
        more = ["--state", state] if state is not None else []
        self.drive("capture", name, "--tokens", ",".join(map(str, prompt)), "--capture", ",".join(names),
                   "--output", directory, "--batch", 256, *more, cache=cache, fa=fa, extra=extra)
        found = {}
        for item in json.loads((directory / "manifest.json").read_text())["captures"]:
            found.setdefault(item["name"], []).append((item["op"], np.load(directory / item["file"])))
        return found

    def spec(self, name, runtime=None):
        """The model's spec on the driver's llama.cpp revision (cls.revision)."""
        return kvstate.ModelSpec.from_gguf(self.models[name], runtime=runtime, llama_revision=self.revision)

    def move(self, spec, state, offset):
        """Every section moved to positions + offset (K rows through pre_rope_keys at the old
        positions and cache_keys at the new ones, M-RoPE cells' (x, y) with them, recurrent cells
        relabelled), with `offset` filler tokens in front so the state still ends at its token count."""
        for part, layout in zip(state.state.parts, spec.layouts(state.state.flags)):
            if layout.kind != "kv":
                part.positions = (part.positions + offset).astype(np.int32)
                continue
            stream = part.stream()
            moved = stream.positions + offset
            for il in part.layers:
                rows = stream.k[il]
                pre = kvstate.pre_rope_keys(spec, il, rows, stream.positions, role=layout.role)
                stream.k[il] = kvstate.cache_keys(spec, il, pre, moved, role=layout.role).astype(rows.dtype)
            stream.positions = moved.astype(np.int32)
            if stream.ext is not None and spec.rope_type in ("imrope", "mrope"):
                stream.ext = np.asarray(stream.ext) + np.asarray([offset, offset, 0], dtype=np.int32)
        state.token_words = kvstate.token_words(np.concatenate([np.zeros(offset, np.int64), state.tokens]))

    def rerotate(self, spec, state):
        """Every K row through pre_rope_keys and back through cache_keys (in place)."""
        for part, layout in zip(state.state.parts, spec.layouts(state.state.flags)):
            if layout.kind != "kv":
                continue
            stream = part.stream()
            for il in part.layers:
                rows = stream.k[il]
                pre = kvstate.pre_rope_keys(spec, il, rows, stream.positions, role=layout.role)
                stream.k[il] = kvstate.cache_keys(spec, il, pre, stream.positions, role=layout.role).astype(rows.dtype)


@unittest.skipUnless(DRIVER, "set KVPARK_STATE_DRIVER to enable llama.cpp state-layout checks")
class StateLayoutTests(DriverMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-layouts-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.weights, cls.vocab = {}, {}, {}
        for name, (architecture, options) in TINY.items():
            path = cls.root / f"{name}.gguf"
            cls.weights[name] = create_mla_fixture(path, architecture, **options)
            cls.models[name], cls.vocab[name] = path, 256
        if LLAMA_ARCHS:
            for architecture in ARCHS:
                # these architectures only exist as MoE models in test-llama-archs
                name = f"archs-{architecture}"
                cls.models[name] = llama_archs_model(LLAMA_ARCHS, architecture, cls.root / "archs")
                cls.vocab[name] = 128
            # bailingmoe3 with rope sections that leave pairs to the fourth component (position 0 for
            # text: unrotated) and wrap around their sum, where a7b94df2c's fixture sections [8, 12, 12, 0]
            # cover every pair (text M-RoPE = NEOX); 85d5703a3 never reads them (NORM, cells without ext)
            path = cls.root / "bailingmoe3-sections.gguf"
            copy_with_metadata(cls.models["archs-bailingmoe3"], path,
                               values={"bailingmoe3.rope.dimension_sections": list(SECTIONS_PAST_W)})
            cls.models["archs-bailingmoe3-sections"], cls.vocab["archs-bailingmoe3-sections"] = path, 128
        cls.revision = driver_revision(DRIVER, cls.root / "revision")
        cls.rng = np.random.default_rng(2026)
        # the later fixture draws from its own stream, so every other one keeps the prompts the
        # tolerances were set on (see test_rerotated_states_restore_like_native)
        cls.rngs = {"archs-bailingmoe3-sections": np.random.default_rng(2027)}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_driver_options(self):
        directory = self.workdir()
        prompt = ",".join(map(str, self.ids("deepseek2", 6)))
        state = directory / "bare.state"
        self.drive("save", "deepseek2", "--tokens", prompt, "--state", state, "--no-token-list")
        parsed = kvstate.read_file(state, self.spec("deepseek2"))
        self.assertEqual((len(parsed.token_words), parsed.state.part("kv").stream().n_cells), (0, 6))
        for args, fragment in ((["--tokens", prompt, "--state", state], "output exists"),
                               (["--tokens", prompt, "--state", directory / "x", "--next", prompt, "--logits",
                                 directory / "x.npy", "--batch", "4"], "one ubatch"),
                               (["--tokens", prompt, "--state", directory / "y", "--next", prompt], "together"),
                               (["--tokens", "1,-2", "--state", directory / "z"], "invalid token id"),
                               (["--tokens", "999", "--state", directory / "z"], "outside the vocabulary")):
            with self.subTest(fragment=fragment):
                result = subprocess.run([DRIVER, "--mode", "save", "--model", str(self.models["deepseek2"]),
                                         *map(str, args)], capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(fragment, result.stderr)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["bare.state"])
        # a leftover (or planted) .partial is refused rather than written through, and kept
        planted = directory / "w.state.partial"
        planted.symlink_to(directory / "elsewhere")
        result = subprocess.run([DRIVER, "--mode", "save", "--model", str(self.models["deepseek2"]), "--tokens", prompt,
                                 "--state", str(directory / "w.state")], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("temporary file exists", result.stderr)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["bare.state", "w.state.partial"])

    # -- layouts -----------------------------------------------------------

    def test_native_saves_parse_and_write_byte_identically(self):
        # deepseek2's nextn layer: kept (zero rows) by 85d5703a3's plain cache, filtered out by a7b94df2c's
        nextn = (0, 1, 2) if self.revision == "85d5703a3" else (0, 1)
        expected = {"deepseek2": [("kv", (0, 1))], "deepseek2-nextn": [("kv", nextn)],
                    "deepseek32": [("kv", (0, 1)), ("indexer", (0, 1))],
                    "glm-dsa": [("kv", (0, 1, 2, 3)), ("indexer", (0, 2))],
                    "dots3note": [("base", (0, 2)), ("indexer", (0, 2)), ("swa", (1, 3))],
                    "archs-hy_v4": [("kv", (0, 1)), ("indexer", (0,))],
                    "archs-kimi-k3": [("attn", (0,)), ("recurrent", (1,))],
                    "archs-kimi-linear": [("attn", (0,)), ("recurrent", (1,))],
                    "archs-bailingmoe3": [("attn", (0,)), ("recurrent", (1,))],
                    "archs-bailingmoe3-sections": [("attn", (0,)), ("recurrent", (1,))]}
        for name in self.models:
            spec = self.spec(name)
            prompt = self.ids(name, 13)
            for cache in ("f16", "f32"):
                for fa in ("off", "on"):
                    with self.subTest(model=name, cache=cache, flash_attn=fa):
                        path = self.save(name, prompt, cache=cache, fa=fa) / "native.state"
                        data = path.read_bytes()
                        parsed = kvstate.parse_file(data, spec)
                        self.assertEqual(parsed.to_bytes(), data)
                        np.testing.assert_array_equal(parsed.tokens, prompt)
                        # a spec of either revision reads the save as the driver's revision where their
                        # layouts of the model differ (deepseek2-nextn, bailingmoe3 with rope sections)
                        for revision in kvstate.LLAMA_REVISIONS:
                            other = spec.for_revision(revision)
                            reading = kvstate.parse_file(data, other).state.spec
                            self.assertEqual(reading, spec if len(spec.layout_variants()) > 1 else other)
                        if name in expected:
                            self.assertEqual([(p.name, p.layers) for p in parsed.state.parts], expected[name])
                        for part, layout in zip(parsed.state.parts, spec.layouts(0)):
                            if layout.kind != "kv":
                                self.assertEqual((part.positions.tolist(), part.n_layer), ([12], spec.n_layer_trunk))
                                continue
                            stream = part.stream()
                            # v_trans = !flash_attn even though an MLA cache stores no V at all
                            self.assertEqual((stream.v_trans, stream.v), (fa == "off", {}))
                            self.assertEqual(sorted(stream.k_types.values())[0], kvstate.DTYPE_NAMES[cache])
                            last = 13 if not layout.sliding else spec.n_swa
                            np.testing.assert_array_equal(stream.positions, np.arange(13 - last, 13))
                            for il in part.layers:
                                if spec.layer_types[il] == "nextn":
                                    self.assertFalse(stream.k[il].any(), "the trunk never writes nextn rows")
                                else:
                                    self.assertTrue(stream.k[il].any())

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the K-only hybrid fixtures")
    def test_converter_value_length_plays_no_part(self):
        # conversion/kimi_linear.py and bailingmoe3.py leave the base class's value_length (the
        # KDA head_dim, or nothing); the MLA cache stores no V, so llama.cpp saves the same bytes
        for name, edit in (("archs-kimi-linear", {"remove": ("kimi-linear.attention.value_length",)}),
                           ("archs-bailingmoe3", {"values": {"bailingmoe3.attention.value_length": 128}})):
            with self.subTest(model=name):
                prompt, next_ids = self.ids(name, 9), self.ids(name, 2)
                copy = self.workdir() / "converted.gguf"
                copy_with_metadata(self.models[name], copy, **edit)
                self.models[name + "-converted"], self.vocab[name + "-converted"] = copy, self.vocab[name]
                self.addCleanup(self.models.pop, name + "-converted")
                spec = self.spec(name + "-converted")
                self.assertEqual(spec, self.spec(name))
                native = self.save(name, prompt, next_ids=next_ids)
                converted = self.save(name + "-converted", prompt, next_ids=next_ids)
                data = (converted / "native.state").read_bytes()
                self.assertEqual(data, (native / "native.state").read_bytes())
                self.assertEqual(kvstate.parse_file(data, spec).to_bytes(), data)
                np.testing.assert_array_equal(np.load(converted / "native.npy"), np.load(native / "native.npy"))

    def test_row_size_invariants(self):
        f16 = {}
        for name in ("deepseek2", "dots3note", "deepseek32"):
            f16[name] = kvstate.read_file(self.save(name, self.ids(name, 9), cache="f16") / "native.state",
                                          self.spec(name))
        # [c_kv 32 | k_pe 16] rows; the dots3note sliding layers have a 24-wide latent
        self.assertEqual(f16["deepseek2"].state.part("kv").stream().k[1].shape, (9, 1, 48))
        dots = f16["dots3note"].state
        self.assertEqual((dots.part("base").stream().k[2].shape, dots.part("swa").stream().k[3].shape,
                          dots.part("indexer").stream().k[0].shape), ((9, 1, 48), (8, 1, 40), (9, 1, 64)))
        # LayerNorm (unit weight, zero bias), rotation and H keep every indexer key at norm sqrt(64)
        keys = f16["deepseek32"].state.part("indexer").stream().k[1].astype(np.float32)
        np.testing.assert_allclose(np.linalg.norm(keys, axis=-1), 8.0, rtol=2e-3)
        if not LLAMA_ARCHS:
            return
        for name, recurrent in (("archs-kimi-linear", (4 * 3 * 3 * 1 * 128, 4 * 128 * 128)),
                                ("archs-kimi-k3", (4 * 3 * 3 * 1 * 128, 4 * 128 * 128)),
                                ("archs-bailingmoe3", (4 * 3 * 3 * 1 * 128, 4 * 128 * 128))):
            with self.subTest(model=name):
                spec = self.spec(name)
                self.assertEqual(spec.recurrent_row_bytes, recurrent)
                state = kvstate.read_file(self.save(name, self.ids(name, 5)) / "native.state", spec).state
                self.assertEqual(state.part("attn").stream().k[0].shape, (5, 1, 576))
                self.assertEqual(state.part("recurrent").s[1].row_bytes, recurrent[1])

    # -- RoPE --------------------------------------------------------------

    def check_mla_rope(self, name, *, runtime=None, extra=()):
        spec = self.spec(name, runtime)
        prompt = self.ids(name, 70)
        positions = np.arange(len(prompt))
        found = self.capture(name, prompt, ["Kcur", "k_pe", "indexer_k"], extra=extra)
        checked = 0
        for il in spec.layers_of("full", "sliding"):
            kcur = [a for op, a in found.get(f"Kcur-{il}", []) if op == "CONCAT"]
            if not kcur:
                continue
            row = kcur[0].reshape(len(prompt), 1, -1)
            ops = [op for op, _ in found[f"k_pe-{il}"]]
            pre_pe = found[f"k_pe-{il}"][ops.index("ROPE") - 1][1].reshape(len(prompt), 1, -1)
            pre = np.concatenate([row[..., :spec.kv_lora[il]], pre_pe], axis=-1)
            self.assertTrue(bits_equal(kvstate.cache_keys(spec, il, pre, positions), row), f"{name} layer {il}")
            np.testing.assert_allclose(kvstate.pre_rope_keys(spec, il, row, positions), pre, atol=5e-7)
            checked += 1
        for il in spec.indexer.layers if spec.indexer else ():
            nodes = found[f"indexer_k-{il}"]
            ops = [op for op, _ in nodes]
            at = ops.index("ROPE")
            pre, roped = (nodes[at + d][1].reshape(len(prompt), 1, -1) for d in (-1, 0))
            self.assertTrue(bits_equal(spec.rope(il, "indexer").apply(pre, positions), roped), f"{name} indexer {il}")
            # H is applied with a float32 mul_mat: equal within summation-order rounding
            stored = nodes[at + 1][1].reshape(len(prompt), 1, -1)
            np.testing.assert_allclose(kvstate.cache_keys(spec, il, pre, positions, role="indexer"), stored, atol=1e-6)
            np.testing.assert_allclose(kvstate.pre_rope_keys(spec, il, stored, positions, role="indexer"), pre,
                                       atol=1e-6)
            checked += 1
        return checked

    def test_mla_rope_matches_cpu_captures_bit_for_bit(self):
        for name in ("deepseek2", "deepseek2-q0", "deepseek2-yarn", "mistral4-yarn", "mistral4-yarn-m1",
                     "deepseek32", "deepseek32-yarn", "glm-dsa", "glm-dsa-yarn", "dots3note"):
            with self.subTest(model=name):
                self.assertGreaterEqual(self.check_mla_rope(name), 2)
        # the receiving server's --rope-scaling/--rope-freq-scale/--yarn-orig-ctx, as a spec runtime
        for name in ("deepseek2", "dots3note"):
            with self.subTest(model=name, runtime="yarn"):
                spec = self.spec(name, RUNTIME_YARN)
                self.assertEqual((spec.rope_ext_factor, spec.rope_freq_scale), (1.0, 0.125))
                self.assertGreaterEqual(self.check_mla_rope(name, runtime=RUNTIME_YARN, extra=RUNTIME_YARN_FLAGS), 2)
        for name in ("archs-deepseek2", "archs-mistral4", "archs-deepseek32", "archs-glm-dsa"):
            if name in self.models:
                with self.subTest(model=name):
                    self.assertGreaterEqual(self.check_mla_rope(name), 2)

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the hy_v4 fixture")
    def test_hy_v4_indexer_tail_rope_matches_the_cpu_build(self):
        # [nope | rope] indexer keys (ggml_rope_set_offset), NORM pairs, no Hadamard; the LayerNorm
        # output is norm_w + bias (the bias add is unnamed)
        name = "archs-hy_v4"
        for runtime, extra in ((None, ()), (RUNTIME_YARN, RUNTIME_YARN_FLAGS)):
            with self.subTest(runtime=runtime):
                spec = self.spec(name, runtime)
                prompt = self.ids(name, 40)
                found = self.capture(name, prompt, ["norm_w-0", "indexer_k-0"], extra=extra)
                bias = gguf_tensors.read_tensor(self.models[name], "blk.0.indexer.k_norm.bias")
                pre = (found["norm_w-0"][0][1] + bias.astype(np.float32)).astype(np.float32).reshape(len(prompt), 1, -1)
                stored = [a for op, a in found["indexer_k-0"] if op == "ROPE"][0].reshape(len(prompt), 1, -1)
                got = kvstate.cache_keys(spec, 0, pre, np.arange(len(prompt)), role="indexer")
                self.assertTrue(bits_equal(got, stored))
                np.testing.assert_array_equal(got[..., :64], pre[..., :64])

    @unittest.skipUnless(LLAMA_ARCHS, "set KVPARK_LLAMA_ARCHS for the K-only hybrid and hy_v4 fixtures")
    def test_k_pe_rope_from_the_latent_projection(self):
        # bailingmoe3 and hy_v4 name no k_pe node before RoPE and the NoPE kimi graphs none at
        # all, so the pre-RoPE tail is rebuilt from the attn_norm capture and the GGUF's
        # kv_a_mqa weight ([c_kv | k_pe] = W x before c_kv's RMSNorm) and rotated by cache_keys;
        # it must match the stored rows up to the float32 matmul rounding (restoring re-rotated
        # rows cannot see a wrong RoPE: pre_rope_keys and cache_keys share it)
        cases = (("archs-bailingmoe3", None, ()), ("archs-bailingmoe3", RUNTIME_YARN, RUNTIME_YARN_FLAGS),
                 ("archs-bailingmoe3-sections", None, ()),
                 ("archs-bailingmoe3-sections", RUNTIME_YARN, RUNTIME_YARN_FLAGS),
                 ("archs-hy_v4", None, ()), ("archs-hy_v4", RUNTIME_YARN, RUNTIME_YARN_FLAGS),
                 ("archs-kimi-k3", None, ()), ("archs-kimi-linear", None, ()))
        for name, runtime, extra in cases:
            spec = self.spec(name, runtime)
            prompt = self.ids(name, 24)
            positions = np.arange(len(prompt))
            found = self.capture(name, prompt, ["attn_norm"], extra=extra)
            state = kvstate.read_file(self.save(name, prompt, extra=extra) / "native.state", spec)
            stream = state.state.parts[0].stream()
            for il in spec.layers_of("full"):
                with self.subTest(model=name, runtime=runtime, layer=il):
                    lora, rows = spec.kv_lora[il], stream.k[il]
                    x = found[f"attn_norm-{il}"][0][1].reshape(len(prompt), -1).astype(np.float64)
                    w = gguf_tensors.read_tensor(self.models[name], f"blk.{il}.attn_kv_a_mqa.weight").astype(np.float64)
                    k_pe = (x @ w.T)[:, None, lora:].astype(np.float32)
                    pre = np.concatenate([rows[..., :lora], k_pe], axis=-1)
                    got = kvstate.cache_keys(spec, il, pre, positions)
                    scale = float(np.abs(rows[..., lora:]).max())
                    np.testing.assert_allclose(got[..., lora:], rows[..., lora:], rtol=0, atol=1e-5 * scale)
                    np.testing.assert_allclose(kvstate.pre_rope_keys(spec, il, rows, positions)[..., lora:], k_pe,
                                               rtol=0, atol=1e-5 * scale)
                    rope = spec.rope(il)
                    if rope is None:
                        self.assertTrue(name.startswith("archs-kimi"))
                        continue
                    # the check separates parameters: a 7x base misses by orders of magnitude more
                    wrong = dataclasses.replace(rope, base=rope.base * 7).apply(pre, positions)
                    self.assertGreater(np.abs(wrong[..., lora:] - rows[..., lora:]).max(), 1e-2 * scale)
                    if not name.startswith("archs-bailingmoe3"):
                        continue
                    sections = SECTIONS_PAST_W if name.endswith("-sections") else (8, 12, 12, 0)
                    self.assertEqual((rope.mode, rope.sections),
                                     ("norm", None) if self.revision == "85d5703a3" else ("mrope", sections))
                    if rope.mode == "mrope" and 3 in kvstate.rope_components(rope.n_dims // 2, rope.sections,
                                                                              interleaved=False):
                        # and the unrotated fourth-component pairs: NEOX over every pair misses
                        neox = dataclasses.replace(rope, mode="neox", sections=None).apply(pre, positions)
                        self.assertGreater(np.abs(neox[..., lora:] - rows[..., lora:]).max(), 1e-2 * scale)

    # -- restore equivalence -------------------------------------------------

    def test_rerotated_states_restore_like_native(self):
        # with flash attention on, llama.cpp casts an f32 cache to f16 for the kernel
        # (src/llama-graph.cpp:2655-2658), so a re-rotation that moves a key by one f32 ulp across an
        # f16 rounding boundary moves the logits by about 1e-5 of their scale: rare (none in 20
        # random dots3note prompts), but one prompt drawn from this class's stream in another
        # order missed by 1.85e-5 on both pins (base section only; indexer and SWA exact)
        names = ["deepseek2", "deepseek2-q0", "deepseek2-yarn", "deepseek2-nextn", "mistral4-yarn",
                 "deepseek32-yarn", "glm-dsa-yarn", "dots3note"]
        names += [n for n in ("archs-kimi-k3", "archs-kimi-linear", "archs-bailingmoe3", "archs-bailingmoe3-sections",
                              "archs-hy_v4", "archs-mistral4") if n in self.models]
        for name in names:
            spec = self.spec(name)
            # 40 cells: past the indexer top_k (8) and the dots3note window (8)
            prompt, next_ids = self.ids(name, 40), self.ids(name, 4)
            for cache in ("f32", "f16"):
                for fa in ("off", "on"):
                    with self.subTest(model=name, cache=cache, flash_attn=fa):
                        directory = self.save(name, prompt, next_ids=next_ids, cache=cache, fa=fa)
                        native = np.load(directory / "native.npy")
                        state = kvstate.read_file(directory / "native.state", spec)
                        self.rerotate(spec, state)
                        kvstate.write_file(directory / "rerotated.state", state)
                        logits, report = self.restore(name, directory / "rerotated.state", next_ids, cache=cache, fa=fa)
                        self.assertEqual(report["next_pos"], 40)
                        scale = max(1.0, float(np.abs(native).max()))
                        np.testing.assert_allclose(logits, native, rtol=0, atol=2e-6 * scale)

    def test_builders_and_runtime_yarn_restore_like_native(self):
        # build_dense_state from pre-RoPE rows (nextn zero-filled, indexer keys re-rotated) and a
        # spec carrying the receiving context's YaRN settings
        cases = (("deepseek2-nextn", None, ()), ("deepseek32-yarn", None, ()),
                 ("deepseek2", RUNTIME_YARN, RUNTIME_YARN_FLAGS))
        for name, runtime, extra in cases:
            with self.subTest(model=name, runtime=runtime):
                spec = self.spec(name, runtime)
                prompt, next_ids = self.ids(name, 30), self.ids(name, 3)
                directory = self.save(name, prompt, next_ids=next_ids, extra=extra)
                native = np.load(directory / "native.npy")
                state = kvstate.read_file(directory / "native.state", spec)
                positions = np.arange(30)
                keys, indexer = {}, None
                stream = state.state.part("kv").stream()
                for il in spec.layers_of("full"):
                    keys[il] = kvstate.pre_rope_keys(spec, il, stream.k[il], positions)
                if spec.indexer is not None:
                    ix = state.state.part("indexer").stream()
                    indexer = {il: kvstate.pre_rope_keys(spec, il, ix.k[il], positions, role="indexer")
                               for il in spec.indexer.layers}
                # flash attention off: the receiving cache has v_trans set, even with no V stored
                built = kvstate.build_dense_state(spec, prompt, positions, keys, indexer_k=indexer, dtype="f32",
                                                  v_trans=True)
                kvstate.write_file(directory / "built.state", built)
                logits, _ = self.restore(name, directory / "built.state", next_ids, extra=extra)
                np.testing.assert_allclose(logits, native, rtol=0, atol=2e-6 * max(1.0, float(np.abs(native).max())))

    def test_states_moved_to_other_positions_restore_like_native(self):
        # RoPE attention depends on relative positions only, so keys re-rotated to positions + 37
        # restore to the native continuation (decoded from 37 further on) up to rounding, while a
        # wrong rotation does not: unlike a same-position re-rotation this checks the parameters
        # (a 7x base misses by 3e-2 to 0.8 of the logit scale on these fixtures). f16 caches stay
        # at or below the indexer top_k: f16 rounding of moved indexer keys can flip a top-k pick.
        cases = [("deepseek2", None, ()), ("deepseek2-yarn", None, ()), ("mistral4-yarn", None, ()),
                 ("deepseek32-yarn", None, ()), ("glm-dsa-yarn", None, ()), ("dots3note", None, ()),
                 ("deepseek2", RUNTIME_YARN, RUNTIME_YARN_FLAGS)]
        # hybrids: the recurrent cell moves too (their test-llama-archs logits barely depend on
        # attention, so these check the plumbing; test_k_pe_rope_from_the_latent_projection the RoPE)
        cases += [(n, None, ()) for n in ("archs-bailingmoe3", "archs-bailingmoe3-sections", "archs-kimi-linear",
                                          "archs-hy_v4") if n in self.models]
        offset = 37
        for name, runtime, extra in cases:
            spec = self.spec(name, runtime)
            for n_prompt, cache, fa, tolerance in ((30, "f32", "off", 2e-6), (6, "f16", "on", 2e-3)):
                with self.subTest(model=name, runtime=runtime, cells=n_prompt, cache=cache, flash_attn=fa):
                    prompt, next_ids = self.ids(name, n_prompt), self.ids(name, 3)
                    directory = self.save(name, prompt, next_ids=next_ids, cache=cache, fa=fa, extra=extra)
                    native = np.load(directory / "native.npy")
                    state = kvstate.read_file(directory / "native.state", spec)
                    self.move(spec, state, offset)
                    kvstate.write_file(directory / "moved.state", state)
                    logits, report = self.restore(name, directory / "moved.state", next_ids, cache=cache, fa=fa,
                                                  extra=extra)
                    self.assertEqual(report["next_pos"], n_prompt + offset)
                    np.testing.assert_allclose(logits, native, rtol=0, atol=tolerance * max(1.0, float(np.abs(native).max())))

    def test_dsa_indexer_keys_matter_only_above_top_k(self):
        names = ["deepseek32", "glm-dsa", "dots3note"] + [n for n in ("archs-hy_v4",) if n in self.models]
        for name in names:
            spec = self.spec(name)
            self.assertEqual(spec.indexer.top_k, 8)
            for n_prompt in (5, 6, 30):
                with self.subTest(model=name, cells=n_prompt + 2):
                    prompt, next_ids = self.ids(name, n_prompt), self.ids(name, 2)
                    directory = self.save(name, prompt, next_ids=next_ids)
                    native = np.load(directory / "native.npy")
                    state = kvstate.read_file(directory / "native.state", spec)
                    stream = state.state.part("indexer").stream()
                    for il in stream.k:
                        stream.k[il] = np.zeros_like(stream.k[il])
                    kvstate.write_file(directory / "zeroed.state", state)
                    logits, _ = self.restore(name, directory / "zeroed.state", next_ids)
                    if n_prompt + 2 <= spec.indexer.top_k:
                        # every cell is selected whatever the scores: bit-identical outputs
                        np.testing.assert_array_equal(logits, native)
                    else:
                        self.assertGreater(np.abs(logits - native).max(), 1e-3 * np.abs(native).max())

    # -- MLA helpers on native rows --------------------------------------------

    def test_latent_rms_shell_and_k_pe_rerotation(self):
        for name in ("deepseek2", "dots3note", "deepseek2-yarn"):
            spec = self.spec(name)
            prompt, next_ids = self.ids(name, 24), self.ids(name, 3)
            directory = self.save(name, prompt, next_ids=next_ids)
            native = np.load(directory / "native.npy")
            state = kvstate.read_file(directory / "native.state", spec)
            for part in state.state.parts:
                if part.name == "indexer":
                    continue
                stream = part.stream()
                positions = stream.positions
                for il in part.layers:
                    with self.subTest(model=name, layer=il):
                        g = self.weights[name][f"blk.{il}.attn_kv_a_norm.weight"]
                        c_kv, k_pe = kvstate.mla_split(spec, il, stream.k[il])
                        # cached latents sit on the RMSNorm shell: RMS(c_kv / g) = 1
                        np.testing.assert_allclose(np.sqrt(np.mean((c_kv / g) ** 2, axis=-1)), 1.0, rtol=1e-5)
                        np.testing.assert_allclose(kvstate.mla_renormalize_latent(c_kv, g), c_kv, rtol=1e-5, atol=1e-6)
                        # a decoder that lands off the shell (scale and noise) is put back on it
                        noisy = 3.0 * c_kv + 1e-4 * self.rng.normal(size=c_kv.shape)
                        fixed = kvstate.mla_renormalize_latent(noisy, g)
                        np.testing.assert_allclose(fixed, c_kv, rtol=0, atol=2e-4)
                        # k_pe moved to other positions and back is the same row
                        pre = kvstate.pre_rope_keys(spec, il, stream.k[il], positions)
                        moved = kvstate.cache_keys(spec, il, pre, positions + 1000)
                        back = kvstate.pre_rope_keys(spec, il, moved, positions + 1000)
                        np.testing.assert_allclose(back, pre, atol=1e-5)
                        np.testing.assert_array_equal(moved[..., :spec.kv_lora[il]], c_kv)
                        stream.k[il] = np.concatenate([fixed, k_pe], axis=-1).astype(np.float32)
            kvstate.write_file(directory / "renormalized.state", state)
            logits, _ = self.restore(name, directory / "renormalized.state", next_ids)
            with self.subTest(model=name, check="renormalized latents restore"):
                np.testing.assert_allclose(logits, native, rtol=0, atol=2e-3)



def driver_has(*options):
    """Whether the state driver's usage lists every option (older builds lack the DeepSeek V4 ones)."""
    if not DRIVER:
        return False
    usage = subprocess.run([DRIVER, "--help"], capture_output=True, text=True, timeout=60).stdout
    return all(option in usage for option in options)


# test-llama-archs' deepseek4 model: layers 0-1 window only, 2 CSA (+ LID), 3 HCA; window 32
DSV4_RUNTIME_YARN = {"rope_scaling_type": "yarn", "rope_freq_scale": 0.25, "yarn_orig_ctx": 64}
DSV4_RUNTIME_YARN_FLAGS = ["--rope-scaling", "yarn", "--rope-freq-scale", "0.25", "--yarn-orig-ctx", "64"]


@unittest.skipUnless(DRIVER and LLAMA_ARCHS, "set KVPARK_STATE_DRIVER and KVPARK_LLAMA_ARCHS for DeepSeek V4 saves")
@unittest.skipUnless(driver_has("--parallel", "--partial", "q8_0"), "the state driver predates the DeepSeek V4 options")
class Dsv4LayoutTests(DriverMixin, unittest.TestCase):
    """DeepSeek V4 states (kvstate.CompressSpec) of test-llama-archs' random deepseek4 model: byte-exact
    codecs for full saves and PARTIAL_ONLY blobs, every cache layout (f16/f32/q8_0, flash attention,
    -np, slot, rollback snapshots, an MTP layer), and what the rows hold (RoPE'd latents, rings)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-dsv4-")
        cls.root = Path(cls.temp.name)
        cls.models = {"dsv4": llama_archs_model(LLAMA_ARCHS, "deepseek4", cls.root / "archs"),
                      "dsv4-mtp": cls.root / "deepseek4-mtp.gguf"}
        cls.revision = driver_revision(DRIVER, cls.root / "revision")
        create_dsv4_mtp_fixture(cls.models["dsv4"], cls.models["dsv4-mtp"])
        cls.vocab = {"dsv4": 128, "dsv4-mtp": 128}
        cls.rng = np.random.default_rng(2028)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def save_both(self, name, prompt, *, cache="f16", fa="off", extra=()):
        """(full save bytes, PARTIAL_ONLY blob bytes) of one decode of `prompt`."""
        directory = self.workdir()
        self.drive("save", name, "--tokens", ",".join(map(str, prompt)), "--state", directory / "native.state",
                   "--partial", directory / "native.partial", "--ctx", 1024, cache=cache, fa=fa, extra=extra)
        return (directory / "native.state").read_bytes(), (directory / "native.partial").read_bytes()

    def test_native_saves_parse_and_write_byte_identically(self):
        n = 300
        prompt = self.ids("dsv4", n)
        layouts = [("f16", "off", ()), ("f16", "on", ()), ("f32", "off", ()), ("f32", "on", ()), ("q8_0", "on", ()),
                   ("f16", "off", ("--parallel", "2", "--seq", "1")), ("q8_0", "on", ("--parallel", "2", "--rs-seq", "3"))]
        for name in self.models:
            spec = self.spec(name)
            for cache, fa, extra in layouts:
                with self.subTest(model=name, cache=cache, flash_attn=fa, extra=extra):
                    data, partial = self.save_both(name, prompt, cache=cache, fa=fa, extra=extra)
                    parsed = kvstate.parse_file(data, spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    np.testing.assert_array_equal(parsed.tokens, prompt)
                    seq = int(extra[extra.index("--seq") + 1]) if "--seq" in extra else 0
                    blob = kvstate.parse_blob(partial, spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
                    self.assertEqual((blob.to_bytes(), blob.seq_id), (partial, seq))
                    parallel = int(extra[extra.index("--parallel") + 1]) if "--parallel" in extra else 1
                    layout = {entry["name"]: entry for entry in kvstate.template_layout(parsed)}
                    # the raw cache writes one stream per sequence (never unified), the rest one
                    self.assertEqual([layout[s]["n_stream"] for s in ("base", "swa", "csa", "csa_state")],
                                     [parallel, parallel, 1, 1])
                    self.assertEqual((layout["swa"]["stream"], layout["swa"]["dtype"], layout["swa"]["v_trans"]),
                                     (seq, cache, fa == "off"))
                    base, swa = parsed.state.part("base").stream(), parsed.state.part("swa").stream()
                    np.testing.assert_array_equal(base.positions, np.arange(n))
                    np.testing.assert_array_equal(swa.positions, np.arange(n - spec.n_swa, n))
                    for pool in spec.pools():
                        self.assertEqual(parsed.state.part(pool.name).n_rows, n // pool.ratio)
                    for layer in swa.k:
                        nextn = spec.layer_types[layer] == "nextn"
                        values = swa.k[layer]["qs"] if cache == "q8_0" else swa.k[layer]
                        self.assertEqual(bool(np.any(values)), not nextn, f"layer {layer}")
                    # the PARTIAL_ONLY blob holds the same window and rings
                    for mine, full in zip(blob.state.parts, [parsed.state.part(l.name) for l in
                                                             spec.layouts(kvstate.FLAGS_PARTIAL_ONLY)]):
                        if isinstance(mine, kvstate.KVCache):
                            self.assertTrue(all(np.array_equal(mine.stream().k[l], full.stream().k[l])
                                                for l in mine.layers))
                        else:
                            self.assertTrue(all(np.array_equal(mine.kv[l], full.kv[l]) and
                                                np.array_equal(mine.score[l], full.score[l]) for l in mine.layers))
        self.assertEqual(self.spec("dsv4-mtp").layer_types, ("sliding",) * 4 + ("nextn",))

    def test_rows_hold_roped_latents_and_rings_hold_the_compressor_inputs(self):
        n = 300
        prompt = self.ids("dsv4", n)
        names = ["kv_norm", "kv", "csa_state_kv", "csa_state_score_ape", "hca_state_kv", "hca_state_score_ape",
                 "lid_state_kv", "lid_state_score_ape"]
        for runtime, flags in ((None, ()), (DSV4_RUNTIME_YARN, DSV4_RUNTIME_YARN_FLAGS)):
            for cache, fa in (("f16", "off"), ("q8_0", "on")):
                with self.subTest(runtime=runtime, cache=cache):
                    spec = self.spec("dsv4", runtime)
                    state_path = self.workdir() / "captured.state"
                    found = self.capture("dsv4", prompt, names, state=state_path, cache=cache, fa=fa,
                                         extra=[*flags, "--ctx", "512"])
                    rows = {name: np.concatenate([a for _, a in found[name]]) for name in found}
                    parsed = kvstate.read_file(state_path, spec)
                    swa = parsed.state.part("swa").stream()
                    for layer in range(4):
                        latent = rows[f"kv_norm-{layer}"].reshape(n, 1, -1)
                        cached = spec.rope(layer).apply(latent, np.arange(n))
                        # NORM RoPE on the last n_rot dims, the compress base and YaRN on compressed layers
                        self.assertTrue(bits_equal(cached, rows[f"kv-{layer}"].reshape(n, 1, -1)), f"layer {layer}")
                        stored = swa.k[layer]
                        window = cached[n - spec.n_swa:]
                        if cache == "q8_0":
                            # a quantized cache Hadamard-rotates the rows, then the x86 build's quantizer
                            h = kvstate.hadamard(64).astype(np.float64)
                            rotated = (window.reshape(-1, 64).astype(np.float64) @ h.T).astype(np.float32)
                            np.testing.assert_array_equal(stored, kvstate.quantize_q8_0(rotated).reshape(stored.shape))
                        else:
                            np.testing.assert_array_equal(stored, window.astype(np.float16))
                    for pool in spec.pools():
                        ring = parsed.state.part(pool.name + "_state")
                        for layer in pool.layers:
                            a = rows[f"{pool.name}_state_kv-{layer}"].reshape(n, -1)
                            s = rows[f"{pool.name}_state_score_ape-{layer}"].reshape(n, -1)
                            last = np.arange(n - pool.state_size, n)
                            np.testing.assert_array_equal(ring.kv[layer][0][last % pool.state_size], a[last])
                            np.testing.assert_array_equal(ring.score[layer][0][last % pool.state_size], s[last])

@unittest.skipUnless(DRIVER, "set KVPARK_STATE_DRIVER to enable llama.cpp state-layout checks")
class RecurrentFamilyTests(DriverMixin, unittest.TestCase):
    """Hybrids (GatedDeltaNet, Mamba, Mamba-2, short-conv, lightning next to GQA attention) and
    recurrent-only models: codecs, row sizes, R0 restores, truncation and the hybrids' RoPE."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-recurrent-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.vocab = {}, {}
        for architecture in RWKV_FIXTURE_ARCHITECTURES:
            path = cls.root / f"{architecture}.gguf"
            create_rwkv_fixture(path, architecture)
            cls.models[architecture], cls.vocab[architecture] = path, 258
        if LLAMA_ARCHS:
            for architecture in RECURRENT_ARCHS:
                if architecture in RWKV_FIXTURE_ARCHITECTURES:
                    continue
                first = kvstate.ARCHITECTURE_SINCE.get(architecture)
                order = kvstate.LLAMA_REVISIONS.index
                if first is not None and order(driver_revision(DRIVER, cls.root / "revision")) < order(first):
                    continue                    # glm5-next: a7b94df2c on
                cls.models[architecture] = llama_archs_model(LLAMA_ARCHS, architecture, cls.root / "archs")
                cls.vocab[architecture] = 128
        cls.revision = driver_revision(DRIVER, cls.root / "revision")
        cls.rng = np.random.default_rng(2027)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_saves_parse_and_write_byte_identically(self):
        for name in self.models:
            spec = self.spec(name)
            prompt = self.ids(name, 13)
            for cache in ("f16", "f32"):
                for fa in ("off", "on"):
                    with self.subTest(model=name, cache=cache, flash_attn=fa):
                        data = (self.save(name, prompt, cache=cache, fa=fa) / "native.state").read_bytes()
                        parsed = kvstate.parse_file(data, spec)
                        self.assertEqual(parsed.to_bytes(), data)
                        np.testing.assert_array_equal(parsed.tokens, prompt)
                        self.assertEqual([(p.name, p.layers) for p in parsed.state.parts], FAMILY_SECTIONS[name])
                        for part, layout in zip(parsed.state.parts, spec.layouts(0)):
                            if layout.kind == "recurrent":
                                self.assertEqual((part.positions.tolist(), part.n_layer), ([12], spec.n_layer_trunk))
                                continue
                            stream = part.stream()
                            self.assertEqual(stream.v_trans, fa == "off")
                            np.testing.assert_array_equal(stream.positions, np.arange(13))
                            # M-RoPE and PLE cells carry (x, y, token): x = y = pos for text
                            self.assertEqual(stream.ext is not None, layout.ext)
                            if layout.ext:
                                np.testing.assert_array_equal(stream.ext, np.stack([np.arange(13)] * 2 + [prompt], 1))
                            for il in part.layers:
                                self.assertEqual(stream.k_types[il], kvstate.DTYPE_NAMES[cache])
                                self.assertTrue(stream.k[il].any())
                                if layout.role == "indexer" and self.revision == "85d5703a3":
                                    # 85d5703a3's qwen4exp indexer cache stores raw keys and V rows nothing writes
                                    self.assertFalse(stream.v[il].any())
                                elif layout.role == "indexer":
                                    # a7b94df2c's stores K only, [key | pooled] with k-pooling (the fixture's
                                    # compress ratio 4): a pooled key in the last row of each complete block
                                    # (the graph's padding entries may write other rows' slots too)
                                    self.assertEqual(stream.v, {})
                                    slots = stream.k[il][..., spec.indexer.head_size:]
                                    self.assertEqual(slots.shape[-1],
                                                     spec.indexer.head_size * (spec.indexer_row_slots - 1))
                                    for last in range(spec.indexer.kpool - 1, 13, spec.indexer.kpool or 13):
                                        self.assertTrue(slots[last].any(), f"pooled key of the block ending at {last}")
        # the layer kinds behind those sections
        expected = {"falcon-h1": ("parallel", "parallel"), "nemotron_h": ("full", "recurrent", "mlp"),
                    "lfm2": ("sliding", "recurrent"), "mamba": ("recurrent", "recurrent")}
        for name, kinds in expected.items():
            if name in self.models:
                self.assertEqual(self.spec(name).layer_types, kinds)

    def test_recurrent_row_sizes_of_every_family(self):
        # recurrent_row_floats / ple_rows_from_metadata against what llama_memory_recurrent writes,
        # walked without kvstate's shapes, including 0-byte rows (MiniMax-01 R, LFM2 S), the
        # qwen4exp PLE row and the RWKV token-shift rows; then the spec and the parsed rows
        known = {"minimax-01": (0, 131072, 0), "lfm2": (2048, 0, 0), "qwen4exp": (9216, 131072, 36864),
                 "rwkv6": (512, 4096, 0), "rwkv7": (512, 4096, 0), "rwkv6qwen2": (256, 4096, 0),
                 "arwkv7": (256, 4096, 0), "mamba": (6144, 262144, 0), "mamba2": (12288, 262144, 0)}
        for name, (family, order) in RECURRENT_ARCHS.items():
            if name not in self.models:
                continue
            with self.subTest(architecture=name):
                meta = gguf_tensors.read_metadata(self.models[name], max_array=1 << 16)
                r_floats, s_floats = kvstate.recurrent_row_floats(meta, family)
                p_floats, ple_layers = kvstate.ple_rows_from_metadata(meta)
                data = (self.save(name, list(range(1, 8))) / "native.state").read_bytes()
                # the fixtures' PLE layer list is [0], the first recurrent layer
                found = walk_payload(data, order, set(ple_layers))
                self.assertIsNotNone(found, "payload does not walk")
                rows, n_layer = found
                self.assertEqual((rows["r"], rows["s"], rows["p"]),
                                 ({4 * r_floats}, {4 * s_floats}, {4 * p_floats} if p_floats else set()))
                nextn = int(meta.get(f"{name}.nextn_predict_layers", 0))
                self.assertEqual(n_layer, int(meta[f"{name}.block_count"]) - nextn)
                spec = self.spec(name)
                self.assertEqual((spec.recurrent_row_bytes, spec.recurrent_p_bytes),
                                 ((4 * r_floats, 4 * s_floats), 4 * p_floats))
                if name in known:
                    self.assertEqual((4 * r_floats, 4 * s_floats, 4 * p_floats), known[name])
                state = kvstate.parse_file(data, spec).state
                for part, layout in zip(state.parts, spec.layouts(0)):
                    if layout.kind == "recurrent":
                        self.assertEqual({rows.row_bytes for rows in part.r.values()}, {4 * r_floats})
                        self.assertEqual({rows.row_bytes for rows in part.s.values()}, {4 * s_floats})
                        self.assertEqual(sorted(part.p), list(ple_layers))
                        continue
                    stream = part.stream()
                    for il in part.layers:
                        self.assertEqual(stream.k[il].shape, (7,) + spec.kv_shape(il, "k", layout.role))
                        if spec.kv_shape(il, "v", layout.role) is not None:
                            self.assertEqual(stream.v[il].shape, (7,) + spec.kv_shape(il, "v", layout.role))

    def test_attention_rope_matches_the_cpu_cache_rows(self):
        # the pre-RoPE keys of one capture run, rotated by cache_keys, are the rows the same run
        # saved, bit for bit (f32 cache): with the GGUF's RoPE and with the receiving server's
        # YaRN settings; granitehybrid includes its rope_freqs frequency factors
        checked = 0
        for name, node in PRE_ROPE_KEYS.items():
            if name not in self.models:
                continue
            for runtime, extra in ((None, ()), (RUNTIME_YARN, RUNTIME_YARN_FLAGS)):
                spec = self.spec(name, runtime)
                if runtime is not None and spec.rope_type == "none":
                    continue
                prompt = self.ids(name, 40)
                saved = self.workdir() / "captured.state"
                found = self.capture(name, prompt, [node], extra=extra, state=saved)
                state = kvstate.read_file(saved, spec)
                for part, layout in zip(state.state.parts, spec.layouts(0)):
                    if layout.role != "attention" or not part.layers:
                        continue
                    stream = part.stream()
                    positions = np.asarray(stream.positions)
                    for il in part.layers:
                        with self.subTest(model=name, runtime=runtime, layer=il):
                            nodes = [array for op, array in found[f"{node}-{il}"] if op != "ROPE"]
                            pre = nodes[-1].reshape((len(prompt),) + spec.kv_shape(il, "k"))[positions]
                            rows = stream.k[il]
                            self.assertTrue(bits_equal(kvstate.cache_keys(spec, il, pre, positions), rows))
                            np.testing.assert_allclose(kvstate.pre_rope_keys(spec, il, rows, positions), pre,
                                                       rtol=0, atol=1e-6 * max(1.0, float(np.abs(pre).max())))
                            rope = spec.rope(il)
                            if rope is None:
                                self.assertTrue(bits_equal(rows, pre))
                            else:
                                # the comparison separates parameters: a 7x base misses
                                wrong = dataclasses.replace(rope, base=rope.base * 7).apply(pre, positions)
                                self.assertGreater(np.abs(wrong - rows).max(), 1e-2 * np.abs(rows).max())
                            checked += 1
        if LLAMA_ARCHS:
            self.assertGreaterEqual(checked, 20)

    def test_rerotated_states_restore_like_native(self):
        # every attention row through pre_rope_keys and back (the qwen4exp indexer keys pass through)
        for name in self.models:
            spec = self.spec(name)
            if not spec.attention_layers:
                continue
            prompt, next_ids = self.ids(name, 20), self.ids(name, 3)
            for cache, fa in (("f32", "off"), ("f16", "on")):
                with self.subTest(model=name, cache=cache, flash_attn=fa):
                    directory = self.save(name, prompt, next_ids=next_ids, cache=cache, fa=fa)
                    native = np.load(directory / "native.npy")
                    state = kvstate.read_file(directory / "native.state", spec)
                    self.rerotate(spec, state)
                    kvstate.write_file(directory / "rerotated.state", state)
                    logits, report = self.restore(name, directory / "rerotated.state", next_ids, cache=cache, fa=fa)
                    self.assertEqual(report["next_pos"], 20)
                    scale = max(1.0, float(np.abs(native).max()))
                    np.testing.assert_allclose(logits, native, rtol=0, atol=2e-6 * scale)

    def test_nemotron_mtp_block_sits_in_neither_section(self):
        # a Nemotron-3 MoE GGUF with its MTP head as the converter writes it: the nextn block's
        # feed_forward_length is the MoE width, so nemotron's filters (attention: n_ff == 0) leave it
        # out of both sections of the main context, whose sections are the trunk's
        if "nemotron_h_moe" not in self.models:
            self.skipTest("set KVPARK_LLAMA_ARCHS for test-llama-archs' nemotron_h_moe")
        path = self.root / "nemotron_h_moe-mtp.gguf"
        if not path.exists():
            create_nemotron_mtp_fixture(self.models["nemotron_h_moe"], path)
        self.models, self.vocab = {**self.models, "mtp": path}, {**self.vocab, "mtp": self.vocab["nemotron_h_moe"]}
        meta = gguf_tensors.read_metadata(path, max_array=1 << 16)
        self.assertEqual((meta["nemotron_h_moe.nextn_predict_layers"], meta["nemotron_h_moe.expert_count"]), (1, 2))
        spec = self.spec("mtp")
        self.assertEqual((spec.layer_types, spec.n_layer_trunk), (("full", "recurrent", "mlp"), 3))
        prompt, next_ids = self.ids("mtp", 20), self.ids("mtp", 4)
        for cache in ("f16", "f32"):
            for fa in ("off", "on"):
                with self.subTest(cache=cache, flash_attn=fa):
                    directory = self.save("mtp", prompt, next_ids=next_ids, cache=cache, fa=fa)
                    data = (directory / "native.state").read_bytes()
                    parsed = kvstate.parse_file(data, spec)
                    self.assertEqual(parsed.to_bytes(), data)
                    self.assertEqual([(p.name, p.layers) for p in parsed.state.parts], FAMILY_SECTIONS["nemotron_h_moe"])
                    self.assertEqual(parsed.state.part("recurrent").n_layer, 3)
                    # truncated (R0) states restore and decode on, KV rows as a native prefix save
                    truncated = kvstate.truncate_state(parsed, 15, recurrent="reset")
                    kvstate.write_file(directory / "cut.state", truncated)
                    logits, report = self.restore("mtp", directory / "cut.state", next_ids, cache=cache, fa=fa)
                    self.assertEqual(report["next_pos"], 15)
                    self.assertTrue(np.isfinite(logits).all())
                    prefix = kvstate.read_file(self.save("mtp", prompt[:15], cache=cache, fa=fa) / "native.state", spec)
                    assert_zero_recurrent_rows(self, truncated, prefix)
                    a, b = truncated.state.part("attn").stream(), prefix.state.part("attn").stream()
                    np.testing.assert_array_equal(a.positions, b.positions)
                    self.assertTrue(bits_equal(a.k[0], b.k[0]) and bits_equal(a.v[0], b.v[0]))
        # with n_ff == 0 the nextn layer would sit in the attention cache: refused
        meta["nemotron_h_moe.feed_forward_length"] = list(meta["nemotron_h_moe.feed_forward_length"][:-1]) + [0]
        with self.assertRaisesRegex(kvstate.StateFormatError, "nextn layers would sit in the attention cache"):
            kvstate.ModelSpec.from_metadata(meta)

    def test_r0_and_truncated_states_restore_and_keep_decoding(self):
        # R0 (recurrent_reset at t0 - 1 for every recurrent section) and truncate_state at a cut
        # before the end (which needs R0) restore through llama_state_seq_load_file and decode on
        for name in self.models:
            spec = self.spec(name)
            prompt, next_ids = self.ids(name, 20), self.ids(name, 4)
            state = None                # (never the previous model's state when this one's save fails)
            with self.subTest(model=name, state="R0"):
                directory = self.save(name, prompt, next_ids=next_ids)
                state = kvstate.read_file(directory / "native.state", spec)
                parts = [kvstate.recurrent_reset(part, 19) if isinstance(part, kvstate.RecurrentState) else part
                         for part in state.state.parts]
                r0 = kvstate.StateFile(state.token_words, kvstate.SequenceState(spec, state.state.flags, parts))
                assert_zero_recurrent_rows(self, r0, state)
                kvstate.write_file(directory / "r0.state", r0)
                logits, report = self.restore(name, directory / "r0.state", next_ids)
                self.assertEqual(report["next_pos"], 20)
                self.assertTrue(np.isfinite(logits).all())
                if spec.memory == "recurrent" and name not in ("rwkv7", "arwkv7"):
                    # nothing but recurrent rows: R0 is a fresh sequence, so the continuation decodes
                    # as from scratch (RWKV6 up to ubatch-split rounding; the random RWKV7 fixtures
                    # move by ~1e-2 with the thread count or ubatch split, so they are left out)
                    fresh = np.load(self.save(name, next_ids[:1], next_ids=next_ids[1:]) / "native.npy")
                    tolerance = 0.0 if name.startswith("mamba") else 1e-6
                    np.testing.assert_allclose(logits[1:], fresh, rtol=0,
                                               atol=tolerance * max(1.0, float(np.abs(fresh).max())))
            if state is None:
                continue
            with self.subTest(model=name, state="truncated"):
                cut = 15
                with self.assertRaisesRegex(kvstate.StateFormatError, "needs R0"):
                    kvstate.truncate_state(state, cut)
                truncated = kvstate.truncate_state(state, cut, recurrent="reset")
                kvstate.write_file(directory / "cut.state", truncated)
                logits, report = self.restore(name, directory / "cut.state", next_ids)
                self.assertEqual(report["next_pos"], cut)
                self.assertTrue(np.isfinite(logits).all())
                # the KV sections are those of a native save of the prefix, bit for bit
                prefix = kvstate.read_file(self.save(name, prompt[:cut]) / "native.state", spec)
                np.testing.assert_array_equal(truncated.token_words, prefix.token_words)
                assert_zero_recurrent_rows(self, truncated, prefix)
                for mine, theirs in zip(truncated.state.parts, prefix.state.parts):
                    if isinstance(mine, kvstate.RecurrentState):
                        self.assertEqual(mine.positions.tolist(), [cut - 1])
                        continue
                    a, b = mine.stream(), theirs.stream()
                    np.testing.assert_array_equal(a.positions, b.positions)
                    self.assertEqual(a.ext is None, b.ext is None)
                    if a.ext is not None:
                        np.testing.assert_array_equal(a.ext, b.ext)
                    for il in mine.layers:
                        self.assertTrue(bits_equal(a.k[il], b.k[il]), f"layer {il} K")
                        if il in b.v:
                            self.assertTrue(bits_equal(a.v[il], b.v[il]), f"layer {il} V")



# GQA family (kvstate.GQA_ARCHITECTURES): gguf-py Gemma models (both pins) -> create_gemma_fixture options
GEMMA_TINY = {"gemma4": ("gemma4", {}), "gemma4-shared": ("gemma4", {"layers": 8, "shared_kv_layers": 2}),
              "gemma4-v": ("gemma4", {"alternative_attention": False}), "gemma3-tiny": ("gemma3", {})}
# test-llama-archs models (85d5703a3's writes no gemma3, gpt-oss... see GqaFamilyTests.setUpClass)
GQA_ARCHS = ("llama", "mistral3", "qwen3moe", "hy_v3", "phi3", "gemma3", "gpt-oss", "laguna", "mimo2")
# copies of them with other RoPE settings: name -> (base model, metadata values without the arch prefix)
GQA_VARIANTS = {
    # Laguna-S-2.1's geometry scaled down: full layers YaRN over a partial head, plain RoPE on sliding ones
    "laguna-yarn": ("laguna", {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                               "rope.scaling.original_context_length": 64, "rope.dimension_count": 40,
                               "rope.freq_base": 500000.0, "rope.freq_base_swa": 1000.0}),
    # gpt-oss runs YaRN on every layer, its sliding layers with the GGUF's scale whatever the context's
    "gpt-oss-yarn": ("gpt-oss", {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                                 "rope.scaling.original_context_length": 64, "rope.freq_base": 150000.0}),
    # mistral3 reads the YaRN betas and log multiplier from the GGUF (mistral3.cpp:7-9)
    "mistral3-yarn": ("mistral3", {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                                   "rope.scaling.original_context_length": 64, "rope.scaling.yarn_beta_fast": 16.0,
                                   "rope.scaling.yarn_beta_slow": 2.0, "rope.scaling.yarn_log_multiplier": 0.8}),
    # Gemma 3's linear scaling of the global layers (sliding layers keep scale 1 and their own base)
    "gemma3-linear": ("gemma3", {"rope.scaling.type": "linear", "rope.scaling.factor": 8.0,
                                 "rope.freq_base_swa": 500.0}),
    # MiMo's partial NEOX (64 of the head) and its own sliding base
    "mimo2-partial": ("mimo2", {"rope.dimension_count": 64, "rope.freq_base_swa": 777.0}),
}
# Gemma's FFN GELU reads its input rounded to float16 on the CPU (ggml_vec_gelu_f32 with GGML_GELU_FP16,
# ggml-cpu/vec.h:1000-1014), so a key moved by one float32 ulp can flip a GELU input across an f16
# rounding boundary: the tiny Gemma models (weights N(0, 0.05)) move their logits by up to ~3e-4 of
# their scale when a re-rotation changes the last bit of ~10% of the keys (1e-6 or less for SiLU
# models). Restore tolerances (f32, f16 caches) of those models, and of every other:
GELU_TOLERANCE, TOLERANCE = (2e-3, 2e-2), (2e-6, 2e-3)
# the graph node holding each GQA architecture's keys right before RoPE: (name, op); hy_v3's k_norm
# output is unnamed, so it is the k RMSNorm ("norm", the layer's second 3-D one) times attn_k_norm
GQA_PRE_ROPE = {"gemma3": ("Kcur_normed", "MUL"), "gemma4": ("Kcur_normed", "MUL"), "laguna": ("Kcur_normed", "MUL"),
                "qwen3moe": ("Kcur_normed", "MUL"), "llama": ("Kcur", "RESHAPE"), "mistral3": ("Kcur", "RESHAPE"),
                "phi3": ("Kcur", "RESHAPE"), "gpt-oss": ("Kcur", "RESHAPE"), "mimo2": ("Kcur", "MUL_MAT"),
                "hy_v3": ("norm", "RMS_NORM")}


@unittest.skipUnless(DRIVER, "set KVPARK_STATE_DRIVER to enable llama.cpp state-layout checks")
class GqaFamilyTests(DriverMixin, unittest.TestCase):
    """Plain-attention models that take the context's RoPE (kvstate.GQA_ARCHITECTURES): one cache
    (llama, mistral3, qwen3moe, hy_v3, phi3) or a global and a sliding-window one (gemma3, gemma4,
    gpt-oss, laguna, mimo2) with per-layer KV heads and head sizes, per-layer-kind RoPE (gemma4's
    proportional RoPE on its full layers, laguna's plain RoPE on its sliding ones, frequency factors)
    and gemma4's shared KV layers: byte-exact codecs, RoPE bit for bit against the CPU build, restores
    of re-rotated, moved, truncated and built states. The Gemma models come from
    tiny_fixture.create_gemma_fixture (both pins); the others from test-llama-archs (KVPARK_LLAMA_ARCHS)
    where its build writes them."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-gqa-")
        cls.root = Path(cls.temp.name)
        cls.models, cls.vocab, cls.weights = {}, {}, {}
        for name, (architecture, options) in GEMMA_TINY.items():
            cls.models[name] = cls.root / f"{name}.gguf"
            cls.weights[name] = create_gemma_fixture(cls.models[name], architecture, **options)
            cls.vocab[name] = 256
        cls.missing = []
        for architecture in GQA_ARCHS if LLAMA_ARCHS else ():
            try:
                path = llama_archs_model(LLAMA_ARCHS, architecture, cls.root / "archs")
            except RuntimeError:
                cls.missing.append(architecture)        # 85d5703a3's test-llama-archs saves no such model
                continue
            if architecture == "phi3":
                # get_rope_factors would pick rope_factors_long or _short by the context size (refused):
                # Phi-3 GGUFs without longrope carry neither
                plain = cls.root / "phi3-plain.gguf"
                copy_with_metadata(path, plain, drop_tensors=["rope_factors_long.weight", "rope_factors_short.weight"])
                cls.refused_phi3, path = path, plain
            cls.models[architecture], cls.vocab[architecture] = path, 128
        for name, (base, values) in GQA_VARIANTS.items():
            if base in cls.models:
                cls.models[name] = cls.root / f"{name}.gguf"
                copy_with_metadata(cls.models[base], cls.models[name],
                                   values={f"{base}.{key}": value for key, value in values.items()})
                cls.vocab[name] = cls.vocab[base]
        cls.revision = driver_revision(DRIVER, cls.root / "revision")
        cls.rng = np.random.default_rng(2029)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_specs_of_the_fixtures(self):
        gemma = self.spec("gemma4")
        # the last of every three layers is full: 1 KV head of 32 with proportional RoPE (4 of its 16
        # pairs rotate, factor 1e30 leaves the rest), sliding layers 4 heads of 16 rotated whole
        self.assertEqual(gemma.layer_types, ("sliding", "sliding", "full") * 2)
        self.assertEqual((gemma.memory, gemma.n_swa, gemma.n_rot, gemma.n_rot_swa), ("iswa", 48, 32, 16))
        self.assertEqual([gemma.head_shape(il, "k") for il in (0, 2)], [(4, 16), (1, 32)])
        full, sliding = gemma.rope(2), gemma.rope(0)
        self.assertEqual((full.mode, full.n_dims, full.base, sliding.n_dims, sliding.base, sliding.freq_factors),
                         ("neox", 32, 1e6, 16, 1e4, None))
        self.assertEqual(full.freq_factors, (1.0,) * 4 + (float(np.float32(1e30)),) * 12)
        shared = self.spec("gemma4-shared")
        self.assertEqual((shared.layer_types[-2:], shared.sections(0)),
                         (("shared", "shared"), [("base", "kv", (2, 5)), ("swa", "kv", (0, 1, 3, 4))]))
        with self.assertRaisesRegex(ValueError, "no attention cache"):
            shared.rope(7)
        if "phi3" in self.models:
            with self.assertRaisesRegex(kvstate.StateFormatError, "by the context size"):
                kvstate.ModelSpec.from_gguf(self.refused_phi3)
        if "laguna-yarn" in self.models:
            laguna = self.spec("laguna-yarn")
            full, sliding = laguna.rope(0), laguna.rope(1)
            self.assertEqual((full.n_dims, full.ext_factor, full.freq_scale, full.base),
                             (40, 1.0, 0.25, 500000.0))
            self.assertEqual((sliding.n_dims, sliding.ext_factor, sliding.attn_factor, sliding.freq_scale, sliding.base),
                             (80, 0.0, 1.0, 1.0, 1000.0))

    def test_native_saves_parse_and_write_byte_identically(self):
        for name in self.models:
            spec = self.spec(name)
            prompt = self.ids(name, 40)
            for cache in ("f16", "f32"):
                for fa in ("off", "on"):
                    with self.subTest(model=name, cache=cache, flash_attn=fa):
                        directory = self.workdir()
                        self.drive("save", name, "--tokens", ",".join(map(str, prompt)), "--state",
                                   directory / "native.state", "--partial", directory / "native.partial",
                                   cache=cache, fa=fa)
                        data = (directory / "native.state").read_bytes()
                        parsed = kvstate.parse_file(data, spec)
                        self.assertEqual(parsed.to_bytes(), data)
                        np.testing.assert_array_equal(parsed.tokens, prompt)
                        # one layout on both revisions: either spec reads it as itself
                        self.assertEqual(spec.layout_variants(), [spec])
                        for revision in kvstate.LLAMA_REVISIONS:
                            self.assertEqual(kvstate.parse_file(data, spec.for_revision(revision)).state.spec.llama_revision,
                                             revision)
                        self.assertEqual([(p.name, p.layers) for p in parsed.state.parts],
                                         [(name_, layers) for name_, _, layers in spec.sections(0)])
                        for part, layout in zip(parsed.state.parts, spec.layouts(0)):
                            stream = part.stream()
                            self.assertEqual((stream.v_trans, stream.ext), (fa == "off", None))
                            first = max(0, 40 - spec.n_swa) if layout.sliding else 0
                            np.testing.assert_array_equal(stream.positions, np.arange(first, 40))
                            for il in part.layers:
                                self.assertEqual((stream.k[il].shape[1:], stream.v[il].shape[1:]),
                                                 (spec.head_shape(il, "k"), spec.head_shape(il, "v")))
                                self.assertTrue(stream.k[il].any() and stream.v[il].any())
                                self.assertEqual(stream.k_types[il], kvstate.DTYPE_NAMES[cache])
                        # a server checkpoint's PARTIAL_ONLY blob: the sliding-window section of iSWA
                        # memory, the whole cache of a plain one (llama_kv_cache ignores the flags)
                        partial = (directory / "native.partial").read_bytes()
                        blob = kvstate.parse_blob(partial, spec, flags=kvstate.FLAGS_PARTIAL_ONLY)
                        self.assertEqual(blob.to_bytes(), partial)
                        self.assertEqual([p.name for p in blob.state.parts], ["swa"] if spec.memory == "iswa" else ["kv"])

    def pre_rope(self, name, spec, found, layer, n):
        """A layer's keys right before RoPE from one capture run's nodes (GQA_PRE_ROPE)."""
        node, op = GQA_PRE_ROPE[spec.architecture]
        nodes = [a for o, a in found[f"{node}-{layer}"] if o == op]
        if spec.architecture == "hy_v3":
            # attn_norm and ffn_norm are 2-D, then q's and k's RMSNorm: k's is the second 3-D one
            norm = [a for a in nodes if a.ndim == 3][1]
            weight = gguf_tensors.read_tensor(self.models[name], f"blk.{layer}.attn_k_norm.weight").astype(np.float32)
            return (norm.reshape((n,) + spec.kv_shape(layer, "k")) * weight).astype(np.float32)
        return nodes[0].reshape((n,) + spec.kv_shape(layer, "k"))

    def test_rope_matches_the_cpu_cache_rows_bit_for_bit(self):
        # the pre-RoPE keys of one capture run, rotated by cache_keys, are the rows the same run saved
        # (f32 cache), with the GGUF's RoPE and with the receiving server's YaRN settings; a 7x base misses
        cases = [(name, None, ()) for name in self.models]
        cases += [(name, RUNTIME_YARN, RUNTIME_YARN_FLAGS) for name in ("gemma4", "gpt-oss-yarn", "laguna-yarn", "llama")
                  if name in self.models]
        for name, runtime, extra in cases:
            spec = self.spec(name, runtime)
            prompt = self.ids(name, 40)
            saved = self.workdir() / "captured.state"
            node = GQA_PRE_ROPE[spec.architecture][0]
            found = self.capture(name, prompt, [node], extra=extra, state=saved)
            state = kvstate.read_file(saved, spec)
            for part in state.state.parts:
                stream = part.stream()
                positions = np.asarray(stream.positions)
                for il in part.layers:
                    with self.subTest(model=name, runtime=runtime, layer=il):
                        pre = self.pre_rope(name, spec, found, il, len(prompt))[positions]
                        rows = stream.k[il]
                        self.assertTrue(bits_equal(kvstate.cache_keys(spec, il, pre, positions), rows))
                        np.testing.assert_allclose(kvstate.pre_rope_keys(spec, il, rows, positions), pre, rtol=0,
                                                   atol=1e-6 * max(1.0, float(np.abs(pre).max())))
                        rope = spec.rope(il)
                        wrong = dataclasses.replace(rope, base=rope.base * 7).apply(pre, positions)
                        self.assertGreater(np.abs(wrong - rows).max(), 1e-2 * np.abs(rows).max())

    def tolerance(self, name, cache):
        """The restore tolerance of a model and cache type, in units of the logit scale (GELU_TOLERANCE)."""
        return (GELU_TOLERANCE if name in GEMMA_TINY else TOLERANCE)[cache == "f16"]

    def test_rerotated_states_restore_like_native(self):
        for name in self.models:
            spec = self.spec(name)
            prompt, next_ids = self.ids(name, 40), self.ids(name, 4)
            for cache, fa in (("f32", "off"), ("f32", "on"), ("f16", "on")):
                with self.subTest(model=name, cache=cache, flash_attn=fa):
                    directory = self.save(name, prompt, next_ids=next_ids, cache=cache, fa=fa)
                    native = np.load(directory / "native.npy")
                    state = kvstate.read_file(directory / "native.state", spec)
                    self.rerotate(spec, state)
                    kvstate.write_file(directory / "rerotated.state", state)
                    logits, report = self.restore(name, directory / "rerotated.state", next_ids, cache=cache, fa=fa)
                    self.assertEqual(report["next_pos"], 40)
                    # f32: one rounding of each key (an f16 cache keeps the native bits)
                    tolerance = self.tolerance(name, "f32") if cache == "f32" else 0.0
                    np.testing.assert_allclose(logits, native, rtol=0,
                                               atol=tolerance * max(1.0, float(np.abs(native).max())))

    def test_states_moved_to_other_positions_restore_like_native(self):
        # attention depends on relative positions only (the sliding windows too), so every key moved to
        # positions + 37 restores to the native continuation up to rounding (DriverMixin.move); in the
        # Gemma models, whose logits hear attention, keys moved with a wrong RoPE (7x every base) miss
        # by far more, so there the comparison checks the parameters. (test-llama-archs' models barely
        # hear attention: a wrong RoPE moves their logits by ~1e-7 of their scale. Their RoPE is checked
        # bit for bit against llama.cpp's rows instead.)
        offset = 37
        for name in self.models:
            spec = self.spec(name)
            for n_prompt, cache, fa in ((24, "f32", "off"), (6, "f16", "on")):
                with self.subTest(model=name, cells=n_prompt, cache=cache, flash_attn=fa):
                    prompt, next_ids = self.ids(name, n_prompt), self.ids(name, 3)
                    directory = self.save(name, prompt, next_ids=next_ids, cache=cache, fa=fa)
                    native = np.load(directory / "native.npy")
                    scale = max(1.0, float(np.abs(native).max()))
                    misses = []
                    for label, moved_spec in (("moved", spec), ("wrong", dataclasses.replace(
                            spec, rope_freq_base=spec.rope_freq_base * 7, rope_freq_base_swa=spec.rope_freq_base_swa * 7))):
                        state = kvstate.read_file(directory / "native.state", spec)
                        # (a wrong spec turns each key by its wrong angle of `offset` positions)
                        self.move(moved_spec, state, offset)
                        kvstate.write_file(directory / f"{label}.state", state)
                        logits, report = self.restore(name, directory / f"{label}.state", next_ids, cache=cache, fa=fa)
                        self.assertEqual(report["next_pos"], n_prompt + offset)
                        misses.append(float(np.abs(logits - native).max()) / scale)
                    self.assertLessEqual(misses[0], self.tolerance(name, cache))
                    if cache == "f32" and name in GEMMA_TINY:
                        self.assertGreater(misses[1], 10 * self.tolerance(name, cache))

    def test_truncated_states_are_native_prefix_saves(self):
        # truncate_state keeps what a save of the prefix holds (every sliding window still whole), byte for byte
        for name in self.models:
            spec = self.spec(name)
            n = min(30, spec.n_swa) if spec.n_swa else 30
            cut = n - 9
            prompt, next_ids = self.ids(name, n), self.ids(name, 3)
            for cache, fa in (("f32", "off"), ("f16", "on")):
                with self.subTest(model=name, cache=cache, flash_attn=fa):
                    state = kvstate.read_file(self.save(name, prompt, cache=cache, fa=fa) / "native.state", spec)
                    truncated = kvstate.truncate_state(state, cut)
                    prefix = self.save(name, prompt[:cut], next_ids=next_ids, cache=cache, fa=fa)
                    self.assertEqual(truncated.to_bytes(), (prefix / "native.state").read_bytes())
                    path = prefix / "cut.state"
                    kvstate.write_file(path, truncated)
                    logits, _ = self.restore(name, path, next_ids, cache=cache, fa=fa)
                    np.testing.assert_array_equal(logits, np.load(prefix / "native.npy"))

    def test_built_plain_caches_restore_like_native(self):
        # build_dense_state from pre-RoPE rows (and V) of the plain-cache models, with the GGUF's RoPE and
        # the receiving server's YaRN
        cases = [(name, None, ()) for name in ("llama", "mistral3", "mistral3-yarn", "qwen3moe", "hy_v3", "phi3")
                 if name in self.models]
        cases += [(name, RUNTIME_YARN, RUNTIME_YARN_FLAGS) for name in ("llama", "qwen3moe") if name in self.models]
        for name, runtime, extra in cases:
            with self.subTest(model=name, runtime=runtime):
                spec = self.spec(name, runtime)
                prompt, next_ids = self.ids(name, 30), self.ids(name, 3)
                directory = self.save(name, prompt, next_ids=next_ids, extra=extra)
                native = np.load(directory / "native.npy")
                stream = kvstate.read_file(directory / "native.state", spec).state.part("kv").stream()
                positions = np.arange(30)
                keys = {il: kvstate.pre_rope_keys(spec, il, stream.k[il], positions) for il in spec.layers_of("full")}
                built = kvstate.build_dense_state(spec, prompt, positions, keys, dict(stream.v), dtype="f32", v_trans=True)
                kvstate.write_file(directory / "built.state", built)
                logits, _ = self.restore(name, directory / "built.state", next_ids, extra=extra)
                np.testing.assert_allclose(logits, native, rtol=0, atol=2e-6 * max(1.0, float(np.abs(native).max())))


if __name__ == "__main__":
    unittest.main()
