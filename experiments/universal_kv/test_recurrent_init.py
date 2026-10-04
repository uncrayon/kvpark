"""recurrent_init: strategies for the recurrent cell at t0 - 1, on synthetic states (and the
helpers of r0_experiments that need no model)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import r0_experiments
import recurrent_init
from recurrent_init import ALL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import kvstate  # noqa: E402

R_FLOATS, S_FLOATS = 6, 16
# Qwen3.5-like hybrid: recurrent layers 0, 2 (R 6 floats, S 16 floats), attention layers 1, 3
HYBRID = kvstate.ModelSpec("qwen35", ("recurrent", "full", "recurrent", "full"), (0, 2, 0, 2), (0, 16, 0, 16),
                           (0, 16, 0, 16), rope_type="imrope", n_rot=8, rope_freq_base=1e7,
                           rope_sections=(1, 1, 1, 1), recurrent_row_bytes=(4 * R_FLOATS, 4 * S_FLOATS))


def rows(values):
    values = np.asarray(values, dtype="<f4")
    return kvstate.OpaqueRows(kvstate.GGML_TYPE_F32, values.shape[-1] * 4, values.tobytes())


def part(r, s, position=0, p=None, layers=(0, 2), name="recurrent"):
    """A recurrent section of len(r[layer]) cells: r, s (and p) {layer: [cells, floats]}."""
    cells = len(next(iter(r.values())))
    p_rows = {}
    for layer, values in (p or {}).items():
        built = rows(values)
        p_rows[layer] = kvstate.OpaqueRows(None, built.row_bytes, built.data)
    return kvstate.RecurrentState(name, tuple(layers), np.full(cells, position, dtype=np.int32), ((),) * cells,
                                  4, {l: rows(r[l]) for l in layers}, {l: rows(s[l]) for l in layers}, p_rows)


def random_part(rng, position=0, scale=1.0, shift=0.0):
    return part({l: rng.standard_normal((1, R_FLOATS)) * scale + shift for l in (0, 2)},
                {l: rng.standard_normal((1, S_FLOATS)) * scale + shift for l in (0, 2)}, position)


def hybrid_state(rng, n=9, recurrent=None):
    """A Qwen3.5-like slot file of n tokens: attention KV and one recurrent cell at n - 1."""
    tokens = np.arange(n) + 100
    positions = np.arange(n)
    attention = HYBRID.layers_of("full")
    k = {l: rng.standard_normal((n, 2, 16)).astype(np.float32) for l in attention}
    v = {l: rng.standard_normal((n, 2, 16)).astype(np.float32) for l in attention}
    keys = {l: kvstate.cache_keys(HYBRID, l, k[l], positions) for l in attention}
    stream = kvstate.build_kv_stream(HYBRID, attention, positions, keys, v, dtype="f16", v_trans=False,
                                     tokens_at=tokens)
    recurrent = recurrent or random_part(rng, n - 1)
    parts = [kvstate.KVCache("attn", attention, [stream]), recurrent]
    state = kvstate.StateFile(kvstate.token_words(tokens), kvstate.SequenceState(HYBRID, kvstate.FLAGS_NONE, parts))
    return kvstate.parse_file(state.to_bytes(), HYBRID)


def floats(rows_):
    return {key: {layer: np.array(values) for layer, values in by_layer.items()} for key, by_layer in rows_.items()}


class StrategyTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(recurrent_init.parse_strategy("zero"), recurrent_init.Rule("zero", "zero", "zero"))
        self.assertEqual(recurrent_init.parse_strategy("mean_conv"), recurrent_init.Rule("mean_conv", "mean", "zero"))
        self.assertEqual(recurrent_init.parse_strategy("mean_s0.25").s_scale, 0.25)
        self.assertEqual(recurrent_init.parse_strategy("prefix").needs, {"prefix"})
        self.assertEqual(recurrent_init.parse_strategy("zero").needs, set())
        for bad in ("one", "mean_s", "mean_s-1", "prefix_s0.5"):
            with self.subTest(bad), self.assertRaises(ValueError):
                recurrent_init.parse_strategy(bad)


class RowsTest(unittest.TestCase):
    def test_cell_rows_read_the_last_cell_by_default(self):
        r = {l: np.arange(2 * R_FLOATS).reshape(2, R_FLOATS) + 10 * l for l in (0, 2)}
        s = {l: -np.arange(2 * S_FLOATS).reshape(2, S_FLOATS) - 10 * l for l in (0, 2)}
        cells = part(r, s)
        last, first = recurrent_init.cell_rows(cells), recurrent_init.cell_rows(cells, 0)
        for l in (0, 2):
            np.testing.assert_array_equal(last["r"][l], r[l][1])
            np.testing.assert_array_equal(first["s"][l], s[l][0])
        with self.assertRaises(IndexError):
            recurrent_init.cell_rows(cells, 2)

    def test_non_f32_rows_are_refused(self):
        bad = random_part(np.random.default_rng(0))
        bad.r[0] = kvstate.OpaqueRows(kvstate.GGML_TYPE_F16, bad.r[0].row_bytes, bad.r[0].data)
        with self.assertRaises(ValueError):
            recurrent_init.cell_rows(bad)

    def test_section_writes_the_template_layout_and_zero_for_missing_rows(self):
        template = random_part(np.random.default_rng(1), position=40)
        values = np.linspace(-1, 1, S_FLOATS)
        built = recurrent_init.section(template, 7, {"s": {2: values}})
        self.assertEqual(built.positions.tolist(), [7])
        self.assertEqual(recurrent_init.Layout.of(built), recurrent_init.Layout.of(template))
        out = recurrent_init.cell_rows(built)
        np.testing.assert_array_equal(out["s"][2], values.astype(np.float32))
        self.assertFalse(out["s"][0].any() or out["r"][0].any() or out["r"][2].any())
        with self.assertRaises(ValueError):
            recurrent_init.section(template, 7, {"s": {2: np.zeros(S_FLOATS + 1)}})
        with self.assertRaises(ValueError):
            recurrent_init.section(template, 7, {"s": {2: np.full(S_FLOATS, np.nan)}})


class MeanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def samples(self, n=12, seed=2):
        rng = np.random.default_rng(seed)
        return [(random_part(rng, scale=0.5, shift=1.0 if i % 3 else -2.0), "tools" if i % 3 == 0 else "chat",
                 1024 * (1 + i)) for i in range(n)]

    def test_mean_per_group_and_pooled(self):
        samples = self.samples()
        accumulator = recurrent_init.MeanAccumulator()
        for state, group, depth in samples:
            accumulator.add(state, group=group, depth=depth)
        for group in ("chat", "tools", ALL):
            chosen = [s for s, g, _ in samples if group == ALL or g == group]
            mean = accumulator.mean(group)
            for key in ("r", "s"):
                for layer in (0, 2):
                    expected = np.mean([recurrent_init.cell_rows(s)[key][layer] for s in chosen], axis=0)
                    np.testing.assert_allclose(mean[key][layer], expected, rtol=1e-6, atol=1e-6)
        stats = accumulator.stats()
        self.assertEqual(stats["count"], len(samples))
        self.assertEqual(stats["depth"], [1024, 1024 * len(samples)])
        rows_s = np.array([[recurrent_init.cell_rows(s)["s"][l] for l in (0, 2)] for s, _, _ in samples], np.float64)
        mean_s = rows_s.mean(axis=0)
        explained = 1 - np.sum((rows_s - mean_s) ** 2) / np.sum(rows_s ** 2)
        self.assertAlmostEqual(stats["s"]["explained"], explained, places=6)
        self.assertEqual(accumulator.stats("tools")["count"], 4)
        with self.assertRaises(KeyError):
            accumulator.mean("none")

    def test_build_mean_takes_states_or_tuples(self):
        samples = self.samples(6)
        built = recurrent_init.build_mean([s for s, _, _ in samples[:3]] + [(s, g) for s, g, _ in samples[3:]])
        self.assertEqual(sorted(built.groups), [ALL, "chat", "tools"])
        self.assertEqual(built.groups[ALL]["count"], 6)
        self.assertEqual(built.groups["chat"]["count"] + built.groups["tools"]["count"], 3)
        with self.assertRaises(ValueError):
            recurrent_init.build_mean([])

    def test_a_state_of_another_layout_is_refused(self):
        accumulator = recurrent_init.MeanAccumulator()
        accumulator.add(random_part(np.random.default_rng(3)))
        other = part({l: np.zeros((1, R_FLOATS + 2)) for l in (0, 2)}, {l: np.zeros((1, S_FLOATS)) for l in (0, 2)})
        with self.assertRaises(ValueError):
            accumulator.add(other)

    def test_save_and_load(self):
        samples = self.samples()
        mean = recurrent_init.build_mean(samples, model={"gguf": "tiny.gguf"})
        for dtype, tolerance in (("float32", 0), ("float16", 2e-3)):
            path = mean.save(self.dir / f"{dtype}.npz", dtype=dtype)
            loaded = recurrent_init.RecurrentMean.load(path)
            self.assertEqual(loaded.layout, mean.layout)
            self.assertEqual(loaded.model, {"gguf": "tiny.gguf"})
            self.assertEqual(loaded.groups, json.loads(json.dumps(mean.groups)))
            for group in (ALL, "chat", "tools"):
                for key in ("r", "s"):
                    for layer in (0, 2):
                        got, want = loaded.rows(group)[key][layer], mean.rows(group)[key][layer]
                        self.assertEqual(got.dtype, np.float32)
                        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
        with self.assertRaises(FileExistsError):
            mean.save(self.dir / "float16.npz")
        only = recurrent_init.RecurrentMean.load(self.dir / "float32.npz", groups=["tools"])
        self.assertEqual(sorted(only.means), ["tools"])
        np.testing.assert_array_equal(only.rows("tools")["s"][2], mean.rows("tools")["s"][2])
        with self.assertRaises(KeyError):
            only.rows()
        with self.assertRaises(KeyError):
            recurrent_init.RecurrentMean.load(self.dir / "float32.npz", groups=["none"])

    def test_a_subset_of_groups_saves_and_loads(self):
        mean = recurrent_init.build_mean(self.samples(), model={"gguf": "tiny.gguf"})
        full = mean.save(self.dir / "full.npz", dtype="float32")
        self.assertEqual(recurrent_init.RecurrentMean.stored_groups(full), [ALL, "chat", "tools"])
        subset = recurrent_init.RecurrentMean.load(full, groups=["chat", "tools"]).save(self.dir / "subset.npz")
        self.assertEqual(recurrent_init.RecurrentMean.stored_groups(subset), ["chat", "tools"])
        loaded = recurrent_init.RecurrentMean.load(subset)
        self.assertEqual(sorted(loaded.means), ["chat", "tools"])
        self.assertEqual(loaded.groups["tools"], json.loads(json.dumps(mean.groups["tools"])))
        self.assertEqual(loaded.model, {"gguf": "tiny.gguf"})
        np.testing.assert_array_equal(loaded.rows("chat")["r"][0], mean.rows("chat")["r"][0].astype(np.float16))
        with self.assertRaises(KeyError):
            recurrent_init.RecurrentMean.load(subset, groups=[ALL])
        self.assertEqual(recurrent_init.match_group(recurrent_init.RecurrentMean.stored_groups(subset), []), "chat")

    def test_a_group_listed_without_rows_is_refused(self):
        path = recurrent_init.build_mean(self.samples(3)).save(self.dir / "mean.npz")
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        meta = json.loads(str(arrays.pop("meta")))
        meta["groups"]["other"] = {"count": 1, "depth": [None, None]}
        broken = self.dir / "broken.npz"
        np.savez(broken, meta=np.asarray(json.dumps(meta)), **arrays)
        recurrent_init.RecurrentMean.load(broken, groups=[ALL, "tools"])
        with self.assertRaisesRegex(KeyError, "other/r/0"):
            recurrent_init.RecurrentMean.load(broken)

    def test_match_group(self):
        self.assertEqual(recurrent_init.match_group([ALL, "chat", "tools"], [{"type": "function"}]), "tools")
        self.assertEqual(recurrent_init.match_group([ALL, "chat", "tools"], None), "chat")
        self.assertEqual(recurrent_init.match_group([ALL, "chat"], [{"type": "function"}]), ALL)
        self.assertIsNone(recurrent_init.match_group(["chat"], [{"type": "function"}]))
        # a tools group not calibrated on the deployment's own agent prompts: the pooled mean for tools
        tools = [{"type": "function"}]
        self.assertEqual(recurrent_init.match_group([ALL, "chat", "tools"], tools, tools_group=False), ALL)
        self.assertEqual(recurrent_init.match_group([ALL, "chat", "tools"], None, tools_group=False), "chat")
        self.assertIsNone(recurrent_init.match_group(["chat", "tools"], tools, tools_group=False))

    def test_translate_options_for_evaluations(self):
        path = recurrent_init.build_mean(self.samples(), model={"gguf": "tiny.gguf"}).save(self.dir / "options.npz")
        self.assertEqual(recurrent_init.translate_options(), {"recurrent": "zero"})
        means = {}
        options = recurrent_init.translate_options("mean", path, tools=[{"type": "function"}], means=means)
        self.assertEqual((options["recurrent"], options["recurrent_group"], sorted(options["recurrent_mean"].means)),
                         ("mean", ALL, [ALL]))
        self.assertIs(recurrent_init.translate_options("mean_conv", path, ALL, means=means)["recurrent_mean"],
                      options["recurrent_mean"])                                # loaded once per group
        self.assertEqual(recurrent_init.translate_options("mean", path)["recurrent_group"], "chat")
        with self.assertRaisesRegex(ValueError, "needs a recurrent mean file"):
            recurrent_init.translate_options("mean")
        with self.assertRaisesRegex(ValueError, "no group 'agents'"):
            recurrent_init.translate_options("mean", path, "agents")

    def test_values_beyond_float16_stay_float32(self):
        big = part({l: np.full((1, R_FLOATS), 1e6) for l in (0, 2)}, {l: np.full((1, S_FLOATS), 0.5) for l in (0, 2)})
        path = recurrent_init.build_mean([big]).save(self.dir / "big.npz")
        loaded = recurrent_init.RecurrentMean.load(path)
        np.testing.assert_array_equal(loaded.rows()["r"][0], np.full(R_FLOATS, 1e6, np.float32))
        with np.load(path) as data:
            self.assertEqual(data[f"{ALL}/r/0"].dtype, np.float32)
            self.assertEqual(data[f"{ALL}/s/0"].dtype, np.float16)


class InitialStateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.template = random_part(rng, position=30)
        self.mean = recurrent_init.build_mean([random_part(rng, shift=0.3) for _ in range(5)]
                                              + [(random_part(rng, shift=-4.0), "tools")])
        self.prefix = part({l: rng.standard_normal((2, R_FLOATS)) for l in (0, 2)},
                           {l: rng.standard_normal((2, S_FLOATS)) for l in (0, 2)}, position=11)

    def build(self, strategy, **options):
        options = {"mean": self.mean, "prefix": self.prefix, **options}
        built = recurrent_init.initial_state(self.template, 99, strategy, **options)
        self.assertEqual(built.positions.tolist(), [99])
        self.assertEqual(built.n_cells, 1)
        self.assertEqual(recurrent_init.Layout.of(built), recurrent_init.Layout.of(self.template))
        return floats(recurrent_init.cell_rows(built))

    def test_zero_is_kvstate_recurrent_reset(self):
        built = recurrent_init.initial_state(self.template, 99, "zero")
        reset = kvstate.recurrent_reset(self.template, 99)
        self.assertEqual(kvstate._write_recurrent(HYBRID, built), kvstate._write_recurrent(HYBRID, reset))

    def test_mean_strategies(self):
        mean = self.mean.rows()
        full, conv, half = self.build("mean"), self.build("mean_conv"), self.build("mean_s0.5")
        tools = self.build("mean", group="tools")
        for layer in (0, 2):
            np.testing.assert_array_equal(full["r"][layer], mean["r"][layer])
            np.testing.assert_array_equal(full["s"][layer], mean["s"][layer])
            np.testing.assert_array_equal(conv["r"][layer], mean["r"][layer])
            self.assertFalse(conv["s"][layer].any())
            np.testing.assert_allclose(half["s"][layer], 0.5 * mean["s"][layer], rtol=1e-7)
            np.testing.assert_array_equal(tools["s"][layer], self.mean.rows("tools")["s"][layer])
        with self.assertRaises(KeyError):
            self.build("mean", group="chat")

    def test_prefix_takes_the_prefix_states_last_cell(self):
        built, last = self.build("prefix"), recurrent_init.cell_rows(self.prefix)
        for key in ("r", "s"):
            for layer in (0, 2):
                np.testing.assert_array_equal(built[key][layer], last[key][layer])

    def test_missing_or_foreign_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            recurrent_init.initial_state(self.template, 99, "mean")
        with self.assertRaises(ValueError):
            recurrent_init.initial_state(self.template, 99, "prefix")
        other = part({l: np.zeros((1, R_FLOATS)) for l in (0, 1)}, {l: np.zeros((1, S_FLOATS)) for l in (0, 1)},
                     layers=(0, 1))
        foreign = recurrent_init.build_mean([other])
        with self.assertRaises(ValueError):
            recurrent_init.initial_state(self.template, 99, "mean", mean=foreign)
        with self.assertRaises(ValueError):
            recurrent_init.initial_state(self.template, 99, "prefix", prefix=other)
        with self.assertRaises(ValueError):
            foreign.check(self.template)
        self.mean.check(self.template)

    def test_ple_rows_follow_the_conv_rows_and_empty_rows_work(self):
        rng = np.random.default_rng(5)

        def ple_part(scale):
            return part({l: rng.standard_normal((1, R_FLOATS)) * scale for l in (0, 2)},
                        {l: np.zeros((1, 0)) for l in (0, 2)}, p={2: rng.standard_normal((1, 3)) * scale})
        template = ple_part(1.0)
        self.assertEqual(recurrent_init.Layout.of(template).p_layers, (2,))
        samples = [ple_part(1.0) for _ in range(4)]
        mean = recurrent_init.build_mean(samples)
        prefix = ple_part(2.0)
        for strategy, source in (("mean", mean.rows()), ("mean_conv", mean.rows()),
                                 ("prefix", recurrent_init.cell_rows(prefix)), ("zero", None)):
            built = recurrent_init.initial_state(template, 5, strategy, mean=mean, prefix=prefix)
            self.assertIsNone(built.p[2].ggml_type)
            out = recurrent_init.cell_rows(built)
            self.assertEqual(out["s"][0].size, 0)
            if source is None:
                self.assertFalse(out["p"][2].any())
            else:
                np.testing.assert_array_equal(out["p"][2], source["p"][2])
                np.testing.assert_array_equal(out["r"][0], source["r"][0])

    def test_state_distance(self):
        rng = np.random.default_rng(6)
        reference = random_part(rng)
        zero = recurrent_init.initial_state(reference, 0, "zero")
        self.assertEqual(recurrent_init.state_distance(reference, reference)["s"]["relative"], 0.0)
        self.assertAlmostEqual(recurrent_init.state_distance(reference, zero)["s"]["relative"], 1.0)
        self.assertAlmostEqual(recurrent_init.state_distance(reference, zero)["r"]["layers"][2], 1.0)


class StateFileTest(unittest.TestCase):
    def test_the_write_path_gives_a_native_save_back(self):
        rng = np.random.default_rng(9)
        native = hybrid_state(rng)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "9.state"
            r0_experiments.state_writer.write(path, native)
            self.assertTrue(r0_experiments.same_bytes(path, native))
            r0_experiments.check_rewrite(path, native, 9)
            with self.assertRaises(SystemExit):
                r0_experiments.check_rewrite(path, native, 8)
            other = recurrent_init.replace_recurrent(native, recurrent_init.initial_state(native, 8, "zero"))
            self.assertFalse(r0_experiments.same_bytes(path, other))
            with open(path, "ab") as stream:
                stream.write(b"\0")
            self.assertFalse(r0_experiments.same_bytes(path, native))

    def test_replace_recurrent_keeps_everything_else(self):
        rng = np.random.default_rng(7)
        native = hybrid_state(rng)
        mean = recurrent_init.build_mean([hybrid_state(rng) for _ in range(3)])
        built = recurrent_init.initial_state(native, 8, "mean", mean=mean)
        replaced = recurrent_init.replace_recurrent(native, built)
        replaced.check_positions()
        data = replaced.to_bytes()
        reparsed = kvstate.parse_file(data, HYBRID)
        np.testing.assert_array_equal(reparsed.tokens, native.tokens)
        tail = len(kvstate._write_recurrent(HYBRID, built))
        self.assertEqual(data[:-tail], native.to_bytes()[:-tail])
        rows_ = recurrent_init.cell_rows(recurrent_init.recurrent_part(reparsed))
        np.testing.assert_array_equal(rows_["s"][2], mean.rows()["s"][2])
        with self.assertRaises(ValueError):
            other = part({l: np.zeros((1, R_FLOATS)) for l in (0, 1)}, {l: np.zeros((1, S_FLOATS)) for l in (0, 1)},
                         layers=(0, 1))
            recurrent_init.replace_recurrent(native, other)

    def test_recurrent_part_needs_exactly_one_section(self):
        rng = np.random.default_rng(8)
        native = hybrid_state(rng)
        self.assertIs(recurrent_init.recurrent_part(native), native.state.parts[1])
        no_recurrent = kvstate.StateFile(native.token_words, kvstate.SequenceState(HYBRID, 0, native.state.parts[:1]))
        with self.assertRaises(ValueError):
            recurrent_init.recurrent_part(no_recurrent)


class ExperimentHelpersTest(unittest.TestCase):
    def test_small_helpers(self):
        self.assertEqual(r0_experiments.common_prefix_bytes(b"abcdef", b"abcxef"), 3)
        self.assertEqual(r0_experiments.common_prefix_bytes(b"abc", b"abcdef"), 3)
        self.assertEqual(r0_experiments.common_prefix_bytes(b"", b"a"), 0)
        self.assertEqual(r0_experiments.t0_for(10615, 2048), 8192)
        self.assertEqual(r0_experiments.t0_for(1500, 2048), 512)
        self.assertEqual(r0_experiments.t0_for(10615, 2048, 1), 8567)
        self.assertEqual(r0_experiments.sanity_tails({1024: 9216, 2048: 8192, 4096: 6144}), {2048})
        self.assertEqual(r0_experiments.sanity_tails({512: 10103, 1024: 9591}), {512, 1024})
        self.assertEqual(r0_experiments.strategy_list("zero, mean:match,mean_s0.5"),
                         [("zero", "zero", ALL), ("mean-match", "mean", "match"), ("mean_s0.5", "mean_s0.5", ALL)])
        with self.assertRaises(ValueError):
            r0_experiments.strategy_list("zero,bogus")

    def test_the_prefix_is_saved_by_its_own_record(self):
        did = "ab" * 32
        doc = {"record": {"document_id": did, "text": "t"}, "tokens": {"token_id": np.arange(5)},
               "t0s": {1024: 4096, 2048: 3072}, "H": 48}
        main, prefix = r0_experiments.save_records([doc])
        self.assertEqual(main["save_at"], [3072, 4096])
        self.assertEqual(prefix["save_at"], [48])
        self.assertEqual(prefix["document_id"], r0_experiments.prefix_id(did))
        self.assertRegex(prefix["document_id"], "^[0-9a-f]{64}$")
        self.assertNotEqual(prefix["document_id"], did)
        self.assertEqual((prefix["text"], prefix["expected_token_sha256"]), (main["text"], main["expected_token_sha256"]))

    def test_calibration_sessions_join_same_tool_lists(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]

        def conversation(identifier, source, with_tools, split="train"):
            messages = [{"role": "system", "content": "sys " + identifier}, {"role": "user", "content": "q " + identifier}]
            if with_tools:
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call0", "type": "function", "function": {"name": "f", "arguments": {"x": "1"}}}]})
                messages.append({"role": "tool", "tool_call_id": "call0", "name": "f", "content": "r"})
            messages.append({"role": "assistant", "content": "a " * 30 + identifier})
            return {"id": identifier, "source": source, "split": split, "tools": tools if with_tools else None,
                    "messages": messages}
        rows = [conversation(f"c{i}", "chat-src", False) for i in range(5)] \
            + [conversation(f"t{i}", "tool-src", True) for i in range(3)] \
            + [conversation("x", "chat-src", False, split="test"), conversation("c4", "chat-src", False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            sessions = r0_experiments.calibration_conversations([path], concat_chars=130, exclude={"c0"})
        self.assertEqual([s["id"] for s in sessions], ["concat:c1+c2", "concat:c3+c4", "concat:t0+t1", "t2"])
        joined = sessions[2]
        self.assertEqual(joined["tools"], tools)
        self.assertEqual([m["role"] for m in joined["messages"]].count("system"), 1)
        self.assertEqual([m.get("tool_call_id") for m in joined["messages"] if m["role"] == "tool"],
                         ["c0-call0", "c1-call0"])
        import chat
        for session in sessions:
            chat.validate_conversation(session)
        # conversations without a system message keep every turn
        bare = [conversation(f"b{i}", "bare-src", False) for i in range(2)]
        for item in bare:
            item["messages"] = item["messages"][1:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bare.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in bare))
            joined, = r0_experiments.calibration_conversations([path], concat_chars=10 ** 6, exclude=set())
        self.assertEqual([m["content"].split()[-1] for m in joined["messages"]], ["b0", "b0", "b1", "b1"])
        self.assertEqual([m["role"] for m in joined["messages"]], ["user", "assistant"] * 2)

    def test_calibrate_cannot_resume(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            r0_experiments.main(["calibrate", "--probe", "p", "--output", "o", "--model", "m", "--name", "n",
                                 "--corpus", "c", "--resume"])
        self.assertIn("--resume", stderr.getvalue())

    def test_summary_tables(self):
        def document(kind, deltas, n=4, tail=2100, sanity=0.0):
            scores = {"n_scored": n, "native_nll": 1.0, "sanity_2048": {"nll_delta": sanity, "nll_sum_delta": sanity * n,
                                                                        "kl": abs(sanity), "top1": 1.0}}
            for label, delta in deltas.items():
                scores[f"{label}_2048"] = {"nll_delta": delta, "nll_sum_delta": delta * n, "kl": abs(delta),
                                           "top1": 0.9 if label == "zero" else 0.5 + n / 100}
            return {"kind": kind, "scores": scores, "tail": {"2048": tail},
                    "state_distance": {f"{label}_2048": {"s": 0.5, "r": 0.1} for label in deltas}}
        report = {"tails": [2048], "strategies": ["zero", "mean"],
                  "documents": [document("chat", {"zero": 0.04, "mean": 0.02}, tail=2080),
                                document("chat", {"zero": 0.06, "mean": 0.03}, n=12, tail=2120, sanity=-0.002),
                                document("agent", {"zero": 0.1, "mean": 0.12})]}
        summary = r0_experiments.summarize_reports([report])
        chat_mean = summary["chat"]["2048"]["mean"]
        self.assertAlmostEqual(chat_mean["nll_delta"], (0.02 * 4 + 0.03 * 12) / 16)
        self.assertAlmostEqual(chat_mean["vs_zero"], -(0.02 * 4 + 0.03 * 12) / 16)
        self.assertAlmostEqual(chat_mean["kl_doc_mean"], 0.025)
        self.assertAlmostEqual(chat_mean["top1"], (0.54 * 4 + 0.62 * 12) / 16)
        self.assertEqual(chat_mean["tail_range"], [2080, 2120])
        self.assertEqual(summary["all"]["2048"]["zero"]["docs"], 3)
        sanity = summary["chat"]["2048"]["sanity"]
        self.assertAlmostEqual(sanity["max_abs"], 0.002)
        self.assertAlmostEqual(sanity["nll_delta"], -0.002 * 12 / 16)
        self.assertAlmostEqual(sanity["kl"], 0.002 * 12 / 16)
        self.assertNotIn("vs_zero", summary["all"]["2048"]["zero"])
        self.assertAlmostEqual(chat_mean["kl"], chat_mean["nll_delta"])
        self.assertAlmostEqual(chat_mean["kl_vs_zero"], chat_mean["vs_zero"])
        self.assertEqual(chat_mean["kl_wins"], 2)
        text = r0_experiments.summary_text(summary)
        self.assertIn("| mean | 2 | 2100 (2080-2120) | +0.0275", text)
        self.assertIn("(sanity over 2 docs: dNLL -0.0015, max |dNLL| 0.002", text)


if __name__ == "__main__":
    unittest.main()
