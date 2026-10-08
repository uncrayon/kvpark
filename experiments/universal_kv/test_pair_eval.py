"""pair_eval.py's renderings: the templates' defaults, and as llama-server renders a chat
(--server-render). No model is loaded: a GGUF holding only a tokenizer and a chat template."""

import sys
import tempfile
import unittest
from pathlib import Path

import pair_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import tiny_fixture  # noqa: E402

# the shape of Qwen3.6's template: earlier turns' reasoning only with preserve_thinking, the last
# turn's always; enable_thinking only in the generation prompt
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n"
            "{% if m.reasoning_content is defined and m.reasoning_content and "
            "((preserve_thinking is defined and preserve_thinking is true) or loop.last) %}"
            "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}{% endif %}")
CONVERSATION = {"id": "c1", "source": "eval", "messages": [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer.", "reasoning_content": "Earlier reasoning."},
    {"role": "user", "content": "Second question?"},
    {"role": "assistant", "content": "Second answer.", "reasoning_content": "Last reasoning."}]}


def template_gguf(path, template=TEMPLATE):
    gguf = tiny_fixture._gguf_module()
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["a", "b", "<|endoftext|>", "<|im_end|>"])
    writer.add_token_types([1, 1, 3, 3])
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_chat_template(template)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class RenderTest(unittest.TestCase):
    def test_the_server_keeps_earlier_reasoning_where_the_template_reads_it(self):
        with tempfile.TemporaryDirectory() as directory:
            gguf = template_gguf(Path(directory) / "m.gguf")
            default = pair_eval.render(CONVERSATION, gguf)
            server = pair_eval.render(CONVERSATION, gguf, server=True)
            again = pair_eval.render(CONVERSATION, gguf, server=True)
        self.assertNotIn("Earlier reasoning.", default["text"])
        self.assertIn("<think>Earlier reasoning.</think>First answer.", server["text"])
        self.assertIn("<think>Last reasoning.</think>Second answer.", default["text"])
        self.assertEqual(server["text"], again["text"])                     # the clock is fixed
        self.assertEqual(default["document_id"], server["document_id"])      # the same conversation
        self.assertGreater(len(server["segments"]), len(default["segments"]))  # the reasoning is a segment there

    def test_a_template_without_those_variables_renders_the_same(self):
        plain = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        with tempfile.TemporaryDirectory() as directory:
            gguf = template_gguf(Path(directory) / "m.gguf", plain)
            self.assertEqual(pair_eval.render(CONVERSATION, gguf)["text"],
                             pair_eval.render(CONVERSATION, gguf, server=True)["text"])


def side(sha):
    return {"files": {"f32": {"file": f"{sha}.npy", "sha256": sha}, "f16": {"file": f"{sha}.f16.npy", "sha256": sha + "h"}}}


class ExtraHubTest(unittest.TestCase):
    def test_labels_and_conditions(self):
        self.assertEqual(pair_eval.parse_extra_hubs(["consensus=/h/a", "match_z=/h/b"]),
                         {"consensus": Path("/h/a"), "match_z": Path("/h/b")})
        for bad in (["x"], ["=/h"], ["a=/h", "a=/g"], ["a.b=/h"], ["a@z=/h"], ["é=/h"]):
            with self.assertRaises(ValueError, msg=bad):
                pair_eval.parse_extra_hubs(bad)
        self.assertEqual(pair_eval.parse_extra_conditions("hub_prefix"), ["hub_prefix"])
        self.assertEqual(pair_eval.parse_extra_conditions("hub_prefix,hub"), ["hub", "hub_prefix"])
        for bad in ("", "floor", "hub,hub_table_prefix"):
            with self.assertRaises(ValueError, msg=bad):
                pair_eval.parse_extra_conditions(bad)

    def test_the_floor_follows_the_target_side_only(self):
        models = {"qwen3": {}, "qwen3.6-35b-a3b": {}}
        v2 = {"format_version": 2, "models": models, "sides": {
            "qwen3.k.encoder": side("e1"), "qwen3.k.decoder": side("d1"), "qwen3.v.decoder": side("d2"),
            "qwen3.6-35b-a3b.k.decoder": side("x1"), "qwen3.6-35b-a3b.k.encoder": side("x2")}}
        digests = pair_eval.target_side_digests(v2, "qwen3")
        self.assertEqual(digests, {"qwen3.k.decoder": "d1", "qwen3.v.decoder": "d2"})   # not the longer name's
        changed = {**v2, "sides": {**v2["sides"], "qwen3.k.encoder": side("e9"), "qwen3.6-35b-a3b.k.encoder": side("x9")}}
        self.assertEqual(pair_eval.target_side_digests(changed, "qwen3"), digests)      # an encoder: the same floor
        changed["sides"]["qwen3.v.decoder"] = side("d9")
        self.assertNotEqual(pair_eval.target_side_digests(changed, "qwen3"), digests)
        v2_heads = {**v2, "files": {"qwen3.csa": {"file": "qwen3.csa.npz", "sha256": "h1"}}}
        self.assertEqual(pair_eval.target_side_digests(v2_heads, "qwen3")["qwen3.csa"], "h1")
        v1 = {"format_version": 1, "models": models, "files": {
            "qwen3.k": {"file": "qwen3.k.npz", "sha256": "f1"}, "qwen3.indexer": {"file": "i.npz", "sha256": "f2"},
            "qwen3.6-35b-a3b.k": {"file": "o.npz", "sha256": "f3"}}}
        self.assertEqual(pair_eval.target_side_digests(v1, "qwen3"), {"qwen3.k": "f1", "qwen3.indexer": "f2"})


class ResumeTest(unittest.TestCase):
    def test_saved_states_are_kept_only_for_the_same_input_and_when_complete(self):
        records = [{"document_id": "d1", "text": "x", "save_at": [512], "expected_token_sha256": "s1"},
                   {"document_id": "d2", "text": "y", "save_at": [1024], "expected_token_sha256": "s2"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "states-t"
            self.assertFalse(pair_eval.saved_states(output, records))
            output.with_name("states-t.input.jsonl").write_text(
                "".join(pair_eval.json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            for document, at in (("d1", 512), ("d2", 1024)):
                (output / "run" / document).mkdir(parents=True)
                (output / "run" / document / f"{at}.state").write_bytes(b"s")
            self.assertTrue(pair_eval.saved_states(output, records))
            self.assertFalse(pair_eval.saved_states(output, records[:1]))            # another input
            (output / "run" / "d2" / "1024.state").unlink()
            self.assertFalse(pair_eval.saved_states(output, records))                # a state missing
            (output.parent / ".states-t.partial-x1").mkdir()                       # what a stopped probe leaves
            output.with_name("states-t.log").write_text("log")
            other = output.with_name("states-s")
            other.mkdir()
            pair_eval.remove_probe_output(output)
            self.assertEqual(sorted(p.name for p in output.parent.iterdir()), ["states-s"])

    def test_scores_are_kept_only_for_translations_built_from_the_same_inputs(self):
        """A resumed run reuses a document's scores only when its hubs (every file of the source and the target,
        by label), its scoring input and its options are those the scores were made from: an --extra-hub label
        pointing at a bundle whose source encoder changed since is scored again."""
        models = {"qwen3": {}, "m": {}, "other": {}}
        manifest = {"format_version": 2, "models": models, "sides": {
            "m.k.encoder": side("e1"), "qwen3.k.decoder": side("d1"), "other.k.encoder": side("o1")}}
        digests = pair_eval.model_digests(manifest, ("m", "qwen3"))
        self.assertEqual(set(digests), {"m.k.encoder", "qwen3.k.decoder"})                 # not another model's
        item = {"document_id": "d1", "text": "x", "conditions": [{"name": "hub_prefix.r", "state_file": "/b/x.state"}]}
        options = {"table": "t1", "recurrent": "mean", "recurrent_mean": "m1", "recurrent_group": None, "tail": 2048}
        digest = pair_eval.build_digest({"": digests, "r": digests}, item, options)
        reencoded = {**manifest, "sides": {**manifest["sides"], "m.k.encoder": side("e2")}}
        other_changed = {**manifest, "sides": {**manifest["sides"], "other.k.encoder": side("o2")}}
        self.assertNotEqual(pair_eval.build_digest({"": digests, "r": pair_eval.model_digests(reencoded, ("m", "qwen3"))},
                                                   item, options), digest)
        self.assertEqual(pair_eval.build_digest({"": digests, "r": pair_eval.model_digests(other_changed, ("m", "qwen3"))},
                                                item, options), digest)
        self.assertNotEqual(pair_eval.build_digest({"": digests, "r": digests}, item, {**options, "tail": 1024}), digest)
        v1 = {"format_version": 1, "models": models, "files": {"m.k": {"file": "m.k.npz", "sha256": "f1"}}}
        self.assertEqual(pair_eval.model_digests(v1, ("m",)), {"m.k": "f1"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores" / "d1"
            (output / "scores" / "d1").mkdir(parents=True)
            for name in ("native", "hub_prefix.r"):
                (output / "scores" / "d1" / f"{name}.npy").write_bytes(b"x")
            names = ("native", "hub_prefix.r")
            self.assertFalse(pair_eval.kept_scores(output, "d1", names, digest))           # no record: a run before
            pair_eval.write_json_file(output.with_name("d1.built.json"), {"sha256": digest})
            self.assertTrue(pair_eval.kept_scores(output, "d1", names, digest))
            self.assertFalse(pair_eval.kept_scores(output, "d1", names, "another"))
            self.assertFalse(pair_eval.kept_scores(output, "d1", names + ("floor",), digest))  # a condition missing
            self.assertIsNone(pair_eval.stored_build(output, "d1", names, digest))    # no translation report saved
            pair_eval.write_json_file(output.with_name("d1.built.json"), {
                "sha256": digest, "build": {"mapped_cells": 10, "fill_cells": 2, "seconds": 1.0}})
            self.assertEqual(pair_eval.stored_build(output, "d1", names, digest)["mapped_cells"], 10)
            self.assertIsNone(pair_eval.stored_build(output, "d1", names, "another"))   # other inputs: translated
            pair_eval.remove_probe_output(output)
            self.assertFalse(output.with_name("d1.built.json").exists())
            self.assertIsNone(pair_eval.stored_build(output, "d1", names, digest))

class ResumeMainTest(unittest.TestCase):
    """pair_eval.main --resume with the probe, the models and the translator replaced by stand-ins: a run stopped
    after scoring some documents translates and scores only the others when resumed, and its report equals an
    uninterrupted run's."""

    def run_main(self, root, stop_after=None, resume=True):
        import types
        from unittest import mock
        import numpy as np
        ran = {"translated": [], "scored": []}
        documents = [f"{n:064x}" for n in (1, 2, 3)]
        corpus = root / "corpus.jsonl"
        corpus.write_text("".join(pair_eval.json.dumps({"id": d, "source": "chat", "messages": []}) + "\n"
                                  for d in documents))

        def render(conversation, gguf, server=False):
            return {"document_id": conversation["id"], "text": "x" * 50, "segments": [(0, 0, 10), (1, 10, 50)],
                    "segment_keys": ["0:system", "1:user"]}

        def tokenize(probe, gguf, records, ctx, workdir):
            return {r["document_id"]: {"token_id": list(range(40)), "byte_end": list(range(1, 41))} for r in records}

        def probe(probe_path, artifacts, mode, model, records, output, *, ctx, extra=(), guard=None):
            output.mkdir(parents=True, exist_ok=True)
            path = output.parent / (output.name + ".input.jsonl")
            path.write_text("".join(pair_eval.json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            for r in records:
                if mode == "save-state":
                    (output / r["document_id"]).mkdir(exist_ok=True)
                    for at in r["save_at"]:
                        (output / r["document_id"] / f"{at}.state").write_bytes(b"s")
                    continue
                if stop_after is not None and len(ran["scored"]) == stop_after:
                    raise SystemExit("stopped by the guard")
                ran["scored"].append(r["document_id"])
                for c in r["conditions"]:
                    (output / "scores" / r["document_id"]).mkdir(parents=True, exist_ok=True)
                    value = 0.1 if c["name"] == "native" else 0.2
                    np.save(output / "scores" / r["document_id"] / f"{c['name']}.npy", np.full((4, 3), value))

        def translate_(source_state, side, target, hubs, **options):
            ran["translated"].append(side.document_id)
            return types.SimpleNamespace(to_bytes=lambda: b"t"), {"mapped_cells": 7, "fill_cells": 3, "seconds": 0.1}

        def side(name, spec, record, tokens, gguf=None):
            return types.SimpleNamespace(document_id=record["document_id"])
        manifest = {"format_version": 2, "models": {"a": {}, "b": {}}, "sides": {}}
        spec = types.SimpleNamespace(n_swa=None)
        patches = [mock.patch.object(pair_eval, "render", render),
                   mock.patch.object(pair_eval.collect_v2, "tokenize", tokenize),
                   mock.patch.object(pair_eval, "probe", probe),
                   mock.patch.object(pair_eval, "verify_runtime", lambda path: ("rev", None, [])),
                   mock.patch.object(pair_eval.kvstate.ModelSpec, "from_gguf", lambda gguf, llama_revision=None: spec),
                   mock.patch.object(pair_eval.kvstate, "llama_revision_of", lambda revision: revision),
                   mock.patch.object(pair_eval.kvstate, "read_file", lambda path, spec, strict=True: object()),
                   mock.patch.object(pair_eval.kvstate, "write_new", lambda path, data: Path(path).write_bytes(data)),
                   mock.patch.object(pair_eval.handoff, "handoff_position", lambda record, tokens: 2000),
                   mock.patch.object(pair_eval.handoff, "final_message_start", lambda record: 10),
                   mock.patch.object(pair_eval.translate, "translate", translate_),
                   mock.patch.object(pair_eval.translate, "Side", side),
                   mock.patch.object(pair_eval.recurrent_init, "translate_options", lambda *a, **k: {}),
                   mock.patch.object(pair_eval.hub_module, "open_hub",
                                     lambda directory: types.SimpleNamespace(manifest=manifest, names=["a", "b"],
                                                                             directory=directory))]
        argv = ["--corpus", str(corpus), "--hub", str(root / "hub"), "--source", "a=/a.gguf", "--target", "b=/b.gguf",
                "--probe", str(root / "probe"), "--output", str(root / "run"), "--min-prefix", "1000",
                "--max-prefix", "3000", "--tail", "0"] + (["--resume"] if resume else [])
        import contextlib
        import io
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                pair_eval.main(argv)
            except SystemExit as error:
                ran["stopped"] = str(error)
        return ran

    def test_a_resumed_run_translates_and_scores_only_what_is_left(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_main(root, stop_after=2)
            self.assertEqual(first.get("stopped"), "stopped by the guard")
            self.assertEqual(len(set(first["translated"])), 3)
            resumed = self.run_main(root)
            self.assertNotIn("stopped", resumed)
            self.assertEqual(resumed["scored"], [f"{3:064x}"])                       # only the unscored one
            self.assertEqual(set(resumed["translated"]), {f"{3:064x}"})              # and only it translated
            report = pair_eval.json.loads((root / "run" / "report.json").read_text())
            self.assertEqual(len(report["documents"]), 3)
            self.assertTrue(all(row["mapped_cells"] == 7 and row["H"] == 10 for row in report["documents"]))
            self.assertIn("built_from", report)
            with tempfile.TemporaryDirectory() as other:
                whole = self.run_main(Path(other))                                   # an uninterrupted run
                self.assertEqual(len(whole["scored"]), 3)
                again = pair_eval.json.loads((Path(other) / "run" / "report.json").read_text())
            self.assertEqual(again["summary"], report["summary"])
            with tempfile.TemporaryDirectory() as other:
                single = self.run_main(Path(other), resume=False)                     # one probe run for all
                self.assertEqual(len(single["scored"]), 3)
                once = pair_eval.json.loads((Path(other) / "run" / "report.json").read_text())
            self.assertEqual(once["summary"], report["summary"])


if __name__ == "__main__":
    unittest.main()
