"""Tiny models of the MLA/DSA family and the K-only hybrids, llama.cpp drivers and exact hubs for
the end-to-end tests in test_families.py.

Models come from tiny_fixture.create_mla_fixture (gguf-py: byte vocabulary, dense, random
weights) and from llama.cpp's test-llama-archs (random MoE models without a vocabulary:
token ids only), and amplified copies of either (Models.amplified: latents on the RMS shell and
attention loud enough in the logits for a restore to show it). The kvpark state driver
(linear_transfer/state_driver.cpp) saves and restores their states and decodes after a restore;
kvpark-kv-probe captures hub views of the gguf-py ones.
Every helper here is driven by environment variables (DRIVER, LLAMA_ARCHS, PROBE) and the tests
skip without them.

An *identity hub* holds one model whose encoder and decoder are exact identities per layer
(z is the concatenation of the model's own pre-RoPE K rows, or value rows), so translating a
model into itself through it reproduces the native cache up to the RoPE round trip and the
MLA latent shell: what remains between the restored and the native logits is the translator's
own arithmetic.
"""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import gguf_tensors  # noqa: E402
import hub as hub_module  # noqa: E402
import kvstate  # noqa: E402
from tiny_fixture import YARN_V2, copy_with_metadata, create_mla_fixture  # noqa: E402

DRIVER = os.environ.get("KVPARK_STATE_DRIVER")
_RUNS = itertools.count()
LLAMA_ARCHS = os.environ.get("KVPARK_LLAMA_ARCHS")
PROBE = os.environ.get("KVPARK_KV_PROBE")

# gguf-py fixtures: name -> (architecture, create_mla_fixture options); indexer top_k 8, byte vocabulary
TINY = {
    "deepseek2": ("deepseek2", {}),
    "deepseek2-yarn": ("deepseek2", {"yarn": YARN_V2}),
    "deepseek2-nextn": ("deepseek2", {"nextn": 1}),
    "mistral4-yarn": ("mistral4", {"yarn": YARN_V2}),
    "deepseek32": ("deepseek32", {}),
    "glm-dsa": ("glm-dsa", {"layers": 4, "indexer_types": [1, 0, 1, 0]}),
    "dots3note": ("dots3note", {"layers": 4}),
}
# test-llama-archs models (random weights, vocabulary of 128 ids, no tokenizer): the K-only MLA hybrids
# and hy_v4 (576/512 rows, MoE), and the hybrids with GQA attention next to recurrent layers
ARCHS = ("kimi-linear", "kimi-k3", "bailingmoe3", "hy_v4", "qwen3next", "qwen35moe", "qwen4exp", "minimax-01", "jamba",
         "falcon-h1", "granitehybrid", "nemotron_h", "nemotron_h_moe", "plamo2", "lfm2", "lfm2moe")
# The test-llama-archs weights keep MLA latents far inside the RMS shell (mean square << eps, where
# real models sit far outside it) and leave attention a whisper in the logits (zeroing every
# attention row moves them by ~1e-3 of their scale). Models.amplified copies a model with the c_kv
# outputs of attn_kv_a_mqa scaled by LATENT_GAIN and attn_output scaled by OUTPUT_GAIN.
LATENT_GAIN, OUTPUT_GAIN = 1000.0, 100.0


class Models:
    """GGUF paths by name under one directory, written on first use."""

    def __init__(self, root):
        self.root = Path(root)
        self.paths, self.vocab = {}, {}

    def path(self, name):
        if name not in self.paths:
            if name in TINY:
                architecture, options = TINY[name]
                path = self.root / f"{name}.gguf"
                create_mla_fixture(path, architecture, **options)
                self.paths[name], self.vocab[name] = path, 256
            elif name.startswith("archs-") and name[6:] in ARCHS:
                if not LLAMA_ARCHS:
                    raise unittest.SkipTest("set KVPARK_LLAMA_ARCHS for the test-llama-archs fixtures")
                out = self.root / "archs"
                out.mkdir(exist_ok=True)
                result = subprocess.run([LLAMA_ARCHS, "-a", name[6:], "-s", "7", "-o", str(out)], capture_output=True,
                                        text=True, timeout=600)
                if result.returncode != 0:
                    raise RuntimeError(f"test-llama-archs {name[6:]}: {result.stderr[-400:]}")
                # <arch>-moe.gguf or <arch>-dense.gguf, whichever the arch has
                self.paths[name], self.vocab[name] = sorted(out.glob(f"{name[6:]}-*.gguf"))[0], 128
            else:
                raise KeyError(name)
        return self.paths[name]

    def spec(self, name, runtime=None):
        return kvstate.ModelSpec.from_gguf(self.path(name), runtime=runtime)

    def amplified(self, name):
        """The name of a copy of model `name` with LATENT_GAIN and OUTPUT_GAIN applied (written on first use)."""
        key = name + "-amplified"
        if key not in self.paths:
            source, spec = str(self.path(name)), self.spec(name)
            _, tensors = gguf_tensors.reader(source)
            scaled = {}
            for layer in spec.attention_layers:
                latent, output = f"blk.{layer}.attn_kv_a_mqa.weight", f"blk.{layer}.attn_output.weight"
                if spec.k_only(layer) and latent in tensors:
                    weight = gguf_tensors.read_tensor(source, latent).copy()     # [kv_lora + rope, n_embd]
                    weight[:spec.kv_lora[layer]] *= LATENT_GAIN
                    scaled[latent] = weight
                if output in tensors:
                    scaled[output] = gguf_tensors.read_tensor(source, output) * OUTPUT_GAIN
            path = self.root / f"{key}.gguf"
            copy_with_metadata(source, path, drop_tensors=list(scaled), add_tensors=scaled)
            self.paths[key], self.vocab[key] = path, self.vocab[name]
        return key


class Driver:
    """kvpark-state-driver runs (save, restore-decode, capture) with one cache layout."""

    def __init__(self, models, workdir, *, cache="f32", fa="off", extra=()):
        self.models, self.workdir = models, Path(workdir)
        self.cache, self.fa, self.extra = cache, fa, list(extra)

    def run(self, mode, name, *args):
        command = [DRIVER, "--mode", mode, "--model", str(self.models.path(name)), "--cache-type", self.cache,
                   "--flash-attn", self.fa, "--threads", "4", *map(str, args), *self.extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"state driver {mode} {name}: {result.stderr[-1500:]}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    def fresh(self, stem):
        """A new path under the work directory (the driver never overwrites)."""
        return self.workdir / f"{stem}-{next(_RUNS)}"

    def save(self, name, prompt, next_ids=None):
        """(state path, native logits of next_ids or None)."""
        state = self.fresh("native").with_suffix(".state")
        args = ["--tokens", ",".join(map(str, prompt)), "--state", state]
        if next_ids is not None:
            args += ["--next", ",".join(map(str, next_ids)), "--logits", state.with_suffix(".npy")]
        self.run("save", name, *args)
        return state, (np.load(state.with_suffix(".npy")) if next_ids is not None else None)

    def restore(self, name, state, next_ids):
        """(logits after restoring `state` and decoding next_ids, the driver report)."""
        logits = Path(state).with_suffix(".restored.npy")
        report = self.run("restore-decode", name, "--state", state, "--next", ",".join(map(str, next_ids)),
                          "--logits", logits)
        return np.load(logits), report


def ascii_ids(rng, n, vocab):
    """n printable ASCII token ids (bytes of the gguf-py byte vocabulary; plain ids below 128 otherwise)."""
    return [int(t) for t in rng.integers(33, min(vocab, 127), n)]


def byte_side(ids, document_id="doc"):
    """(record, tokens) of a prompt whose token i is the byte ids[i]: one chat segment covering it all, so
    every token of a model translated into itself aligns."""
    text = bytes(ids).decode("ascii")
    record = {"document_id": document_id, "text": text, "segments": [[0, 0, len(text)]], "segment_keys": ["0:content"],
              "split": "test", "final_message_segments": []}
    tokens = {"token_id": np.asarray(ids, dtype=np.int32), "byte_end": np.arange(1, len(ids) + 1, dtype=np.int64),
              "special": np.zeros(len(ids), dtype=np.uint8)}
    return record, tokens


def view_widths(spec):
    """{kind: {layer: width}} of a model's hub views (an MLA row and its c_kv view), trunk attention layers."""
    layers = [layer for layer in spec.attention_layers if spec.layer_types[layer] != "nextn"]
    return {kind: {layer: int(np.prod(spec.head_shape(layer, kind))) for layer in layers} for kind in hub_module.KINDS}


def identity_hub(name, spec):
    """{kind: HubKind} of one model whose encoder and decoders are identities: z = the concatenated
    rows of its layers, decoded back to each layer's rows exactly."""
    hubs = {}
    for kind, widths in view_widths(spec).items():
        layers = sorted(widths)
        sizes = [widths[layer] for layer in layers]
        total = sum(sizes)
        eye = np.eye(total, dtype=np.float32)
        offsets = np.cumsum([0] + sizes)
        transform = hub_module.ViewTransform(layers, [np.zeros(w, np.float32) for w in sizes],
                                             [np.eye(w, dtype=np.float32) for w in sizes], eye.copy(),
                                             np.ones(total, np.float32))
        decoders = [np.ascontiguousarray(eye[:, offsets[i]:offsets[i + 1]]) for i in range(len(layers))]
        hubs[kind] = hub_module.HubKind([name], {name: transform}, {name: eye.copy()}, np.ones(total),
                                        {name: [np.zeros(w, np.float32) for w in sizes]},
                                        {name: np.zeros(total, np.float32)}, {name: decoders}, {name: {"alpha": 0.0}})
    return hubs


class StateRowsView:
    """A hub.CaptureView stand-in over native f32 states of one DSA model: K = pre-RoPE cache rows of
    every attention layer, ik = pre-RoPE indexer keys (kvstate.pre_rope_keys), rows of every state
    in order, documents split train/dev by `splits` (one per state)."""

    def __init__(self, name, spec, states, splits):
        self.name, self.spec = name, spec
        self.data = {}
        rows, split_of = {"k": {}, hub_module.INDEXER_KEY: {}}, []
        for state, split in zip(states, splits):
            for part, layout in zip(state.state.parts, spec.layouts(state.state.flags)):
                if layout.kind != "kv":
                    continue
                stream = part.stream()
                positions = np.asarray(stream.positions, dtype=np.int64)
                key = "k" if layout.role == "attention" else hub_module.INDEXER_KEY
                for layer in part.layers:
                    if spec.layer_types[layer] == "nextn":
                        continue
                    pre = kvstate.pre_rope_keys(spec, layer, stream.k[layer], positions, role=layout.role)
                    rows[key].setdefault(layer, []).append(pre.reshape(len(positions), -1))
            split_of += [split] * len(state.tokens)
        self.data = {(key, layer): np.concatenate(blocks) for key, by_layer in rows.items()
                     for layer, blocks in by_layer.items()}
        self.splits = np.asarray(split_of)

    def layers(self, kind):
        return sorted(layer for key, layer in self.data if key == kind)

    def width(self, kind, layer):
        return self.data[kind, layer].shape[1]

    def rows(self, kind, layer, rows):
        return self.data[kind, layer][rows]

    def split_rows(self, split):
        return np.flatnonzero(self.splits == split)


# a chat template without special tokens: every model renders (and tokenizes) the same bytes
TEMPLATE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")
WORDS = ("tide", "harbor", "moon", "north", "coast", "ebb", "wave", "shore", "boat", "wind", "rain", "fog", "sail",
         "rope", "anchor", "gull", "salt", "reef", "dune", "cliff")


def conversations(count, seed=0, splits=None):
    """Synthetic ASCII conversations (system, user, assistant) in chat.py's corpus format."""
    rng = np.random.default_rng(seed)
    splits = splits or ["train"] * count
    out = []
    for number, split in enumerate(splits):
        def words(n):
            return " ".join(WORDS[i] for i in rng.integers(0, len(WORDS), n)).capitalize() + "."
        out.append({"id": f"families:{seed}:{number}", "source": "families", "split": split, "tools": None,
                    "messages": [{"role": "system", "content": "You answer briefly about the sea."},
                                 {"role": "user", "content": words(int(rng.integers(8, 16)))},
                                 {"role": "assistant", "content": words(int(rng.integers(10, 20)))}]})
    return out


def rendered_records(conversations_, model_name="fixture"):
    """chat.py rendered records of conversations with TEMPLATE."""
    import chat
    return [chat.rendered_record(c, model_name, chat.render(c, TEMPLATE)) for c in conversations_]


def probe_supports(model, workdir):
    """Whether PROBE knows the model's architecture (probes built before its table row refuse it)."""
    docs = Path(workdir) / "probe-supports.jsonl"
    docs.write_text(json.dumps({"document_id": "0" * 64, "text": "probe"}) + "\n", encoding="utf-8")
    result = subprocess.run([PROBE, "--mode", "tokenize", "--model", str(model), "--input", str(docs), "--output",
                             str(Path(workdir) / "probe-supports")], capture_output=True, text=True, timeout=120)
    return "kv probe supports only" not in result.stderr


def probe_collect(model, records, output, *, keys=None, ctx=512):
    """A kvpark-kv-probe collect directory (k/v, or `keys`) of rendered records."""
    output = Path(output)
    docs = output.with_name(output.name + ".input.jsonl")
    docs.write_text("".join(json.dumps({"document_id": r["document_id"], "text": r["text"]}) + "\n" for r in records),
                    encoding="utf-8")
    command = [PROBE, "--mode", "collect", "--model", str(model), "--input", str(docs), "--output", str(output),
               "--ctx", str(ctx), "--batch", "64", "--threads", "4"]
    if keys is not None:
        spec = output.with_name(output.name + ".spec.json")
        spec.write_text(json.dumps({key: "all" for key in keys}), encoding="utf-8")
        command += ["--capture-spec", str(spec)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"kv probe collect: {result.stderr[-1500:]}")
    return output
