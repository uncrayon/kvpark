"""Which switches the bridge can translate, and why it skips the others.

Each rule turns a switch the translator cannot handle into a skip with a reason; the target
then prefills the conversation natively, which is always correct. Architecture rules live in
one table (``RULES``) that later stages extend as the engine learns new cache families;
``TRANSLATABLE`` lists the architectures the engine translates end to end today: its state
codec (experiments/linear_transfer/kvstate.py ``ARCHITECTURES``) reads and writes them, the
translator fills every section their states hold (MLA latent rows, DSA indexer keys,
recurrent rows reset to zero) and the hub can hold views of their caches (kv_probe captures,
or captures from saved states where a graph names no pre-RoPE key). Each was translated into
itself through an exact hub on tiny fixtures and restored in llama.cpp
(experiments/universal_kv/test_families.py), on both pinned runtimes (85d5703a3, which the
production Qwen3.8 server still runs, and a7b94df2c): the codec reads either's states, and their
layouts differ only for deepseek2/mistral4 nextn layers, qwen4exp's indexer and bailingmoe3 with
rope sections. The GQA family (``GQA``: one plain or a global and a sliding-window cache, per-layer
KV heads and head sizes, per-layer-kind RoPE; Gemma 4 with its proportional RoPE and shared KV
layers, Laguna with plain RoPE on its sliding layers) was checked the same way on test-llama-archs
models and gguf-py Gemma models, and the real Gemma-4 26B-A4B and 31B saves parse and write back
byte for byte; Llama, Mistral 3 and Gemma 4 rotate with their GGUF's rope_freqs.weight, which the
engine reads into its model profiles (Phi-3/4 LongRoPE is skipped, see ``model``). GLM5-Next
(``glm5-next``, a7b94df2c only: NoPE MLA latents, KDA, k-pool indexer rows holding a per-token
gate) was checked on a test-llama-archs model. A DSA target whose prompt is longer than its indexer top_k also needs the hub's indexer
ridge; without one the engine refuses the job with "no indexer keys" (``indexer`` below decides
the same before the engine runs; it also covers qwen4exp's and GLM5-Next's k-pool indexers).

DeepSeek V4 (``deepseek4``) is translatable as a target only: its pooled caches are synthesized
from per-token compressor inputs the hub decodes (experiments/universal_kv/dsv4_target.py, which
translates [0, T0) with T0 rounded down to a multiple of 128; the rest is native tail). As a
source it keeps per-token latents only for its last 128 tokens, so it is skipped. A DeepSeek V4
target needs the hub's head decoders for it (``heads``), and a translated slot of it is restored
only on a server without a draft (``draft``).

Beyond its architecture a model is skipped for what its GGUF holds (``model``): a vocabulary that
appends EOS to every text it tokenizes (``tokenizer.ggml.add_eos_token``: an embedding model such as
gte-Qwen2, whose chat prompts the engine cannot spell back), and LongRoPE (Phi-3/4, and Llama or
Mistral 3 GGUFs converted with it), whose frequency factors (``rope_factors_long``/``_short``) llama.cpp
picks by the context size, which the engine does not model. The engine reads the vocabularies of ``TOKENIZERS`` (GPT-2 byte-level BPE) and, from its
"vocabularies" feature on, ``FEATURE_TOKENIZERS`` (Gemma 4's BPE and SentencePiece); ``tokenizer``
skips the others, and those for an engine without the feature.

The architecture and the rest come from the model's GGUF header (``general.architecture``, the
tokenizer keys, the RoPE scaling type, and for the architectures that take RoPE factor tensors the
tensor names), read here with a minimal standard-library GGUF reader.
"""

import functools
import os
from pathlib import Path
import re
import struct

# no per-token cache at all: the whole conversation lives in a fixed-size recurrent state
PURE_RECURRENT = frozenset({"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"})
# what the engine translates: Qwen3.5/3.8 hybrids (gated DeltaNet + attention), Muse-Glimmer
# (sliding window + global attention) and dense GQA Qwen3; the MLA family (deepseek2, mistral4,
# hy_v4; DSA: deepseek32, glm-dsa, dots3note) and its K-only hybrids with KDA layers (kimi-linear,
# kimi-k3, bailingmoe3, glm5-next); the GQA hybrids with GatedDeltaNet, Mamba, short-conv or lightning
# layers; the GQA family (Llama, Mistral 3, Qwen3-MoE, Hy3, Phi-3/4, Gemma 3/4, gpt-oss, Laguna, MiMo)
MLA = frozenset({"deepseek2", "mistral4", "hy_v4", "deepseek32", "glm-dsa", "dots3note", "kimi-linear", "kimi-k3",
                 "bailingmoe3", "glm5-next"})
GQA_HYBRIDS = frozenset({"qwen3next", "qwen35moe", "qwen4exp", "minimax-01", "jamba", "falcon-h1", "granitehybrid",
                         "nemotron_h", "nemotron_h_moe", "plamo2", "lfm2", "lfm2moe"})
# plain-attention models that take the server's RoPE settings (kvstate.py GQA_ARCHITECTURES): one cache, or
# a global and a sliding-window one
GQA = frozenset({"llama", "mistral3", "qwen3moe", "hy_v3", "phi3", "gemma3", "gemma4", "gpt-oss", "laguna", "mimo2"})
# translated into, never out of (RULES skips them as sources)
TARGET_ONLY = frozenset({"deepseek4"})
TRANSLATABLE = frozenset({"qwen35", "muse-glimmer", "qwen3"}) | MLA | GQA_HYBRIDS | GQA | TARGET_ONLY
# hybrids: a recurrent state (gated DeltaNet, Mamba, short-conv, lightning or KDA layers) beside the
# attention cache, which a translated state holds as one cell at T0 - 1: zeros, or the model's mean
# of native cells (kvpark bridge add --recurrent-mean)
HYBRID = frozenset({"qwen35", "kimi-linear", "kimi-k3", "bailingmoe3", "glm5-next"}) | GQA_HYBRIDS
# the hub's head kinds of a DeepSeek V4 target (experiments/universal_kv/hub.py HEAD_KINDS)
DSV4_HEADS = ("csa", "hca", "lid")
# tokenizer.ggml.model values the engine reads (experiments/universal_kv/engine.py TOKENIZER_MODELS):
# GPT-2 byte-level BPE always; Gemma 4's BPE and SentencePiece ("llama": Jamba) from the engine's
# "vocabularies" feature on (an older engine fails on them after the source's slot was saved)
TOKENIZERS = frozenset({"gpt2"})
FEATURE_TOKENIZERS = {"vocabularies": frozenset({"gemma4", "llama"})}
# LongRoPE: llama_model::get_rope_factors picks one of these by the context size when the GGUF has no
# rope_freqs.weight (src/llama-model.cpp:2343-2356 at a7b94df2c), which the engine does not model
# (kvstate.gguf_rope_freq_factors refuses them for ROPE_FACTOR_ARCHITECTURES, and kvstate.rope_runtime a
# "longrope" rope.scaling.type for every architecture). phi3 loads them for LongRoPE models, llama and
# mistral3 when the GGUF's scaling type is longrope (src/models/llama.cpp:61, mistral3.cpp:56)
LONGROPE_TENSORS = ("rope_factors_long.weight", "rope_factors_short.weight")
# the architectures whose RoPE takes factor tensors (kvstate.ROPE_FACTOR_TENSOR_ARCHITECTURES)
ROPE_FACTOR_ARCHITECTURES = frozenset({"granitehybrid", "llama", "mistral3", "gemma4", "phi3"})

# (side, architectures, skip code, reason); side is "source", "target" or "either"; the first
# matching rule wins, and architectures no rule names must be in TRANSLATABLE
RULES = (
    ("either", PURE_RECURRENT, "no-per-token-cache",
     "{model} ({arch}) keeps one fixed-size recurrent state and no per-token cache, so there is "
     "nothing to translate"),
    ("source", frozenset({"deepseek4"}), "dsv4-source",
     "{model} ({arch}) keeps per-token latents only for its last 128 tokens and pools the rest, "
     "so its cache cannot be read back token by token"),
)

# /props "modalities" keys of the OpenAI content parts llama-server accepts
# (tools/server/server-common.cpp:1154-1199 at llama.cpp a7b94df2c)
MODALITIES = ("vision", "audio", "video")


def check(source, target):
    """None when the engine can translate between these models, else (skip code, reason).

    `source` and `target` are (name, architecture) pairs; an architecture of None means the
    GGUF could not be read."""
    for side, (name, arch) in (("source", source), ("target", target)):
        for rule_side, archs, code, reason in RULES:
            if rule_side in (side, "either") and arch in archs:
                return code, reason.format(model=name, arch=arch)
    for name, arch in (source, target):
        if arch is None:
            return "unsupported-arch", f"cannot read the architecture of {name}'s GGUF file"
        if arch not in TRANSLATABLE:
            return "unsupported-arch", (f"the bridge engine cannot translate {arch} caches yet ({name}; "
                                        f"supported: {', '.join(sorted(TRANSLATABLE))})")
    return None


def model(name, arch, gguf):
    """None when nothing in the model's GGUF beyond its architecture stops a translation, else (skip
    code, reason): "embedding-model" for a vocabulary that appends EOS to every text it tokenizes (an
    embedding model such as gte-Qwen2; the engine could not spell its prompts back), "rope-factors" for
    LongRoPE (a "longrope" rope.scaling.type, or LONGROPE_TENSORS without rope_freqs.weight in a
    ROPE_FACTOR_ARCHITECTURES model: Phi-3/4, LongRoPE Llama or Mistral 3). A GGUF that cannot be read
    passes (``check`` reports it)."""
    facts = gguf_facts(gguf)
    if facts is None:
        return None
    if facts["add_eos"]:
        return "embedding-model", (f"{name}'s vocabulary appends EOS to every text it tokenizes "
                                   "(tokenizer.ggml.add_eos_token: an embedding model such as gte-Qwen2), so the "
                                   "bridge does not switch conversations into or out of it")
    if facts["longrope"]:
        return "rope-factors", (f"{name} ({arch}) uses LongRoPE: llama.cpp picks its frequency factors "
                                "(rope_factors_long/short) by the context size, which the bridge engine does not model")
    return None


def tokenizer(source, target, features):
    """None when the engine reads both models' vocabularies, else ("tokenizer", reason). `source` and
    `target` are (name, tokenizer.ggml.model) pairs (None: unread, left to the engine); `features` are
    those of the engine that runs the job (engine_client.features): TOKENIZERS always, and those of
    FEATURE_TOKENIZERS whose feature it has."""
    readable = set(TOKENIZERS).union(*(models for feature, models in FEATURE_TOKENIZERS.items() if feature in features))
    for name, tokenizer_model in (source, target):
        if tokenizer_model is None or tokenizer_model in readable:
            continue
        needed = next((feature for feature, models in FEATURE_TOKENIZERS.items() if tokenizer_model in models), None)
        if needed is not None:
            return "tokenizer", (f"{name}'s vocabulary ({tokenizer_model}) needs an engine with the \"{needed}\" "
                                 "feature; this engine reads GPT-2 byte-level BPE vocabularies only")
        return "tokenizer", (f"the bridge engine cannot read {name}'s vocabulary yet (tokenizer model "
                             f"{tokenizer_model}; it reads {', '.join(sorted(readable))})")
    return None


def indexer(target, gguf, ridges, prompt):
    """None when the engine can fill the target's lightning-indexer keys, else (skip code, reason).

    A DSA target (deepseek32, glm-dsa, hy_v4 with an indexer, dots3note; qwen4exp's block indexer)
    caches indexer keys that only matter once a query sees more than ``attention.indexer.top_k``
    cells. Without the hub's indexer ridge for the target the engine writes zero keys, which are
    exact only while every query of the prompt sees at most top_k cells, so it refuses a longer
    prompt (experiments/universal_kv/translate.py). `ridges` names the models with a ridge (the hub
    manifest's "indexer_ridges"); `prompt` is the length of the target's prompt in tokens (the
    whole conversation the target will process, the natively prefilled tail included, not just the
    translated part). Checked before the engine runs, from the GGUF header alone. A DeepSeek V4
    target keeps its indexer keys in pooled LID rows, which its head decoders fill (``heads``)."""
    arch = gguf_architecture(gguf)
    if arch in TARGET_ONLY:
        return None
    top_k = gguf_value(gguf, f"{arch}.attention.indexer.top_k") or 0
    if not isinstance(top_k, int) or top_k <= 0 or target in ridges or prompt <= top_k:
        return None
    return "no-indexer-keys", (f"{target} ({arch}) needs indexer keys for a prompt of {prompt} tokens (its indexer "
                               f"reads the keys beyond {top_k} cells) and the hub has no indexer ridge for it")


def heads(target, arch, hub_manifest):
    """None when the hub can fill a DeepSeek V4 target's compressed caches, else (skip code, reason).

    The engine decodes each compressed layer's compressor inputs with the hub's head decoders for
    the target (manifest "heads": kind -> {"models": {name: ...}}, written by fit_hub.py --heads);
    without them there is nothing to pool. Every DeepSeek V4 model (V4, V4-Pro, V4-Flash) has CSA
    layers (with their lightning indexer) and HCA layers, so all of DSV4_HEADS are required: with
    one missing the engine would refuse the job. Other architectures need none."""
    if arch not in TARGET_ONLY:
        return None
    found = {kind for kind, entry in ((hub_manifest or {}).get("heads") or {}).items()
             if target in ((entry or {}).get("models") or {})}
    missing = [kind for kind in DSV4_HEADS if kind not in found]
    if not missing:
        return None
    what = "no head decoders" if not found else f"no {', '.join(missing)} head decoders"
    return "no-head-decoders", (f"{target} ({arch}) keeps pooled caches that the engine synthesizes from per-token "
                                f"compressor inputs, and the hub has {what} for it (fit_hub.py --heads)")


def draft(target, arch, layout):
    """None when a translated slot of this target can be restored with its draft setup, else (skip code,
    reason). `layout` is resume.classify_draft of the target's template .resume draft blob, or for an
    upstream server (no .resume) a DraftLayout named after its speculative types (blob None); None
    when the draft setup is unknown, which passes.

    DeepSeek V4 targets: no draft only. Both of its drafts keep their own sliding window over target
    features: the MTP head (layers >= n_layer, src/llama-model.cpp:2598-2618) and a DSpark draft
    (DFlash graph_dsv4, 2637-2659). Each is a plain iSWA cache over hparams that are not MLA, so it
    holds K and V rows (V never written by DSpark), and classify_draft reads both as a DFlash-shaped
    draft: an empty base and the window (what the kvpark-patched server saved for draft-mtp on the
    test fixture: one K/V layer). Neither has been restored around a translated DeepSeek V4 slot, so
    both are skipped (the research's plan, an empty draft rebuilt by the native tail, is unvalidated
    too), like every draft classify_draft calls "other". MTP and DSpark drafting also make the target
    keep rollback snapshots (n_rs_seq = the draft length): only the active plane is saved and a
    restore zeroes the others, which the native tail (a decode, never a rollback) would tolerate.
    Other architectures: the bridge's general draft rules (a Gemma 4 target's MTP head keeps no cells of its
    own, classify_draft's "shared" kind, and restores with the 8-byte blob of a native save)."""
    if arch not in TARGET_ONLY or layout is None or layout.kind == "none":
        return None
    what = ("its own sliding K/V window of target features: a DSpark draft or the MTP head"
            if layout.kind == "dflash" else layout.note)
    return "draft-unsupported", (f"{target} ({arch}) runs a draft whose state the bridge has not restored around a "
                                 f"translated DeepSeek V4 slot ({what}); run it without a draft to translate")


def media(conversation_media, *, source, target, source_modalities, target_modalities, source_saw,
          target_arch=None, replaced=()):
    """None when a conversation with media can be translated up to its first media message,
    else (skip code, reason).

    `conversation_media` maps each modality to the index of the first message holding it;
    the modalities are /props "modalities" of each server; `source_saw` is the index of the
    last message the source server has processed (its slot holds media from those);
    `replaced`: the modalities the target lacks whose media the bridge replaces by text
    surrogates (surrogates.py; the media policy is not "skip"), so the target needs no encoder
    for them, while the source's cache may still hold them."""
    for modality in sorted(conversation_media):
        if not target_modalities.get(modality) and modality not in replaced:
            return f"target-no-{modality}", (f"the conversation has {modality} input, {target}'s server has no "
                                             f"{modality} encoder (mmproj), and the media policy for it is skip "
                                             "(kvpark bridge media)")
    rule = pooled_media(target, target_arch, set(conversation_media) - set(replaced))
    if rule:
        return rule
    for modality, first in sorted(conversation_media.items()):
        if first <= source_saw and source_modalities.get(modality):
            return "media-source", (f"{source}'s cache holds the conversation's {modality} input, which the bridge "
                                    "engine cannot read yet")
    return None


def pooled_media(target, target_arch, modalities):
    """("dsv4-media", reason) when a DeepSeek V4 target would read media (`modalities`, those left
    for the target itself after any surrogates), else None."""
    if target_arch in TARGET_ONLY and modalities:
        # media after a synthesized pooled cache (blocks of 128 positions) has never been restored
        return "dsv4-media", (f"{target} ({target_arch}) pools its cache in blocks; media after a translated "
                              "pooled cache is untested, so the target prefills this conversation natively")
    return None


# -- GGUF header ---------------------------------------------------------------------------------

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_KV = 1 << 16
MAX_TENSORS = 1 << 24


def _read(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return data


def _string(stream):
    (size,) = struct.unpack("<Q", _read(stream, 8))
    if size > 1 << 30:
        raise ValueError("implausible GGUF string")
    return _read(stream, size)


def _skip(stream, kind):
    if kind in _SCALARS:
        stream.seek(struct.calcsize(_SCALARS[kind]), 1)
    elif kind == _STRING:
        (size,) = struct.unpack("<Q", _read(stream, 8))
        stream.seek(size, 1)
    elif kind == _ARRAY:
        item, count = struct.unpack("<IQ", _read(stream, 12))
        if item in _SCALARS:
            stream.seek(count * struct.calcsize(_SCALARS[item]), 1)
        elif item == _STRING:
            for _ in range(count):
                _skip(stream, _STRING)
        else:
            raise ValueError("unsupported GGUF array type")
    else:
        raise ValueError(f"unknown GGUF value type {kind}")


def _header(stream, path):
    """(tensor count, key-value count) of a GGUF file (v2/v3, little-endian) at its start."""
    if _read(stream, 4) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    version, n_tensors, n_kv = struct.unpack("<IQQ", _read(stream, 20))
    if version not in (2, 3) or n_kv > MAX_KV or n_tensors > MAX_TENSORS:
        raise ValueError(f"{path}: unsupported GGUF version {version}")
    return n_tensors, n_kv


def _value(stream, kind):
    if kind == _STRING:
        return _string(stream).decode("utf-8")
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(stream, struct.calcsize(fmt)))[0]
    _skip(stream, kind)
    return None


def gguf_values(path, keys):
    """{key: value} of the scalar or string metadata `keys` a GGUF file holds (arrays read as None)."""
    wanted, found = {key.encode(): key for key in keys}, {}
    with open(path, "rb") as stream:
        _, n_kv = _header(stream, path)
        for _ in range(n_kv):
            if len(found) == len(wanted):
                break
            name = _string(stream)
            (kind,) = struct.unpack("<I", _read(stream, 4))
            if name in wanted:
                found[wanted[name]] = _value(stream, kind)
            else:
                _skip(stream, kind)
    return found


def gguf_value(path, key):
    """A scalar or string metadata value of a GGUF file (v2/v3, little-endian), or None."""
    return gguf_values(path, [key]).get(key)


def gguf_tensor_names(path):
    """The names of the tensors a GGUF file holds (its tensor infos, after the metadata)."""
    names = []
    with open(path, "rb") as stream:
        n_tensors, n_kv = _header(stream, path)
        for _ in range(n_kv):
            _string(stream)
            (kind,) = struct.unpack("<I", _read(stream, 4))
            _skip(stream, kind)
        for _ in range(n_tensors):
            names.append(_string(stream).decode("utf-8"))
            (n_dims,) = struct.unpack("<I", _read(stream, 4))
            if n_dims > 8:
                raise ValueError(f"{path}: implausible tensor info")
            stream.seek(8 * n_dims + 4 + 8, 1)          # dimensions, type, offset
    return names


def _split_files(path, count):
    """Every file of a split GGUF (llama_split_path names) given its first; [path] when not split."""
    match = re.match(r"^(?P<prefix>.+)-(?P<number>\d{5})-of-(?P<count>\d{5})\.gguf$", Path(path).name)
    if not isinstance(count, int) or count <= 1 or match is None:
        return [Path(path)]
    return [Path(path).with_name(f"{match['prefix']}-{n:05d}-of-{count:05d}.gguf") for n in range(1, count + 1)]


def gguf_facts(path):
    """What the bridge reads from a model's GGUF besides its architecture, or None when it cannot be
    read: {"tokenizer": tokenizer.ggml.model or None, "add_eos": the vocabulary appends EOS to every
    text, "longrope": its rope.scaling.type is "longrope", or a ROPE_FACTOR_ARCHITECTURES GGUF holds
    LONGROPE_TENSORS and no rope_freqs.weight (in any file of a split GGUF)}. Kept per file identity
    (path, size, modification time)."""
    try:
        status = os.stat(path)
        return dict(_facts(os.path.realpath(path), status.st_size, status.st_mtime_ns))
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return None


@functools.lru_cache(maxsize=32)
def _facts(path, size, mtime_ns):
    # the architecture first (llama.cpp's converters write it among the first keys): it names the scaling key
    arch = gguf_values(path, ("general.architecture",)).get("general.architecture")
    scaling_key = f"{arch}.rope.scaling.type" if isinstance(arch, str) else None
    values = gguf_values(path, ("tokenizer.ggml.model", "tokenizer.ggml.add_eos_token", "split.count")
                         + ((scaling_key,) if scaling_key else ()))
    tokenizer_model = values.get("tokenizer.ggml.model")
    facts = {"tokenizer": tokenizer_model if isinstance(tokenizer_model, str) else None,
             "add_eos": bool(values.get("tokenizer.ggml.add_eos_token")),
             "longrope": scaling_key is not None and values.get(scaling_key) == "longrope"}
    if not facts["longrope"] and arch in ROPE_FACTOR_ARCHITECTURES:
        names = set()
        for part in _split_files(path, values.get("split.count")):
            if part.is_file():
                names.update(gguf_tensor_names(part))
        facts["longrope"] = "rope_freqs.weight" not in names and any(name in names for name in LONGROPE_TENSORS)
    return facts


def gguf_architecture(path):
    """general.architecture of a GGUF model (llama.cpp's converters write it first)."""
    arch = gguf_value(path, "general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{path} has no general.architecture")
    return arch
