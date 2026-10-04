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
(experiments/universal_kv/test_families.py). A DSA target whose prompt is longer than its
indexer top_k also needs the hub's indexer ridge; without one the engine refuses the job with
"no indexer keys" (``indexer`` below decides the same before the engine runs).

DeepSeek V4 (``deepseek4``) is translatable as a target only: its pooled caches are synthesized
from per-token compressor inputs the hub decodes (experiments/universal_kv/dsv4_target.py, which
translates [0, T0) with T0 rounded down to a multiple of 128; the rest is native tail). As a
source it keeps per-token latents only for its last 128 tokens, so it is skipped. A DeepSeek V4
target needs the hub's head decoders for it (``heads``), and a translated slot of it is restored
only on a server without a draft (``draft``).

The architecture comes from the model's GGUF header (``general.architecture``), read here with
a minimal standard-library GGUF reader.
"""

import struct

# no per-token cache at all: the whole conversation lives in a fixed-size recurrent state
PURE_RECURRENT = frozenset({"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"})
# what the engine translates: Qwen3.5/3.8 hybrids (gated DeltaNet + attention), Muse-Glimmer
# (sliding window + global attention) and dense GQA Qwen3; the MLA family (deepseek2, mistral4,
# hy_v4; DSA: deepseek32, glm-dsa, dots3note) and its K-only hybrids with KDA layers (kimi-linear,
# kimi-k3, bailingmoe3); the GQA hybrids with GatedDeltaNet, Mamba, short-conv or lightning layers
MLA = frozenset({"deepseek2", "mistral4", "hy_v4", "deepseek32", "glm-dsa", "dots3note", "kimi-linear", "kimi-k3",
                 "bailingmoe3"})
GQA_HYBRIDS = frozenset({"qwen3next", "qwen35moe", "qwen4exp", "minimax-01", "jamba", "falcon-h1", "granitehybrid",
                         "nemotron_h", "nemotron_h_moe", "plamo2", "lfm2", "lfm2moe"})
# translated into, never out of (RULES skips them as sources)
TARGET_ONLY = frozenset({"deepseek4"})
TRANSLATABLE = frozenset({"qwen35", "muse-glimmer", "qwen3"}) | MLA | GQA_HYBRIDS | TARGET_ONLY
# hybrids: a recurrent state (gated DeltaNet, Mamba, short-conv, lightning or KDA layers) beside the
# attention cache, which a translated state holds as one cell at T0 - 1: zeros, or the model's mean
# of native cells (kvpark bridge add --recurrent-mean)
HYBRID = frozenset({"qwen35", "kimi-linear", "kimi-k3", "bailingmoe3"}) | GQA_HYBRIDS
# the hub's head kinds of a DeepSeek V4 target (experiments/universal_kv/hub.py HEAD_KINDS)
DSV4_HEADS = ("csa", "hca", "lid")

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
# (tools/server/server-common.cpp:1213-1260)
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
    features: the MTP head (layers >= n_layer, src/llama-model.cpp:2425-2445) and a DSpark draft
    (DFlash graph_dsv4, 2462-2483). Each is a plain iSWA cache over hparams that are not MLA, so it
    holds K and V rows (V never written by DSpark), and classify_draft reads both as a DFlash-shaped
    draft: an empty base and the window (what the kvpark-patched server saved for draft-mtp on the
    test fixture: one K/V layer). Neither has been restored around a translated DeepSeek V4 slot, so
    both are skipped (the research's plan, an empty draft rebuilt by the native tail, is unvalidated
    too), like every draft classify_draft calls "other". MTP and DSpark drafting also make the target
    keep rollback snapshots (n_rs_seq = the draft length): only the active plane is saved and a
    restore zeroes the others, which the native tail (a decode, never a rollback) would tolerate.
    Other architectures: the bridge's general draft rules."""
    if arch not in TARGET_ONLY or layout is None or layout.kind == "none":
        return None
    what = ("its own sliding K/V window of target features: a DSpark draft or the MTP head"
            if layout.kind == "dflash" else layout.note)
    return "draft-unsupported", (f"{target} ({arch}) runs a draft whose state the bridge has not restored around a "
                                 f"translated DeepSeek V4 slot ({what}); run it without a draft to translate")


def media(conversation_media, *, source, target, source_modalities, target_modalities, source_saw,
          target_arch=None):
    """None when a conversation with media can be translated up to its first media message,
    else (skip code, reason).

    `conversation_media` maps each modality to the index of the first message holding it;
    the modalities are /props "modalities" of each server; `source_saw` is the index of the
    last message the source server has processed (its slot holds media from those)."""
    for modality in sorted(conversation_media):
        if not target_modalities.get(modality):
            return f"target-no-{modality}", (f"the conversation has {modality} input and {target}'s server has no "
                                             f"{modality} encoder (mmproj)")
    if target_arch in TARGET_ONLY:
        # media after a synthesized pooled cache (blocks of 128 positions) has never been restored
        return "dsv4-media", (f"{target} ({target_arch}) pools its cache in blocks; media after a translated "
                              "pooled cache is untested, so the target prefills this conversation natively")
    for modality, first in sorted(conversation_media.items()):
        if first <= source_saw and source_modalities.get(modality):
            return "media-source", (f"{source}'s cache holds the conversation's {modality} input, which the bridge "
                                    "engine cannot read yet")
    return None


# -- GGUF header ---------------------------------------------------------------------------------

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_KV = 1 << 16


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


def gguf_value(path, key):
    """A scalar or string metadata value of a GGUF file (v2/v3, little-endian), or None."""
    with open(path, "rb") as stream:
        if _read(stream, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, _, n_kv = struct.unpack("<IQQ", _read(stream, 20))
        if version not in (2, 3) or n_kv > MAX_KV:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        wanted = key.encode()
        for _ in range(n_kv):
            name = _string(stream)
            (kind,) = struct.unpack("<I", _read(stream, 4))
            if name != wanted:
                _skip(stream, kind)
                continue
            if kind == _STRING:
                return _string(stream).decode("utf-8")
            if kind in _SCALARS:
                fmt = _SCALARS[kind]
                return struct.unpack(fmt, _read(stream, struct.calcsize(fmt)))[0]
            return None
    return None


def gguf_architecture(path):
    """general.architecture of a GGUF model (llama.cpp's converters write it first)."""
    arch = gguf_value(path, "general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{path} has no general.architecture")
    return arch
