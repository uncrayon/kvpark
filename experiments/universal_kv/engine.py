"""Bridge engine: translate a live conversation's slot file from one model to another.

    python engine.py JOB.json        (prints a JSON report; writes job["target"]["output"])

Job:

    {"hub": DIR, "tail": 2048, "fill": "mean", "device": "auto",       # device: auto | cpu | gpu
     "conversation": {"messages": [...], "tools": null,      # chat.py rules; ends with the new message
                      "template_options": null},             # optional: the request's template fields
     "source": {"name": "qwen38", "gguf": PATH, "state": SLOT_FILE, "runtime": null,
                "build_info": null},                           # optional: the server's /props build_info
     "target": {"name": "muse", "gguf": PATH, "template_state": SLOT_FILE, "output": NEW_FILE,
                "tokens": [[id, "piece" | [bytes]], ...],      # the target server's /tokenize
                "runtime": null,                               # optional: the server's RoPE settings
                "build_info": null,                            # optional: the server's /props build_info
                "template_kwargs": null,                       # optional: the server's template flags
                "max_t0_message": null,                        # optional: T0 stops before message k
                "recurrent_mean": null,                        # optional (hybrid targets): a mean npz
                "recurrent": null,                             # optional: the recurrent cell's strategy
                "recurrent_tools": false}}                     # optional: use the mean's tools group

"runtime" holds a server's RoPE settings where they differ from its GGUF (llama_context_params
names, kvstate.RUNTIME_ROPE_KEYS: rope_scaling_type, rope_freq_base, rope_freq_scale,
yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow, yarn_orig_ctx; None = unset),
passed to kvstate.ModelSpec.from_metadata: the RoPE that de-rotates the source's keys and rotates
the target's must be the one each server applies (the MLA family, the hybrids and the GQA family
accept overrides). The RoPE frequency factors a GGUF keeps as tensors (rope_freqs.weight of
granitehybrid, llama, mistral3 and gemma4; kvstate.ROPE_FACTOR_TENSOR_ARCHITECTURES) are read once
into its profile, so specs rotate with them as ModelSpec.from_gguf's do; a GGUF whose factors
kvstate refuses (LongRoPE rope_factors_long/short: phi3's, or a llama's) gets no spec. "build_info" (llama-server's
/props build_info, "b<number>-<commit>") names the llama.cpp pin the server runs
(kvstate.llama_revision_of; job_revision), whose layouts and RoPE the side's spec takes; without it,
or when it names no known pin, the spec is kvstate's default revision. Either way the states are
read in the layout they hold (kvstate.parse_file tries the other pin's reading too, and translate
takes the template's and the source state's).

The target record is the conversation rendered with the target's own chat template and the
generation prompt (what the target server will process next); its tokens come from the target
server, so the built cache matches the server's tokenization exactly. The rendering mirrors the
template context llama-server builds for the request: "template_options" holds the request's
chat_template_kwargs and reasoning_effort (kvpark.bridge.messages.template_options), mapped as
the server maps them (chat.server_context: Qwen3.8 writes the reasoning effort into its system
prompt and drops it for enable_thinking: false; Muse reads it as its reasoning strength and does
not read enable_thinking, so reasoning_effort "none" gives its default strength). The server's
own settings (--reasoning-effort, --chat-template-kwargs, the thinking switch, the date) are not
visible over HTTP ("template_kwargs" gives the template kwargs its flags set, where the bridge
knows them: kvpark.bridge.messages.server_template_kwargs), so the candidates of
chat.server_contexts are tried in order and the first whose bytes the target tokens spell is
used (report["rendering"]); when none matches, the job
fails with TemplateMismatch (a ValueError; `engine.py JOB` exits with TEMPLATE_MISMATCH), so
the bridge can translate the template's default rendering instead. The source record is
rebuilt from the source slot itself: its token ids decoded with the source vocabulary, and
every message segment located in that text, where the source's own template renders it in the
part of the text that rendering spells (source_spans: the history up to the reply the source
generated), else in order (located_record: the in-order occurrences covering the most bytes;
report["source_segments_rendered"] and ["source_segments_found"]). A template that re-renders
history differently (dropped reasoning, other whitespace, arguments written JSON-escaped) only
costs alignment, never correctness; both records span a segment's text inside the whitespace at
its edges (chat.render, live), which templates often trim.
Vocabularies (vocabulary, TOKENIZER_MODELS: GPT-2 byte-level BPE, Gemma 4's BPE, SentencePiece)
are decoded as llama.cpp's token_to_piece prints them, and the space a SentencePiece tokenizer
puts before each text after a special token (add_space_prefix) is dropped from both sides'
pieces (spelled_pieces), so that the pieces join to the text.

A source slot that holds media (images or audio its server's projector encoded) is read up to
its first medium (kvstate.media_cut and truncate_state: the media cells and everything after
them are dropped, recurrent rows reset, sliding sections kept as saved); its text before the
media is what the source rows are aligned and translated from (report["source_media"]), and T0
stops after the content of the last message segment found in it (report["source_media"]
["t0_cap"]). The target's own cut (max_t0_message below) keeps T0 before the first message
with media anyway; the source's cap matters when the request no longer holds the source's
media (a client that replaced an earlier image by text).

T0 = min(P - tail, first token of the last message), P = the target prompt length, so the
newest message and at least `tail` tokens are prefilled natively by the target. With
max_t0_message = k, T0 also stops where message k starts (the first token after the content of
the messages before it, so its role header and any media it holds are prefilled natively): the
bridge passes the message holding a conversation's first image. When message k or a later one
holds media (content parts instead of a string, which chat.py does not render), only messages
[0, k) are validated and rendered, and the target tokens (the server's tokenization of the whole
prompt, media markers included) need only start with their content; the translated cells
[0, T0) lie inside it.

A hybrid target (Qwen3.5/3.8, Ornith, every model with a recurrent section) keeps one recurrent
cell, which the translated state holds at T0 - 1 for the native tail to carry on from
(recurrent_init): "recurrent" names the strategy, by default "mean" when "recurrent_mean" gives
the target's mean npz (r0_experiments.py calibrate; RESULTS.md section 7: closer to the native
cell than zero on every measured document), else "zero" (R0). Its group follows RESULTS.md
section 7's recommendation (recurrent_init.match_group): "chat" for a conversation without tools;
with tools the pooled "all", or "tools" when "recurrent_tools" says the file's tools group was
calibrated on this deployment's own agent prompts; "all" when the file lacks the chosen group,
and zero, with a note, when it lacks that too or cannot be read (moved, deleted or damaged since
it was registered; report["recurrent"]). Loaded groups are kept per file identity
(recurrent_mean: 0.16 s for a Qwen3.8 group, once per engine_daemon.py).

The hub is read lazily (hub.open_hub: only the two sides of each kind this switch needs, K only
into an MLA target; a format 2 bundle from convert_hub.py reads fastest). An MLA target's latent
norm weights come from its GGUF; a DSA target whose prompt P exceeds its indexer top_k needs the
hub's indexer ridge for it, else translate raises translate.IndexerKeysUnavailable (a ValueError
whose message starts "no indexer keys"; kvpark.bridge.capabilities.indexer decides the same from
the GGUF header and P before a switch). The arithmetic runs on a backend
(backends/): job["device"] or $KVPARK_ENGINE_DEVICE, "auto" = the GPU when torch has one;
a GPU failure (unsupported model, float16 overflow) falls back to numpy, noted in the report
(with report["device_fault"] when the error looks like a broken device, after which
engine_daemon.py stays on the CPU). A DeepSeek V4 target always translates on numpy
(translate.numpy_only: it pools on the CPU), with T0 rounded down to a multiple of 128
(report["t0"], report["t0_requested"]). The report carries numpy's BLAS and a warning when it is
the reference BLAS, and FEATURES (report["features"], and engine_daemon.py's ping): what this
engine handles that earlier ones did not, so the bridge skips such switches only for an older
engine. engine_daemon.py keeps all of this resident between switches.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import bisect
from collections import OrderedDict
import functools
import itertools
import json
import mmap
from pathlib import Path
import sys
import threading
import time
from typing import NamedTuple
import zipfile

import jinja2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import backends  # noqa: E402
from backends.numpy_backend import blas_info  # noqa: E402
import hub as hub_module  # noqa: E402
import recurrent_init  # noqa: E402
import state_writer  # noqa: E402
import translate  # noqa: E402

import chat  # noqa: E402
import gguf_tensors  # noqa: E402
import kvstate  # noqa: E402

# llama.cpp token types (llama_token_type, the GGUF's tokenizer.ggml.token_type)
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
# tokens llama.cpp prints as their text and reads as special fragments (attr_special and
# user-defined: llama_vocab::impl::token_to_piece, cache_special_tokens)
RAW_TYPES = (UNKNOWN, CONTROL, USER_DEFINED)
# tokenizer.ggml.model values the engine reads (llama_vocab::impl::load): byte-level BPE,
# Gemma 4's BPE with SentencePiece pieces, SentencePiece ("llama": Jamba)
TOKENIZER_MODELS = ("gpt2", "gemma4", "llama")
# BPE pre-tokenizers whose pieces write a space as U+2581 (escape_whitespaces in llama-vocab.cpp)
ESCAPED_PRE = ("gemma4", "granite-embed-multi-311m", "mmbert", "sarvam-moe")
PROFILE_FORMAT = 4     # 2: the EOS text and rope_freqs.weight (frequency factors) are kept too;
                       # 3: the pre-tokenizer and add_space_prefix (non-BPE vocabularies);
                       # 4: the frequency factors of every kvstate.ROPE_FACTOR_TENSOR_ARCHITECTURES model
MAX_ARRAY = 1 << 16    # per-layer metadata arrays are read whole (the vocabulary is kept apart)
# what this engine handles beyond earlier ones (the bridge skips those switches for engines
# without them): "template-options" renders the target with the request's template options;
# "media-source" reads a source slot that holds media up to its first medium; "recurrent-mean"
# starts a hybrid target's recurrent cell from target["recurrent_mean"]; "vocabularies" reads
# vocabularies other than GPT-2 byte-level BPE (TOKENIZER_MODELS: Gemma 4's BPE, SentencePiece)
FEATURES = ("media-source", "template-options", "recurrent-mean", "vocabularies")
TEMPLATE_MISMATCH = 4  # exit status of `engine.py JOB` when the job fails with TemplateMismatch
MEAN_CACHE = 4         # recurrent mean groups kept per process (a Qwen3.8 group: 157 MB in float32)
LOCATE_OCCURRENCES = 1024  # located_record: a segment text found more often in the source is not placed


class TemplateMismatch(ValueError):
    """The target server's tokens spell none of the renderings this engine can make of the
    conversation (the server renders with settings or a template the engine cannot mirror)."""


def _json_value(value):
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _identity(gguf_path):
    """(resolved path, size, mtime_ns): what the in-memory caches below are keyed on, so a
    long-lived process (engine_daemon.py) notices a GGUF replaced at the same path."""
    path = Path(gguf_path).resolve()
    status = path.stat()
    return str(path), status.st_size, status.st_mtime_ns


def profile(gguf_path, cache_dir=None):
    """What the engine needs from a GGUF, parsed once and cached per file identity: the metadata
    summary (state spec), the vocabulary with token types, tokenizer model, pre-tokenizer and
    add_space_prefix, the chat template, BOS and EOS text, and the RoPE frequency factors of a
    kvstate.ROPE_FACTOR_TENSOR_ARCHITECTURES model (kvstate.gguf_rope_freq_factors: [] without them;
    {"refused": reason} when kvstate refuses them; None for other architectures)."""
    return _profile(*_identity(gguf_path), cache_dir)


@functools.lru_cache(maxsize=8)
def _profile(path, size, mtime_ns, cache_dir):
    path = Path(path)
    key = f"{path.name}-{size}-{mtime_ns}"
    cached = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cached is not None and cached.is_file():
        data = json.loads(cached.read_text(encoding="utf-8"))
        # profiles written before per-layer arrays were kept whole summarize arrays of more than
        # 64 entries (a deep model's per-layer head counts or indexer types): built again
        if data.get("format") == PROFILE_FORMAT and not _summarized(data.get("metadata", {})):
            return data
    reader, _ = gguf_tensors.reader(str(path))

    def tokenizer_key(name):
        field = reader.fields.get(f"tokenizer.ggml.{name}")
        return field.contents() if field is not None else None
    model = tokenizer_key("model")
    types = reader.fields.get("tokenizer.ggml.token_type")
    info = chat.gguf_tokenizer_info(str(path))
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    # the vocabulary is kept below; the other arrays (per-layer settings) whole
    metadata = {name: gguf_tensors._field_value(item, MAX_ARRAY) for name, item in reader.fields.items()
                if not name.startswith(("GGUF.", "tokenizer."))}
    factors = None
    if metadata.get("general.architecture") in kvstate.ROPE_FACTOR_TENSOR_ARCHITECTURES:
        # tensors the metadata does not show, from whichever file of a split GGUF holds them
        # (kvstate.ModelSpec.from_gguf reads them alike)
        try:
            found = kvstate.gguf_rope_freq_factors(path, kvstate.ModelSpec.from_metadata(_json_value(metadata)))
            factors = None if found is None else list(found)
        except kvstate.StateFormatError as error:
            factors = {"refused": str(error)}
    data = {"format": PROFILE_FORMAT, "file": key,
            "metadata": _json_value(metadata),
            "tokenizer_model": model, "tokenizer_pre": tokenizer_key("pre"),
            "add_space_prefix": tokenizer_key("add_space_prefix"),
            "tokens": list(reader.fields["tokenizer.ggml.tokens"].contents()),
            "token_types": list(types.contents()) if types is not None else [],
            "chat_template": info["chat_template"], "bos_token": info["bos_token"], "eos_token": info["eos_token"],
            "rope_freq_factors": factors}
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        pending = cached.with_name(cached.name + f".{os.getpid()}.tmp")
        pending.write_text(json.dumps(data), encoding="utf-8")
        os.replace(pending, cached)
    return data


def _summarized(metadata):
    """Whether a profile's metadata summarizes an array other than the vocabulary's by its length."""
    return any(isinstance(value, dict) and "array_length" in value
               for key, value in metadata.items() if not key.startswith("tokenizer."))


_PROFILE_CACHE = [None]


def spec_of(gguf_path, runtime=None, llama_revision=None):
    """The model's kvstate.ModelSpec, as ModelSpec.from_gguf builds it (its RoPE frequency factors
    included); runtime: the serving context's RoPE settings (see the module docstring);
    llama_revision: the pin the server runs (job_revision), else kvstate's default."""
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("a runtime must be a mapping of RoPE settings")
    data = profile(str(gguf_path), _PROFILE_CACHE[0])
    spec = kvstate.ModelSpec.from_metadata(data["metadata"], runtime=runtime, llama_revision=llama_revision)
    factors = data.get("rope_freq_factors")
    if isinstance(factors, dict):
        raise kvstate.StateFormatError(factors["refused"])
    if factors is not None and spec.rope_type != "none":
        spec = kvstate.ModelSpec.from_metadata(data["metadata"], runtime=runtime, rope_freq_factors=factors,
                                               llama_revision=llama_revision)
    return spec


def job_revision(side):
    """The kvstate.LLAMA_REVISIONS pin a job side's server runs: its "build_info" (/props) when that
    names a known pin, else None (kvstate's default)."""
    info = side.get("build_info")
    if info is None:
        return None
    if not isinstance(info, str):
        raise ValueError("a build_info must be llama-server's /props build_info string")
    return kvstate.llama_revision_of(info)


@functools.lru_cache(maxsize=None)
def _byte_decoder():
    """GPT-2 byte-level BPE: unicode character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Vocabulary(NamedTuple):
    """A GGUF's vocabulary as llama.cpp reads it (llama_vocab::impl::load).

    spelling: how a normal token's string spells its bytes, "bpe" (GPT-2 byte-level characters)
    or "spm" (SentencePiece: U+2581 for a space; byte tokens <0xNN> carry one byte each), for
    SentencePiece vocabularies and BPE ones that escape whitespace (Gemma 4). space_prefix: the
    tokenizer puts a space before the text that starts a prompt or follows a special token
    (a SentencePiece vocabulary's add_space_prefix, true unless the GGUF says false), which the
    pieces print and the text does not hold (spelled_pieces)."""

    tokens: tuple
    types: tuple
    model: str
    spelling: str
    space_prefix: bool


def vocabulary(gguf_path):
    """The Vocabulary of a GGUF (TOKENIZER_MODELS: byte-level BPE, Gemma 4's BPE, SentencePiece)."""
    return _vocabulary(*_identity(gguf_path), _PROFILE_CACHE[0])


@functools.lru_cache(maxsize=8)
def _vocabulary(path, size, mtime_ns, cache_dir):
    data = _profile(path, size, mtime_ns, cache_dir)
    model = data["tokenizer_model"]
    if model not in TOKENIZER_MODELS:
        raise ValueError(f"{path}: tokenizer model {model!r} is not supported yet "
                         f"(supported: {', '.join(TOKENIZER_MODELS)})")
    spm = model == "llama"
    escaped = spm or model == "gemma4" or data.get("tokenizer_pre") in ESCAPED_PRE
    prefix = data.get("add_space_prefix")
    return Vocabulary(tuple(data["tokens"]), tuple(data["token_types"]), model, "spm" if escaped else "bpe",
                      spm and prefix is not False)


def _token_type(vocab, token_id):
    return vocab.types[token_id] if vocab.types else NORMAL


def _piece(vocab, token_id):
    """llama_vocab::impl::token_to_piece(token, special=true) of one token: special and
    user-defined tokens as their text, normal tokens decoded (GPT-2 byte characters, or U+2581
    as a space), byte tokens as their byte, unused and undefined tokens as nothing."""
    text = vocab.tokens[token_id] or f"[EMPTY_{token_id}]"      # llama.cpp renames empty strings
    kind = _token_type(vocab, token_id)
    if kind in RAW_TYPES:
        return text.encode("utf-8")
    if kind == NORMAL:
        if vocab.spelling == "spm":
            return text.replace("\u2581", " ").encode("utf-8")
        decoder = _byte_decoder()
        if all(ch in decoder for ch in text):
            return bytes(decoder[ch] for ch in text)
        # llama_decode_text: a character outside the byte map becomes a marker naming it
        return b"".join(bytes([decoder[ch]]) if ch in decoder else
                        f"[UNK_BYTE_0x{ch.encode('utf-8').hex()}{text}]".encode("utf-8") for ch in text)
    if kind == BYTE:
        return bytes([int(text[3:5], 16)])
    return b""


def token_bytes(gguf_path, ids, *, spelled=False):
    """Each token's bytes as llama-server's /tokenize pieces give them (llama_token_to_piece with
    special=true); spelled=True: as spelled_pieces returns them (their join is the text)."""
    vocab = vocabulary(str(gguf_path))
    pieces = [_piece(vocab, int(token_id)) for token_id in ids]
    return spelled_pieces(gguf_path, ids, pieces) if spelled else pieces


def spelled_pieces(gguf_path, ids, pieces):
    """The pieces as the tokenized text spells them. A SentencePiece tokenizer with
    add_space_prefix tokenizes the text at the start of a prompt and after each special
    token with a space in front (llama_vocab::impl::tokenize: is_prev_special), which that
    text's first piece prints: it is dropped there, so the pieces join to the text the server
    tokenized. Every other vocabulary's pieces are returned as they are.

    "Special" is read from the GGUF token types (RAW_TYPES), not from the attributes llama.cpp
    derives at load (it marks some NORMAL end-of-turn tokens CONTROL too; no local SentencePiece
    vocabulary has such a token), and text holding a literal U+2581 cannot be spelled back (it
    tokenizes as a space): either way the target pieces do not spell the rendering and the
    engine refuses the switch (TemplateMismatch)."""
    vocab = vocabulary(str(gguf_path))
    pieces = list(pieces)
    if not vocab.space_prefix:
        return pieces
    after_special = True
    for index, token_id in enumerate(ids):
        kind = _token_type(vocab, int(token_id))
        if after_special and kind == NORMAL and pieces[index].startswith(b" "):
            pieces[index] = pieces[index][1:]
        after_special = kind in RAW_TYPES
    return pieces


def token_arrays(gguf_path, ids, pieces):
    """token_id / byte_end / special from token ids and their byte pieces."""
    types = vocabulary(str(gguf_path)).types
    ends = np.cumsum([len(piece) for piece in pieces]).astype(np.int64)
    special = np.asarray([1 if types and types[int(i)] == CONTROL else 0 for i in ids], dtype=np.uint8)
    return {"token_id": np.asarray(ids, dtype=np.int32), "byte_end": ends, "special": special}


def _common_prefix(a, b):
    """Length of the longest common prefix of two byte strings."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def source_spans(conversation, source_profile, options, ids, pieces, gguf):
    """The spans the source's own template gives the segments in its slot's text: chat.render
    (live, no generation prompt) in each chat.server_contexts candidate for the request's options,
    the one sharing the longest prefix with the text (after a leading control token, BOS, when
    the rendering lacks it), and of its spans those inside that prefix, as [[segment, start,
    end]]. The slot holds the previous request and the reply the source generated, so the prefix
    usually runs to that reply; [] when no candidate renders."""
    joined = b"".join(pieces)
    types = vocabulary(str(gguf)).types
    shifts = [0] + ([len(pieces[0])] if len(ids) and types and types[int(ids[0])] == CONTROL else [])
    try:
        contexts = chat.server_contexts(options, eos_token=source_profile.get("eos_token") or "")
    except ValueError:
        return []
    best = (0, [])
    for _, context in contexts:
        try:
            rendered = chat.render(conversation, source_profile["chat_template"], bos_token=source_profile["bos_token"],
                                   extra_context=context, live=True)
        except (ValueError, jinja2.TemplateError):
            continue
        text = rendered.text.encode("utf-8")
        for shift in shifts:
            common = _common_prefix(text, joined[shift:])
            if common > best[0]:
                best = (common, [[n, a + shift, b + shift] for n, a, b in rendered.segments if b <= common])
    return best[1]


def located_record(conversation, document_id, text_bytes, known=None):
    """A rendered-record stand-in for text the source model actually processed. `known` spans
    (source_spans: where the source's own template puts them) are kept; every later segment's
    core (chat.render's live spans: the text inside the whitespace at its edges) is located after
    them, in segment order and without overlaps, choosing the occurrences that cover the most
    bytes (the earliest on ties). A segment the source template renders elsewhere or not at all
    (a tool-call argument that also appears in a later answer, content written after its tool
    calls) then costs only itself, not the segments after it; one found more than
    LOCATE_OCCURRENCES times is too ambiguous to place."""
    known = sorted(known or [], key=lambda span: span[1])
    after, last = (known[-1][2], max(n for n, _, _ in known)) if known else (0, -1)
    keys, found = [], []                    # found: (segment, start, end) occurrences, by segment
    for number, (key, value) in enumerate(chat.segment_texts(conversation)):
        keys.append(key)
        needle = value.strip(chat.EDGE_SPACE).encode("utf-8")
        starts, at = [], text_bytes.find(needle, after) if needle and number > last else -1
        while at >= 0 and len(starts) <= LOCATE_OCCURRENCES:
            starts.append(at)
            at = text_bytes.find(needle, at + 1)
        if len(starts) <= LOCATE_OCCURRENCES:
            found.extend((number, start, start + len(needle)) for start in starts)
    # longest chain: best[i] = (bytes covered by the best chain ending in occurrence i, -its end)
    # over the earlier segments' occurrences ending before i starts (a Fenwick tree of maxima)
    ends = sorted({end for _, _, end in found})
    tree, best, previous = [None] * (len(ends) + 1), [], []
    for _, group in itertools.groupby(range(len(found)), key=lambda i: found[i][0]):
        group = list(group)
        for i in group:
            number, start, end = found[i]
            node, chain = bisect.bisect_right(ends, start), None
            while node:
                if tree[node] is not None and (chain is None or tree[node] > chain):
                    chain = tree[node]
                node -= node & -node
            best.append(((chain[0] if chain else 0) + end - start, -end))
            previous.append(chain[2] if chain else None)
        for i in group:                     # a segment's own occurrences never chain
            node = bisect.bisect_left(ends, found[i][2]) + 1
            while node <= len(ends):
                if tree[node] is None or (*best[i], i) > tree[node]:
                    tree[node] = (*best[i], i)
                node += node & -node
    spans, i = [], max(range(len(best)), key=best.__getitem__, default=None)
    while i is not None:
        spans.append(list(found[i]))
        i = previous[i]
    return {"document_id": document_id, "text": text_bytes.decode("utf-8", errors="strict"),
            "segments": [list(span) for span in known] + spans[::-1], "segment_keys": keys, "split": "test",
            "final_message_segments": []}


def target_pieces(tokens, gguf_path=None):
    """(ids, byte pieces) of a target server's /tokenize answer; with the target's GGUF, the
    pieces as the text spells them (spelled_pieces)."""
    ids, pieces = [], []
    for token_id, piece in tokens:
        ids.append(int(token_id))
        pieces.append(bytes(piece) if isinstance(piece, list) else piece.encode("utf-8"))
    return ids, (spelled_pieces(gguf_path, ids, pieces) if gguf_path is not None else pieces)


_MEANS = OrderedDict()             # (path, size, mtime_ns, group) -> recurrent_init.RecurrentMean
_MEANS_LOCK = threading.Lock()


@functools.lru_cache(maxsize=16)
def _mean_groups(path, size, mtime_ns):
    return tuple(recurrent_init.RecurrentMean.stored_groups(path))


def recurrent_mean(path, group):
    """One group of a recurrent mean npz (recurrent_init.RecurrentMean), kept per file identity and
    group (the MEAN_CACHE most recently used): engine_daemon.py reads each once."""
    key = (*_identity(path), group)
    with _MEANS_LOCK:
        if key in _MEANS:
            _MEANS.move_to_end(key)
            return _MEANS[key]
    mean = recurrent_init.RecurrentMean.load(key[0], groups=[group])
    with _MEANS_LOCK:
        _MEANS[key] = mean
        while len(_MEANS) > MEAN_CACHE:
            _MEANS.popitem(last=False)
    return mean


def _mean_bytes(mean):
    return int(sum(np.asarray(values).nbytes for rows in mean.means.values() for kind in rows.values()
                   for values in kind.values()))


def recurrent_means():
    """[{"file", "group", "bytes"}] of the recurrent means kept, oldest first."""
    with _MEANS_LOCK:
        return [{"file": Path(key[0]).name, "group": key[3], "bytes": _mean_bytes(mean)}
                for key, mean in _MEANS.items()]


def drop_recurrent_means():
    """Forget the kept recurrent means; returns their bytes."""
    with _MEANS_LOCK:
        released = sum(_mean_bytes(mean) for mean in _MEANS.values())
        _MEANS.clear()
    return released


def recurrent_options(target, tools):
    """(translate's recurrent options, report entry) for a job's hybrid target: the strategy
    (target["recurrent"], default "mean" with a target["recurrent_mean"] file, else "zero") and,
    when it reads the mean, the conversation's group (see the module docstring) and its rows; zero
    with a note when the file holds no usable group or cannot be read."""
    path = target.get("recurrent_mean")
    rule = recurrent_init.parse_strategy(target.get("recurrent") or ("mean" if path else "zero"))
    if "mean" not in rule.needs:
        return {"recurrent": rule.name}, {"strategy": rule.name}
    if not path:
        raise ValueError(f"the recurrent strategy {rule.name} needs the target's recurrent_mean file")
    entry = {"file": Path(path).name}

    def zero(note):
        return {"recurrent": "zero"}, {**entry, "strategy": "zero", "requested": rule.name, "note": note}
    try:
        groups = _mean_groups(*_identity(path))
        use_tools = bool(target.get("recurrent_tools"))
        group = recurrent_init.match_group(groups, bool(tools), tools_group=use_tools)
        if group is None:
            wanted = ("tools" if use_tools else recurrent_init.ALL) if tools else "chat"
            return zero(f"the mean holds neither the {wanted} group nor the pooled one (groups: {', '.join(groups)})")
        mean = recurrent_mean(path, group)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        # a registered file since moved, deleted or damaged: the switch still saves the prefill
        return zero(f"the mean cannot be read ({type(error).__name__}: {error})"[:300])
    return ({"recurrent": rule.name, "recurrent_mean": mean, "recurrent_group": group},
            {**entry, "strategy": rule.name, "group": group})


def read_state(path, spec):
    """A parsed slot file whose arrays view a read-only memory map: no 1 GB copy, and pages the
    translation never reads (a hybrid source's recurrent rows) are never read from the file."""
    with open(path, "rb") as stream:
        try:
            data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):           # empty files, file systems without mmap
            return kvstate.read_file(path, spec)
    return kvstate.parse_file(data, spec)


def message_start_token(record, tokens, message):
    """Index of the first token past the content of the messages before `message` (0 if none)."""
    ends = [end for number, _, end in record["segments"]
            if int(record["segment_keys"][number].split(":", 1)[0]) < message]
    return int(np.searchsorted(tokens["byte_end"], max(ends, default=0), side="right")) if ends else 0


def _fallback_worthy(error):
    """GPU errors after which the same job is retried on numpy."""
    return isinstance(error, (NotImplementedError, FloatingPointError, RuntimeError, MemoryError))


def device_fault(error):
    """Whether a GPU error looks like a broken device or context (not an out-of-memory, a
    float16 overflow or an unsupported model): engine_daemon.py then stops using the GPU."""
    return (isinstance(error, RuntimeError) and not isinstance(error, NotImplementedError)
            and type(error).__name__ != "OutOfMemoryError" and "out of memory" not in str(error).lower())


def has_media(message):
    """Whether a message carries content parts (images, audio) instead of text."""
    return isinstance(message, dict) and isinstance(message.get("content"), list)


def _spelling_shift(joined, text, ids, pieces, gguf, *, prefix):
    """Bytes the target tokens hold before the rendering: 0, or one leading control token (BOS);
    None when the tokens do not spell it.

    llama-server renders without BOS text and adds BOS as a token; chat.py renders BOS text only
    when the template uses it, so exactly one leading control token is accepted either way. With
    prefix, the tokens need only start with `text` (the rendering of the messages before a
    max_t0_message cap, when later messages hold media)."""
    def spells(at):
        return joined[at:].startswith(text) if prefix else joined[at:] == text
    if spells(0):
        return 0
    types = vocabulary(str(gguf)).types
    lead = pieces[0] if pieces and types and types[ids[0]] == CONTROL else b""
    if lead and joined.startswith(lead) and spells(len(lead)):
        return len(lead)
    return None


def render_target(conversation, target_profile, options, ids, pieces, gguf, *, media, server_kwargs=None):
    """(rendered, shift, name): the conversation rendered as the target server renders it for a
    request with these template options (the first of chat.server_contexts whose bytes the target
    tokens spell, see _spelling_shift; name is that context's; server_kwargs: the template kwargs
    of the server's own flags, when known); raises TemplateMismatch when none does. With media
    only the messages before the cap are rendered (no generation prompt), and the tokens need
    only start with their content."""
    joined = b"".join(pieces)
    try:
        contexts = chat.server_contexts(options, eos_token=target_profile.get("eos_token") or "",
                                        server_kwargs=server_kwargs)
    except ValueError as error:
        raise TemplateMismatch(f"the request's template options cannot be rendered: {error}") from None
    first = None
    for name, context in contexts:
        try:
            rendered = chat.render(conversation, target_profile["chat_template"], bos_token=target_profile["bos_token"],
                                   add_generation_prompt=not media, extra_context=context, live=True)
        except (ValueError, jinja2.TemplateError) as error:
            # e.g. `content + eos_token` fails without the variables only the server context sets
            first = first or f"rendering with the {name} context failed: {type(error).__name__}: {error}"
            continue
        text = rendered.text.encode("utf-8")
        if media:
            # only [0, T0) is translated: the tokens must spell the content of the messages before
            # the cap (what follows it in this rendering, the end of the last message, may differ)
            text = text[:max((end for _, _, end in rendered.segments), default=0)]
        shift = _spelling_shift(joined, text, ids, pieces, gguf, prefix=media)
        if shift is not None:
            return rendered, shift, name
        first = first or f"{len(joined)} vs {len(text)} bytes"
    what = "the rendering of the messages before max_t0_message" if media else "the target rendering"
    how = " with the request's template options" if options else ""
    raise TemplateMismatch(f"the target server's tokens do not spell {what}{how} ({first}); the templates differ")


def run(job, *, hub=None, backend=None, log=None):
    """Translate one job; hub (a hub.Hub) and backend are passed by the daemon to keep them resident."""
    started = time.perf_counter()
    messages, tools = job["conversation"]["messages"], job["conversation"].get("tools")
    src, tgt = job["source"], job["target"]
    cap = tgt.get("max_t0_message")
    if cap is not None and (type(cap) is not int or not isinstance(messages, list) or not 0 <= cap < len(messages)):
        raise ValueError("max_t0_message must be the index of a conversation message")
    # media in message `cap` or later: only the text messages before it are rendered and translated
    media = cap is not None and any(has_media(message) for message in messages[cap:])
    if media and cap == 0:
        raise ValueError("conversation too short to hand over (T0 = 0)")
    template_options = job["conversation"].get("template_options") or None
    if template_options is not None and not isinstance(template_options, dict):
        raise ValueError("template_options must be an object")
    server_kwargs = tgt.get("template_kwargs")
    if server_kwargs is not None and not isinstance(server_kwargs, dict):
        raise ValueError("the target's template_kwargs must be an object")
    conversation = {"messages": messages[:cap] if media else messages, "tools": tools}
    chat.validate_conversation(conversation, live=True)
    document_id = chat.document_id(conversation)
    _PROFILE_CACHE[0] = job.get("cache_dir")
    timings = {}
    t = time.perf_counter()
    backend = backend or backends.select(job.get("device"))
    timings["backend_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    if hub is None:
        hub = hub_module.open_hub(job["hub"], cache_dir=job.get("cache_dir"))
    timings["hub_open_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    source_spec = spec_of(src["gguf"], src.get("runtime"), llama_revision=job_revision(src))
    target_spec = spec_of(tgt["gguf"], tgt.get("runtime"), llama_revision=job_revision(tgt))
    target_profile = profile(str(tgt["gguf"]), _PROFILE_CACHE[0])
    timings["profiles_s"] = round(time.perf_counter() - t, 3)
    if translate.numpy_only(target_spec) and backend.name != "numpy":
        # a DeepSeek V4 target pools on the CPU: the numpy backend keeps its hub sides resident
        backend = backends.select("cpu")
    t = time.perf_counter()
    ids, pieces = target_pieces(tgt["tokens"], tgt["gguf"])
    joined = b"".join(pieces)
    rendered, shift, rendering = render_target(conversation, target_profile, template_options, ids, pieces,
                                               tgt["gguf"], media=media, server_kwargs=server_kwargs)
    timings["render_s"] = round(time.perf_counter() - t, 3)
    target_record = {"document_id": document_id, "text": joined.decode("utf-8"),
                     "segments": [[n, start + shift, end + shift] for n, start, end in rendered.segments],
                     "segment_keys": rendered.segment_keys, "split": "test",
                     "final_message_segments": rendered.final_message_segments}
    target_tokens = token_arrays(tgt["gguf"], ids, pieces)

    t_read = time.perf_counter()
    source_state = read_state(src["state"], source_spec)
    source_media = source_state.media
    if source_media.media:
        # the source's server encoded media: its text before the first medium is what is read
        cut = kvstate.media_cut(source_media)
        if cut == 0:
            raise ValueError("the source slot starts with media: it holds no text to translate")
        source_state = kvstate.truncate_state(source_state, cut, recurrent="reset", sliding="keep")
        source_media = {"chunks": len(source_media.media), "entries": int(len(source_media.tokens)), "read": cut}
    else:
        source_media = None
    timings["read_source_s"] = round(time.perf_counter() - t_read, 3)
    source_ids = np.asarray(source_state.tokens)
    source_piece_bytes = token_bytes(src["gguf"], source_ids, spelled=True)
    known = source_spans(conversation, profile(str(src["gguf"]), _PROFILE_CACHE[0]), template_options, source_ids,
                         source_piece_bytes, src["gguf"])
    source_record = located_record(conversation, document_id, b"".join(source_piece_bytes), known=known)
    source_tokens = token_arrays(src["gguf"], source_ids, source_piece_bytes)

    P = len(ids)
    t0 = P - int(job.get("tail", 2048))
    if not media:
        last = len(messages) - 1
        last_start = min((start for (number, start, _) in target_record["segments"]
                          if rendered.segment_keys[number].startswith(f"{last}:")), default=len(joined))
        t0 = min(t0, int(np.searchsorted(target_tokens["byte_end"], last_start, side="right")))
    if cap is not None:
        t0 = min(t0, message_start_token(target_record, target_tokens, cap))
    if source_media:
        # the source's text stops at its first medium: T0 stops after the content of the last
        # segment found in it (a request that no longer holds that medium, a client replacing an
        # earlier image by text, sets no cap, and the rest would be translated from nothing)
        located = max((number for number, _, _ in source_record["segments"]), default=-1)
        covered = max((end for number, _, end in target_record["segments"] if number <= located), default=0)
        source_media["t0_cap"] = int(np.searchsorted(target_tokens["byte_end"], covered, side="right"))
        t0 = min(t0, source_media["t0_cap"])
    if t0 < (target_spec.n_swa + 1 if target_spec.n_swa else 1):
        raise ValueError(f"conversation too short to hand over (T0 = {t0})")
    # memory-mapped like the source: a hybrid's template and native prefix are ~150 MB saves whose
    # recurrent rows are never read (0.12 -> 0.05 s each for Qwen3.8, more on a busy machine)
    template_state = read_state(tgt["template_state"], target_spec)
    prefix_state, prefix_tokens = None, int(tgt.get("prefix_tokens") or 0)
    if tgt.get("prefix_state") and prefix_tokens:
        prefix_state = read_state(tgt["prefix_state"], target_spec)
    timings["inputs_s"] = round(time.perf_counter() - t, 3)
    recurrent = None
    if any(isinstance(part, kvstate.RecurrentState) for part in template_state.state.parts):
        t = time.perf_counter()
        recurrent, recurrent_entry = recurrent_options(tgt, tools)
        timings["recurrent_s"] = round(time.perf_counter() - t, 3)

    source_side = translate.Side(src["name"], source_spec, source_record, source_tokens, gguf=src["gguf"])
    target_side = translate.Side(tgt["name"], target_spec, target_record, target_tokens, gguf=tgt["gguf"])
    options = dict(t0=t0, template=template_state, fill=job.get("fill", "mean"), native_state=prefix_state,
                   native_prefix=prefix_tokens if prefix_state is not None else 0, **(recurrent or {}))
    fallback = None
    try:
        state, report = _translate(source_state, source_side, target_side, hub, backend, options, timings)
    except Exception as error:      # noqa: BLE001  (GPU trouble: the numpy reference still works)
        if backend.name == "numpy" or not _fallback_worthy(error):
            raise
        fallback = f"{type(error).__name__}: {error}"
        fault = device_fault(error)
    if fallback is not None:
        # retried past the except block: its traceback, and the failed attempt's output arrays its
        # frames hold, are released first
        if log:
            log({"event": "fallback", "backend": backend.name, "error": fallback, "device_fault": fault})
        try:
            backend._released()             # the failed attempt's cached device memory (host RAM on an APU)
        except Exception:                   # noqa: BLE001  (a broken context may refuse even this)
            pass
        backend = backends.select("cpu")
        state, report = _translate(source_state, source_side, target_side, hub, backend, options, timings)
    del source_state
    t = time.perf_counter()
    written = state_writer.write(tgt["output"], state)
    timings["write_s"] = round(time.perf_counter() - t, 3)
    del state
    blas = blas_info()
    # the translated part may end before the T0 asked for (a DeepSeek V4 target rounds it down to
    # whole 128-token blocks): the restore, the .resume and the native tail follow the report's
    t0 = int(report.get("t0", t0))
    report.update(engine_timings=timings, P=P, t0=t0, tail=P - t0, source_tokens=int(len(source_ids)),
                  source_segments_found=len(source_record["segments"]), source_segments_rendered=len(known),
                  segments=len(source_record["segment_keys"]), output=str(tgt["output"]), output_bytes=written,
                  backend_info=backend.describe(), blas=blas["name"], blas_gflops=blas["gflops"],
                  engine_seconds=round(time.perf_counter() - started, 3))
    report.update(rendering=rendering, features=list(FEATURES))
    if template_options:
        report["template_options"] = sorted(template_options)
    if recurrent is not None:
        report["recurrent"] = {**report.get("recurrent", {}), **recurrent_entry}
    if server_kwargs is not None:
        report["template_kwargs"] = sorted(server_kwargs)
    warnings = [w for w in [blas.get("warning")] if w]
    if media:
        report["media_after"] = cap         # messages from here on hold media: rendered by the target only
    if source_media:
        report["source_media"] = source_media
    if fallback:
        report["fallback"] = fallback
        report["device_fault"] = fault
        warnings.append(f"{fallback}; translated on the CPU")
    if warnings:
        report["warnings"] = warnings
    return report


def _translate(source_state, source, target, hub, backend, options, timings):
    """translate.translate with the hub sides loaded first (timed apart from the arithmetic)."""
    t = time.perf_counter()
    for spec in (source.spec, target.spec):
        backend.supports(spec)
    # an MLA target decodes K only; a DeepSeek V4 target's head kinds decode the K latent
    for kind in translate.target_kinds(target.spec):
        if kind in translate.encoder_kinds(target.spec):
            backend.encoder(hub, kind, source.name)
        backend.decoder(hub, kind, target.name)
    backend.synchronize()
    timings["hub_load_s"] = round(time.perf_counter() - t, 3)
    state, report = translate.translate(source_state, source, target, hub, backend=backend, **options)
    split = report["timings"]
    timings.update(prepare_s=split.get("prepare_s", 0.0),
                   compute_s=round(split.get("encode_s", 0.0) + split.get("decode_s", 0.0), 3),
                   assemble_s=split.get("assemble_s", 0.0))
    return state, report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("KVPARK_ENGINE_TRACE"):      # debugging: SIGUSR1 dumps every thread's stack
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, file=open(os.environ["KVPARK_ENGINE_TRACE"], "a"), all_threads=True)
    if len(argv) != 1:
        raise SystemExit("usage: engine.py JOB.json")
    try:
        report = run(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    except TemplateMismatch as error:
        # its own exit status: the bridge then translates the template's default rendering
        print(f"TemplateMismatch: {error}", file=sys.stderr)
        return TEMPLATE_MISMATCH
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
