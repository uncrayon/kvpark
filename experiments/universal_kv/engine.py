"""Bridge engine: translate a live conversation's slot file from one model to another.

    python engine.py JOB.json        (prints a JSON report; writes job["target"]["output"])

Job:

    {"hub": DIR, "tail": 2048, "fill": "mean", "device": "auto",       # device: auto | cpu | gpu
     "conversation": {"messages": [...], "tools": null,      # chat.py rules; ends with the new message
                      "template_options": null},             # optional: the request's template fields
     "source": {"name": "qwen38", "gguf": PATH, "state": SLOT_FILE, "runtime": null},
     "target": {"name": "muse", "gguf": PATH, "template_state": SLOT_FILE, "output": NEW_FILE,
                "tokens": [[id, "piece" | [bytes]], ...],      # the target server's /tokenize
                "runtime": null,                               # optional: the server's RoPE settings
                "template_kwargs": null,                       # optional: the server's template flags
                "max_t0_message": null,                        # optional: T0 stops before message k
                "recurrent_mean": null,                        # optional (hybrid targets): a mean npz
                "recurrent": null,                             # optional: the recurrent cell's strategy
                "recurrent_tools": false}}                     # optional: use the mean's tools group

"runtime" holds a server's RoPE settings where they differ from its GGUF (llama_context_params
names, kvstate.RUNTIME_ROPE_KEYS: rope_scaling_type, rope_freq_base, rope_freq_scale,
yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow, yarn_orig_ctx; None = unset),
passed to kvstate.ModelSpec.from_metadata: the RoPE that de-rotates the source's keys and rotates
the target's must be the one each server applies (only the MLA family and the hybrids accept
overrides). A GGUF's rope_freqs.weight (granitehybrid's frequency factors) is read once into its
profile, so specs rotate with it as ModelSpec.from_gguf's do.

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
every message segment located in that text in order. A template that re-renders history
differently (dropped reasoning, other whitespace) only costs alignment, never correctness.

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

from collections import OrderedDict
import functools
import json
import mmap
from pathlib import Path
import sys
import threading
import time
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

CONTROL = 3            # llama.cpp token type: control (special) token
PROFILE_FORMAT = 2     # 2: the EOS text and rope_freqs.weight (frequency factors) are kept too
MAX_ARRAY = 1 << 16    # per-layer metadata arrays are read whole (the vocabulary is kept apart)
# what this engine handles beyond earlier ones (the bridge skips those switches for engines
# without them): "template-options" renders the target with the request's template options;
# "media-source" reads a source slot that holds media up to its first medium; "recurrent-mean"
# starts a hybrid target's recurrent cell from target["recurrent_mean"]
FEATURES = ("media-source", "template-options", "recurrent-mean")
TEMPLATE_MISMATCH = 4  # exit status of `engine.py JOB` when the job fails with TemplateMismatch
MEAN_CACHE = 4         # recurrent mean groups kept per process (a Qwen3.8 group: 157 MB in float32)


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
    summary (state spec), the vocabulary with token types, the chat template, BOS and EOS text,
    and granitehybrid's RoPE frequency factors (rope_freqs.weight; [] without one, None for
    other architectures)."""
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
    reader, tensors = gguf_tensors.reader(str(path))
    field = reader.fields.get("tokenizer.ggml.model")
    model = field.contents() if field is not None else None
    types = reader.fields.get("tokenizer.ggml.token_type")
    info = chat.gguf_tokenizer_info(str(path))
    if info["chat_template"] is None:
        raise ValueError("model has no tokenizer.chat_template metadata")
    # the vocabulary is kept below; the other arrays (per-layer settings) whole
    metadata = {name: gguf_tensors._field_value(item, MAX_ARRAY) for name, item in reader.fields.items()
                if not name.startswith(("GGUF.", "tokenizer."))}
    factors = None
    if metadata.get("general.architecture") == "granitehybrid":
        # an optional tensor the metadata does not show (kvstate.ModelSpec.from_gguf reads it alike)
        factors = []
        if "rope_freqs.weight" in tensors:
            factors = [float(x) for x in np.asarray(gguf_tensors.read_tensor(str(path), "rope_freqs.weight"),
                                                    dtype=np.float32).ravel()]
    data = {"format": PROFILE_FORMAT, "file": key,
            "metadata": _json_value(metadata),
            "tokenizer_model": model, "tokens": list(reader.fields["tokenizer.ggml.tokens"].contents()),
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


def spec_of(gguf_path, runtime=None):
    """The model's kvstate.ModelSpec, as ModelSpec.from_gguf builds it (the frequency factors of a
    rotating granitehybrid included); runtime: the serving context's RoPE settings (see the
    module docstring)."""
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("a runtime must be a mapping of RoPE settings")
    data = profile(str(gguf_path), _PROFILE_CACHE[0])
    spec = kvstate.ModelSpec.from_metadata(data["metadata"], runtime=runtime)
    factors = data.get("rope_freq_factors")
    if factors is not None and spec.architecture == "granitehybrid" and spec.rope_type != "none":
        spec = kvstate.ModelSpec.from_metadata(data["metadata"], runtime=runtime, rope_freq_factors=factors)
    return spec


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


def vocabulary(gguf_path):
    """(token strings, token types, tokenizer model) of a GGUF (byte-level BPE only)."""
    return _vocabulary(*_identity(gguf_path), _PROFILE_CACHE[0])


@functools.lru_cache(maxsize=8)
def _vocabulary(path, size, mtime_ns, cache_dir):
    data = _profile(path, size, mtime_ns, cache_dir)
    if data["tokenizer_model"] != "gpt2":
        raise ValueError(f"{path}: tokenizer model {data['tokenizer_model']!r} is not supported yet "
                         "(byte-level BPE only)")
    return tuple(data["tokens"]), tuple(data["token_types"]), data["tokenizer_model"]


def token_bytes(gguf_path, ids):
    tokens, types, _ = vocabulary(str(gguf_path))
    decoder = _byte_decoder()
    out = []
    for token_id in ids:
        text = tokens[int(token_id)]
        if types and types[int(token_id)] == CONTROL:
            out.append(text.encode("utf-8"))
        else:
            out.append(bytes(decoder[ch] for ch in text) if all(ch in decoder for ch in text) else text.encode("utf-8"))
    return out


def token_arrays(gguf_path, ids, pieces):
    """token_id / byte_end / special from token ids and their byte pieces."""
    _, types, _ = vocabulary(str(gguf_path))
    ends = np.cumsum([len(piece) for piece in pieces]).astype(np.int64)
    special = np.asarray([1 if types and types[int(i)] == CONTROL else 0 for i in ids], dtype=np.uint8)
    return {"token_id": np.asarray(ids, dtype=np.int32), "byte_end": ends, "special": special}


def located_record(conversation, document_id, text_bytes):
    """A rendered-record stand-in for text the source model actually processed: segments located in order."""
    keys, spans, cursor = [], [], 0
    for number, (key, value) in enumerate(chat.segment_texts(conversation)):
        keys.append(key)
        needle = value.encode("utf-8")
        found = text_bytes.find(needle, cursor)
        if found >= 0:
            spans.append([number, found, found + len(needle)])
            cursor = found + len(needle)
    return {"document_id": document_id, "text": text_bytes.decode("utf-8", errors="strict"), "segments": spans,
            "segment_keys": keys, "split": "test", "final_message_segments": []}


def target_pieces(tokens):
    ids, pieces = [], []
    for token_id, piece in tokens:
        ids.append(int(token_id))
        pieces.append(bytes(piece) if isinstance(piece, list) else piece.encode("utf-8"))
    return ids, pieces


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
    _, types, _ = vocabulary(str(gguf))
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
    source_spec, target_spec = spec_of(src["gguf"], src.get("runtime")), spec_of(tgt["gguf"], tgt.get("runtime"))
    target_profile = profile(str(tgt["gguf"]), _PROFILE_CACHE[0])
    timings["profiles_s"] = round(time.perf_counter() - t, 3)
    if translate.numpy_only(target_spec) and backend.name != "numpy":
        # a DeepSeek V4 target pools on the CPU: the numpy backend keeps its hub sides resident
        backend = backends.select("cpu")
    t = time.perf_counter()
    ids, pieces = target_pieces(tgt["tokens"])
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
    source_piece_bytes = token_bytes(src["gguf"], source_ids)
    source_record = located_record(conversation, document_id, b"".join(source_piece_bytes))
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
                  source_segments_found=len(source_record["segments"]),
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
