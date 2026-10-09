"""Bridge configuration: registered models (server, weights, slot directory) and the engine."""

import ast
import json
import math
import os
from pathlib import Path
import re
import struct
from urllib.parse import urlsplit
import zipfile

from ..platforms import data_directory
from . import capabilities

FORMAT = 1
HUB_FORMATS = (1, 2)               # hub manifest format_version: 1 (fit_hub.py), 2 (convert_hub.py)
ENGINE_DEVICES = ("auto", "cpu", "gpu", "vulkan")
ENGINE_MODES = ("daemon", "oneshot")
# a model server's RoPE settings that change its cache, by llama_context_params name (the
# engine builds the model's cache layout with them); rope_scaling_type is none | linear | yarn
RUNTIME_KEYS = ("rope_scaling_type", "rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor",
                "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx")
ROPE_SCALING_TYPES = ("none", "linear", "yarn")
# architectures whose state codec takes RoPE from the GGUF only and refuses those settings
# (experiments/linear_transfer/kvstate.py, ModelSpec._dense_from_metadata)
FIXED_ROPE_ARCHS = ("muse-glimmer", "qwen35", "qwen3")
RECURRENT_MEAN_FORMAT = 1          # experiments/universal_kv/recurrent_init.py FORMAT_VERSION
# hybrids with gated DeltaNet layers on a fixed pattern (every full_attention_interval-th layer is
# attention): their recurrent layout follows from the GGUF header (kvstate.ModelSpec)
GDN_INTERVAL_ARCHS = ("qwen35", "qwen35moe", "qwen3next")
# what the bridge does with media a target's server cannot take (surrogates.py), by /props modality:
# the `kvpark bridge media` option and its values, the first the default
MEDIA_POLICIES = {"vision": ("image", ("caption", "skip")), "audio": ("audio", ("transcribe", "skip")),
                  "video": ("video", ("describe", "skip"))}
# the path that makes an audio describer an OpenAI-compatible speech-to-text endpoint
TRANSCRIPTION_PATH = "/audio/transcriptions"


def default_path():
    return Path(os.environ.get("KVPARK_BRIDGE_CONFIG", data_directory() / "kvpark" / "bridge" / "bridge.json"))


def empty():
    return {"format": FORMAT, "models": {}, "hub": None, "engine": None, "tail": 2048, "fill": "mean"}


def load(path=None):
    path = Path(path or default_path())
    if not path.is_file():
        return empty()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported bridge configuration format")
    return {**empty(), **config}


def save(config, path=None):
    path = Path(path or default_path())
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_name(path.name + ".tmp")
    # private like the rest of the bridge directory (model paths, server addresses)
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    os.replace(pending, path)


def check_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path not in ("", "/"):
        raise ValueError("model URL must be a local http://127.0.0.1:<port> llama-server address")
    return f"http://{parsed.hostname}:{parsed.port or 80}"


def is_transcription_url(url):
    """Whether a describer is a speech-to-text endpoint (its URL ends in /audio/transcriptions)."""
    return isinstance(url, str) and urlsplit(url).path.rstrip("/").endswith(TRANSCRIPTION_PATH)


def check_describer_url(url, modality):
    """A describer given by URL: a local llama-server address (check_url) or, for audio, a local
    OpenAI-compatible speech-to-text endpoint, http://127.0.0.1:<port>/v1/audio/transcriptions."""
    if modality != "audio" or not is_transcription_url(url):
        return check_url(url)
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.query:
        raise ValueError("a transcription endpoint must be a local http://127.0.0.1:<port>/v1/audio/transcriptions "
                         "address")
    return f"http://{parsed.hostname}:{parsed.port or 80}{parsed.path.rstrip('/')}"


def check_runtime(runtime):
    """The RoPE settings of a model's server ({RUNTIME_KEYS name: value}, None values dropped), checked."""
    out = {}
    for key, value in (runtime or {}).items():
        if value is None:
            continue
        if key not in RUNTIME_KEYS:
            raise ValueError(f"unknown RoPE setting {key!r}")
        if key == "rope_scaling_type":
            if value not in ROPE_SCALING_TYPES:
                raise ValueError(f"rope_scaling_type must be one of {', '.join(ROPE_SCALING_TYPES)}")
        elif key == "yarn_orig_ctx":
            if type(value) is not int or value < 0:
                raise ValueError("yarn_orig_ctx must be a nonnegative token count")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        out[key] = value
    return out


def recurrent_mean_meta(path):
    """The meta record of a recurrent mean file (experiments/universal_kv/recurrent_init.py
    RecurrentMean.save: an npz whose meta.npy holds one JSON string: format_version, layout, model,
    groups), read with the standard library; ValueError when the file is not one."""
    try:
        with zipfile.ZipFile(path) as archive:
            data = archive.read("meta.npy")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"{path} is not a recurrent mean file ({error})") from None
    try:
        # NPY: magic, version, header length (u16 in version 1, u32 after), a dict literal, the data
        if data[:6] != b"\x93NUMPY" or data[6] not in (1, 2, 3):
            raise ValueError("not an NPY array")
        size, start = ((struct.unpack_from("<H", data, 8)[0], 10) if data[6] == 1
                       else (struct.unpack_from("<I", data, 8)[0], 12))
        header = ast.literal_eval(data[start:start + size].decode("latin-1"))
        descr = str(header.get("descr")) if isinstance(header, dict) else ""
        if not re.fullmatch(r"[<>]U\d+", descr) or header.get("shape") != ():
            raise ValueError("meta is not one string")
        text = data[start + size:].decode("utf-32-le" if descr[0] == "<" else "utf-32-be").rstrip("\x00")
        meta = json.loads(text)
    except (ValueError, SyntaxError, struct.error, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not a recurrent mean file ({error})") from None
    if (not isinstance(meta, dict) or meta.get("format_version") != RECURRENT_MEAN_FORMAT
            or not isinstance(meta.get("groups"), dict) or not meta["groups"]
            or not isinstance(meta.get("layout"), dict)):
        raise ValueError(f"{path} is not a recurrent mean file of format {RECURRENT_MEAN_FORMAT}")
    return meta


def check_recurrent_mean(path, gguf, arch):
    """The meta of a recurrent mean file (recurrent_mean_meta), checked against the model as far as
    its GGUF header shows: the architecture it was built for, the layer count of the recurrent
    section (the trunk: block_count less nextn layers) and, for gated DeltaNet hybrids on a fixed
    pattern (GDN_INTERVAL_ARCHS), its recurrent layers and row sizes. The engine checks the whole
    layout against the target's own slot save at every switch."""
    meta = recurrent_mean_meta(path)
    built_for = (meta.get("model") or {}).get("architecture")
    if built_for and built_for != arch:
        raise ValueError(f"{path} is the recurrent mean of a {built_for} model, not of this {arch} model")
    layout = meta["layout"]

    def value(key):
        return capabilities.gguf_value(gguf, f"{arch}.{key}")
    blocks, nextn = value("block_count"), value("nextn_predict_layers") or 0
    if isinstance(blocks, int) and layout.get("n_layer") != blocks - nextn:
        raise ValueError(f"{path} is the recurrent mean of a {layout.get('n_layer')}-layer model; this one has "
                         f"{blocks - nextn} layers")
    interval = value("full_attention_interval")
    if arch in GDN_INTERVAL_ARCHS and isinstance(blocks, int) and isinstance(interval, int) and interval > 0:
        kernel, inner, state, groups = (value(f"ssm.{key}") for key in ("conv_kernel", "inner_size", "state_size",
                                                                         "group_count"))
        layers = [il for il in range(blocks - nextn) if (il + 1) % interval]
        expected = {"layers": layers}
        if all(isinstance(x, int) for x in (kernel, inner, state, groups)):
            # conv taps of the last K - 1 qkv projections; per head the delta-rule matrix (float32)
            expected.update(r_bytes=[4 * (kernel - 1) * (inner + 2 * groups * state)] * len(layers),
                            s_bytes=[4 * state * inner] * len(layers))
        for key, wanted in expected.items():
            if layout.get(key) != wanted:
                raise ValueError(f"{path} is the recurrent mean of another model: its recurrent layout's {key} "
                                 "differs from this model's")
    return meta


def add_model(config, name, *, url, gguf, slot_dir, hub_name=None, tail=None, native_prefix=False,
              min_handover=None, draft_window=None, runtime=None, template_kwargs=None, recurrent_mean=None,
              recurrent_tools=False, source=True):
    if not name or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("model names use letters, digits, '-', '_' and '.'")
    gguf, slot_dir = Path(gguf).expanduser().resolve(), Path(slot_dir).expanduser().resolve()
    if not gguf.is_file():
        raise ValueError(f"{gguf} is not a file")
    if not slot_dir.is_dir():
        raise ValueError(f"{slot_dir} is not a directory (the server's --slot-save-path)")
    if tail is not None and (type(tail) is not int or tail < 0):
        raise ValueError("tail must be a nonnegative token count")
    if draft_window is not None and (type(draft_window) is not int or draft_window < 1):
        raise ValueError("draft_window must be a positive token count")
    runtime = check_runtime(runtime)
    if template_kwargs is not None:
        if not isinstance(template_kwargs, dict) or not all(isinstance(key, str) for key in template_kwargs):
            raise ValueError("template_kwargs must be an object (messages.server_template_kwargs)")
        try:
            json.dumps(template_kwargs)
        except (TypeError, ValueError):
            raise ValueError("template_kwargs must hold JSON values") from None
    try:
        arch = capabilities.gguf_architecture(gguf)
    except (OSError, ValueError, struct.error) as error:
        raise ValueError(f"{gguf} is not a readable GGUF model: {error}") from None
    if runtime and arch in FIXED_ROPE_ARCHS:
        # every switch would save the source's slot and then fail in the engine
        raise ValueError(f"the bridge engine reads {arch} caches with the GGUF's own RoPE settings; a {arch} server "
                         "run with --rope-*/--yarn-* flags cannot take part in switches")
    if native_prefix and arch in capabilities.TARGET_ONLY:
        # the engine synthesizes a DeepSeek V4 cache whole: every switch into it would fail after
        # the target's prefix prefill
        raise ValueError(f"the bridge engine copies no native prefix into a {arch} cache; register {name} "
                         "without native_prefix")
    if recurrent_tools and not recurrent_mean:
        raise ValueError("recurrent_tools needs a recurrent mean")
    if not isinstance(source, bool):
        raise ValueError("source must be true or false")
    if recurrent_mean is not None:
        if arch not in capabilities.HYBRID:
            raise ValueError(f"only hybrid models keep a recurrent state; {arch} has none to start from a mean")
        recurrent_mean = Path(recurrent_mean).expanduser().resolve()
        if not recurrent_mean.is_file():
            raise ValueError(f"{recurrent_mean} is not a file")
        meta = check_recurrent_mean(recurrent_mean, gguf, arch)
        if recurrent_tools and "tools" not in meta["groups"]:
            raise ValueError(f"{recurrent_mean} has no tools group (groups: {', '.join(sorted(meta['groups']))})")
    config["models"][name] = {"url": check_url(url), "gguf": str(gguf), "slot_dir": str(slot_dir),
                              "hub_name": hub_name or name, "arch": arch}
    if tail is not None:
        # this model as a target re-reads at least `tail` recent tokens itself (dense models need more)
        config["models"][name]["tail"] = tail
    if native_prefix:
        # as a target, prefill the prompt start (system message) natively and copy it exactly
        config["models"][name]["native_prefix"] = True
    if min_handover is not None:
        if type(min_handover) is not int or min_handover < 0:
            raise ValueError("min_handover must be a nonnegative token count")
        # as a target, translate only prompts longer than tail + this (small models prefill fast)
        config["models"][name]["min_handover"] = min_handover
    if draft_window is not None:
        # as a target with a DFlash draft: the draft's sliding window (the native tail covers it)
        config["models"][name]["draft_window"] = draft_window
    if runtime:
        # the server's RoPE flags (--rope-scaling, --rope-freq-base, --yarn-*): its cache is rotated
        # with these, so the engine reads and writes it with them (job["source"/"target"]["runtime"])
        config["models"][name]["runtime"] = runtime
    if template_kwargs is not None:
        # the server's chat template flags (--chat-template-kwargs, --reasoning, --reasoning-effort,
        # --no-reasoning-preserve): as a target, the engine renders the conversation with them
        # (job["target"]["template_kwargs"]) as the server does
        config["models"][name]["template_kwargs"] = template_kwargs
    if recurrent_mean is not None:
        # as a hybrid target, the translated state's recurrent cell starts from this mean of the
        # model's native cells (job["target"]["recurrent_mean"]); the engine picks the group: chat
        # without tools, the pooled mean with them, or the tools group when recurrent_tools says it
        # was calibrated on this deployment's own agent prompts
        config["models"][name]["recurrent_mean"] = str(recurrent_mean)
        if recurrent_tools:
            config["models"][name]["recurrent_tools"] = True
    if not source:
        # a target only: switches out of this model are skipped ("source-refused") and the target prefills the
        # conversation natively. For a model whose cache the hub translates worse than an information-free cache
        # (experiments/universal_kv/RESULTS.md section 10: Nemotron-3.5 as the servers render)
        config["models"][name]["source"] = False
    return config


def model_arch(config, name, *, path=None):
    """The model's GGUF architecture, or None when the file cannot be read.

    Entries registered before the architecture was recorded get it here, and it is written
    back to the configuration file at `path` (best effort)."""
    cfg = model(config, name)
    if cfg.get("arch"):
        return cfg["arch"]
    try:
        cfg["arch"] = capabilities.gguf_architecture(cfg["gguf"])
    except (OSError, ValueError, struct.error):
        return None
    try:
        stored = load(path)
        entry = stored["models"].get(name)
        if entry is not None and entry.get("gguf") == cfg["gguf"] and not entry.get("arch"):
            entry["arch"] = cfg["arch"]
            save(stored, path)
    except (OSError, ValueError):
        pass
    return cfg["arch"]


def set_engine(config, *, python=None, script=None, hub=None, device=None, mode=None, idle_unload=None,
               idle_exit=None, threads=None):
    """Set the engine worker (a Python with numpy, and torch for the GPU; the engine script), the
    hub bundle and the engine's options. Settings not given keep their configured values; the
    first time, python, script and hub are required.

    device: auto | cpu | gpu; mode: daemon (a resident engine, engine_client.py) | oneshot (a
    fresh process per switch); idle_unload / idle_exit: the daemon's idle seconds before it
    drops its hub sides / exits; threads: BLAS threads of the CPU path (0: the default)."""
    engine = dict(config.get("engine") or {})
    if not engine and (python is None or script is None or hub is None and not config.get("hub")):
        raise ValueError("the first engine setting needs the engine Python, script and hub")
    if python is not None:
        # never resolve the interpreter: a venv's bin/python is a symlink to the base interpreter,
        # and running that instead silently drops the venv's packages (numpy and its BLAS)
        engine["python"] = str(Path(os.path.abspath(Path(python).expanduser())))
    if script is not None:
        engine["script"] = str(Path(script).expanduser().resolve())
    for key, what in (("python", "engine Python"), ("script", "engine script")):
        if not Path(engine[key]).is_file():
            raise ValueError(f"{what} {engine[key]} is not a file")
    if hub is not None:
        hub = Path(hub).expanduser().resolve()
        hub_format(hub)
        config["hub"] = str(hub)
    for key, value, choices in (("device", device, ENGINE_DEVICES), ("mode", mode, ENGINE_MODES)):
        if value is not None:
            if value not in choices:
                raise ValueError(f"engine {key} must be one of {', '.join(choices)}")
            engine[key] = value
    for key, value in (("idle_unload", idle_unload), ("idle_exit", idle_exit)):
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive number of seconds")
            engine[key] = value
    if threads is not None:
        if type(threads) is not int or threads < 0:
            raise ValueError("threads must be a nonnegative count (0: the default)")
        if threads:
            engine["threads"] = threads
        else:
            engine.pop("threads", None)
    config["engine"] = engine
    return config


def set_media(config, *, policy=None, target=None, describers=None, cache_mib=None, transcription_model=None):
    """Set what the bridge does with media a target's server cannot take (surrogates.py).

    policy: {"image": "caption"|"skip", "audio": "transcribe"|"skip", "video": "describe"|"skip"}
    (None values are left alone, "default" drops the setting), for every target or, with `target`,
    for that registered model as a target; describers: {modality ("vision", "audio", "video"): a
    registered model name or a local llama-server URL (for audio also a local speech-to-text
    endpoint, .../v1/audio/transcriptions), or None to drop it}; cache_mib: the size cap of the
    surrogate cache; transcription_model: the "model" a speech-to-text endpoint is asked for
    ("" drops it: surrogates.TRANSCRIPTION_MODEL). Settings not given keep their values."""
    media = dict(config.get("media") or {})
    choices = dict(MEDIA_POLICIES.values())
    changes = {key: value for key, value in (policy or {}).items() if value is not None}
    for key, value in changes.items():
        if key not in choices:
            raise ValueError(f"unknown media kind {key!r} (image, audio or video)")
        if value not in (*choices[key], "default"):
            raise ValueError(f"--{key} must be one of {', '.join(choices[key])} or default")
    if target is not None:
        model(config, target)                              # a registered model, or ValueError
        overrides = dict(media.get("models") or {})
        entry = dict(overrides.get(target) or {})
        _apply_policy(entry, changes)
        if entry:
            overrides[target] = entry
        else:
            overrides.pop(target, None)
        media["models"] = overrides
    else:
        settings = dict(media.get("policy") or {})
        _apply_policy(settings, changes)
        media["policy"] = settings
    if describers:
        chosen = dict(media.get("describers") or {})
        for modality, value in describers.items():
            if modality not in MEDIA_POLICIES:
                raise ValueError(f"unknown modality {modality!r} (vision, audio or video)")
            if value is None:
                chosen.pop(modality, None)
            else:
                # a registered model is saved and restored around its side requests; any other
                # describer is a local server whose resident conversation the request replaces
                chosen[modality] = value if value in config["models"] else check_describer_url(value, modality)
        media["describers"] = chosen
    if cache_mib is not None:
        if type(cache_mib) not in (int, float) or not math.isfinite(cache_mib) or cache_mib <= 0:
            raise ValueError("the surrogate cache size must be a positive number of MiB")
        media["cache_mib"] = cache_mib
    if transcription_model is not None:
        if not isinstance(transcription_model, str):
            raise ValueError("the transcription model is a name")
        media["transcription_model"] = transcription_model.strip()
    config["media"] = {key: value for key, value in media.items() if value or key == "cache_mib"}
    return config


def _apply_policy(settings, changes):
    for key, value in changes.items():
        if value == "default":
            settings.pop(key, None)
        else:
            settings[key] = value


def forget_model_media(config, name):
    """Drop a removed model's media settings: its overrides as a target, and describers naming it."""
    media = config.get("media") or {}
    models = {key: value for key, value in (media.get("models") or {}).items() if key != name}
    describers = {key: value for key, value in (media.get("describers") or {}).items() if value != name}
    for key, value in (("models", models), ("describers", describers)):
        if value:
            media[key] = value
        else:
            media.pop(key, None)


def hub_format(directory):
    """The format_version of a hub bundle's manifest (1 or 2); ValueError for anything else."""
    path = Path(directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{directory} is not a hub bundle (no manifest.json)") from None
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read: {error}") from None
    version = manifest.get("format_version") if isinstance(manifest, dict) else None
    if version not in HUB_FORMATS:
        raise ValueError(f"{directory} is a hub bundle of format {version!r}; the bridge reads formats "
                         f"{' and '.join(map(str, HUB_FORMATS))}")
    return version


def hub_manifest(config):
    """The configured hub bundle's manifest ({} without a hub)."""
    if not config.get("hub"):
        return {}
    return json.loads((Path(config["hub"]) / "manifest.json").read_text(encoding="utf-8"))


def hub_models(config):
    """Model names the configured hub knows (empty without a hub); formats 1 and 2 list them alike."""
    return set(hub_manifest(config).get("models", {}))


def hub_indexer_ridges(config):
    """Hub model names with an indexer ridge (fit_hub.py --indexer-ridge): the DSA targets the
    engine can give indexer keys beyond their top_k (capabilities.indexer)."""
    return set(hub_manifest(config).get("indexer_ridges") or {})


def model(config, name):
    try:
        return config["models"][name]
    except KeyError:
        raise ValueError(f"unknown bridge model {name!r}; add it with kvpark bridge add") from None
