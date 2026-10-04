"""Live tests of `kvpark bridge switch` on real model servers.

    python live_bridge.py hops   --servers SERVERS.json --route qwen38,muse,qwen3-4b,qwen38 --engine gpu=PY:gpu ...
    python live_bridge.py drafts --servers SERVERS.json --source muse --target qwen38 --engine gpu=PY:gpu ...
    python live_bridge.py media  --servers SERVERS.json --source muse --target qwen38 --engine gpu=PY:gpu ...
    python live_bridge.py drafts-control --servers SERVERS.json --source muse --target qwen38 --engine gpu=PY:gpu ...
    python live_bridge.py media-control  --servers SERVERS.json --source muse --target qwen38 --engine gpu=PY:gpu ...
    python live_bridge.py options --servers SERVERS.json --source muse --target qwen38 [--extra-target qwen3-4b] ...
    python live_bridge.py --model NAME=GGUF ... --server BINARY ...     (hops with one upstream build, as before)

Starts throwaway llama-servers (private ports and slot directories; nothing deployed is
touched) and registers them in private bridge configurations, one per --engine
(LABEL=PYTHON:DEVICE: the engine daemon run by that Python on that device; the configurations
live under --bridge-root, reached by a relative path so the daemon's socket path stays short).
SERVERS.json maps each model name to {"gguf", "binary", "args" (every server flag but
--model, --alias, --port, --ctx-size and --slot-save-path; default: COMMON), "ctx", "env",
"register" (kvpark bridge add settings, e.g. {"native_prefix": true, "recurrent_mean": NPZ})}.

hops    per conversation (held-out corpus, final assistant message removed): prime the first
        model of --route (it processes the conversation as it would before answering); then
        for every hop A -> B: a native reference on B (erase, full prefill of the same tokens,
        32 generated tokens), and per engine: prewarm the pair (what staging a switch does),
        kvpark.bridge.switch.switch() (save A's slot, translate, restore into B, warm request),
        32 generated tokens. Records cache reuse (cache_n == T0), engine mode and device,
        engine, restore and tail-prefill seconds against the native prefill, the load average
        and MemAvailable next to every timing, and how many generated tokens agree.
drafts  speculative decoding on the target after a translated restore: per conversation the
        target generates --n-predict tokens (temperature 0) after a native prefill and after a
        switch from the source (no warm request, as Hermes switches), with the draft counts of
        each (timings draft_n / draft_n_accepted); then the target's log is scanned for the
        draft-position warning (speculative.cpp: "ctx_dft pos_max=...") and position errors.
media   switches into a vision target with an image (a generated chart): (i) in the newest
        user message, (ii) in an earlier user message (T0 stops there); the source never saw
        the image (it has no vision), so it is primed on the conversation without it. The real
        chat request then reports the reuse (cache_n against T0), its time and its answer,
        against a native run; finally a switch back into the source must skip (target-no-vision).
drafts-control  separates the draft state from the translation: per conversation the target
        generates --n-predict tokens (temperature 0) after (i) a native prefill, (ii) a switch
        from the source (as drafts), and after restoring its own cache of [0, T0) (a native
        prefill of the switch's T0 tokens, saved, erased, restored as a switch restores) with
        (iii) the draft state the bridge synthesizes for a translated restore (DFlash: the
        aligned dummy; MTP: the empty marker) or (iv) the server's own .resume. (iii) and (iv)
        have native target cells, so they show what the synthesized draft state alone costs.
media-control  media case (ii) (the image in an earlier message) on --limit conversations:
        native, a switch from the source, and the target's own cache of [0, T0) restored with the
        synthesized draft state: what the image and the reply look like after a restore with
        native values in front of the image, against translated ones.
options switches whose real chat request carries template options (TEMPLATE_OPTIONS: none,
        what Hermes sends, enable_thinking off), with one 27-30B model loaded at a time: the
        source primes each conversation alone and its slots are saved and kept, then the
        source stops and a stub server (StubSource) hands those slots to the bridge as the
        source's saves while the target (and --extra-target models) run. Per conversation and
        options: the switch (no warm request, as Hermes switches), then the real chat request
        with the same options (/v1/chat/completions, a few tokens): its cache_n against T0, the
        engine's rendering (report["engine"]["rendering"]) and the prefill time; one native
        chat request per conversation (the first options) for the prefill it saves. With
        --extra-target: the same conversations into those targets with the default rendering,
        and from the target too (primed first), the jobs kept with KVPARK_BRIDGE_KEEP_JOB in
        --keep-jobs (for bench_engine.py job).

Every server start waits for memory (wait_for_memory): MemAvailable must exceed the model files
(the model, a draft model, a projector) plus 25 GiB (the box hard-freezes rather than OOM-killing); it polls every 60 s for up to 90
minutes and records the wait.

Counts, timings and token ids only for the corpus conversations (no text); the media answers
(about the generated chart) are kept.
"""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from kvpark.bridge import engine_client, messages as bridge_messages, registry, resume  # noqa: E402
from kvpark.bridge import switch as bridge_switch  # noqa: E402
from kvpark.bridge.client import LlamaServer  # noqa: E402
from kvpark.bridge.messages import conversation_from_openai  # noqa: E402
from kvpark.bridge.switch import switch  # noqa: E402

COMMON = ["--n-gpu-layers", "all", "--split-mode", "none", "--parallel", "1", "--threads", "16",
          "--threads-batch", "16", "--batch-size", "2048", "--ubatch-size", "512", "--flash-attn", "on",
          "--cache-type-k", "f16", "--cache-type-v", "f16", "--fit", "off", "--cache-ram", "0",
          "--host", "127.0.0.1", "--jinja", "--no-ui", "--offline", "--log-colors", "off", "--slots"]
# what the target's log says when DFlash draft positions trail the target (common/speculative.cpp
# begin()), and llama_batch's refusal of non-consecutive positions (src/llama-batch.cpp)
LOG_PATTERNS = {"draft_pos_warning": "ctx_dft pos_max=", "position_error": "inconsistent sequence positions",
                "decode_failed": "failed to decode", "mtmd_failed": "failed to process mtmd chunk"}
MEMORY_HEADROOM_MIB = 25 * 1024     # MemAvailable must exceed a server's model files by this before it loads
MEMORY_POLL_S, MEMORY_WAIT_S = 60, 90 * 60
# llama-server options whose value is another file the server loads (a draft model, a projector)
MODEL_FILE_OPTIONS = {"-md", "--model-draft", "--spec-draft-model", "-mm", "--mmproj", "--lora"}
# the template options of the options mode's chat requests: none; what Hermes sends for the main
# model (agent.reasoning_effort, a custom provider: top-level reasoning_effort); with the provider's
# extra_body chat_template_kwargs (preserve_reasoning, ~/.hermes/config.yaml); its auxiliary
# compression calls (enable_thinking off); the main model with reasoning turned off (effort "none")
TEMPLATE_OPTIONS = {"none": None,
                    "hermes": {"reasoning_effort": "medium"},
                    "hermes-provider": {"reasoning_effort": "medium",
                                        "chat_template_kwargs": {"preserve_reasoning": True}},
                    "thinking-off": {"chat_template_kwargs": {"enable_thinking": False}},
                    "hermes-off": {"reasoning_effort": "none"}}


def mem_available_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024


def load_average():
    return [round(x, 2) for x in os.getloadavg()]


def model_files(gguf, args=()):
    """The files a server with these arguments loads: the model and every MODEL_FILE_OPTIONS value."""
    files, args = [Path(gguf)], [str(arg) for arg in args or ()]
    for index, arg in enumerate(args):
        option, _, value = arg.partition("=")
        if option in MODEL_FILE_OPTIONS:
            value = value or (args[index + 1] if index + 1 < len(args) else "")
            if value and Path(value).is_file():
                files.append(Path(value))
    return files


def wait_for_memory(name, gguf, *, args=(), log=print, poll=MEMORY_POLL_S, limit=MEMORY_WAIT_S):
    """Block until MemAvailable exceeds the server's model files (the model, and its draft model
    and projector from `args`) plus MEMORY_HEADROOM_MIB (polling every `poll` seconds, at most
    `limit`); returns the seconds waited, raises SystemExit after that."""
    needed = int(sum(path.stat().st_size for path in model_files(gguf, args)) / 2**20) + MEMORY_HEADROOM_MIB
    started = time.monotonic()
    while mem_available_mib() < needed:
        waited = time.monotonic() - started
        log(json.dumps({"memory_wait": name, "mem_available_mib": mem_available_mib(), "needed_mib": needed,
                        "waited_s": round(waited)}))
        if waited >= limit:
            raise SystemExit(f"MemAvailable {mem_available_mib()} MiB < {needed} MiB for {round(waited)} s; "
                             f"not starting {name}")
        time.sleep(poll)
    return round(time.monotonic() - started, 1)


def machine():
    """The load average and MemAvailable, recorded next to every timing (other work on the box changes them)."""
    return {"load_average": load_average(), "mem_available_mib": mem_available_mib()}


class Server:
    def __init__(self, name, binary, gguf, port, ctx, slot_dir, log, args=None, env=None):
        self.name, self.port = name, port
        self.argv = [binary, "--model", gguf, "--alias", name, "--port", str(port), "--ctx-size", str(ctx),
                     "--slot-save-path", str(slot_dir), *(COMMON if args is None else args)]
        self.log, self.process, self.env = log, None, env

    def start(self, timeout=900):
        self.process = subprocess.Popen(self.argv, stdout=open(self.log, "ab"), stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True,
                                        env={**os.environ, **(self.env or {})})
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited; see {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as reply:
                    if reply.status == 200:
                        return round(time.monotonic() - started, 1)
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(f"{self.name} did not start")

    def stop(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)

    def log_offset(self):
        return Path(self.log).stat().st_size

    def scan_log(self, since=0):
        """Lines of the server log after byte `since` that match LOG_PATTERNS, counted (with up to 3 samples)."""
        with open(self.log, "rb") as stream:
            stream.seek(since)
            text = stream.read().decode("utf-8", errors="replace")
        out = {}
        for key, pattern in LOG_PATTERNS.items():
            lines = [line.strip()[:300] for line in text.splitlines() if pattern in line]
            out[key] = {"count": len(lines), "samples": lines[:3]}
        return out


def conversation_for(item):
    """The held-out conversation without its final assistant message (that is what the target answers)."""
    messages = item["messages"][:-1]
    return {"messages": messages, "tools": item.get("tools")}


def server_specs(args):
    """{name: spec} from --servers, or from --model/--server (one upstream build, COMMON flags)."""
    if args.servers:
        specs = json.loads(Path(args.servers).read_text(encoding="utf-8"))
    else:
        if not args.model or not args.server:
            raise SystemExit("give --servers SERVERS.json, or --model NAME=GGUF and --server BINARY")
        specs = {name: {"gguf": gguf, "binary": str(args.server)}
                 for name, gguf in (item.split("=", 1) for item in args.model)}
        for name in specs:
            specs[name]["register"] = {"native_prefix": name in args.native_prefix_models.split(",")}
    for spec in specs.values():
        spec.setdefault("ctx", args.ctx)
    return specs


def engines(args):
    """[(label, python, device)] from --engine LABEL=PYTHON:DEVICE (default: --engine-python, device auto)."""
    out = []
    for item in args.engine or [f"default={args.engine_python}:auto"]:
        label, rest = item.split("=", 1)
        python, _, device = rest.rpartition(":")
        if device not in registry.ENGINE_DEVICES:
            python, device = rest, "auto"
        out.append((label, python, device))
    return out


class Rig:
    """Running servers and one bridge configuration per engine."""

    def __init__(self, args, names):
        self.args = args
        self.specs = {name: spec for name, spec in server_specs(args).items() if name in names}
        missing = set(names) - set(self.specs)
        if missing:
            raise SystemExit(f"no server spec for {sorted(missing)}")
        self.servers, self.configs = {}, {}

    def start(self):
        args = self.args
        for number, (name, spec) in enumerate(self.specs.items()):
            slot_dir = Path(args.slot_root) / name
            slot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            for stale in slot_dir.iterdir():
                stale.unlink()
            # the model's weights plus headroom for its KV cache, the engine and the desktop;
            # the box hard-freezes rather than OOM-killing near MemAvailable 0
            waited = wait_for_memory(name, spec["gguf"], args=spec.get("args"))
            server = Server(name, spec["binary"], spec["gguf"], spec.get("port", args.base_port + number), spec["ctx"],
                            slot_dir, args.output / f"{name}.log", args=spec.get("args"), env=spec.get("env"))
            load = server.start()
            self.servers[name] = server
            print(json.dumps({"server": name, "port": server.port, "load_s": load, "memory_wait_s": waited,
                              **machine()}), flush=True)
        for label, python, device in engines(args):
            path = Path(f"bridge-{label}") / "bridge.json"           # relative to --bridge-root (cwd)
            config = registry.empty()
            for name, spec in self.specs.items():
                server = self.servers[name]
                registry.add_model(config, name, url=f"http://127.0.0.1:{server.port}", gguf=spec["gguf"],
                                   slot_dir=Path(args.slot_root) / name, **(spec.get("register") or {}))
            registry.set_engine(config, python=python, script=Path(__file__).with_name("engine.py"), hub=args.hub,
                                device=device)
            config["tail"] = args.tail
            registry.save(config, path)
            self.configs[label] = (registry.load(path), path)

    def stop(self):
        for label, (_, path) in self.configs.items():
            engine_client.stop(path.parent)
        for server in self.servers.values():
            server.stop()

    def client(self, name):
        return LlamaServer(f"http://127.0.0.1:{self.servers[name].port}")


def choose(args, server, items, *, limit, min_tokens, max_tokens):
    """(item, conversation, target-token count) of the first `limit` corpus items in the token range."""
    chosen = []
    for item in items:
        conversation = conversation_for(item)
        try:
            prompt = server.apply_template(conversation["messages"], conversation.get("tools"))
        except Exception:      # noqa: BLE001  (a template that refuses it)
            continue
        n_tokens = len(server.tokenize(prompt, add_special=True))
        if min_tokens <= n_tokens < max_tokens:
            chosen.append((item, conversation, n_tokens))
        if len(chosen) >= limit:
            break
    return chosen


def corpus(args):
    items = [json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()]
    if args.ids:
        wanted = args.ids.split(",")
        items = [item for item in items if item.get("id") in wanted]
    if args.sources:
        keep = set(args.sources.split(","))
        items = [item for item in items if item.get("source") in keep]
    return items


def target_tokens(server, conversation):
    return [t for t, _ in server.tokenize(server.apply_template(conversation["messages"], conversation.get("tools")),
                                         add_special=True)]


def prime(server, conversation):
    """The model processes the conversation as it would before answering (its slot holds it)."""
    server.slot_action("erase")
    tokens = target_tokens(server, conversation)
    reply, seconds = server.complete(tokens, n_predict=1)
    return {"tokens": len(tokens), "seconds": round(seconds, 2), "prompt_ms": reply["timings"].get("prompt_ms")}


def generation(reply):
    """Counts of a completion: prompt reuse, prefill and generation speed, drafting."""
    timings = reply.get("timings") or {}
    out = {key: timings.get(key) for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                             "predicted_per_second", "draft_n", "draft_n_accepted")}
    if out["draft_n"]:
        out["acceptance"] = round(out["draft_n_accepted"] / out["draft_n"], 4)
    return out


def agree(a, b):
    same = 0
    for x, y in zip(a or [], b or []):
        if x != y:
            break
        same += 1
    return same


def switch_record(report, prewarmed, before):
    """The fields of a switch report the tables use."""
    engine = report.get("engine") or {}
    timings = report.get("timings") or {}
    warm = report.get("warm") or {}
    return {"t0": engine.get("t0"), "P": report.get("target_prompt_tokens"), "restored": report.get("restored_tokens"),
            "reused": report.get("reused_tokens"), "reuse_ok": report.get("reuse_ok"), "skipped": report.get("skipped"),
            "note": report.get("note"), "engine_mode": report.get("engine_mode"), "backend": engine.get("backend"),
            "device": engine.get("device"), "fallback": engine.get("fallback"),
            "daemon_error": report.get("daemon_error"), "engine_seconds": engine.get("engine_seconds"),
            "engine_timings": engine.get("engine_timings"), "daemon": engine.get("daemon"),
            "timings": timings, "warm_prompt_ms": warm.get("prompt_ms"), "draft": report.get("draft"),
            "media": report.get("media"), "tail": engine.get("tail"), "recurrent": report.get("recurrent"),
            "prewarm": prewarmed, "before": before}


def prewarm(config, path, a, b):
    started = time.monotonic()
    reply = engine_client.prewarm(config, [a, b], path.parent)
    return {"seconds": round(time.monotonic() - started, 3), "ok": bool(reply),
            **({key: reply.get(key) for key in ("load_s", "warm_up_s", "backend")} if reply else {})}


# -- hops -----------------------------------------------------------------------------------------

def hops(args):
    route = args.route.split(",")
    rig = Rig(args, set(route))
    results = []
    try:
        rig.start()
        first = rig.client(route[0])
        chosen = choose(args, first, corpus(args), limit=args.limit, min_tokens=args.min_tokens,
                        max_tokens=min(spec["ctx"] for spec in rig.specs.values()) - 512)
        for item, conversation, _ in chosen:
            record = {"id": item.get("id"), "source": item.get("source"), "hops": []}
            record["prime"] = {"model": route[0], **prime(first, conversation), **machine()}
            for a, b in zip(route, route[1:]):
                hop = {"from": a, "to": b, "engines": {}}
                target = rig.client(b)
                t_tokens = target_tokens(target, conversation)
                target.slot_action("erase")
                before = machine()
                native, wall = target.complete(t_tokens, n_predict=32)
                hop["native"] = {**generation(native), "wall_s": round(wall, 2), "tokens": len(t_tokens), **before}
                for label, (config, path) in rig.configs.items():
                    entry = {}
                    try:
                        warmed = prewarm(config, path, a, b) if args.prewarm else None
                        before = machine()
                        report = switch(config, conversation, a, b, config_path=path)
                        entry = switch_record(report, warmed, before)
                        bridged, _ = target.complete(t_tokens, n_predict=32)
                        entry["generation"] = generation(bridged)
                        entry["agree_prefix"] = agree(bridged.get("tokens"), native.get("tokens"))
                    except Exception as error:      # noqa: BLE001  (record and continue)
                        entry["error"] = f"{type(error).__name__}: {error}"[:400]
                    hop["engines"][label] = entry
                    print(json.dumps({"from": a, "to": b, "engine": label, "t0": entry.get("t0"),
                                      "reuse_ok": entry.get("reuse_ok"), "engine_mode": entry.get("engine_mode"),
                                      "device": entry.get("device"),
                                      "engine_s": (entry.get("timings") or {}).get("engine_s"),
                                      "restore_s": (entry.get("timings") or {}).get("restore_s"),
                                      "warm_ms": entry.get("warm_prompt_ms"),
                                      "total_s": (entry.get("timings") or {}).get("total_s"),
                                      "native_ms": hop["native"].get("prompt_ms"), "error": entry.get("error")}),
                          flush=True)
                record["hops"].append(hop)
                if any("error" in entry for entry in hop["engines"].values()):
                    break
            results.append(record)
            with open(args.output / "hops.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
    finally:
        rig.stop()
    return results


# -- drafts ---------------------------------------------------------------------------------------

def drafts(args):
    rig = Rig(args, {args.source, args.target})
    results = []
    try:
        rig.start()
        source, target = rig.client(args.source), rig.client(args.target)
        label, (config, path) = next(iter(rig.configs.items()))
        chosen = choose(args, target, corpus(args), limit=args.limit, min_tokens=args.min_tokens,
                        max_tokens=min(spec["ctx"] for spec in rig.specs.values()) - args.n_predict - 512)
        offset = rig.servers[args.target].log_offset()
        for item, conversation, _ in chosen:
            record = {"id": item.get("id"), "source": item.get("source"), "engine": label}
            record["prime"] = {"model": args.source, **prime(source, conversation)}
            tokens = target_tokens(target, conversation)
            target.slot_action("erase")
            native, _ = target.complete(tokens, n_predict=args.n_predict)
            record["native"] = {**generation(native), **machine()}
            mark = rig.servers[args.target].log_offset()
            try:
                warmed = prewarm(config, path, args.source, args.target)
                before = machine()
                report = switch(config, conversation, args.source, args.target, config_path=path, warm=False)
                record["switch"] = switch_record(report, warmed, before)
                bridged, _ = target.complete(tokens, n_predict=args.n_predict)
                record["bridged"] = {**generation(bridged), **machine()}
                record["bridged"]["reuse_ok"] = record["bridged"]["cache_n"] == record["switch"]["t0"]
                record["agree_prefix"] = agree(bridged.get("tokens"), native.get("tokens"))
                record["log_after_switch"] = rig.servers[args.target].scan_log(mark)
            except Exception as error:      # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"[:400]
            results.append(record)
            print(json.dumps({"id": record["id"], "native": record["native"].get("acceptance"),
                              "bridged": (record.get("bridged") or {}).get("acceptance"),
                              "cache_n": (record.get("bridged") or {}).get("cache_n"),
                              "t0": (record.get("switch") or {}).get("t0"), "error": record.get("error")}), flush=True)
            with open(args.output / "drafts.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        summary = {"log": rig.servers[args.target].scan_log(offset), "argv": rig.servers[args.target].argv[1:]}
        (args.output / "drafts-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    finally:
        rig.stop()
    return results


# -- media ----------------------------------------------------------------------------------------

def chart_png(width=480, height=320):
    """A bar chart as PNG bytes (stdlib only): five bars of heights 3, 7, 4, 9, 5 on white, with axes.

    Colors: blue, orange, green, red, purple, left to right."""
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    values = [3, 7, 4, 9, 5]
    pixels = [[(255, 255, 255)] * width for _ in range(height)]
    left, bottom, top = 40, height - 30, 20
    for y in range(top, bottom + 1):                    # y axis
        pixels[y][left] = (0, 0, 0)
    for x in range(left, width - 20):                   # x axis
        pixels[bottom][x] = (0, 0, 0)
    bar, gap = 60, 25
    for i, (value, color) in enumerate(zip(values, colors)):
        x0 = left + gap + i * (bar + gap)
        y0 = bottom - int(value / 10 * (bottom - top))
        for y in range(y0, bottom):
            for x in range(x0, x0 + bar):
                pixels[y][x] = color
    raw = b"".join(b"\x00" + bytes(c for pixel in row for c in pixel) for row in pixels)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def image_part(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


QUESTION = "How many bars does this chart have, and which color is the tallest bar?"


def without_images(messages):
    """The conversation as a text-only model saw it: image parts dropped, text parts joined."""
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part.get("type") == "text")
        out.append({**message, "content": content})
    return out


def chat(server, messages, tools, max_tokens):
    """The real chat request, with the server's default template settings (the options mode sends
    requests with their own template options: chat_options)."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "id_slot": 0,
            "cache_prompt": True}
    if tools:
        body["tools"] = tools
    reply, seconds = server._call("POST", "/v1/chat/completions", body)
    message = reply["choices"][0]["message"]
    return {**generation(reply), "wall_s": round(seconds, 2), "answer": (message.get("content") or "")[:600],
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "reasoning_head": (message.get("reasoning_content") or "")[:300],
            "finish_reason": reply["choices"][0].get("finish_reason")}


def media(args):
    rig = Rig(args, {args.source, args.target})
    results = []
    png = chart_png()
    (args.output / "chart.png").write_bytes(png)
    try:
        rig.start()
        source, target = rig.client(args.source), rig.client(args.target)
        label, (config, path) = next(iter(rig.configs.items()))
        (item, base, _), = choose(args, target, corpus(args), limit=1, min_tokens=args.min_tokens,
                                  max_tokens=min(spec["ctx"] for spec in rig.specs.values()) - 4096)
        history = item["messages"]                     # the whole conversation, ending with an answer
        users = [i for i, m in enumerate(history) if m["role"] == "user"]
        early = users[int(len(users) * 0.6)]           # (ii): an image in a user message 60% of the way in
        cases = {
            "newest": history + [{"role": "user", "content": [{"type": "text", "text": QUESTION}, image_part(png)]}],
            "earlier": [({**m, "content": [{"type": "text", "text": m["content"]}, image_part(png)]} if i == early else m)
                        for i, m in enumerate(history)] + [{"role": "user", "content": QUESTION + " (the chart I sent "
                                                                                       "earlier)"}],
        }
        for name, messages in cases.items():
            record = {"case": name, "id": item.get("id"), "messages": len(messages), "engine": label}
            if name == "earlier":
                record["image_message"] = early
            try:
                record["prime"] = {"model": args.source,
                                   **prime(source, {"messages": without_images(messages), "tools": base.get("tools")})}
                target.slot_action("erase")
                record["native"] = {**chat(target, messages, base.get("tools"), args.max_tokens), **machine()}
                warmed = prewarm(config, path, args.source, args.target)
                before = machine()
                report = switch(config, conversation_from_openai(messages, base.get("tools")), args.source, args.target,
                                config_path=path, warm=False)
                record["switch"] = switch_record(report, warmed, before)
                record["bridged"] = {**chat(target, messages, base.get("tools"), args.max_tokens), **machine()}
                record["bridged"]["reuse_ok"] = record["bridged"]["cache_n"] == record["switch"]["t0"]
                record["same_answer"] = record["bridged"]["answer"] == record["native"]["answer"]
            except Exception as error:      # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"[:400]
            results.append(record)
            print(json.dumps({k: record.get(k) for k in ("case", "error", "same_answer")}
                             | {"t0": (record.get("switch") or {}).get("t0"),
                                "cache_n": (record.get("bridged") or {}).get("cache_n")}), flush=True)
        # back into the text-only source: the image must make the switch skip
        report = switch(config, conversation_from_openai(cases["newest"], base.get("tools")), args.target, args.source,
                        config_path=path, warm=False)
        results.append({"case": "into-text-only", "skipped": report.get("skipped"), "note": report.get("note")})
        print(json.dumps(results[-1]), flush=True)
        # a request that turns thinking off renders differently from the start (Qwen3.8 writes the
        # reasoning effort into the prompt): an engine with the "template-options" feature renders it
        # with the request's options, an older one is skipped rather than translate unused cells
        report = switch(config, conversation_from_openai(cases["newest"], base.get("tools")), args.source, args.target,
                        config_path=path, warm=False, template_options={"chat_template_kwargs": {"enable_thinking": False}})
        results.append({"case": "thinking-off", "skipped": report.get("skipped"), "note": report.get("note")})
        print(json.dumps(results[-1]), flush=True)
        with open(args.output / "media.jsonl", "a", encoding="utf-8") as stream:
            for record in results:
                stream.write(json.dumps(record) + "\n")
        (args.output / "media-log.json").write_text(json.dumps(rig.servers[args.target].scan_log(), indent=1) + "\n",
                                                    encoding="utf-8")
    finally:
        rig.stop()
    return results


# -- controls: the target's own cache restored like a translated one -------------------------------

def synthesized_resume(rig, name, config, path):
    """(the .resume bytes builder for t0, draft kind): the draft state the bridge writes for a
    translated restore into `name`, from its cached layout template (made if needed)."""
    server, cfg = rig.client(name), config["models"][name]
    template, template_resume, speculative = bridge_switch.layout_template(server, cfg, path.parent.resolve())
    if template_resume is None:
        return None, f"upstream ({speculative})"
    n_tokens = bridge_switch._template_tokens(template)
    data = template_resume.read_bytes()
    kind = resume.classify_draft(resume.parse_resume(data).draft, n_tokens=n_tokens).kind
    return (lambda t0: resume.translated_resume(data, t0, n_tokens=n_tokens)[0]), kind


def restore_own(server, tokens, t0, slot_dir, *, resume_bytes=None):
    """Restore the server's own cache of tokens[:t0] as a switch restores a translated one: a
    native prefill of [0, t0) (one generated token, which the slot does not keep), saved,
    erased, restored. resume_bytes replaces the saved .resume (its draft state) first."""
    server.slot_action("erase")
    _, prefill = server.complete(tokens[:t0], n_predict=1)
    name = f"kvcontrol-{os.getpid()}-{t0}.bin"
    path = Path(slot_dir) / name
    sidecar = Path(str(path) + ".resume")
    try:
        saved, _ = server.slot_action("save", name)
        if resume_bytes is not None:
            sidecar.unlink(missing_ok=True)
            resume.write_private(sidecar, resume_bytes)
        server.slot_action("erase")
        restored, seconds = server.slot_action("restore", name)
    finally:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    return {"t0": t0, "saved": saved.get("n_saved"), "restored": restored.get("n_restored"),
            "prefill_s": round(prefill, 2), "restore_s": round(seconds, 3)}


def drafts_control(args):
    rig = Rig(args, {args.source, args.target})
    results = []
    try:
        rig.start()
        source, target = rig.client(args.source), rig.client(args.target)
        label, (config, path) = next(iter(rig.configs.items()))
        synthesize, kind = synthesized_resume(rig, args.target, config, path)
        slot_dir = Path(config["models"][args.target]["slot_dir"])
        chosen = choose(args, target, corpus(args), limit=args.limit, min_tokens=args.min_tokens,
                        max_tokens=min(spec["ctx"] for spec in rig.specs.values()) - args.n_predict - 512)
        offset = rig.servers[args.target].log_offset()
        for item, conversation, _ in chosen:
            record = {"id": item.get("id"), "source": item.get("source"), "engine": label, "draft": kind}
            tokens = target_tokens(target, conversation)
            try:
                target.slot_action("erase")
                native, _ = target.complete(tokens, n_predict=args.n_predict)
                record["native"] = {**generation(native), **machine()}
                record["prime"] = {"model": args.source, **prime(source, conversation)}
                warmed = prewarm(config, path, args.source, args.target)
                report = switch(config, conversation, args.source, args.target, config_path=path, warm=False)
                record["switch"] = switch_record(report, warmed, machine())
                t0 = record["switch"]["t0"]
                generated = {}
                for case, data in (("bridged", None), ("own-synthesized", synthesize), ("own-native", None)):
                    mark = rig.servers[args.target].log_offset()
                    if case != "bridged":
                        record[case + "-restore"] = restore_own(target, tokens, t0, slot_dir,
                                                                resume_bytes=data(t0) if data else None)
                    reply, _ = target.complete(tokens, n_predict=args.n_predict)
                    generated[case] = reply.get("tokens")
                    record[case] = {**generation(reply), **machine(),
                                    "reuse_ok": (reply.get("timings") or {}).get("cache_n") == t0,
                                    "agree_prefix": agree(reply.get("tokens"), native.get("tokens")),
                                    "log": rig.servers[args.target].scan_log(mark)}
                # the same target cells, two draft states: does the text differ at all?
                record["own-synthesized"]["agree_own_native"] = agree(generated["own-synthesized"],
                                                                      generated["own-native"])
            except Exception as error:      # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"[:400]
            results.append(record)
            print(json.dumps({"id": record["id"], "error": record.get("error"),
                              **{case: {key: (record.get(case) or {}).get(key) for key in ("acceptance", "cache_n",
                                                                                         "agree_prefix")}
                                 for case in ("native", "bridged", "own-synthesized", "own-native")}}), flush=True)
            with open(args.output / "drafts-control.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        summary = {"log": rig.servers[args.target].scan_log(offset), "argv": rig.servers[args.target].argv[1:]}
        (args.output / "drafts-control-summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                                 encoding="utf-8")
    finally:
        rig.stop()
    return results


def media_control(args):
    rig = Rig(args, {args.source, args.target})
    results = []
    png = chart_png()
    (args.output / "chart.png").write_bytes(png)
    try:
        rig.start()
        source, target = rig.client(args.source), rig.client(args.target)
        label, (config, path) = next(iter(rig.configs.items()))
        synthesize, kind = synthesized_resume(rig, args.target, config, path)
        slot_dir = Path(config["models"][args.target]["slot_dir"])
        marker = target.props().get("media_marker")
        chosen = choose(args, target, corpus(args), limit=args.limit, min_tokens=args.min_tokens,
                        max_tokens=min(spec["ctx"] for spec in rig.specs.values()) - 4096)
        offset = rig.servers[args.target].log_offset()
        for item, base, _ in chosen:
            history = item["messages"]
            users = [i for i, m in enumerate(history) if m["role"] == "user"]
            early = users[int(len(users) * 0.6)]          # as media's case (ii)
            messages = ([({**m, "content": [{"type": "text", "text": m["content"]}, image_part(png)]} if i == early
                          else m) for i, m in enumerate(history)]
                        + [{"role": "user", "content": QUESTION + " (the chart I sent earlier)"}])
            record = {"case": "earlier", "id": item.get("id"), "messages": len(messages), "image_message": early,
                      "engine": label, "draft": kind}
            try:
                target.slot_action("erase")
                record["native"] = {**chat(target, messages, base.get("tools"), args.max_tokens), **machine()}
                record["prime"] = {"model": args.source,
                                   **prime(source, {"messages": without_images(messages), "tools": base.get("tools")})}
                warmed = prewarm(config, path, args.source, args.target)
                conversation = conversation_from_openai(messages, base.get("tools"))
                report = switch(config, conversation, args.source, args.target, config_path=path, warm=False)
                record["switch"] = switch_record(report, warmed, machine())
                t0 = record["switch"]["t0"]
                mark = rig.servers[args.target].log_offset()
                record["bridged"] = {**chat(target, messages, base.get("tools"), args.max_tokens), **machine()}
                record["bridged"]["log"] = rig.servers[args.target].scan_log(mark)
                # the same [0, t0) with the target's own values, tokenized as the bridge does
                flat = bridge_messages.flatten(conversation, marker=marker,
                                               text_join=bridge_switch._text_join(target, conversation))
                tokens = [t for t, _ in target.tokenize(target.apply_template(flat["messages"], flat.get("tools")),
                                                       add_special=True, parse_special=True)]
                record["own-restore"] = restore_own(target, tokens, t0, slot_dir,
                                                    resume_bytes=synthesize(t0) if synthesize else None)
                mark = rig.servers[args.target].log_offset()
                record["own"] = {**chat(target, messages, base.get("tools"), args.max_tokens), **machine()}
                record["own"]["log"] = rig.servers[args.target].scan_log(mark)
                for case in ("bridged", "own"):
                    record[case]["reuse_ok"] = record[case]["cache_n"] == t0
                    record[case]["same_answer"] = record[case]["answer"] == record["native"]["answer"]
            except Exception as error:      # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"[:400]
            results.append(record)
            print(json.dumps({"id": record["id"], "error": record.get("error"),
                              "t0": (record.get("switch") or {}).get("t0"),
                              **{case: {key: (record.get(case) or {}).get(key) for key in ("cache_n", "same_answer",
                                                                                         "reasoning_chars")}
                                 for case in ("native", "bridged", "own")}}), flush=True)
            with open(args.output / "media-control.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        # request template options: with an engine older than the "template-options" feature, T0
        # stops where the request's rendering departs from the default one (Qwen3.8:
        # enable_thinking false from its system prompt on, so a skip; an effort its template maps
        # to the server's default changes nothing); a newer engine renders with them (options mode)
        item, _, _ = chosen[0]
        conversation = conversation_for(item)
        prime(source, conversation)
        for options in ({"chat_template_kwargs": {"enable_thinking": False}}, {"reasoning_effort": "high"}):
            report = switch(config, conversation, args.source, args.target, config_path=path, template_options=options)
            results.append({"case": "template-options", "id": item.get("id"), "options": options,
                            "skipped": report.get("skipped"), "template_options": report.get("template_options"),
                            "t0": (report.get("engine") or {}).get("t0"), "reuse_ok": report.get("reuse_ok"),
                            "note": report.get("note")})
            print(json.dumps(results[-1]), flush=True)
        (args.output / "media-control-log.json").write_text(
            json.dumps(rig.servers[args.target].scan_log(offset), indent=1) + "\n", encoding="utf-8")
    finally:
        rig.stop()
    return results


# -- options: request template options, one 27-30B model at a time ----------------------------------

class StubSource:
    """The source's llama-server as far as a switch uses it, after the real one has stopped: a slot
    save copies the slot file the real server saved for the selected conversation (select) into
    the slot directory under the requested name; /props and /health answer as a text model."""

    def __init__(self, port, slot_dir):
        self.slot_dir, self.current, self.saves = Path(slot_dir), None, 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    return self.reply(200, {"status": "ok"})
                if self.path == "/props":
                    return self.reply(200, {"build_info": "stub", "total_slots": 1,
                                            "modalities": {"vision": False, "audio": False, "video": False}})
                return self.reply(404, {"error": "not served by the stub"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path != "/slots/0?action=save" or stub.current is None:
                    return self.reply(400, {"error": "the stub only saves the selected conversation's slot"})
                path = stub.slot_dir / Path(body["filename"]).name
                shutil.copyfile(stub.current["file"], path)
                stub.saves += 1
                return self.reply(200, {"id_slot": 0, "filename": path.name, "n_saved": stub.current["tokens"],
                                        "n_written": path.stat().st_size, "timings": {"save_ms": 0.0}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def select(self, path, tokens):
        self.current = {"file": Path(path), "tokens": tokens}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def chat_options(server, conversation, options, max_tokens=4):
    """The real chat request with these template options, as Hermes' OpenAI client sends them
    (top-level reasoning_effort and chat_template_kwargs): its prompt reuse and prefill time."""
    body = {"messages": conversation["messages"], "max_tokens": max_tokens, "temperature": 0, "id_slot": 0,
            "cache_prompt": True, **(options or {})}
    if conversation.get("tools"):
        body["tools"] = conversation["tools"]
    reply, seconds = server._call("POST", "/v1/chat/completions", body)
    return {**generation(reply), "wall_s": round(seconds, 2)}


def options(args):
    specs = server_specs(args)
    extras = [name for name in (args.extra_target or "").split(",") if name]
    missing = {args.source, args.target, *extras} - set(specs)
    if missing:
        raise SystemExit(f"no server spec for {sorted(missing)}")
    slot_root = Path(args.slot_root)
    held = slot_root / "held"
    held.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []

    def start(name, number):
        spec = specs[name]
        slot_dir = slot_root / name
        slot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for stale in slot_dir.iterdir():
            stale.unlink()
        waited = wait_for_memory(name, spec["gguf"], args=spec.get("args"))
        server = Server(name, spec["binary"], spec["gguf"], spec.get("port", args.base_port + number), spec["ctx"],
                        slot_dir, args.output / f"{name}.log", args=spec.get("args"), env=spec.get("env"))
        load = server.start()
        print(json.dumps({"server": name, "port": server.port, "load_s": load, "memory_wait_s": waited,
                          **machine()}), flush=True)
        return server

    # 1. the source alone primes every conversation; its slot saves are kept for the stub
    source = start(args.source, 0)
    primed = []
    try:
        client = LlamaServer(f"http://127.0.0.1:{source.port}")
        chosen = choose(args, client, corpus(args), limit=args.limit, min_tokens=args.min_tokens,
                        max_tokens=min(specs[name]["ctx"] for name in (args.source, args.target, *extras)) - 512)
        for number, (item, conversation, _) in enumerate(chosen):
            record = prime(client, conversation)
            name = f"primed-{number}.bin"
            client.slot_action("save", name)
            os.replace(slot_root / args.source / name, held / name)
            Path(slot_root / args.source / (name + ".resume")).unlink(missing_ok=True)
            primed.append((item, conversation, held / name, record["tokens"]))
            print(json.dumps({"primed": item.get("id"), "model": args.source, **record}), flush=True)
    finally:
        source.stop()
    # 2. the targets, with the stub in the source's place
    servers, stub = {}, None
    try:
        for number, name in enumerate([args.target, *extras], start=1):
            servers[name] = start(name, number)
        stub_dir = slot_root / f"{args.source}-stub"
        stub_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        stub = StubSource(args.base_port + 9, stub_dir)
        (label, python, device), = engines(args)
        path = Path(f"bridge-{label}") / "bridge.json"           # relative to --bridge-root (cwd)
        config = registry.empty()
        registry.add_model(config, args.source, url=stub.url, gguf=specs[args.source]["gguf"], slot_dir=stub_dir,
                           **(specs[args.source].get("register") or {}))
        for name, server in servers.items():
            registry.add_model(config, name, url=f"http://127.0.0.1:{server.port}", gguf=specs[name]["gguf"],
                               slot_dir=slot_root / name, **(specs[name].get("register") or {}))
        registry.set_engine(config, python=python, script=Path(__file__).with_name("engine.py"), hub=args.hub,
                            device=device)
        config["tail"] = args.tail
        registry.save(config, path)
        config = registry.load(path)
        target = LlamaServer(f"http://127.0.0.1:{servers[args.target].port}")
        variants = [name for name in args.options.split(",") if name]
        for item, conversation, slot, tokens in primed:
            record = {"id": item.get("id"), "source_tokens": tokens, "engine": label, "options": {}}
            try:
                # what the switch saves: the native prefill of the first options' request
                target.slot_action("erase")
                record["native"] = {"options": variants[0], **chat_options(target, conversation,
                                                                           TEMPLATE_OPTIONS[variants[0]]), **machine()}
                for variant in variants:
                    entry = {}
                    try:
                        stub.select(slot, tokens)
                        warmed = prewarm(config, path, args.source, args.target)
                        before = machine()
                        report = switch(config, conversation, args.source, args.target, config_path=path, warm=False,
                                        template_options=TEMPLATE_OPTIONS[variant])
                        entry = switch_record(report, warmed, before)
                        entry["template_options"] = report.get("template_options")
                        entry["rendering"] = (report.get("engine") or {}).get("rendering")
                        if not report.get("skipped"):
                            entry["chat"] = {**chat_options(target, conversation, TEMPLATE_OPTIONS[variant]),
                                             **machine()}
                            entry["reuse_ok"] = entry["chat"]["cache_n"] == entry["t0"]
                    except Exception as error:      # noqa: BLE001  (record and continue)
                        entry["error"] = f"{type(error).__name__}: {error}"[:400]
                    record["options"][variant] = entry
                    print(json.dumps({"id": record["id"], "options": variant, "skipped": entry.get("skipped"),
                                      "rendering": entry.get("rendering"), "t0": entry.get("t0"),
                                      "P": entry.get("P"), "cache_n": (entry.get("chat") or {}).get("cache_n"),
                                      "reuse_ok": entry.get("reuse_ok"), "recurrent": entry.get("recurrent"),
                                      "engine_s": (entry.get("timings") or {}).get("engine_s"),
                                      "prompt_ms": (entry.get("chat") or {}).get("prompt_ms"),
                                      "native_ms": record["native"].get("prompt_ms"), "error": entry.get("error")}),
                          flush=True)
                # into the extra targets (dense Qwen3-4B: the assemble profile), from the stub's
                # source and from the target primed natively; the jobs are kept for bench_engine.py
                if extras:
                    target.slot_action("erase")
                    prime(target, conversation)
                for name in extras:
                    for origin in (args.source, args.target):
                        key = f"{origin}->{name}"
                        entry = {}
                        try:
                            stub.select(slot, tokens)
                            warmed = prewarm(config, path, origin, name)
                            before = machine()
                            keep = {"KVPARK_BRIDGE_KEEP_JOB": str(args.keep_jobs / f"{record['id']}-{key}".replace(":", "_"))
                                    } if args.keep_jobs else {}
                            previous = {k: os.environ.get(k) for k in keep}
                            os.environ.update(keep)
                            try:
                                report = switch(config, conversation, origin, name, config_path=path, warm=True)
                            finally:
                                for k, v in previous.items():
                                    if v is None:
                                        os.environ.pop(k, None)
                                    else:
                                        os.environ[k] = v
                            entry = switch_record(report, warmed, before)
                        except Exception as error:      # noqa: BLE001
                            entry["error"] = f"{type(error).__name__}: {error}"[:400]
                        record.setdefault("extra", {})[key] = entry
                        print(json.dumps({"id": record["id"], "switch": key, "t0": entry.get("t0"),
                                          "reuse_ok": entry.get("reuse_ok"), "recurrent": entry.get("recurrent"),
                                          "engine_timings": entry.get("engine_timings"), "error": entry.get("error")}),
                              flush=True)
            except Exception as error:      # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"[:400]
            results.append(record)
            with open(args.output / "options.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        (args.output / "options-log.json").write_text(
            json.dumps({name: server.scan_log() for name, server in servers.items()}, indent=1) + "\n",
            encoding="utf-8")
        if args.after:
            # e.g. bench_engine.py job on the kept jobs, with the servers still loaded as during a switch
            with open(args.output / "after.log", "ab") as log:
                started = time.monotonic()
                code = subprocess.call(shlex.split(args.after), stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"after": args.after[:200], "exit": code, "seconds": round(time.monotonic() - started)}),
                  flush=True)
    finally:
        try:
            engine_client.stop(Path("bridge-" + engines(args)[0][0]))
        finally:
            if stub is not None:
                stub.close()
            for server in servers.values():
                server.stop()
            for path in held.iterdir():
                path.unlink()
    return results


MODES = {"hops": hops, "drafts": drafts, "media": media, "drafts-control": drafts_control,
         "media-control": media_control, "options": options}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    mode = argv.pop(0) if argv and argv[0] in MODES else "hops"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--servers", type=Path, help="SERVERS.json (see above)")
    parser.add_argument("--model", action="append", metavar="NAME=GGUF", help="without --servers: servers to start")
    parser.add_argument("--server", type=Path, help="without --servers: the llama-server binary")
    parser.add_argument("--route", help="hops: e.g. qwen38,muse,qwen3-4b,qwen38")
    parser.add_argument("--source", help="drafts, media: the model the conversation comes from")
    parser.add_argument("--target", help="drafts, media: the model it moves to")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ids", help="comma list of conversation ids (default: first --limit that fit)")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--sources", help="comma list of conversation 'source' values to keep")
    parser.add_argument("--min-tokens", type=int, default=6000)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--engine", action="append", metavar="LABEL=PYTHON:DEVICE",
                        help="an engine configuration (repeat for several: every hop runs each)")
    parser.add_argument("--engine-python", type=Path, help="without --engine: the engine Python (device auto)")
    parser.add_argument("--no-prewarm", dest="prewarm", action="store_false",
                        help="hops: do not prewarm the engine before each switch")
    parser.add_argument("--ctx", type=int, default=36864, help="default --ctx-size of the servers")
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256, help="drafts, drafts-control: tokens generated per run")
    parser.add_argument("--max-tokens", type=int, default=1200,
                        help="media, media-control: answer tokens (reasoning included)")
    parser.add_argument("--base-port", type=int, default=18100)
    parser.add_argument("--native-prefix-models", default="qwen38,qwen3-4b,ornith",
                        help="without --servers: targets that get a native prompt start (measured to help the Qwen "
                        "family, hurt Muse)")
    parser.add_argument("--slot-root", type=Path, help="slot directories (default: OUTPUT/slots)")
    parser.add_argument("--bridge-root", type=Path, help="where the bridge configurations live (default: OUTPUT); "
                        "the process works there, so the engine daemon's socket path stays short")
    parser.add_argument("--options", default=",".join(TEMPLATE_OPTIONS),
                        help=f"options: the template options of the chat requests ({', '.join(TEMPLATE_OPTIONS)}); "
                        "the first also gives the native reference")
    parser.add_argument("--extra-target", help="options: comma list of more targets (small models: they run next to "
                        "the target) switched into with the default rendering, from the source and from the target")
    parser.add_argument("--keep-jobs", type=Path, help="options: keep the extra targets' engine jobs and inputs here "
                        "(KVPARK_BRIDGE_KEEP_JOB; on disk, they hold slot copies)")
    parser.add_argument("--after", help="options: a command run before the servers stop (its output in "
                        "OUTPUT/after.log)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("servers", "corpus", "hub", "server", "engine_python", "keep_jobs"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.slot_root = (args.slot_root or args.output / "slots").resolve()
    bridge_root = (args.bridge_root or args.output).resolve()
    bridge_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if mode == "hops" and not args.route:
        parser.error("hops needs --route")
    if mode != "hops" and not (args.source and args.target):
        parser.error(f"{mode} needs --source and --target")
    os.chdir(bridge_root)
    results = MODES[mode](args)
    (args.output / f"{mode}.json").write_text(json.dumps({"mode": mode, "argv": argv, "results": results}, indent=1)
                                              + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
