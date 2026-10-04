"""`kvpark bridge ...` commands (experimental)."""

import json
import os
from pathlib import Path
import sys

from . import engine_client, messages, registry

# kvpark bridge add's RoPE flags: (flag, llama_context_params name, type); llama-server's own
# flags of the same names (--rope-scale N sets rope_freq_scale = 1 / N, as llama-server does)
ROPE_FLAGS = (("--rope-scaling", "rope_scaling_type", str), ("--rope-freq-base", "rope_freq_base", float),
              ("--rope-freq-scale", "rope_freq_scale", float), ("--yarn-orig-ctx", "yarn_orig_ctx", int),
              ("--yarn-ext-factor", "yarn_ext_factor", float), ("--yarn-attn-factor", "yarn_attn_factor", float),
              ("--yarn-beta-slow", "yarn_beta_slow", float), ("--yarn-beta-fast", "yarn_beta_fast", float))


def add_parser(sub):
    bridge = sub.add_parser("bridge", help="experimental: move a conversation's cache between different models")
    commands = bridge.add_subparsers(dest="bridge_command", required=True)
    commands.add_parser("status", help="show registered models, the hub and the engine")
    add = commands.add_parser("add", help="register a running llama-server model")
    add.add_argument("name")
    add.add_argument("--url", required=True, help="the model's llama-server, http://127.0.0.1:<port>")
    add.add_argument("--gguf", required=True, help="the model's GGUF file")
    add.add_argument("--slot-dir", required=True, help="the server's --slot-save-path")
    add.add_argument("--hub-name", help="the model's name in the hub (default: NAME)")
    add.add_argument("--tail", type=int, help="as a target, re-read at least this many recent tokens natively "
                     "(overrides the bridge default; dense models need more than sliding-window models)")
    add.add_argument("--native-prefix", action="store_true",
                     help="as a target, prefill the prompt start (system message) natively; helps some models")
    add.add_argument("--min-handover", type=int,
                     help="as a target, translate only prompts longer than tail + this many tokens "
                     "(default 1024; raise it for small models, whose native prefill is already fast)")
    add.add_argument("--draft-window", type=int,
                     help="as a target whose server runs a DFlash draft: the draft's sliding window "
                     "(dflash.attention.sliding_window, default 2048); the native tail is kept at least "
                     "this long minus one, so the draft only sees native context")
    add.add_argument("--recurrent-mean", metavar="NPZ",
                     help="as a hybrid target (Qwen3.5/3.8, Ornith, ...): start the translated state's recurrent "
                     "cell from this mean of the model's native cells (experiments/universal_kv/r0_experiments.py "
                     "calibrate) instead of zeros: the chat group for conversations without tools, the pooled mean "
                     "for those with tools. With it chats keep their quality on a shorter native tail "
                     "(experiments/universal_kv/RESULTS.md section 7: 1024 tokens beat zeros with about 4400); "
                     "conversations with tools do so only with --recurrent-tools, as the pooled mean at 1024 is "
                     "worse than zeros at the default 2048 on Qwen3.8's agent traces. The tail is not changed for "
                     "you (--tail), and a DFlash draft keeps its window either way")
    add.add_argument("--recurrent-tools", action="store_true",
                     help="with --recurrent-mean: its tools group was calibrated on this deployment's own agent "
                     "prompts and tools; use it for conversations that declare tools")
    rope = add.add_argument_group("RoPE settings of the model's server, when it runs with any of these "
                                  "llama-server flags (the same values)")
    rope.add_argument("--rope-scaling", choices=registry.ROPE_SCALING_TYPES)
    rope.add_argument("--rope-scale", type=float, metavar="N", help="sets rope_freq_scale = 1/N, as llama-server")
    for flag, _, kind in ROPE_FLAGS[1:]:
        rope.add_argument(flag, type=kind, metavar="N")
    template = add.add_argument_group("chat template settings of the model's server, when it runs with any of "
                                      "these llama-server flags (the same values): as a target, the engine "
                                      "renders the conversation with them")
    template.add_argument("--chat-template-kwargs", metavar="JSON")
    template.add_argument("--reasoning", choices=sorted(messages.REASONING_VALUES), metavar="on|off|auto")
    template.add_argument("--reasoning-effort", metavar="LEVEL")
    template.add_argument("--reasoning-preserve", dest="reasoning_preserve", action="store_true", default=None)
    template.add_argument("--no-reasoning-preserve", dest="reasoning_preserve", action="store_false")
    remove = commands.add_parser("remove", help="forget a registered model")
    remove.add_argument("name")
    engine = commands.add_parser("engine", help="set the translation engine, its hub bundle and options "
                                 "(settings not given are kept; a running engine daemon is stopped)")
    engine.add_argument("--python", help="Python with numpy (and torch for the GPU) for the engine worker; "
                        "required the first time")
    engine.add_argument("--script", help="the engine script (experiments/universal_kv/engine.py); required the "
                        "first time")
    engine.add_argument("--hub", help="hub bundle directory (format 1, or 2 from convert_hub.py); required the "
                        "first time")
    engine.add_argument("--device", choices=registry.ENGINE_DEVICES,
                        help="gpu: torch on a ROCm/CUDA/MPS device; cpu: numpy; auto (default): the GPU when the "
                        "engine Python's torch has one")
    engine.add_argument("--mode", choices=registry.ENGINE_MODES,
                        help="daemon (default): keep the engine resident between switches; oneshot: a fresh "
                        "engine process per switch")
    engine.add_argument("--idle-unload", type=float, metavar="S",
                        help="the daemon drops its hub sides after S idle seconds (default 120)")
    engine.add_argument("--idle-exit", type=float, metavar="S", help="the daemon exits after S idle seconds "
                        "(default 600)")
    engine.add_argument("--threads", type=int, metavar="N", help="BLAS threads of the CPU path (default: one per "
                        "physical core, at most 16; 0 restores the default)")
    engine.add_argument("--tail", type=int, help="native tail in tokens (default 2048)")
    engine.add_argument("--fill", choices=("mean",), help="fill for cells the hub does not cover")
    commands.add_parser("engine-status", help="show the engine daemon: device, resident hub sides, memory, jobs")
    commands.add_parser("engine-stop", help="stop the engine daemon (the next switch starts it again)")
    move = commands.add_parser("switch", help="move a conversation from one model's server to another's")
    move.add_argument("--from", dest="source", required=True)
    move.add_argument("--to", dest="target", required=True)
    move.add_argument("--conversation", required=True, type=Path,
                      help='OpenAI-style JSON {"messages": [...], "tools": [...]} ending with the message the '
                      'target answers next (chat_template_kwargs and reasoning_effort, as the chat request '
                      'will carry them, may be given too)')
    move.add_argument("--tail", type=int)
    move.add_argument("--no-warm", action="store_true", help="skip the warm request that prefills the tail")
    undo = commands.add_parser("undo", help="drop a translated cache from a model's server")
    undo.add_argument("model")
    for parser in commands.choices.values():
        parser.add_argument("--config", type=Path, help="bridge configuration (default: kvpark data dir)")
    return bridge


def daemon_summary(stats):
    """The engine daemon's state for `kvpark bridge status` from its stats reply (None: not running)."""
    if not stats:
        return {"running": False}
    return {"running": True, "pid": stats.get("pid"), "device": stats.get("device"),
            "backends": sorted(stats.get("backends") or {}), "uptime_s": stats.get("uptime_s"),
            "idle_s": stats.get("idle_s"), "jobs": stats.get("jobs"), "failures": stats.get("failures"),
            "resident_bytes": sum(int(b.get("resident_bytes") or 0) for b in (stats.get("backends") or {}).values()),
            "gpu_fault": stats.get("gpu_fault"), "features": stats.get("features") or []}


def engine_warnings(config, engine, daemon, bridge_dir):
    """Problems worth a warning in `kvpark bridge status`: an engine Python that cannot be asked, a
    reference-BLAS numpy, a GPU asked for but missing, a GPU disabled after a fault, a daemon
    that cannot run (switches then run the slower one-shot engine), a format 1 hub."""
    warnings = []
    if (engine or {}).get("error"):
        warnings.append(f"{engine['error']} (switches need a working engine Python: `kvpark bridge engine "
                        "--python`)")
    blas = (engine or {}).get("blas") or {}
    if blas.get("reference"):
        warnings.append("the engine Python's numpy uses the reference BLAS (10-100x slower: a switch takes "
                        "minutes); point `kvpark bridge engine --python` at a Python whose numpy bundles OpenBLAS, "
                        "e.g. experiments/linear_transfer/.venv")
    elif blas.get("warning"):
        warnings.append(f"the engine Python: {blas['warning']}")
    gpu = (engine or {}).get("gpu")
    if (config.get("engine") or {}).get("device") == "gpu" and gpu is not None and not gpu.get("available"):
        warnings.append(f"engine device gpu, but the engine Python has no GPU: {gpu.get('detail')}")
    if daemon.get("gpu_fault"):
        warnings.append(f"the engine daemon stopped using the GPU after an error: {daemon['gpu_fault']}")
    if (config.get("engine") or {}).get("mode", "daemon") == "daemon" and os.name != "nt":
        # Windows always runs the one-shot engine (no AF_UNIX daemon), as docs/bridge.md says
        if not engine_client.daemon_supported(bridge_dir):
            warnings.append(f"the engine daemon cannot run in {bridge_dir}: its socket path must stay under "
                            f"{engine_client.SOCKET_PATH_LIMIT} bytes, so every switch runs the one-shot engine "
                            "(about 10 s slower); use a shorter KVPARK_BRIDGE_CONFIG path")
        elif engine_client._daemon_script(config) is None:
            warnings.append(f"no {engine_client.DAEMON_SCRIPT} next to the engine script: every switch runs the "
                            "one-shot engine (about 10 s slower)")
    try:
        if config.get("hub") and registry.hub_format(config["hub"]) == 1:
            warnings.append("the hub is format 1: convert it once with experiments/universal_kv/convert_hub.py "
                            "for faster loading")
    except ValueError as error:
        warnings.append(str(error))
    return warnings


def run(args):
    path = args.config or registry.default_path()
    config = registry.load(path)
    bridge_dir = Path(path).parent
    command = args.bridge_command
    if command == "status":
        models = registry.hub_models(config) if config.get("hub") else set()
        for name in config["models"]:
            registry.model_arch(config, name, path=path)      # entries registered before it was recorded
        out = {"config": str(path), "hub": config.get("hub"), "hub_models": sorted(models),
               "engine": config.get("engine"), "tail": config.get("tail"),
               "models": {name: {**cfg, "in_hub": cfg["hub_name"] in models}
                          for name, cfg in config["models"].items()}}
        if config.get("engine"):
            stats = engine_client.status(bridge_dir)
            out["engine_daemon"] = daemon_summary(stats)
            engine = (stats or {}).get("engine")
            if engine is None:
                try:          # no daemon to ask: the engine Python reports its numpy and GPU itself
                    engine = engine_client.check(config)
                except engine_client.DaemonError as error:
                    engine = {"error": str(error)}
            out["engine_check"] = engine
            out["warnings"] = engine_warnings(config, engine, out["engine_daemon"], bridge_dir)
            for warning in out["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
        print(json.dumps(out, indent=2))
    elif command == "add":
        runtime = {key: getattr(args, flag[2:].replace("-", "_")) for flag, key, _ in ROPE_FLAGS}
        if args.rope_scale is not None:
            if args.rope_freq_scale is not None:
                raise ValueError("give --rope-scale or --rope-freq-scale, not both")
            if not args.rope_scale > 0:
                raise ValueError("--rope-scale must be positive")
            runtime["rope_freq_scale"] = 1.0 / args.rope_scale
        template_kwargs = messages.server_template_kwargs(
            chat_template_kwargs=args.chat_template_kwargs, reasoning=args.reasoning,
            reasoning_effort=args.reasoning_effort, reasoning_preserve=args.reasoning_preserve)
        registry.save(registry.add_model(config, args.name, url=args.url, gguf=args.gguf, slot_dir=args.slot_dir,
                                         hub_name=args.hub_name, tail=args.tail,
                                         native_prefix=args.native_prefix, min_handover=args.min_handover,
                                         draft_window=args.draft_window, runtime=runtime,
                                         template_kwargs=template_kwargs, recurrent_mean=args.recurrent_mean,
                                         recurrent_tools=args.recurrent_tools), path)
        print(f"registered {args.name} ({config['models'][args.name]['arch']})")
    elif command == "remove":
        if config["models"].pop(args.name, None) is None:
            raise ValueError(f"unknown bridge model {args.name!r}")
        registry.save(config, path)
        print(f"removed {args.name}")
    elif command == "engine":
        registry.set_engine(config, python=args.python, script=args.script, hub=args.hub, device=args.device,
                            mode=args.mode, idle_unload=args.idle_unload, idle_exit=args.idle_exit,
                            threads=args.threads)
        if args.tail:
            config["tail"] = args.tail
        if args.fill:
            config["fill"] = args.fill
        registry.save(config, path)
        # a running daemon keeps the Python, script, device and idle times it was started with
        stopped = engine_client.stop(bridge_dir)
        print(json.dumps({"hub_models": sorted(registry.hub_models(config)), "engine": config["engine"],
                          "daemon_stopped": stopped}))
    elif command == "engine-status":
        stats = engine_client.status(bridge_dir)
        print(json.dumps(stats if stats else {"running": False, "bridge_dir": str(bridge_dir)}, indent=2))
    elif command == "engine-stop":
        print("stopped the engine daemon" if engine_client.stop(bridge_dir) else "no engine daemon is running")
    elif command == "switch":
        from .messages import conversation_from_openai, template_options
        from .switch import switch
        # OpenAI-style messages, as Hermes sends them: content parts become text and media placeholders
        raw = json.loads(args.conversation.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("the conversation file must hold {\"messages\": [...], \"tools\": [...]}")
        conversation = conversation_from_openai(raw.get("messages"), raw.get("tools"))
        report = switch(config, conversation, args.source, args.target, tail=args.tail, warm=not args.no_warm,
                        config_path=path, template_options=template_options(raw),
                        log=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(report, indent=2))
    elif command == "undo":
        from .switch import undo
        undo(config, args.model)
        print(f"erased {args.model}'s slot")
