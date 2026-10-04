#!/usr/bin/env python3
"""Real-model restart test on CPU; creates only private child processes and files."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/acceptance.json"))
    parser.add_argument("--ctx-size", type=positive_integer, default=4096)
    parser.add_argument("--prompt-repetitions", type=positive_integer, default=80)
    parser.add_argument("--max-tokens", type=positive_integer, default=32)
    parser.add_argument("--threads", type=positive_integer, default=4)
    parser.add_argument("--ctx-checkpoints", type=int, default=8)
    parser.add_argument("--checkpoint-min-step", type=positive_integer, default=128)
    parser.add_argument("--batch-size", type=positive_integer)
    parser.add_argument("--ubatch-size", type=positive_integer)
    parser.add_argument("--cache-type-k")
    parser.add_argument("--cache-type-v")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"))
    parser.add_argument("--load-mode", choices=("auto", "none", "mmap", "mlock", "mmap+mlock", "dio"))
    parser.add_argument("--no-repack", action="store_true")
    parser.add_argument("--min-prompt-tokens", type=positive_integer,
                        help="require this many measured cold-prefill tokens; use 2049 to cross a 2048-token window")
    parser.add_argument("--direct-continuation", action="store_true",
                        help="send the continuation immediately after restart, without replaying the original request")
    parser.add_argument("--request-timeout", type=positive_integer, default=1200)
    parser.add_argument("--startup-timeout", type=positive_integer, default=300)
    args = parser.parse_args(argv)
    if args.ctx_checkpoints < 0:
        parser.error("ctx-checkpoints must be nonnegative")
    if args.max_tokens >= args.ctx_size:
        parser.error("max-tokens must be smaller than ctx-size")
    if args.min_prompt_tokens and args.min_prompt_tokens + args.max_tokens >= args.ctx_size:
        parser.error("ctx-size must leave room above min-prompt-tokens and max-tokens")
    if args.batch_size and args.ubatch_size and args.ubatch_size > args.batch_size:
        parser.error("ubatch-size must not exceed batch-size")
    args.server, args.model = args.server.resolve(), args.model.resolve()
    if args.output.exists():
        parser.error("output exists; choose a fresh evidence filename")
    return args


def backend_arguments(args):
    result = ["--device", "none", "--n-gpu-layers", "0", "--ctx-size", str(args.ctx_size),
              "--threads", str(args.threads), "--threads-batch", str(args.threads), "--cache-ram", "0",
              "--ctx-checkpoints", str(args.ctx_checkpoints),
              "--checkpoint-min-step", str(args.checkpoint_min_step), "--jinja"]
    for option in ("batch_size", "ubatch_size", "cache_type_k", "cache_type_v", "flash_attn", "load_mode"):
        value = getattr(args, option)
        if value is not None:
            result.extend(["--" + option.replace("_", "-"), str(value)])
    if args.no_repack:
        result.append("--no-repack")
    return result


def snapshot_evidence(archive, parked):
    metadata = json.loads((archive / (parked["key"] + ".meta")).read_text())
    filename = metadata["filename"]
    if (Path(filename).name != filename or not filename.endswith(".bin")
            or metadata.get("format") != 2 or metadata.get("tokens") != parked["tokens"]):
        raise RuntimeError("published archive metadata does not match the parked state")
    files = {}
    for suffix in ("", ".resume"):
        path = archive / (filename + suffix)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("published snapshot is missing a nonempty .bin/.bin.resume pair")
        files[".bin" + suffix] = path.stat().st_size
    if sum(files.values()) != metadata.get("bytes"):
        raise RuntimeError("published snapshot sizes differ from archive metadata")
    return dict(format=metadata["format"], tokens=metadata["tokens"], files=files)


def generated_messages_match(expected, actual):
    return (isinstance(expected, dict) and isinstance(actual, dict)
            and any(expected.get(field) for field in ("content", "reasoning_content", "tool_calls"))
            and expected == actual)


def acceptance_checks(cold, warm, replay, restored, events, parked_tokens, min_prompt_tokens=None):
    def reused_majority(completion):
        cached, processed = completion["cached_tokens"], completion["processed_tokens"]
        return (type(cached) is int and type(processed) is int
                and processed >= 0 and cached > processed)

    checks = dict(cold_uncached=cold["cached_tokens"] == 0,
                  disk_restore=any(e["action"] == "restored" and e.get("tokens") == parked_tokens for e in events),
                  continuation_reused_majority=reused_majority(restored),
                  continuation_matches=generated_messages_match(warm["message"], restored["message"]))
    if min_prompt_tokens is not None:
        checks["minimum_prompt_tokens"] = (cold["processed_tokens"] or 0) >= min_prompt_tokens
    if replay is not None:
        checks.update(replay_reused_majority=reused_majority(replay),
                      replay_matches=generated_messages_match(cold["message"], replay["message"]))
    return checks


def main(argv=None):
    args = arguments(argv)
    backend_options = backend_arguments(args)
    children, logs = [], []
    with tempfile.TemporaryDirectory(prefix="kvpark-acceptance-") as tmp:
        root = Path(tmp)
        proxy_port, backend_port = free_port(), free_port()
        while proxy_port == backend_port:
            backend_port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("KVPARK_", "LLAMA_ARG_"))}
        env.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")

        def call(port, path, body=None):
            try:
                with urlopen(Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"}), timeout=args.request_timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                detail = exc.read(8192).decode("utf-8", errors="replace")
                exc.msg = f"{exc.reason} from {path}: {detail}"
                raise

        def spawn(argv, name):
            log = (root / name).open("w")
            logs.append(log)
            process = subprocess.Popen([sys.executable, "-m", "kvpark", *argv],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            children.append(process)
            return process

        def ready(process, port, endpoint, log):
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"child failed:\n{(root / log).read_text()[-8000:]}")
                try:
                    actual = port() if callable(port) else port
                    if actual:
                        call(actual, endpoint)
                        return actual
                except OSError:
                    time.sleep(.2)
            raise TimeoutError("private service did not become ready")

        def start():
            nonlocal backend_port, proxy_port
            from kvpark.network import managed_port, proxy_record
            backend = spawn(["backend", "--server", str(args.server), "--model", str(args.model),
                             "--archive-dir", str(root / "archive"), "--port", str(backend_port), "--",
                             *backend_options], "backend.log")
            backend_port = ready(backend, lambda: managed_port(root / "archive"), "/health", "backend.log")
            proxy = spawn(["serve", "--archive-dir", str(root / "archive"), "--port", str(proxy_port),
                           "--upstream-port", str(backend_port)], "proxy.log")
            def bound_port():
                record = proxy_record(root / "archive")
                return int(record["base_url"].rsplit(":", 1)[1]) if record else None
            proxy_port = ready(proxy, bound_port, "/_kvpark/status", "proxy.log")
            if not call(proxy_port, "/_kvpark/doctor")["ok"]:
                raise RuntimeError("runtime diagnostics failed")

        def stop():
            for process in reversed(children):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            children.clear()
            for log in logs:
                log.close()
            logs.clear()

        def completion(messages):
            started = time.monotonic()
            response = call(proxy_port, "/v1/chat/completions", dict(model="local", messages=messages,
                slot_archive_role="foreground", slot_archive_key="acceptance:project:1", max_tokens=args.max_tokens,
                temperature=0, seed=42, reasoning_effort="none"))
            # HTTP EOF can precede the proxy's final bookkeeping by a few ms.
            for _ in range(100):
                status = call(proxy_port, "/_kvpark/status")
                if status["activity"]["phase"] == "idle":
                    break
                time.sleep(.01)
            event = next(e for e in status["recent"] if e["action"] == "completed")
            result = dict(seconds=round(time.monotonic() - started, 3), cached_tokens=event.get("cached_tokens"),
                          processed_tokens=event.get("processed_tokens"), message=response["choices"][0]["message"])
            print(json.dumps(result), flush=True)
            return result

        try:
            print("Starting private CPU backend and proxy; no production services are changed.", flush=True)
            start()
            messages = [dict(role="system", content="Answer briefly. The project code is ORCHID-731."),
                        dict(role="user", content="We plan the workshop and track its materials and schedule.\n" * args.prompt_repetitions +
                             "Reply with the project code only.")]
            cold = completion(messages)
            if cold["cached_tokens"] != 0:
                raise RuntimeError("cold baseline unexpectedly reused tokens")
            status = call(proxy_port, "/_kvpark/status")
            parked = call(proxy_port, "/_kvpark/park", {"key": status["resident_key"]})
            snapshot = snapshot_evidence(root / "archive", parked)
            continued = messages + [cold["message"], dict(role="user", content="What is the project code? Reply with the code only.")]
            warm = completion(continued)
            print("Restarting both processes with RAM prompt cache disabled.", flush=True)
            stop()
            start()
            replay = None if args.direct_continuation else completion(messages)
            restored = completion(continued)
            status = call(proxy_port, "/_kvpark/status")
            checks = acceptance_checks(cold, warm, replay, restored, status["recent"], parked["tokens"], args.min_prompt_tokens)
            checks["snapshot_pair_preserved"] = snapshot_evidence(root / "archive", parked) == snapshot
            evidence = dict(passed=all(checks.values()), checks=checks, model=args.model.name,
                            cold=cold, warm=warm, replay=replay, restored=restored,
                            parked_tokens=parked["tokens"], snapshot=snapshot,
                            configuration=dict(backend_args=backend_options, prompt_repetitions=args.prompt_repetitions,
                                               max_tokens=args.max_tokens, min_prompt_tokens=args.min_prompt_tokens,
                                               direct_continuation=args.direct_continuation),
                            note="CPU functional test; full response times, not TTFT. OS page cache was not flushed.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(evidence, indent=2) + "\n")
            if not evidence["passed"]:
                raise RuntimeError("acceptance failed: " + json.dumps(checks))
            print("PASS: real disk restoration reused most tokens and matched resident output.", flush=True)
        except Exception:
            for name in ("backend.log", "proxy.log"):
                path = root / name
                if path.exists():
                    print(f"{name} (last 4000 characters):\n{path.read_text(errors='replace')[-4000:]}", file=sys.stderr)
            raise
        finally:
            stop()


if __name__ == "__main__":
    main()
