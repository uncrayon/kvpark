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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/acceptance.json"))
    args = parser.parse_args()
    args.server, args.model = args.server.resolve(), args.model.resolve()
    if args.output.exists():
        parser.error("output exists; choose a fresh evidence filename")
    children, logs = [], []
    with tempfile.TemporaryDirectory(prefix="kvpark-acceptance-") as tmp:
        root = Path(tmp)
        proxy_port, backend_port = free_port(), free_port()
        while proxy_port == backend_port:
            backend_port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("KVPARK_", "LLAMA_ARG_"))}
        env.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")

        def call(port, path, body=None):
            with urlopen(Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}), timeout=1200) as response:
                return json.load(response)

        def spawn(argv, name):
            log = (root / name).open("w")
            logs.append(log)
            process = subprocess.Popen([sys.executable, "-m", "kvpark", *argv],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            children.append(process)
            return process

        def ready(process, port, endpoint, log):
            deadline = time.monotonic() + 300
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
                             "--device", "none", "--n-gpu-layers", "0", "--ctx-size", "4096",
                             "--threads", "4", "--threads-batch", "4", "--cache-ram", "0",
                             "--ctx-checkpoints", "8", "--checkpoint-min-step", "128", "--jinja"], "backend.log")
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
                slot_archive_role="foreground", slot_archive_key="acceptance:project:1", max_tokens=32,
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
                        dict(role="user", content="We plan the workshop and track its materials and schedule.\n" * 80 +
                             "Reply with the project code only.")]
            cold = completion(messages)
            if cold["cached_tokens"] != 0:
                raise RuntimeError("cold baseline unexpectedly reused tokens")
            status = call(proxy_port, "/_kvpark/status")
            parked = call(proxy_port, "/_kvpark/park", {"key": status["resident_key"]})
            continued = messages + [cold["message"], dict(role="user", content="What is the project code? Reply with the code only.")]
            warm = completion(continued)
            print("Restarting both processes with RAM prompt cache disabled.", flush=True)
            stop()
            start()
            replay = completion(messages)
            restored = completion(continued)
            status = call(proxy_port, "/_kvpark/status")
            checks = dict(disk_restore=any(e["action"] == "restored" for e in status["recent"]),
                          replay_reused_majority=(replay["cached_tokens"] or 0) > (replay["processed_tokens"] or 0),
                          continuation_reused_majority=(restored["cached_tokens"] or 0) > (restored["processed_tokens"] or 0),
                          replay_matches=bool(cold["message"].get("content")) and replay["message"].get("content") == cold["message"].get("content"),
                          continuation_matches=bool(warm["message"].get("content")) and restored["message"].get("content") == warm["message"].get("content"))
            evidence = dict(passed=all(checks.values()), checks=checks, model=args.model.name,
                            cold=cold, warm=warm, replay=replay, restored=restored,
                            parked_tokens=parked["tokens"],
                            note="CPU functional test; full response times, not TTFT. OS page cache was not flushed.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2) + "\n")
            if not evidence["passed"]:
                raise RuntimeError("acceptance failed: " + json.dumps(checks))
            print("PASS: real disk restoration reused most tokens and matched resident output.", flush=True)
        finally:
            stop()


if __name__ == "__main__":
    main()
