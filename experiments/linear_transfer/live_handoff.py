"""Live Qwen -> Muse handoff check on throwaway llama-server instances.

Measures what the offline evaluation cannot: whether a real llama-server restores a
built Muse slot file and reuses it, how long each step takes, and how DFlash drafting
behaves afterwards. Deployed services and their files are never touched. The script
starts its own Muse server (the production argv from start-muse-server.sh's dry run,
with another binary, port, context size and a private --slot-save-path) and, with
--qwen-server, its own Qwen server (the production argv recorded in kvpark's
runtime.json, without the vision projector, with another port, context size and slot
directory). Both stop when the script ends.

Documents come from run_long.py run directories (plan.json, rendered records, token
captures, saved states). Each document's prompt is its Muse token ids [0, P), sent to
/completion as a token list (P = first token of the final assistant message), greedy,
256 tokens. Per document, in order:

  native    erase slot 0, full prefill: the reference output, TTFT and draft stats;
            then save slot 0 (the stock server file)
  stock     erase, restore the stock file, same request. The stock file holds n_swa
            sliding cells, so the server is expected to re-process the whole prompt
            (cache_n 0); the output is a second native run (determinism baseline).
  oracle    the run's native Muse state at T0 rebuilt by handoff.py with n_swa + 1
            sliding cells (oracle content and fill): mechanics only, expect
            cache_n == T0 and the native output
  handoff   design A: mapped content (mapper), table fill, n_swa + 1 sliding cells,
            built from the live Qwen server's slot file (--qwen-server) or from the
            run's saved Qwen state; expect cache_n == T0

Every built file is unlinked as soon as its restore returns. Outputs (counts, token
ids of generated continuations, timings, memory; no text) go to --output.
"""

import os

if __name__ == "__main__":
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "8")

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attention_eval  # noqa: E402
from captures import Capture  # noqa: E402
import handoff  # noqa: E402
import kvstate  # noqa: E402

LONG = "long"
MUSE_START = Path(os.environ.get("KVPARK_MUSE_START", "start-muse-server.sh"))      # --muse-start
QWEN_RUNTIME = Path(os.environ.get("KVPARK_QWEN_RUNTIME", "runtime.json"))            # --qwen-runtime
STEPS = ("native", "stock", "oracle", "handoff")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def mem_available_mib():
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- servers ------------------------------------------------------------------------------------

def replace_flag(argv, flag, value):
    out, i, found = [], 0, False
    while i < len(argv):
        if argv[i] == flag:
            out += [flag, value]
            i += 2
            found = True
        else:
            out.append(argv[i])
            i += 1
    if not found:
        out += [flag, value]
    return out


def drop_flags(argv, with_value=(), bare=()):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in with_value:
            i += 2
        elif argv[i] in bare:
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def muse_argv(binary, port, ctx, slot_dir, start_script=None):
    env = dict(os.environ, MUSE_DRY_RUN="1")
    lines = subprocess.run(["bash", str(start_script or MUSE_START)], env=env, check=True, capture_output=True,
                           text=True).stdout.splitlines()
    argv = [str(binary)] + lines[1:]
    argv = replace_flag(argv, "--port", str(port))
    argv = replace_flag(argv, "--ctx-size", str(ctx))
    argv = replace_flag(argv, "--alias", "muse-live-handoff")
    argv = drop_flags(argv, with_value=("--slot-save-path",))
    return argv + ["--slot-save-path", str(slot_dir), "--slots"]


def qwen_argv(port, ctx, slot_dir, runtime_json=None):
    argv = list(json.loads(Path(runtime_json or QWEN_RUNTIME).read_text(encoding="utf-8"))["argv"])
    argv = drop_flags(argv, with_value=("--mmproj", "--mmproj-device", "--log-verbosity", "--slot-save-path"),
                      bare=("--mmproj-offload",))
    argv = replace_flag(argv, "--port", str(port))
    argv = replace_flag(argv, "--ctx-size", str(ctx))
    argv = replace_flag(argv, "--alias", "qwen-live-handoff")
    return argv + ["--slot-save-path", str(slot_dir)]


class Server:
    def __init__(self, name, argv, port, log_path):
        self.name, self.argv, self.port, self.log_path = name, argv, port, log_path
        self.process = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=900):
        try:
            urllib.request.urlopen(self.url + "/health", timeout=2)
        except (urllib.error.URLError, OSError):
            pass
        else:
            raise RuntimeError(f"{self.name}: port {self.port} is already serving")
        self.log = open(self.log_path, "ab")
        self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited with {self.process.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as reply:
                    if reply.status == 200:
                        return time.monotonic() - started
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(f"{self.name} did not become healthy in {timeout} s")

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)
        self.log.close()

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def post(self, path, body, timeout=1800):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                payload = json.loads(reply.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{self.name} {path}: HTTP {error.code} {error.read()[:300]!r}") from None
        return payload, time.perf_counter() - started

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as reply:
            return json.loads(reply.read())

    def slot(self, action, filename=None):
        body = {} if filename is None else {"filename": filename}
        return self.post(f"/slots/0?action={action}", body)

    def complete(self, tokens, n_predict):
        """Streamed greedy /completion of a token-id prompt: (tokens, timings, ttft_s, wall_s)."""
        body = {"prompt": [int(t) for t in tokens], "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                "cache_prompt": True, "id_slot": 0, "stream": True, "return_tokens": True, "n_probs": 0}
        request = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        first, generated, final = None, [], None
        with urllib.request.urlopen(request, timeout=3600) as reply:
            for raw in reply:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("tokens"):
                    if first is None:
                        first = time.perf_counter() - started
                    generated += [int(t) for t in chunk["tokens"]]
                if chunk.get("stop"):
                    final = chunk
        if final is None:
            raise RuntimeError(f"{self.name}: completion stream ended without a final chunk")
        return generated, final.get("timings", {}), first, time.perf_counter() - started, final.get("stop_type")


# -- documents ----------------------------------------------------------------------------------

class RunData:
    """Documents, records, tokens and saved states of one run_long.py run directory."""

    def __init__(self, path):
        self.path = Path(path)
        self.plan = json.loads((self.path / LONG / "plan.json").read_text(encoding="utf-8"))
        self._records, self._tokens = {}, {}

    def records(self, role):
        if role not in self._records:
            with open(self.path / LONG / f"rendered-{role}.jsonl", encoding="utf-8") as stream:
                self._records[role] = {r["document_id"]: r for r in map(json.loads, stream)}
        return self._records[role]

    def tokens(self, role, document_id):
        if role not in self._tokens:
            self._tokens[role] = Capture(self.path / LONG / "tokens" / role)
        capture = self._tokens[role]
        rows = capture.rows(document_id)
        return {name: np.array(getattr(capture, name)[rows]) for name in ("token_id", "byte_end", "special")}

    def state(self, role, document_id, n_tokens):
        found = sorted((self.path / LONG / "states" / role).glob(f"group-*/states/{document_id}/{n_tokens}.state"))
        if len(found) != 1:
            raise FileNotFoundError(f"{role} state {document_id[:12]}@{n_tokens}: {len(found)} matches")
        return found[0]


def select_documents(selections, per_bucket, tail, min_t0):
    """[(run, plan document, T0)]: per (run, bucket), up to per_bucket documents spread over the prefix range."""
    chosen = []
    for run, bucket in selections:
        docs = [d for d in run.plan["documents"] if d["bucket"] == bucket and d.get("role") == "primary"]
        usable = []
        for doc in docs:
            t0 = next((c["t0"] for c in doc["conditions"] if c["name"] == f"sanity_{tail}"), None)
            if t0 is not None and t0 >= min_t0:
                usable.append((doc, t0))
        if not usable:
            raise SystemExit(f"{run.path}: no usable document in bucket {bucket}")
        usable.sort(key=lambda item: item[0]["prefix"])
        picks = np.unique(np.round(np.linspace(0, len(usable) - 1, min(per_bucket, len(usable)))).astype(int))
        chosen += [(run, usable[i][0], usable[i][1]) for i in picks]
    return chosen


def agreement(reference, other):
    n = min(len(reference), len(other))
    same = np.asarray(reference[:n]) == np.asarray(other[:n])
    first = int(np.argmin(same)) if n and not same.all() else n
    return {"compared": n, "first_divergence": first, "positionwise": float(same.mean()) if n else None,
            "identical": len(reference) == len(other) and bool(same.all())}


def timing_view(timings):
    keys = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second",
            "draft_n", "draft_n_accepted")
    view = {k: timings.get(k) for k in keys}
    if timings.get("draft_n"):
        view["draft_acceptance"] = timings["draft_n_accepted"] / timings["draft_n"]
    return view


# -- main ---------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="append", required=True, metavar="RUN_DIR:BUCKET",
                        help="run_long.py run directory and plan bucket label, e.g. run-v3:4096-8192 (repeatable)")
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--tail", type=int, default=2048, help="use the plan's sanity_<tail> T0")
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--table", type=Path, help="token table NPZ (default: the run's long/token-table.npz)")
    parser.add_argument("--muse-server", type=Path, required=True, help="upstream llama-server binary for Muse")
    parser.add_argument("--muse-port", type=int, default=18083)
    parser.add_argument("--qwen-server", action="store_true",
                        help="also run the production Qwen binary and hand off from its live slot file")
    parser.add_argument("--qwen-port", type=int, default=18090)
    parser.add_argument("--ctx", type=int, default=36864)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--min-available-mib", type=int, default=40960,
                        help="refuse to start a server below this MemAvailable (the Muse unit's guard)")
    parser.add_argument("--qwen-model", type=Path, default=Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"))
    parser.add_argument("--muse-model", type=Path,
                        default=Path("/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf"))
    parser.add_argument("--muse-start", type=Path, default=MUSE_START,
                        help="the Muse service's start script (its MUSE_DRY_RUN=1 argv is reused)")
    parser.add_argument("--qwen-runtime", type=Path, default=QWEN_RUNTIME,
                        help="kvpark's runtime.json of the Qwen server (its argv is reused)")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    slots = args.output / "slots"
    muse_slots, qwen_slots = slots / "muse", slots / "qwen"
    for path in (slots, muse_slots, qwen_slots):
        path.mkdir(mode=0o700)
    results_path = args.output / "results.jsonl"
    runs, selections = {}, []
    for item in args.select:
        path, _, bucket = item.rpartition(":")
        runs.setdefault(path, RunData(path))
        selections.append((runs[path], bucket))
    muse_spec, qwen_spec = kvstate.ModelSpec.from_gguf(args.muse_model), kvstate.ModelSpec.from_gguf(args.qwen_model)
    documents = select_documents(selections, args.per_bucket, args.tail, min_t0=muse_spec.n_swa + 512)
    if not documents:
        raise SystemExit("no document matches")
    swa_cells = muse_spec.n_swa + 1
    need = max(d["prefix"] + args.n_predict + 16 for _, d, _ in documents)
    if need > args.ctx:
        raise SystemExit(f"--ctx {args.ctx} is smaller than the largest prompt + n_predict ({need})")

    servers = []
    muse = Server("muse", muse_argv(args.muse_server, args.muse_port, args.ctx, muse_slots, args.muse_start),
                  args.muse_port,
                  args.output / "muse-server.log")
    qwen = (Server("qwen", qwen_argv(args.qwen_port, args.ctx, qwen_slots, args.qwen_runtime), args.qwen_port,
                   args.output / "qwen-server.log") if args.qwen_server else None)
    meta = {"started": now_iso(), "documents": [(str(r.path), d["document_id"]) for r, d, _ in documents],
            "muse_argv": muse.argv, "qwen_argv": qwen.argv if qwen else None, "swa_cells": swa_cells,
            "mapper": str(args.mapper.resolve()), "n_predict": args.n_predict, "ctx": args.ctx,
            "code_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                            for name in ("live_handoff.py", "handoff.py", "kvstate.py")}}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    memory = {"before": mem_available_mib()}

    def log(event, **fields):
        print(json.dumps({"time": now_iso(), "event": event, **fields}), flush=True)

    try:
        for server in [s for s in (qwen, muse) if s]:
            available = mem_available_mib()
            if available < args.min_available_mib:
                raise SystemExit(f"MemAvailable {available} MiB < {args.min_available_mib} MiB; not starting {server.name}")
            load_s = server.start()
            memory[f"after_{server.name}_load"] = mem_available_mib()
            servers.append(server)
            log("server_ready", server=server.name, load_s=round(load_s, 1), mem_available_mib=memory[f"after_{server.name}_load"])
        props = muse.get("/props")
        meta["muse_build_info"] = props.get("build_info")
        if qwen is not None:
            meta["qwen_build_info"] = qwen.get("/props").get("build_info")
        mapper = attention_eval.Mapper(args.mapper)
        tables = {}
        for number, (run, doc, t0) in enumerate(documents, 1):
            document_id = doc["document_id"]
            P = doc["prefix"]
            try:
                result = {"document_id": document_id, "run": str(run.path), "bucket": doc["bucket"],
                          "source": doc["source"], "prefix": P, "t0": t0, "qwen_prefix": doc["qwen_prefix"],
                          "steps": {}, "min_mem_available_mib": mem_available_mib()}
                m_tokens = run.tokens("target", document_id)
                q_tokens = run.tokens("source", document_id)
                prompt = m_tokens["token_id"][:P]
                log("document", number=f"{number}/{len(documents)}", document=document_id[:12], bucket=doc["bucket"],
                    prefix=P, t0=t0)

                def run_step(name, restore_file=None):
                    step = {}
                    _, erase_s = muse.slot("erase")
                    step["erase_s"] = round(erase_s, 3)
                    if restore_file is not None:
                        try:
                            reply, restore_s = muse.slot("restore", restore_file.name)
                            step.update(n_restored=reply.get("n_restored"), restore_ms=reply.get("timings", {}).get("restore_ms"),
                                        restore_wall_s=round(restore_s, 3))
                        finally:
                            restore_file.unlink(missing_ok=True)
                    generated, timings, ttft, wall, stop_type = muse.complete(prompt, args.n_predict)
                    step.update(timing_view(timings), ttft_s=round(ttft or 0.0, 3), wall_s=round(wall, 3),
                                n_generated=len(generated), stop_type=stop_type,
                                generated_sha256=hashlib.sha256(np.asarray(generated, "<i4").tobytes()).hexdigest())
                    step["_tokens"] = generated
                    result["min_mem_available_mib"] = min(result["min_mem_available_mib"], mem_available_mib())
                    if not muse.alive():
                        raise RuntimeError(f"muse server died during {name}")
                    log("step", document=document_id[:12], step=name, cache_n=step.get("cache_n"),
                        prompt_n=step.get("prompt_n"), ttft_s=step["ttft_s"], n_restored=step.get("n_restored"),
                        draft_acceptance=step.get("draft_acceptance"))
                    return step

                # native and stock
                result["steps"]["native"] = run_step("native")
                stock_name = f"stock-{document_id[:16]}.bin"
                reply, save_s = muse.slot("save", stock_name)
                result["stock_save"] = {"n_saved": reply.get("n_saved"), "n_written": reply.get("n_written"),
                                        "save_ms": reply.get("timings", {}).get("save_ms"), "wall_s": round(save_s, 3)}
                result["steps"]["stock"] = run_step("stock", muse_slots / stock_name)

                # oracle: the run's native Muse state at T0 with n_swa + 1 sliding cells
                native_path = run.state("target", document_id, t0)
                table_path = args.table or run.path / LONG / "token-table.npz"
                if table_path not in tables:
                    tables[table_path] = handoff.TokenTable(table_path)
                started = time.perf_counter()
                oracle_state = kvstate.read_file(native_path, muse_spec)
                built = handoff.build_handoff(None, run.records("source")[document_id], q_tokens,
                                              run.records("target")[document_id], m_tokens, None, t0=t0,
                                              muse_spec=muse_spec, swa_cells=swa_cells, content_policy="oracle",
                                              fill_policy="oracle", oracle_state=oracle_state, table=tables[table_path],
                                              server_tokens=True)
                oracle_file = muse_slots / f"oracle-{document_id[:16]}.bin"
                kvstate.write_new(oracle_file, built.state.to_bytes())
                result["oracle_build"] = {"seconds": round(time.perf_counter() - started, 3),
                                          "bytes": oracle_file.stat().st_size, "cells": built.report["sections"]}
                del oracle_state, built
                result["steps"]["oracle"] = run_step("oracle", oracle_file)

                # handoff: Qwen slot file (live or saved) -> mapped Muse state
                qwen_info = {}
                if qwen is not None:
                    qwen.slot("erase")
                    q_prompt = q_tokens["token_id"][:doc["qwen_prefix"]]
                    _, q_timings, q_ttft, q_wall, _ = qwen.complete(q_prompt, 1)
                    q_name = f"qwen-{document_id[:16]}.bin"
                    reply, q_save_s = qwen.slot("save", q_name)
                    q_path = qwen_slots / q_name
                    qwen_info = {"source": "live", "prefill_ms": q_timings.get("prompt_ms"), "prompt_n": q_timings.get("prompt_n"),
                                 "save_ms": reply.get("timings", {}).get("save_ms"), "save_wall_s": round(q_save_s, 3),
                                 "n_saved": reply.get("n_saved"), "bytes": q_path.stat().st_size}
                else:
                    q_path = run.state("source", document_id, doc["qwen_prefix"])
                    qwen_info = {"source": "saved", "bytes": q_path.stat().st_size}
                started = time.perf_counter()
                qwen_state = kvstate.read_file(q_path, qwen_spec)
                read_s = time.perf_counter() - started
                cells = len(qwen_state.state.part("attn").stream().positions)
                if len(qwen_state.tokens) == cells + 1:
                    qwen_state = kvstate.StateFile(kvstate.token_words(qwen_state.tokens[:cells],
                                                                       server_packed=qwen_state.server_packed),
                                                   qwen_state.state)
                    qwen_info["dropped_sampled_token"] = True
                qwen_info.update(tokens=len(qwen_state.tokens), cells=cells, read_s=round(read_s, 3))
                started = time.perf_counter()
                built = handoff.build_handoff(qwen_state, run.records("source")[document_id], q_tokens,
                                              run.records("target")[document_id], m_tokens, mapper, t0=t0,
                                              muse_spec=muse_spec, swa_cells=swa_cells, content_policy="mapped",
                                              fill_policy="table", table=tables[table_path], server_tokens=True)
                build_s = time.perf_counter() - started
                handoff_file = muse_slots / f"handoff-{document_id[:16]}.bin"
                data = built.state.to_bytes()
                kvstate.write_new(handoff_file, data)
                write_s = time.perf_counter() - started - build_s
                result["handoff_build"] = {"seconds": round(build_s, 3), "write_s": round(write_s, 3), "bytes": len(data),
                                           "sections": built.report["sections"], "qwen": qwen_info}
                del qwen_state, built, data
                if qwen is not None:
                    q_path.unlink(missing_ok=True)
                    q_path.with_name(q_path.name + ".resume").unlink(missing_ok=True)
                result["steps"]["handoff"] = run_step("handoff", handoff_file)

                native_tokens = result["steps"]["native"]["_tokens"]
                for name in STEPS[1:]:
                    result["steps"][name]["agreement_vs_native"] = agreement(native_tokens, result["steps"][name]["_tokens"])
                result["generated"] = {name: step.pop("_tokens") for name, step in result["steps"].items()}
                result["muse_alive"] = muse.alive()
                with open(results_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result) + "\n")
                log("document_done", document=document_id[:12],
                    agreement={n: result["steps"][n]["agreement_vs_native"]["first_divergence"] for n in STEPS[1:]})
            except Exception as error:  # one document's failure must not end the run
                for path in sorted(slots.rglob("*")):
                    if path.is_file():
                        path.unlink()
                with open(results_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"document_id": document_id, "error": f"{type(error).__name__}: {error}",
                                             "muse_alive": muse.alive(),
                                             "qwen_alive": qwen.alive() if qwen else None}) + "\n")
                log("document_error", document=document_id[:12], error=f"{type(error).__name__}: {error}"[:300])
                if not muse.alive() or (qwen is not None and not qwen.alive()):
                    raise
        memory["after"] = mem_available_mib()
    finally:
        for server in reversed(servers):
            server.stop()
        for path in sorted(slots.rglob("*")):
            if path.is_file():
                path.unlink()
        meta.update(finished=now_iso(), memory=memory)
        (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
