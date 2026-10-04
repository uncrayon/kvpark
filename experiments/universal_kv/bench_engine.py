"""Benchmark the bridge engine on real saved states (linear_transfer run_long run directories).

    python bench_engine.py warm     --hub HUB --direction qwen38:muse --backend torch --runs 4 --output OUT.json
    python bench_engine.py oneshot  --hub HUB --direction qwen38:muse --engine engine.py --runs 2 --output OUT.json
    python bench_engine.py accuracy --hub HUB --direction qwen38:muse --output OUT.json
    python bench_engine.py nll      --hub HUB --docs 6 --probe KV_PROBE --output-dir DIR --output OUT.json
    python bench_engine.py prewarm  --hub HUB --direction qwen38:muse --device gpu --runs 3 --bridge-dir B --output OUT.json
    python bench_engine.py job      --job KEPT/job-X.json --device gpu --runs 4 --output OUT.json [--hugepages off]

Every mode builds a real engine job from a run directory's document: the conversation from the
run's corpus without its final reply (so the target answers it next), the source slot saved
by the run (Qwen3.8 at its prompt length, Muse at its largest saved prefix), a layout
template of the target (a one-token native save) and target tokens: the run's tokenization of
the target rendering, cut where it spells the prompt exactly, with pieces from the GGUF.

  warm      engine.run in this process, repeatedly, with the hub and backend resident (what
            engine_daemon.py does): the first run is cold (side loads, GPU start), the rest warm
  oneshot   `python ENGINE JOB` in a fresh process per run (engine.py, or the snapshot of an
            older engine): the per-switch cost without a daemon
  accuracy  the torch backend against numpy on the same plan: decoded K/V (before RoPE and the
            cache cast) and the stored cache values, per section, as max |err| / max |ref| and
            relative Frobenius error, the worst layer of each
  nll       end to end: per document (the run's T0 for a 2048-token tail, the run's native
            state there as layout template) one translated state from numpy (float32) and one
            from torch (float16), seed-scored on the target with kv_probe (GPU) against its own
            full prefill like run_long's score stage: delta NLL per token of the final reply
  prewarm   the bridge's own path: per run a fresh engine_daemon.py (started by
            kvpark.bridge.engine_client, as a switch starts it), prewarmed for the pair as
            staging does (with and without the warm-up products), then the first and a second
            job through engine_client.run_job: what the first engine job after staging costs.
            --bridge-dir must be short (an AF_UNIX socket path; a relative path works)
  job       a real switch's job, kept by the bridge (KVPARK_BRIDGE_KEEP_JOB: the job, the source
            slot and the layout template side by side; live_bridge.py options --keep-jobs), run
            warm in this process as the daemon runs it (its GPU cache released after each job):
            the assemble stage split into backend.cache_rows (RoPE, cache cast, finite check and
            the device-to-host copies into the output arrays) and the rest, with the kernel's page
            fault, transparent-huge-page and compaction counters of each run (/proc/vmstat): a
            fresh output array is first touched there. --hugepages off allocates numpy arrays
            without MADV_HUGEPAGE (4 KiB faults, never a compaction stall), --heap keep keeps
            freed memory in the process (glibc mallopt) so a later job reuses faulted pages

Reports engine_timings (hub_load, profiles, inputs, prepare, compute, assemble, write), peak
RSS (VmHWM, reset per run) and GPU memory (torch peak allocated), with the 1/5/15-minute load
average next to every timing (another process on the CPU changes CPU timings).
"""

import os
import sys

if __name__ == "__main__":
    _threads = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--threads" and i + 1 < len(sys.argv)),
                    "8")
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_name] = _threads

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "linear_transfer"))

import backends  # noqa: E402
import engine  # noqa: E402
import engine_daemon  # noqa: E402
import hub as hub_module  # noqa: E402
import state_writer  # noqa: E402
import translate  # noqa: E402

import chat  # noqa: E402
import kvstate  # noqa: E402
from live_handoff import RunData  # noqa: E402

REPO = HERE.parents[1]
GGUF = {"qwen38": "/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf",
        "muse": "/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf"}
ROLE = {"qwen38": "source", "muse": "target"}            # roles of the two models in run_long runs
RUN = REPO / "evidence/linear-transfer/run-v3b"
TEMPLATES = REPO / "evidence/universal-kv/live-bridge-3models-v2/bridge/templates"


def load_average():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


class Jobs:
    """Engine jobs for one direction built from a run directory's documents."""

    def __init__(self, args):
        self.args = args
        self.source, self.target = args.direction.split(":")
        self.gguf = {name: args.gguf.get(name, GGUF.get(name)) for name in (self.source, self.target)}
        engine._PROFILE_CACHE[0] = str(args.cache_dir)
        self.spec = {name: engine.spec_of(path) for name, path in self.gguf.items()}
        self.run = RunData(args.run)
        corpus = self.run.plan["corpus"]["path"]
        with open(corpus, encoding="utf-8") as stream:
            self.corpus = {c.get("id"): c for c in map(json.loads, stream)}
        self.template = self._template(self.spec[self.target])
        self.outputs = Path(args.output_dir)
        self.outputs.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _template(spec):
        for path in sorted(TEMPLATES.glob("*.bin")):
            try:
                kvstate.read_file(path, spec)
                return path
            except (ValueError, OSError):
                continue
        raise SystemExit(f"no layout template for {spec.architecture} in {TEMPLATES}")

    def document(self, prefix):
        return next(d for d in self.run.plan["documents"] if d["document_id"].startswith(prefix))

    def source_state(self, doc):
        role = ROLE[self.source]
        if self.source == "qwen38":
            return self.run.state(role, doc["document_id"], doc["qwen_prefix"])
        sizes = sorted({c["t0"] for c in doc["conditions"]}, reverse=True)
        for size in sizes:
            try:
                return self.run.state(role, doc["document_id"], size)
            except FileNotFoundError:
                continue
        raise SystemExit(f"no saved {self.source} state for {doc['document_id'][:12]}")

    def job(self, doc, *, hub, device, tail=2048):
        """An engine job for a document: the conversation up to its final reply."""
        conversation = self.corpus[doc["conversation_id"]]
        conversation = {"messages": conversation["messages"][:-1], "tools": conversation.get("tools")}
        profile = engine.profile(self.gguf[self.target], engine._PROFILE_CACHE[0])
        rendered = chat.render(conversation, profile["chat_template"], bos_token=profile["bos_token"],
                               add_generation_prompt=True, live=True).text.encode("utf-8")
        tokens = self.run.tokens(ROLE[self.target], doc["document_id"])
        pieces = engine.token_bytes(self.gguf[self.target], tokens["token_id"])
        ends = np.cumsum([len(p) for p in pieces])
        lead = len(pieces[0]) if pieces[0] == profile["bos_token"].encode() and not rendered.startswith(pieces[0]) else 0
        n = int(np.searchsorted(ends, len(rendered) + lead, side="right"))
        if b"".join(pieces[:n])[lead:] != rendered:
            raise ValueError(f"{doc['document_id'][:12]}: the run's tokens do not spell the target prompt")
        output = self.outputs / f"bench-{os.getpid()}-{time.monotonic_ns()}.bin"
        return {"hub": str(hub), "tail": tail, "fill": "mean", "device": device, "cache_dir": str(self.args.cache_dir),
                "conversation": conversation,
                "source": {"name": self.source, "gguf": self.gguf[self.source], "state": str(self.source_state(doc))},
                "target": {"name": self.target, "gguf": self.gguf[self.target], "template_state": str(self.template),
                           "output": str(output),
                           "tokens": [[int(i), list(p)] for i, p in zip(tokens["token_id"][:n], pieces[:n])]}}


VMSTAT = ("pgfault", "thp_fault_alloc", "thp_fault_fallback", "compact_stall", "allocstall_movable",
          "allocstall_normal")


def vmstat():
    """The kernel's counters of page faults, THP faults and direct compaction / reclaim (Linux)."""
    out = {}
    try:
        with open("/proc/vmstat", encoding="ascii") as stream:
            for line in stream:
                key, value = line.split()
                if key in VMSTAT:
                    out[key] = int(value)
    except OSError:
        pass
    return out


def kept_job(path, output_dir):
    """A job KVPARK_BRIDGE_KEEP_JOB wrote, its inputs pointed at the copies beside it (the bridge
    deleted the originals); a native prefix that is gone is dropped."""
    path = Path(path)
    job = json.loads(path.read_text(encoding="utf-8"))
    for side, key in (("source", "state"), ("target", "template_state"), ("target", "prefix_state")):
        value = job[side].get(key)
        if not value or Path(value).exists():
            continue
        kept = path.parent / Path(value).name
        if kept.exists():
            job[side][key] = str(kept)
        elif key == "prefix_state":
            job[side].update(prefix_state=None, prefix_tokens=0)
        else:
            raise SystemExit(f"{path}: neither {value} nor {kept} exists")
    Path(output_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    return job


def replay(args):
    """A kept bridge job, run warm repeatedly (see the module docstring)."""
    if args.hugepages == "off":
        np._core.multiarray._set_madvise_hugepage(False)
    if args.heap == "keep":
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        # M_TRIM_THRESHOLD (-1), M_MMAP_THRESHOLD (-3): freed blocks stay in the heap for reuse
        libc.mallopt(-1, ctypes.c_int(2**31 - 1))
        libc.mallopt(-3, ctypes.c_int(32 << 20))
    job = kept_job(args.job, args.output_dir)
    job["cache_dir"] = str(args.cache_dir)
    hub = hub_module.open_hub(args.hub or job["hub"], cache_dir=args.cache_dir)
    backend = backends.select(args.device)
    original = backend.cache_rows
    split = {"cache_rows_s": 0.0, "cache_rows_calls": 0}

    def cache_rows(*a, **k):              # timed (it synchronizes: the finite check reads the device)
        started = time.perf_counter()
        try:
            return original(*a, **k)
        finally:
            split["cache_rows_s"] += time.perf_counter() - started
            split["cache_rows_calls"] += 1
    backend.cache_rows = cache_rows
    out = {"mode": "job", "job": str(args.job), "target": job["target"]["name"], "source": job["source"]["name"],
           "P": len(job["target"]["tokens"]), "hugepages": args.hugepages, "heap": args.heap,
           "backend": backend.describe(), "runs": []}
    for index in range(args.runs):
        run_job = {**job, "device": args.device,
                   "target": {**job["target"], "output": str(Path(args.output_dir) / f"replay-{os.getpid()}-{index}.bin")}}
        split.update(cache_rows_s=0.0, cache_rows_calls=0)
        load, before = load_average(), vmstat()
        report = engine.run(run_job, hub=hub, backend=backend)
        after = vmstat()
        timings = report["engine_timings"]
        item = {"run": index, "load_average": load, "engine_seconds": report["engine_seconds"], **timings,
                "t0": report["t0"], "output_bytes": report["output_bytes"], "backend": report["backend"],
                "cache_rows_s": round(split["cache_rows_s"], 3), "cache_rows_calls": split["cache_rows_calls"],
                "assemble_rest_s": round(timings.get("assemble_s", 0.0) - split["cache_rows_s"], 3),
                "vmstat": {key: after.get(key, 0) - before.get(key, 0) for key in VMSTAT},
                "mem_available_mib": engine_daemon.mem_available() // 2**20 if engine_daemon.mem_available() else None,
                "translate_timings": report["timings"]}
        if report.get("fallback"):
            item["fallback"] = report["fallback"]
        Path(run_job["target"]["output"]).unlink()
        if args.release:
            backend._released()           # as the daemon does after each switch
        out["runs"].append(item)
        print(json.dumps({key: item[key] for key in ("run", "engine_seconds", "compute_s", "assemble_s", "cache_rows_s",
                                                     "write_s", "vmstat")}), flush=True)
    return out


def gpu_memory(backend):
    if backend.name != "torch":
        return {}
    memory = backend.memory()
    return {"gpu_peak_allocated_gib": round(memory.get("peak_allocated", 0) / 2**30, 3),
            "gpu_allocated_gib": round(memory.get("allocated", 0) / 2**30, 3),
            "gpu_reserved_gib": round(memory.get("reserved", 0) / 2**30, 3)}


def warm(args):
    jobs = Jobs(args)
    doc = jobs.document(args.doc)
    hub = hub_module.open_hub(args.hub, cache_dir=args.cache_dir)
    started = time.perf_counter()
    backend = backends.select(args.device)
    out = {"mode": "warm", "direction": args.direction, "document": doc["document_id"][:12], "hub": str(args.hub),
           "hub_format": hub.version, "backend_select_s": round(time.perf_counter() - started, 3),
           "backend": backend.describe(), "runs": []}
    for index in range(args.runs):
        job = jobs.job(doc, hub=args.hub, device=args.device)
        if hasattr(backend, "reset_peak"):
            backend.reset_peak()
        rss = engine_daemon.rss_bytes()
        engine_daemon.reset_peak_rss()
        load = load_average()
        report = engine.run(job, hub=hub, backend=backend)
        item = {"run": index, "load_average": load, "engine_seconds": report["engine_seconds"],
                **report["engine_timings"], "t0": report["t0"], "P": report["P"], "cells": report["cells"],
                "mapped_cells": report["mapped_cells"], "decoded_rows": report["decoded_rows"],
                "output_bytes": report["output_bytes"], "backend": report["backend"],
                "translate_timings": report["timings"],
                "rss_peak_gib": round(engine_daemon.rss_bytes("VmHWM") / 2**30, 3),
                "rss_growth_peak_gib": round((engine_daemon.rss_bytes("VmHWM") - rss) / 2**30, 3),
                **gpu_memory(backend)}
        if report.get("fallback"):
            item["fallback"] = report["fallback"]
        out["runs"].append(item)
        print(json.dumps(item), flush=True)
        Path(job["target"]["output"]).unlink()
    out["resident_bytes"] = backend.resident_bytes()
    return out


def oneshot(args):
    jobs = Jobs(args)
    doc = jobs.document(args.doc)
    out = {"mode": "oneshot", "direction": args.direction, "document": doc["document_id"][:12], "hub": str(args.hub),
           "engine": str(args.engine), "runs": []}
    for index in range(args.runs):
        job = jobs.job(doc, hub=args.hub, device=args.device)
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=jobs.outputs, delete=False) as stream:
            json.dump(job, stream)
        load = load_average()
        started = time.perf_counter()
        result = subprocess.run(["/usr/bin/time", "-f", "%M", sys.executable, str(args.engine), stream.name],
                                capture_output=True, text=True,
                                env={**os.environ, backends.ENV: args.device or "auto"})
        wall = time.perf_counter() - started
        os.unlink(stream.name)
        if result.returncode != 0:
            raise SystemExit(result.stderr[-2000:])
        report = json.loads(result.stdout.strip().splitlines()[-1])
        item = {"run": index, "load_average": load, "process_wall_s": round(wall, 3),
                "engine_seconds": report["engine_seconds"], **report.get("engine_timings", {}),
                "t0": report["t0"], "backend": report.get("backend", "numpy"),
                "max_rss_gib": round(int(result.stderr.strip().splitlines()[-1]) / 2**20, 3)}
        out["runs"].append(item)
        print(json.dumps(item), flush=True)
        Path(job["target"]["output"]).unlink(missing_ok=True)
    return out


def _compare(got, ref, widths):
    offsets = np.cumsum([0] + widths)
    worst = {"max_rel": (0.0, None), "rel_fro": (0.0, None)}
    for layer, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        x, y = got[:, a:b].astype(np.float64), ref[:, a:b].astype(np.float64)
        max_rel = float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-30))
        rel_fro = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))
        if max_rel > worst["max_rel"][0]:
            worst["max_rel"] = (max_rel, layer)
        if rel_fro > worst["rel_fro"][0]:
            worst["rel_fro"] = (rel_fro, layer)
    return {key: float(f"{value:.4g}") for key, (value, _) in worst.items()}


def accuracy(args):
    """torch vs numpy on one plan: decoded values and stored cache values."""
    jobs = Jobs(args)
    doc = jobs.document(args.doc)
    job = jobs.job(doc, hub=args.hub, device="cpu")
    hub = hub_module.open_hub(args.hub, cache_dir=args.cache_dir)
    # the engine's own inputs, then the stages
    numpy_backend, torch_backend = backends.select("cpu"), backends.select("gpu")
    report_numpy = engine.run(job, hub=hub, backend=numpy_backend)
    stored_numpy = kvstate.read_file(job["target"]["output"], jobs.spec[jobs.target])
    job_torch = {**job, "target": {**job["target"], "output": job["target"]["output"] + ".torch"}}
    report_torch = engine.run(job_torch, hub=hub, backend=torch_backend)
    stored_torch = kvstate.read_file(job_torch["target"]["output"], jobs.spec[jobs.target])
    out = {"mode": "accuracy", "direction": args.direction, "document": doc["document_id"][:12],
           "t0": report_numpy["t0"], "stored": {}, "decoded": {}}
    for a, b in zip(stored_torch.state.parts, stored_numpy.state.parts):
        if isinstance(a, kvstate.RecurrentState):
            out["stored"][a.name] = {"bytes_equal": kvstate._write_recurrent(jobs.spec[jobs.target], a)
                                     == kvstate._write_recurrent(jobs.spec[jobs.target], b)}
            continue
        sa, sb = a.stream(), b.stream()
        assert np.array_equal(sa.positions, sb.positions)
        for kind in ("k", "v"):
            x = np.concatenate([(sa.k if kind == "k" else sa.v)[l].reshape(sa.n_cells, -1) for l in a.layers], axis=1)
            y = np.concatenate([(sb.k if kind == "k" else sb.v)[l].reshape(sb.n_cells, -1) for l in a.layers], axis=1)
            widths = [int(np.prod((sa.k if kind == "k" else sa.v)[l].shape[1:])) for l in a.layers]
            identical = float(np.mean(x == y))
            out["stored"][f"{a.name}.{kind}"] = {**_compare(x, y, widths), "f16_identical": round(identical, 4)}
    for path in (job["target"]["output"], job_torch["target"]["output"]):
        Path(path).unlink()
    # decoded values: the same plan through compute on both backends
    source_state = kvstate.read_file(job["source"]["state"], jobs.spec[jobs.source])
    plan = _plan(jobs, job, source_state, hub, report_numpy["t0"])
    reference = translate.compute(plan, hub, numpy_backend)
    got = translate.compute(plan, hub, torch_backend)
    for key, value in reference.items():
        section = next(s for s in plan.sections if s.name == key[0])
        widths = [int(np.prod(jobs.spec[jobs.target].head_shape(l, key[1]))) for l in section.layers]
        out["decoded"][f"{key[0]}.{key[1]}"] = _compare(torch_backend.host(got[key]), value, widths)
    out["reports"] = {"numpy_engine_seconds": report_numpy["engine_seconds"],
                      "torch_engine_seconds": report_torch["engine_seconds"]}
    return out


def _plan(jobs, job, source_state, hub, t0):
    """translate.prepare for an engine job at the engine's T0, with the engine's record and token building."""
    conversation = job["conversation"]
    document_id = chat.document_id(conversation)
    target_gguf, source_gguf = job["target"]["gguf"], job["source"]["gguf"]
    profile = engine.profile(target_gguf, engine._PROFILE_CACHE[0])
    rendered = chat.render(conversation, profile["chat_template"], bos_token=profile["bos_token"],
                           add_generation_prompt=True, live=True)
    ids, pieces = engine.target_pieces(job["target"]["tokens"])
    joined, text = b"".join(pieces), rendered.text.encode()
    shift = len(joined) - len(text)
    record = {"document_id": document_id, "text": joined.decode("utf-8"),
              "segments": [[n, s + shift, e + shift] for n, s, e in rendered.segments],
              "segment_keys": rendered.segment_keys, "split": "test", "final_message_segments": []}
    tokens = engine.token_arrays(target_gguf, ids, pieces)
    source_ids = np.asarray(source_state.tokens)
    source_pieces = engine.token_bytes(source_gguf, source_ids)
    source_record = engine.located_record(conversation, document_id, b"".join(source_pieces))
    source_tokens = engine.token_arrays(source_gguf, source_ids, source_pieces)
    template = kvstate.read_file(job["target"]["template_state"], jobs.spec[jobs.target])
    layers = {kind: translate.source_layers(hub, kind, jobs.source) for kind in translate.KINDS}
    return translate.prepare(source_state, translate.Side(jobs.source, jobs.spec[jobs.source], source_record,
                                                          source_tokens, gguf=source_gguf),
                             translate.Side(jobs.target, jobs.spec[jobs.target], record, tokens, gguf=target_gguf),
                             layers, t0=t0, template=template)


def nll(args):
    """Seed-score delta NLL of float32 (numpy) and float16 (torch) translations on documents of the run."""
    from collect import verify_runtime
    import collect_v2
    jobs = Jobs(args)
    run = jobs.run
    if (jobs.source, jobs.target) != ("qwen38", "muse"):
        raise SystemExit("nll uses the run's saved states: --direction qwen38:muse")
    docs = sorted((d for d in run.plan["documents"] if d["bucket"] == args.bucket and d.get("role") == "primary"),
                  key=lambda d: d["prefix"])
    docs = [docs[i] for i in np.unique(np.round(np.linspace(0, len(docs) - 1, min(args.docs, len(docs))))
                                       .astype(int))]
    hub = hub_module.open_hub(args.hub, cache_dir=args.cache_dir)
    built = Path(args.output_dir) / "nll-states"
    built.mkdir(parents=True, exist_ok=False, mode=0o700)
    backends_by_name = {"numpy": backends.select("cpu"), "torch": backends.select("gpu")}
    inputs, rows = [], []
    for doc in docs:
        did = doc["document_id"]
        t0 = next(c["t0"] for c in doc["conditions"] if c["name"] == "sanity_2048")
        native = run.state("target", did, t0)
        template = kvstate.read_file(native, jobs.spec["muse"])
        source_state = kvstate.read_file(run.state("source", did, doc["qwen_prefix"]), jobs.spec["qwen38"])
        source = translate.Side("qwen38", jobs.spec["qwen38"], run.records("source")[did], run.tokens("source", did),
                                gguf=jobs.gguf["qwen38"])
        target = translate.Side("muse", jobs.spec["muse"], run.records("target")[did], run.tokens("target", did),
                                gguf=jobs.gguf["muse"])
        conditions = [{"name": "native", "state_file": None}, {"name": "sanity", "state_file": str(native)}]
        for name, backend in backends_by_name.items():
            state, report = translate.translate(source_state, source, target, hub, t0=t0, template=template,
                                                backend=backend)
            path = built / f"{did[:16]}-{name}.state"
            state_writer.write(path, state)
            conditions.append({"name": f"hub_{name}", "state_file": str(path)})
        rows.append({"document_id": did, "t0": t0, "P": doc["prefix"], "mapped_cells": report["mapped_cells"],
                     "fill_cells": report["fill_cells"]})
        inputs.append({"document_id": did, "text": run.records("target")[did]["text"],
                       "expected_token_sha256": doc["muse_token_sha256"],
                       "score_ranges": [[doc["final_start_byte"], doc["text_bytes"]]], "conditions": conditions})
        print(json.dumps({"built": did[:12], "t0": t0}), flush=True)
    _, _, artifacts = verify_runtime(args.probe.resolve())
    staging = Path(args.output_dir) / "nll-scores"
    request = Path(args.output_dir) / "nll-input.jsonl"
    request.write_text("".join(json.dumps(item) + "\n" for item in inputs), encoding="utf-8")
    started = time.perf_counter()
    with open(Path(args.output_dir) / "nll-probe.log", "w") as log:
        subprocess.run([str(args.probe), "--mode", "seed-score", "--model", jobs.gguf["muse"], "--input", str(request),
                        "--output", str(staging), "--ctx", "32768", "--batch", "512", "--threads", "16",
                        "--cache-type", "f16", "--flash-attn", "on", "--device", "vulkan", "--max-scored", "256"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=collect_v2.probe_environment(artifacts))
    seconds = time.perf_counter() - started
    names = ("sanity", "hub_numpy", "hub_torch")
    sums = {name: 0.0 for name in names}
    sums["torch_minus_numpy"] = 0.0
    tokens = 0
    for row in rows:
        base = staging / "scores" / row["document_id"]
        scores = {name: np.load(base / f"{name}.npy").astype(np.float64) for name in ("native",) + names}
        native = scores["native"][:, 0]
        row["n_scored"] = int(len(native))
        for name in names:
            row[name] = {"nll_delta": float(np.mean(scores[name][:, 0] - native)), "kl": float(scores[name][:, 1].mean()),
                         "top1": float(scores[name][:, 2].mean())}
            sums[name] += float(np.sum(scores[name][:, 0] - native))
        difference = scores["hub_torch"][:, 0] - scores["hub_numpy"][:, 0]
        row["torch_minus_numpy"] = {"nll_delta": float(difference.mean()), "max_abs": float(np.abs(difference).max())}
        sums["torch_minus_numpy"] += float(difference.sum())
        tokens += len(native)
        print(json.dumps({"document": row["document_id"][:12], **{n: round(row[n]["nll_delta"], 5) for n in names},
                          "torch_minus_numpy": round(row["torch_minus_numpy"]["nll_delta"], 6)}), flush=True)
    for path in built.iterdir():
        path.unlink()
    built.rmdir()
    return {"mode": "nll", "documents": rows, "probe_seconds": round(seconds, 1), "scored_tokens": tokens,
            "token_weighted": {name: value / max(tokens, 1) for name, value in sums.items()}}


def prewarm(args):
    """First and second job after a prewarm, through engine_client and a fresh daemon per run."""
    sys.path.insert(0, str(REPO / "src"))
    from kvpark.bridge import engine_client
    jobs = Jobs(args)
    doc = jobs.document(args.doc)
    bridge = Path(args.bridge_dir)
    bridge.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not engine_client.daemon_supported(bridge):
        raise SystemExit(f"{bridge}: the socket path is too long for the daemon; use a shorter --bridge-dir")
    config = {"engine": {"python": sys.executable, "script": str(HERE / "engine.py"), "device": args.device,
                         "threads": int(args.threads)},
              "hub": str(args.hub), "models": {name: {"hub_name": name, "gguf": jobs.gguf[name]}
                                               for name in (jobs.source, jobs.target)}}
    out = {"mode": "prewarm", "direction": args.direction, "document": doc["document_id"][:12], "hub": str(args.hub),
           "device": args.device, "runs": []}
    for index in range(args.runs):
        for warm_up in (False, True):
            engine_client.stop(bridge)                  # a fresh daemon: nothing resident, no kernels picked
            load = load_average()
            started = time.perf_counter()
            reply = engine_client.prewarm(config, [jobs.source, jobs.target], bridge, warm_up=warm_up) or {}
            item = {"run": index, "warm_up": warm_up, "load_average": load,
                    "prewarm_s": round(time.perf_counter() - started, 3), "load_s": reply.get("load_s"),
                    "warm_up_s": reply.get("warm_up_s"), "backend": reply.get("backend")}
            for which in ("first", "second"):
                job = jobs.job(doc, hub=args.hub, device=args.device)
                job["cache_dir"] = str(bridge / "profiles")          # where switch.py points jobs
                started = time.perf_counter()
                report = engine_client.run_job(config, job, bridge_dir=bridge)
                item[which] = {"client_s": round(time.perf_counter() - started, 3),
                               "engine_seconds": report["engine_seconds"], "engine_mode": report.get("engine_mode"),
                               "backend": report.get("backend"), "engine_timings": report["engine_timings"],
                               "translate_timings": report.get("timings")}
                Path(job["target"]["output"]).unlink(missing_ok=True)
            out["runs"].append(item)
            print(json.dumps(item), flush=True)
    engine_client.stop(bridge)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("warm", "oneshot", "accuracy", "nll", "prewarm", "job"))
    parser.add_argument("--hub", type=Path, help="the hub bundle (job: default the job's)")
    parser.add_argument("--direction", default="qwen38:muse", help="SOURCE:TARGET hub names")
    parser.add_argument("--doc", default="3b211f08c7af", help="document id prefix in the run")
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--gguf", action="append", default=[], help="NAME=PATH (defaults: /srv/models Q8 files)")
    parser.add_argument("--device", default="auto", help="auto | cpu | gpu")
    parser.add_argument("--threads", default="8", help="BLAS threads (read before numpy loads)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--engine", type=Path, default=HERE / "engine.py", help="oneshot: the engine script")
    parser.add_argument("--cache-dir", type=Path, required=True, help="GGUF profile and hub-stamp cache")
    parser.add_argument("--output-dir", type=Path, required=True, help="where translated states are written")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument("--docs", type=int, default=6, help="nll: documents, spread over the bucket's prefixes")
    parser.add_argument("--bucket", default="8192-16384", help="nll: the run's prefix bucket")
    parser.add_argument("--probe", type=Path, default=REPO / "evidence/linear-transfer/kv-probe-build-v1gpu/kvpark-kv-probe",
                        help="nll: the kv_probe build (GPU) for seed-score")
    parser.add_argument("--bridge-dir", default="bench-bridge", help="prewarm: the daemon's bridge directory (short)")
    parser.add_argument("--job", type=Path, help="job: a job-*.json the bridge kept (KVPARK_BRIDGE_KEEP_JOB)")
    parser.add_argument("--hugepages", choices=("on", "off"), default="on",
                        help="job: numpy's MADV_HUGEPAGE on large arrays (numpy's default: on)")
    parser.add_argument("--heap", choices=("default", "keep"), default="default",
                        help="job: keep: freed memory stays in the process (glibc mallopt), reused by later runs")
    parser.add_argument("--no-release", dest="release", action="store_false",
                        help="job: keep the GPU allocator's cached blocks between runs (the daemon releases them)")
    args = parser.parse_args(argv)
    if args.mode != "job" and args.hub is None:
        parser.error(f"{args.mode} needs --hub")
    if args.mode == "job" and args.job is None:
        parser.error("job needs --job")
    args.gguf = dict(item.split("=", 1) for item in args.gguf)
    result = {"warm": warm, "oneshot": oneshot, "accuracy": accuracy, "nll": nll, "prewarm": prewarm,
              "job": replay}[args.mode](args)
    result.update(threads=os.environ.get("OPENBLAS_NUM_THREADS"), load_average_end=load_average(),
                  host=os.uname().nodename if hasattr(os, "uname") else None)
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
