"""Fit a universal KV hub on captured models, evaluate every ordered pair, optionally compare a direct mapper.

  fit_hub.py --model qwen38=CAP1,CAP2@rendered.jsonl --model muse=CAP3,CAP4@rendered.jsonl \
      --output DIR [--hub-dim 1024] [--view-dim 2048] [--layer-rank 512] [--fit-rows 60000] \
      [--direct qwen38:muse=MAPPER_DIR] [--indexer-ridge NAME] [--heads NAME] [--fit-device auto|cpu|gpu]

Models of any cache family mix: the captures hold each attention layer's pre-RoPE K and V (an
MLA model's K is the [c_kv | k_pe] row, its V the c_kv view). --indexer-ridge NAME fits a DSA
model's indexer ridge (hub.fit_indexer_ridge: its captured indexer keys, kv_probe key "ik", from
its K rows) into the bundle; with --base the base bundle's ridges are kept. --heads NAME fits a
DeepSeek V4 model's head decoders (hub.fit_head: its captured compressor inputs, keys csa, hca and
lid, out of the K latent of the other models) into the bundle; with --base the base bundle's
heads are kept.

Onboarding (--base) leaves the base models' adapters untouched: --link-base hard-links their files
from the base bundle (a format 1 bundle) instead of writing them again, so the new bundle costs
only the new model's files, and --evaluate new measures only the pairs that involve the new model
(the base pairs are in the base bundle's evaluation.json). --onboard-encoder picks how the new
model's encoder is fitted (hub.solve_onboard_encoder): reverse (the default since 2026-10-06,
hub.DEFAULT_ONBOARD_ENCODER: the ridge regression onto the mean of the pivots' estimates, rescaled per
dimension so that the encoding is unbiased given that mean instead of shrunk toward it; RESULTS.md
section 9), ridge (onboarding as it was until then: its encodings spread half as far as a jointly fitted
model's and a source recovered 3-11% of the floor's KL), consensus (the ridge regression onto the
consensus the hub latent was fitted as; every model of the base's GCCA must be a pivot), match, tls,
cca or reverse_z. The manifest's settings record it and the models the latent was fitted on
("consensus_views"). --onboard-variance-cap X caps the per-dimension variance of the reverse family's
encodings at X times the pooled pivot estimates' (hub.ONBOARD_VARIANCE_CAP, 4; 0 for none): their scale
is about 1 / R^2 per dimension, unbounded for a dimension the new view barely predicts. The encoders' train
diagnostics (the smallest per-dimension R^2, the largest variance ratio, the dimensions capped) are logged
as stage "onboard-encoder", with a warning (stage "onboard-warning", and on stderr) when the cap binds or a
dimension's R^2 is below hub.ONBOARD_WARN_R2.

--reencode refits only the encoder of a model already onboarded: --base is then that model's bundle
(an onboarding's, e.g. campaign/NAME/hub), --model names the model (first) and the pivots, and
--onboard-encoder the method (hub.reencode_model_kind: one pass over the train spans; the model's
view transforms and decoders, and every other model's adapters, stay exactly as they are, so only
translations out of it change). --link-base then links the files of the base the bundle grew from
(its settings' base), which the new bundle records as its base, so it assembles like an onboarding. The
settings the model's transforms and decoders were fitted with (view_dim, layer_rank, fit_rows,
source_global, indexer_ridge, heads) are the bundle's own, recorded as they were, whatever the command
line says.

Fit placement (--fit-device, fit_backend.py): the fit's projections and its Gram and cross-product
accumulations run on the GPU (gpu: torch with ROCm; float16 products with float32 accumulation summed
in float64, every launch tiled to a few hundredths of a second) or the CPU (cpu: numpy float32 products,
as before); eigh, solve and svd stay LAPACK on the CPU (float64, except float32 for the top eigenvectors
of a large matrix, fit_backend.top_eig, $KVPARK_FIT_LARGE_EIGH). auto (the default) takes the GPU when
torch sees one, so run fits with ~/venvs/rocm/bin/python (the experiment venv has no torch: auto is the
CPU there). README.md "Fit placement" has the measurements. The manifest's settings record the device
(fit_device) and the precision of the large subset eigensolves (large_eigh).

Resource guard (resource_guard.py): BLAS threads are capped at 8 unless set; a GPU fit holds the GPU
lock first (/tmp/kvpark-campaign-gpu.lock, campaign.py's; a fit campaign.py exec runs inherits it) and
the guard's GPU-hang stop reads the kernel log; every fit holds the CPU lock
(/tmp/kvpark-campaign-cpu.lock: one CPU-heavy fit at a time; a fit campaign.py runs inherits
it); it is refused when its estimated peak of anonymous memory (memory_estimate, logged as stage
"memory"; --estimate prints it and exits) exceeds MemAvailable minus 25 GiB (--guard
fit_margin_gib=N); it waits until the CPU is at most 75 C, and every fit loop calls the guard between
chunks (hub.guard_check: sleep at 90 C until 80 C, stop at 95 C twice in a row, stop at a GPU hang), with
a sample every 2 s in OUTPUT.guard.jsonl. A single eigendecomposition of a joint fit runs for minutes
without a check: run a joint refit under `resource_guard.py run --cpu-lock --` (or campaign.py exec
--cpu-lock) so that it can be paused there too.
"""

import os

import resource_guard

if __name__ == "__main__":
    resource_guard.cap_threads(os.environ)

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import fit_backend
import hub

GIB = 1024 ** 3
# -- the memory a fit holds -------------------------------------------------------------------------
# Anonymous memory at the fit's peak (its RSS is mostly the memory-mapped captures, file pages the
# kernel can drop). An onboarding (--base) follows campaign.fit_hub_extra, calibrated on the recorded
# peaks with the 4.6 GiB base hub (LFM2-2.6B 11.0 GiB, DeepSeek-V2-Lite 16.7, Qwen3.6-35B-A3B 13.0):
# ONBOARD_BASE and the base hub, plus per kind (one at a time) 24 bytes per cell of the new model's
# stage-1 covariance (W1 x W1, W1 its per-layer PCA ranks summed), 8 per hub dimension x column of its
# decoder's float64 cross moments, and per dev span and pivot 12 per hub dimension and 4 per column
# (the alpha choice's tiled dev estimates and rows).
# A joint fit holds, per kind, at most: the GCCA's eigendecomposition, the joint covariance C and the
# whitened W C W (P x P float64, P the views' dimensions summed: 28,672 for seven views of 4,096: EIGH_HELD of
# them) and fit_backend.top_eig of it (eigh_bytes: what that path allocates beyond its input); or later the
# decoders: C still held, the projection (P x D), every target's float64 cross moments (D x its columns) and
# Gram (D x D), the dev spans' rows and hub estimates of every model and the alpha choice's tiled copies for
# the widest model (M - 1 copies of its dev rows and of their estimates, float32 and float64, and three
# float64 D x columns solves); or first a model's stage-2 eigendecomposition (its Moments' sums and
# covariance, EIGH_HELD of W1 x W1, and top_eig of it). Each kind's transforms and, while the next kind is
# fitted, the fitted kind's adapters (float32) stay held. JOINT_BASE is the onboarding's ONBOARD_BASE: no
# joint fit has a recorded peak yet; its guard samples (OUTPUT.guard.jsonl, rss_gib and anon_gib) will
# calibrate it.
ONBOARD_BASE = int(4.5 * GIB)
ONBOARD_COVARIANCE_BYTES = 24
ONBOARD_DECODER_BYTES = 8
ONBOARD_DEV_LATENT_BYTES = 12
ONBOARD_DEV_COLUMN_BYTES = 4
# fit_backend.top_eig's allocations beyond its float64 n x n input, in n x n float64 matrices (8 n^2 bytes), by
# the path it takes, the eigenvectors' 12 n k bytes (float32 and float64) on top. Measured in a fresh process
# (VmHWM over the RSS before, scratchpad trackA/topeig_mem.py, 2026-10-07): numpy's eigh of the float64 matrix
# (its input copy, dsyevd's 2 n^2 workspace, every eigenvector) 4.03-4.06 with OpenBLAS 0.3.34; a large subset
# problem (n 4,096-8,192, k n / 7) with scipy 1.18.1 (the ROCm venv the fits run in) 1.16-1.41 in float32 (the
# float32 copy, ssyevr's own copy of it, its work arrays) and 1.24-1.50 in float64, of which 0.21 are the
# eigenvectors; in "float32" without scipy (the experiment venv) 5.49-5.66: numpy.linalg computes a float32 matrix
# in float64, so the float32 copy and its float64 promotion come on top of numpy's eigh (and it took 50 s at 8,192
# against 25 s with scipy). Rounded up.
EIGH_HELD = 2                    # C and W C W (the GCCA); a stage 2's Moments sums and covariance
EIGH_FULL_COPIES = 4.1
EIGH_SUBSET32_COPIES = 1.3
EIGH_SUBSET64_COPIES = 1.4
EIGH_FULL32_COPIES = 5.7
EIGH_COPIES = EIGH_HELD + EIGH_FULL_COPIES       # the numpy path with what is held (6.1: estimates until 2026-10-07)
JOINT_BASE = ONBOARD_BASE


def eigh_bytes(n, k, *, scipy=None, precision=None):
    """What fit_backend.top_eig(cov [n, n] float64, k) allocates beyond its input, by the path it takes there:
    a large subset problem (n >= SUBSET_MIN, k <= n / 2) in fit_backend.LARGE_EIGH precision with scipy's subset
    solver when importable (`scipy`: whether it is, default this process's), else numpy's eigh."""
    import fit_backend
    scipy = fit_backend._scipy_linalg() is not None if scipy is None else scipy
    precision = precision or fit_backend.LARGE_EIGH
    k = min(k, n)
    if not (n >= fit_backend.SUBSET_MIN and 0 < k <= n // 2):
        copies = EIGH_FULL_COPIES
    elif precision == "float32":
        copies = EIGH_SUBSET32_COPIES if scipy else EIGH_FULL32_COPIES
    else:
        copies = EIGH_SUBSET64_COPIES if scipy else EIGH_FULL_COPIES
    return int(copies * 8 * n * n + 12 * n * k)


def parse_model(text):
    name, _, rest = text.partition("=")
    captures, _, records = rest.rpartition("@")
    if not name or not captures or not records:
        raise argparse.ArgumentTypeError("--model NAME=CAPTURE[,CAPTURE...]@RECORDS.jsonl")
    return name, [Path(p) for p in captures.split(",")], Path(records)


def direct_r2(mapper_dir, source, target, spans, kind, split="test", chunk=4096):
    """Variance-weighted R^2 of a fit_layers mapper on the hub's test spans (same rows as the hub)."""
    sys.path.insert(0, str(hub.HERE.parent / "linear_transfer"))
    import attention_eval
    mapper = attention_eval.Mapper(mapper_dir)
    selected = spans.select(split)
    layers = target.layers(kind)
    sse, y_sum, y_sq, n = np.zeros(len(layers)), [None] * len(layers), np.zeros(len(layers)), 0
    for part in hub.chunks(len(selected), chunk):
        rows = selected[part]
        source_rows = {layer: source.captures.gather(kind, layer, spans.rows[source.name][rows])
                       for layer in source.layers(kind)}
        for index, layer in enumerate(layers):
            _, wanted = mapper.source_layers(kind, layer)
            prediction = mapper.predict(kind, layer, {l: source_rows[l] for l in wanted}, target.shape(kind, layer))
            y = target.rows(kind, layer, spans.rows[target.name][rows])
            sse[index] += float(np.sum(np.square(y - prediction.reshape(len(rows), -1)), dtype=np.float64))
            y_sum[index] = y.sum(axis=0, dtype=np.float64) + (0 if y_sum[index] is None else y_sum[index])
            y_sq[index] += float(np.sum(np.square(y.astype(np.float64))))
        n += len(rows)
        mapper.clear()
    total = np.asarray([y_sq[i] - np.sum(np.square(y_sum[i])) / n for i in range(len(layers))])
    return {"r2": float(1 - sse.sum() / total.sum()), "rows": int(n),
            "per_layer": [float(1 - s / t) for s, t in zip(sse, total)]}


def view_shape(widths, layer_rank, view_dim):
    """What a fit holds of one model and kind, from its per-layer widths: stage1 (W1, the per-layer PCA
    ranks summed), view (its view dimension), columns (the widths summed) and bases (the stage-1 bases'
    cells)."""
    ranks = [min(layer_rank, width) for width in widths]
    stage1 = sum(ranks)
    return {"stage1": stage1, "view": min(view_dim, stage1), "columns": sum(widths),
            "bases": sum(width * rank for width, rank in zip(widths, ranks))}


def onboard_bytes(stage1, columns, *, dev_spans, pivots, hub_dim):
    """An onboarding's anonymous memory beyond ONBOARD_BASE and the base hub, for one kind."""
    return (ONBOARD_COVARIANCE_BYTES * stage1 ** 2 + ONBOARD_DECODER_BYTES * hub_dim * columns
            + dev_spans * pivots * (ONBOARD_DEV_LATENT_BYTES * hub_dim + ONBOARD_DEV_COLUMN_BYTES * columns))


def joint_bytes(shapes, *, hub_dim, dev_spans, scipy=None):
    """A joint fit's anonymous memory beyond JOINT_BASE for one kind: {"gcca", "decoder", "stage2",
    "transforms", "adapters"} (bytes; the peak is the transforms plus the largest of the first three,
    the adapters stay held while the next kind is fitted). shapes: view_shape per model; scipy: whether
    top_eig finds scipy (eigh_bytes; default this process's)."""
    views = [s["view"] for s in shapes]
    P, D, M = sum(views), min(hub_dim, sum(views)), len(shapes)
    columns = [s["columns"] for s in shapes]
    widest = max(columns)
    transforms = 4 * sum(s["stage1"] * s["view"] + s["bases"] for s in shapes)
    decoder = (8 * P * P + 8 * P * D + sum(8 * (D * c + D * D) for c in columns)
               + 4 * dev_spans * sum(c + D for c in columns)
               + (M - 1) * dev_spans * (4 * widest + 12 * D) + 3 * 8 * D * widest)
    stage2 = max(EIGH_HELD * 8 * s["stage1"] ** 2 + eigh_bytes(s["stage1"], s["view"], scipy=scipy) for s in shapes)
    return {"gcca": int(EIGH_HELD * 8 * P * P + eigh_bytes(P, D, scipy=scipy)), "decoder": int(decoder),
            "stage2": int(stage2), "transforms": transforms,
            "adapters": transforms + 4 * sum(D * c + v * D for c, v in zip(columns, views))}


def memory_estimate(shapes, *, hub_dim, dev_spans, base_bytes=None, scipy=None):
    """The fit's estimated peak of anonymous memory: {"bytes", "gib", "mode", "phase", "kinds"}.
    shapes: {kind: [view_shape per model]}, the new model first when onboarding; base_bytes: the base
    hub an onboarding loads (None: a joint fit); scipy: as joint_bytes."""
    kinds = {}
    if base_bytes is not None:
        pivots = len(next(iter(shapes.values()))) - 1
        for kind, items in shapes.items():
            kinds[kind] = {"onboard": onboard_bytes(items[0]["stage1"], items[0]["columns"], dev_spans=dev_spans,
                                                    pivots=pivots, hub_dim=hub_dim)}
        kind = max(kinds, key=lambda k: kinds[k]["onboard"])
        total, phase = ONBOARD_BASE + base_bytes + kinds[kind]["onboard"], f"{kind} onboard"
    else:
        total, phase, held = 0, None, 0
        for kind, items in shapes.items():
            parts = kinds[kind] = joint_bytes(items, hub_dim=hub_dim, dev_spans=dev_spans, scipy=scipy)
            name = max(("gcca", "decoder", "stage2"), key=parts.get)
            peak = JOINT_BASE + held + parts["transforms"] + parts[name]
            if peak > total:
                total, phase = peak, f"{kind} {name}"
            held += parts["adapters"]
    return {"bytes": int(total), "gib": round(total / GIB, 1), "mode": "onboard" if base_bytes is not None else "joint",
            "phase": phase, "kinds": {kind: {key: round(value / GIB, 2) for key, value in parts.items()}
                                      for kind, parts in kinds.items()}}


def fit_memory(views, spans, args, base_manifest=None, base_bytes=None):
    """memory_estimate for this command's fit (hub dim of the base when onboarding)."""
    hub_dim = base_manifest["settings"]["hub_dim"] if base_manifest else args.hub_dim
    shapes = {kind: [view_shape([v.width(kind, layer) for layer in v.layers(kind)], args.layer_rank, args.view_dim)
                     for v in views] for kind in hub.KINDS}
    return memory_estimate(shapes, hub_dim=hub_dim, dev_spans=int(len(spans.select("dev"))), base_bytes=base_bytes)


def refuse_short_memory(estimate, settings, log):
    """SystemExit when the fit's estimated peak exceeds MemAvailable minus the guard's fit margin (with
    the guard off: a warning only)."""
    available = resource_guard.read_memory(settings.meminfo)["mem_available"]
    if available is None:
        return
    room = available - settings.fit_margin
    log(json.dumps({"stage": "memory-check", "need_gib": estimate["gib"], "available_gib": round(available / GIB, 1),
                    "margin_gib": settings.fit_margin_gib, "fits": estimate["bytes"] <= room}))
    if estimate["bytes"] <= room:
        return
    message = (f"fit_hub: this fit needs about {estimate['gib']} GiB of anonymous memory at its peak "
               f"({estimate['mode']}, {estimate['phase']}), MemAvailable is {available / GIB:.1f} GiB and "
               f"{settings.fit_margin_gib:g} GiB must stay free")
    if not settings.enabled:
        resource_guard.warn_off(message)
        return
    raise SystemExit(message + ": refused (free memory first, or --guard fit_margin_gib=N; --estimate shows the parts)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-dim", type=int, default=1024)
    parser.add_argument("--view-dim", type=int, default=2048)
    parser.add_argument("--layer-rank", type=int, default=512)
    parser.add_argument("--fit-rows", type=int, default=60000, help="train spans used for stages 1-2")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--direct", action="append", default=[], metavar="SOURCE:TARGET=MAPPER_DIR")
    parser.add_argument("--base", type=Path, help="onboard: an existing hub bundle; --model names the new model "
                        "(first) and pivots already in the hub (the rest)")
    parser.add_argument("--source-global", action="append", default=[], metavar="NAME",
                        help="as a source, this model only provides its full-attention layers (sliding-window models)")
    parser.add_argument("--indexer-ridge", action="append", default=[], metavar="NAME",
                        help="fit this DSA model's indexer ridge from its captured indexer keys (key ik)")
    parser.add_argument("--heads", action="append", default=[], metavar="NAME",
                        help="fit this DeepSeek V4 model's head decoders from its captured compressor inputs "
                        "(keys csa, hca, lid)")
    parser.add_argument("--link-base", action="store_true",
                        help="with --base: hard-link the base models' files instead of writing them again")
    parser.add_argument("--onboard-encoder", choices=hub.ONBOARD_ENCODERS, default=None,
                        help="with --base: how the new model's encoder is fitted (hub.solve_onboard_encoder; default "
                        f"{hub.DEFAULT_ONBOARD_ENCODER})")
    parser.add_argument("--onboard-variance-cap", type=float, default=None, metavar="X",
                        help="with --base: the reverse family's per-dimension variance cap, X times the pooled pivot "
                        f"estimates' (default {hub.ONBOARD_VARIANCE_CAP:g}; 0: none)")
    parser.add_argument("--reencode", action="store_true",
                        help="--base is the bundle of the first --model, already onboarded: fit only its encoder "
                        "again (--onboard-encoder), keeping its transforms and decoders")
    parser.add_argument("--evaluate", choices=("all", "new"), default="all",
                        help="test R^2 of every ordered pair, or (with --base) only those with the new model")
    parser.add_argument("--estimate", action="store_true",
                        help="print the fit's memory estimate (spans and shapes only) and exit")
    parser.add_argument("--fit-device", choices=fit_backend.DEVICES, default="auto",
                        help="where the fit's products run: gpu (torch, under the GPU lock), cpu (numpy) or auto "
                        "(the GPU when torch sees one; fit_backend.py)")
    resource_guard.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        settings = resource_guard.Settings.from_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        args.device = fit_backend.resolve(args.fit_device)
    except RuntimeError as error:
        parser.error(str(error))
    if args.output.exists() and not args.estimate:
        raise SystemExit(f"output exists: {args.output}")
    if (args.link_base or args.evaluate == "new" or args.onboard_encoder or args.reencode
            or args.onboard_variance_cap is not None) and not args.base:
        raise SystemExit("--link-base, --evaluate new, --onboard-encoder, --onboard-variance-cap and --reencode "
                         "need --base")
    if args.base and not args.onboard_encoder:
        args.onboard_encoder = hub.DEFAULT_ONBOARD_ENCODER
    if args.base and args.onboard_variance_cap is None:
        args.onboard_variance_cap = hub.ONBOARD_VARIANCE_CAP
    if args.onboard_variance_cap is not None and args.onboard_variance_cap < 0:
        parser.error("--onboard-variance-cap must be at least 0")
    started = time.perf_counter()
    log_path = args.output.with_name(args.output.name + ".log.jsonl")

    def log(line):
        print(line, flush=True)
        if not args.estimate:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    views = [hub.CaptureView(name, captures, records) for name, captures, records in args.model]
    spans = hub.joint_spans(views)
    log(json.dumps({"stage": "spans", "documents": len(spans.document_ids), "rejected": len(spans.rejected),
                    "spans": {split: int(len(spans.select(split))) for split in ("train", "dev", "test")},
                    "tokens": {v.name: int(v.captures.rows_total) for v in views}}))
    base_manifest = hub.read_manifest(args.base) if args.base else None
    if args.reencode:
        check_reencode(base_manifest, args.base, views[0].name)
        for key, value in fitted_settings(base_manifest).items():
            setattr(args, key, value)       # the bundle's own (its estimate and its record)
    base_bytes = sum(p.stat().st_size for p in Path(args.base).rglob("*")       # what load_hub reads
                     if p.is_file() and p.name != "manifest.json") if args.base else None
    estimate = fit_memory(views, spans, args, base_manifest, base_bytes)
    log(json.dumps({"stage": "memory", **estimate}))
    if args.estimate:
        return 0
    gpu = args.device == "gpu"

    def guard_log(item):
        log(json.dumps({"stage": "guard", **item}))
    # the GPU lock first (resource_guard.check_lock_order), then the CPU lock
    with (resource_guard.gpu_lock(settings.gpu_lock, label=f"fit_hub {args.output}", log=guard_log) if gpu
          else contextlib.nullcontext()), \
            resource_guard.cpu_lock(settings.cpu_lock, label=f"fit_hub {args.output}", log=guard_log):
        refuse_short_memory(estimate, settings, log)
        guard_path = args.output.with_name(args.output.name + ".guard.jsonl")
        with resource_guard.Guard(settings, log_path=guard_path, label="fit_hub", gpu=gpu) as guard:
            guard.wait_cool()
            hub.guard_check = guard.check
            try:
                with fit_backend.using(fit_backend.make(args.device, log=log)) as backend:
                    if not gpu:
                        log(json.dumps({"stage": "fit-backend", **backend.describe()}))
                    result = fit(args, views, spans, log, started)
                    log(json.dumps({"stage": "fit-backend-done", "device": args.device, **backend.stats()}))
                    return result
            finally:
                hub.guard_check = hub._no_check


def check_reencode(manifest, bundle, name):
    """SystemExit unless `bundle` is an onboarding's bundle (format 1, grown from a recorded base) that holds
    `name`, a model onboarded into it (not one of its base's)."""
    settings = manifest.get("settings") or {}
    if manifest["format_version"] != hub.FORMAT:
        raise SystemExit(f"--reencode: {bundle} is not a format 1 bundle")
    if name not in manifest["models"]:
        raise SystemExit(f"--reencode: {bundle} holds no {name} (onboard it first)")
    base = settings.get("base")
    if not base or not (Path(base) / "manifest.json").is_file():
        raise SystemExit(f"--reencode: {bundle} records no base it grew from (an onboarding's bundle is needed)")
    if name in hub.read_manifest(base)["models"]:
        raise SystemExit(f"--reencode: {name} is a model of the base {base}, not an onboarded one")
    return Path(base)


REENCODE_KEPT = ("view_dim", "layer_rank", "fit_rows", "source_global", "indexer_ridge", "heads")


def fitted_settings(manifest):
    """{setting: value} of REENCODE_KEPT that an onboarded bundle's transforms and decoders were fitted with (its
    manifest's settings): a re-encode records them as they were (--reencode fits only the encoder)."""
    settings = manifest.get("settings") or {}
    return {key: settings[key] for key in REENCODE_KEPT if key in settings}


def fit(args, views, spans, log, started):
    """The fit itself (main's, under the CPU lock and the resource guard)."""
    by_name = {v.name: v for v in views}
    source_layers = {}
    for name in args.source_global:
        if args.reencode and name not in by_name:
            continue                # a re-encode's record of the bundle's settings: not used
        types = by_name[name].captures.captures[0].layer_types
        source_layers[name] = {kind: [l for l in by_name[name].layers(kind) if types[l] == "full"] for kind in hub.KINDS}
    hubs, evaluation, base_models, ridges = {}, {}, None, {}
    consensus = [view.name for view in views]
    grown_from = args.base          # the base the new bundle grows from (a re-encoded bundle's own base)
    if args.base:
        hubs, base_manifest = hub.load_hub(args.base)
        base_models = base_manifest["models"]
        if args.reencode:
            grown_from = check_reencode(base_manifest, args.base, views[0].name)
        ridges = hub.load_indexer_ridges(args.base)
        consensus = hub.consensus_views(base_manifest)
        if consensus is None and args.onboard_encoder in hub.CONSENSUS_ENCODERS:
            raise SystemExit(f"{args.base} does not record the models its hub latent was fitted on "
                             f"(--onboard-encoder {args.onboard_encoder} needs them)")
    for kind in hub.KINDS:
        if args.reencode:
            hub.reencode_model_kind(hubs[kind], views[0], views[1:], spans, kind, encoder=args.onboard_encoder,
                                    consensus=consensus, chunk=args.chunk, log=log,
                                    variance_cap=args.onboard_variance_cap)
        elif args.base:
            new, pivots = views[0], views[1:]
            hub.add_model_kind(hubs[kind], new, pivots, spans, kind, layer_rank=args.layer_rank,
                               view_dim=args.view_dim, fit_rows=args.fit_rows, chunk=args.chunk, log=log,
                               source_layers=source_layers.get(new.name, {}).get(kind),
                               encoder=args.onboard_encoder, consensus=consensus,
                               variance_cap=args.onboard_variance_cap)
        else:
            hubs[kind] = hub.fit_hub_kind(views, spans, kind, layer_rank=args.layer_rank, view_dim=args.view_dim,
                                          hub_dim=args.hub_dim, fit_rows=args.fit_rows, chunk=args.chunk, log=log,
                                          source_layers={n: layers[kind] for n, layers in source_layers.items()})
        involving = views[0].name if args.evaluate == "new" else None
        evaluation[kind] = {f"{s}->{t}": item for (s, t), item in
                            hub.evaluate(hubs[kind], views, spans, kind, chunk=args.chunk, involving=involving).items()}
        log(json.dumps({"stage": "test", "kind": kind,
                        "r2": {pair: round(item["r2"], 4) for pair, item in evaluation[kind].items()}}))
        for item in args.direct:
            pair, _, mapper_dir = item.partition("=")
            source, _, target = pair.partition(":")
            result = direct_r2(Path(mapper_dir), by_name[source], by_name[target], spans, kind, chunk=args.chunk)
            evaluation[kind][f"direct {source}->{target}"] = result
            log(json.dumps({"stage": "direct", "kind": kind, "pair": pair, "r2": round(result["r2"], 4)}))
    for name in [] if args.reencode else args.indexer_ridge:      # a re-encode keeps the bundle's ridges
        ridges[name] = hub.fit_indexer_ridge(by_name[name], fit_rows=args.fit_rows, chunk=args.chunk, log=log)
        evaluation.setdefault("indexer", {})[name] = {str(layer): item for layer, item in ridges[name].fit.items()}
    heads = []
    for name in [] if args.reencode else args.heads:
        pivots = [view for view in views if view.name != name]
        for kind in hub.HEAD_KINDS:
            if by_name[name].layers(kind):
                heads.append(hub.fit_head(hubs[hub.HEAD_BASE], pivots, by_name[name], spans, kind, chunk=args.chunk,
                                          log=log))
                evaluation.setdefault("heads", {})[f"{name}.{kind}"] = heads[-1].alphas[name]
        if not any(head.names == [name] for head in heads):
            raise SystemExit(f"{name}: its captures hold no compressor inputs ({', '.join(hub.HEAD_KINDS)})")
    hubs = hub.with_heads(hubs, heads)
    settings = {key: getattr(args, key) for key in ("hub_dim", "view_dim", "layer_rank", "fit_rows", "source_global",
                                                    "indexer_ridge", "heads", "onboard_encoder")}
    settings["onboard_encoder"] = args.onboard_encoder if args.base else None
    settings["onboard_variance_cap"] = args.onboard_variance_cap if args.base else None
    if args.base:               # an onboarding keeps the base's latent: its dimension, not --hub-dim's
        settings["hub_dim"] = int(len(hubs[hub.KINDS[0]].eigenvalues))
    settings["fit_device"] = getattr(args, "device", "cpu")
    settings["large_eigh"] = fit_backend.LARGE_EIGH          # fit_backend.top_eig's precision for large subsets
    settings["consensus_views"] = consensus
    settings["models"] = {name: {"captures": [str(c.resolve()) for c in caps], "records": str(rec.resolve())}
                          for name, caps, rec in args.model}
    settings["base"] = str(Path(grown_from).resolve()) if grown_from else None
    settings["linked_base"] = bool(args.link_base)
    if args.reencode:
        settings["reencoded_from"] = str(args.base.resolve())
    hub.save_hub(args.output, hubs, views, spans, settings, base_models=base_models, indexer_ridges=ridges,
                 link_base=grown_from if args.link_base else None)
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n", encoding="utf-8")
    log(json.dumps({"stage": "done", "seconds": round(time.perf_counter() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
