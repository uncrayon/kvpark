# Universal KV translator (experiment)

One shared per-token latent ("hub") that translates the attention caches of different open
models into each other, so a conversation can move between models without a full prefill.
The product side is `kvpark bridge` ([docs/bridge.md](../../docs/bridge.md)).

- [ARCHITECTURE.md](ARCHITECTURE.md): design, the state families of the top-20 open models,
  what is supported for each llama.cpp architecture and on what it was validated, the engine
  and its daemon, recurrent state, images, draft models, DeepSeek V4, open problems.
- [RESULTS.md](RESULTS.md): measurements on four local models fitted jointly (Qwen3.8-27B,
  Ornith-1.5-9B, Qwen3-4B, Muse-Glimmer-30B), on the twelve onboarded since (sections 9-10; eleven of
  them in hub6) and live switches on real llama-servers; a summary table at the top.
- [survey/SURVEY.md](survey/SURVEY.md): the top-20 open-weights models (Artificial Analysis
  Intelligence Index v4.3, read 2026-10-02) and their cache architectures.

The capture, state codec, RoPE and template code lives in [../linear_transfer](../linear_transfer)
(`kvstate.py`, `chat.py`, `kv_probe.cpp`, `state_driver.cpp`, `template_check.py`,
`vocab_tool.cpp`); its README lists every state layout `kvstate` parses, how `chat.py` follows
llama.cpp's jinja and the template parity checks.

Both pinned llama.cpp runtimes are supported: a7b94df2c (the repository's `runtime/`, built by
`scripts/build_runtime.py`, and the builds in `runtime-next/`) and 85d5703a3 (what the production
Qwen3.8 server still runs; the earlier `runtime/`, kept at `runtime-85d5703a3/`). The state codec
reads and writes both pins' states (they differ for deepseek2/mistral4 nextn layers, qwen4exp's
indexer rows and bailingmoe3 with rope sections), the translator writes the target template's
layout, and an engine job may name each server's build (`build_info`, from `/props`) so its specs
take that pin.

What it translates ([ARCHITECTURE.md, "What the translator supports today"](ARCHITECTURE.md#what-the-translator-supports-today)
has what each was validated on; the bridge's list is `TRANSLATABLE` in
`src/kvpark/bridge/capabilities.py`):

| Family | Architectures |
| --- | --- |
| Dense GQA, sliding window + global | `qwen3`, `muse-glimmer` (real models, live) |
| GQA family | `llama`, `mistral3`, `qwen3moe`, `hy_v3`, `phi3` (none of them LongRoPE), `gemma3`, `gemma4` (real Gemma 4 26B-A4B and 31B onboarded, live), `gpt-oss`, `laguna`, `mimo2` |
| Hybrids with a recurrent cell | `qwen35` (real models, live), `qwen35moe` (real), `qwen3next`, `qwen4exp`, `minimax-01`, `jamba` (real; fails as a source), `falcon-h1` (real), `granitehybrid` (real, live; a target only in practice), `nemotron_h`, `nemotron_h_moe` (real; a target only in practice), `plamo2` (cache only: its PLaMo-2 vocabulary is not read, so the bridge skips it as `tokenizer`), `lfm2` (real), `lfm2moe` |
| MLA, DSA, K-only hybrids | `deepseek2` (real models), `mistral4`, `hy_v4`, `deepseek32`, `glm-dsa`, `dots3note`, `kimi-linear` (real, live), `kimi-k3`, `bailingmoe3` (real), `glm5-next` (a7b94df2c only) |
| Pooled latent (target only) | `deepseek4` |

"real": a local model captured, fitted or onboarded into the hub and measured end to end (RESULTS.md); "live":
switched between running llama-servers through `kvpark bridge`; the rest on random-weight fixtures.

Draft models: DFlash/DSpark (an aligned dummy), MTP (an empty head cache) and the `shared` kind
(Gemma 4's MTP head, which writes no cells of its own: its 8-byte draft blob is kept as it is).
Vocabularies: GPT-2 byte-level BPE, Gemma 4's BPE and SentencePiece (from the engine's
`vocabularies` feature). Chat templates render as llama-server renders them (`chat.py`), checked
by `template_check.py` against `/apply-template` on every local model.

## Code

| File | Purpose |
| --- | --- |
| hub.py | hub fitting (PCA views, multi-view CCA, ridge encoders/decoders, global-layer encoders for sliding-window sources), onboarding a model without refitting (`add_model_kind`; its encoder by `solve_onboard_encoder`, default `reverse`) and fitting only an onboarded model's encoder again (`reencode_model_kind`), evaluation (every pair, or those `involving` one model); DSA indexer ridges (`IndexerRidge`) and DeepSeek V4 head kinds (`HeadKind`: decoders of the compressor inputs of each CSA, HCA and LID layer); translation sides (`EncoderSide` with the fused encoder `(basis2 * scale) @ encoder`, `DecoderSide` with per-section decoders); bundles: `save_hub`/`load_hub` (format 1 and 2; `link_base`: an onboarding's bundle hard-links the base's files), `open_hub` (lazy, side by side, sha256 checked once per file identity), `merge_bundles` (one bundle of a base and models onboarded into it separately, hard links) |
| fit_hub.py | CLI: fit a hub on captured models, or onboard one (`--base`; `--link-base` links the base's files, `--evaluate new` tests only the pairs with the new model, `--onboard-encoder` picks the encoder, `--reencode` fits only an onboarded model's encoder again); `--indexer-ridge NAME` for a DSA model, `--heads NAME` for a DeepSeek V4 model; compares with a pairwise mapper (`--direct`) |
| convert_hub.py | format 1 bundle → format 2: one flat C-contiguous `.npy` per (model, kind, role), fused encoders, float16 copies, sha256 manifest; indexer ridges and head kinds carried over; `--reuse V2` links the sides of models an earlier conversion holds (same files) |
| token_tables.py | per-model token tables (mean K/V per token id over the cells the hub does not cover); `--only NAME` for an onboarded model on its onboarding's spans |
| campaign.py | onboards one model end to end, resumable stage by stage (render with strict template parity, a chunked capture within a size and time budget, onboard, token tables, the recurrent mean where the cell holds a state, evaluation with Qwen3.8 and the Qwen3-4B baseline, a report with paired CIs), every command that loads a model or a hub under the GPU/memory lock, which the command itself holds, the CPU-heavy ones also under the CPU lock, all under the resource guard; `exec` runs one command that way, `assemble` builds one hub of the base and the onboarded models, `reencode` fits onboarded models' encoders again with another method, `status` flags assembled hubs a later onboarding outdated ([Onboarding campaign](#onboarding-campaign-campaignpy)) |
| resource_guard.py | the thermal and resource guard every heavy job runs under (standard library only): sensors (k10temp, amdgpu, NVMe; MemAvailable, swap), `wait_cool`, `Watch` (stops a child's process group while too hot or short of memory, ends it when critical, a JSONL sample every 2 s), `Guard.check` for loops in a process, the CPU lock, BLAS thread caps; CLI `run [--cpu-lock] -- COMMAND` and `status` ([Running heavy jobs safely](#running-heavy-jobs-safely)) |
| translate.py | any supported source slot state → any supported target slot state through the hub: `prepare` (alignment, the positions each target section keeps), `compute` (encode, decode only the cells each section keeps), `assemble` (MLA latent shell, indexer keys, RoPE, cache dtype, native prefix, the recurrent cell); `translate()` runs compute and the per-row part of assemble window by window of target positions (`compute_cache`), touching a GPU job's fresh output pages ahead (`Prefault`); dispatches DeepSeek V4 targets to dsv4_target.py |
| backends/ | the arithmetic: `numpy` (float32, the reference, always available), `torch` (float16 weights, float32 accumulation, RoPE on the device from positions, checked against kvstate on first use; ROCm, CUDA, MPS) and `ggml` (Vulkan through libkvbridge, [ggml/](ggml/README.md), no torch); `backends.select("auto" / "cpu" / "gpu" / "vulkan")`; NORM, NEOX, M-RoPE/IMROPE and frequency factors |
| state_writer.py | writes a translated state part by part: the bytes of `kvstate.write_new(path, state.to_bytes())`, 5x faster, 0600, atomic, no fsync; media token words allowed |
| dsv4_pool.py | DeepSeek V4 pooled rows (CSA, HCA, LID) and compressor rings from per-token compressor inputs (a, s), with the GGUF's APE and norm tensors (numpy) |
| dsv4_target.py | DeepSeek V4 as a target: `build_state` / `synthesize` a state for [0, T0), T0 rounded down to a multiple of 128; `translate`, the translator's path into it (hub latents → window latents and head kinds → pools) |
| recurrent_init.py | the recurrent cell a hybrid target restores at T0 − 1: strategies `zero` (R0), `mean`, `mean_conv`, `mean_s<x>`, `prefix`; `MeanAccumulator` / `build_mean`, the per-model mean npz (`RecurrentMean`, groups `all`, `chat`, `tools`), `match_group` |
| r0_experiments.py | measures those strategies on real hybrid models: `calibrate` (native saves of training conversations → `recurrent-mean.npz`), `run` (native attention + each strategy, seed-scored), `combined` (translated attention + each strategy), `summarize` |
| engine.py | the bridge worker: a live conversation's source slot + the target server's tokens → target slot file; renders the target as llama-server does (request template options, server template flags), reads a source slot holding media up to its first medium, picks the recurrent mean's group; reads GPT-2 byte-level BPE, Gemma 4 BPE and SentencePiece vocabularies (`spelled_pieces`); places the source's segments where its own template renders them (`source_spans`), the rest by search (`located_record`; report `source_segments_rendered`/`_found`); model profiles carry the RoPE frequency factors (`rope_freqs.weight` of llama, mistral3, gemma4, granitehybrid; LongRoPE, phi3's or a llama's, refused); a side's `build_info` picks its pin (`job_revision`); `FEATURES` tell the bridge what it handles (`media-source`, `template-options`, `recurrent-mean`, `vocabularies`); one-shot CLI `engine.py JOB.json`, and `run()` for the daemon |
| engine_daemon.py | the persistent engine: AF_UNIX socket in the bridge directory, JSON lines, one FIFO worker, resident hub sides and recurrent means, memory guard, prewarm with warm-up products, idle unload/exit, error replies with a kind (`job`, `template`, `memory`, `busy`, `internal`, `request`), the GPU switched off for its lifetime after a device fault; `--check` prints the engine Python's BLAS and GPU |
| bench_engine.py | engine benchmarks on real saved states: `warm` (daemon-like), `oneshot`, `accuracy` (torch vs numpy), `nll` (seed-score ΔNLL of float16 vs float32 translations), `prewarm`, `job` (replays a kept switch job with the assemble split and page-fault counters) |
| pair_eval.py | end-to-end evaluation of a model pair (render, tokenize, save states, translate, seed-score); `--recurrent`, `--recurrent-mean`, `--recurrent-group` for hybrid targets; `--ids` (with `--min-documents`) a fixed set of conversations, such as RESULTS.md section 2's; `--server-render` renders both models as llama-server does (its default template kwargs, a fixed clock) instead of with the templates' defaults; `--extra-hub LABEL=DIR` also translates with another bundle on the same documents, states and native scores (conditions `hub_prefix.LABEL`, and its own floors when its target side differs), so bundles compare paired in one run |
| score_translate.py | hub vs a pairwise mapper on the documents of a linear_transfer long run; the same recurrent options |
| recurrent_reset.py | the cost of resetting a hybrid model's recurrent state (R0 alone; superseded by `r0_experiments.py run --strategies zero`) |
| live_bridge.py | live tests on throwaway llama-servers: `hops` (`--cool-c`, `--keep-text`), `drafts`, `media`, `drafts-control`, `media-control`, `options` (request template options, the recurrent mean); a memory guard before each model start; under a guard's Watch (`campaign.py exec`) its servers and the engine daemon stay in its process group ([Running heavy jobs safely](#running-heavy-jobs-safely)) |
| pair_table.py | markdown table of pair_eval reports |
| fixtures_families.py | test helpers: tiny MLA/DSA, K-only hybrid and GQA-family models (test-llama-archs and gguf-py Gemma 3/4), the state driver and probe wrappers, exact identity hubs |
| fixtures_bridge/ | `make_fixtures.py` (tiny Qwen3.5 and Qwen3 targets with DFlash and MTP drafts, and a 5120-wide Qwen3.5 the Qwen3.8 mmproj can feed), `live_resume_check.py` (the bridge's `.resume` sidecars on a kvpark-patched server), `stub_engine.py` (a same-model stand-in engine for live switch checks) |
| test_*.py | `test_hub`, `test_campaign` (campaign.py's planning and resume, capture chunks, parity rules, the lock and process groups, statistics and reports, no model), `test_resource_guard` (the guard on fake sensor trees and fake children: pause, resume, abort, sensor loss, the CPU lock and its order, thread caps, campaign commands under it, fit_hub.py's memory estimate and refusal), `test_pair_eval` (its renderings, no model), `test_translate` (every cache family on synthetic models with hand-built hubs), `test_backends`, `test_ggml_backend`, `test_state_writer`, `test_engine` (rendering, media sources, features, vocabularies, specs and pins), `test_engine_daemon`, `test_recurrent_init`, `test_families` (MLA/DSA/hybrids and the GQA family through llama.cpp), `test_dsv4` (DeepSeek V4 through llama.cpp, the engine and `kvpark.bridge.switch`) |

The product's client for the daemon is
[src/kvpark/bridge/engine_client.py](../../src/kvpark/bridge/engine_client.py) (standard
library only): `run_job` sends a job to the daemon and starts it when nobody answers. When the
daemon is unavailable (it does not start or answer, times out, shuts down, or reports a fault
of its own) the job runs in the one-shot `engine.py JOB.json`; when the daemon says the job
itself failed (`job`, `template`) or its memory guard refused it (`memory`), `run_job` raises
`EngineError` (a `RuntimeError` reading "bridge engine failed: ...", like a failed one-shot
engine) instead of rerunning the job without the guard. Also `status`, `stop`, `prewarm`,
`features`.

## Tests

Run from this directory with the linear_transfer venv (numpy with OpenBLAS), and again with a
torch Python for the GPU backend (`test_backends`' torch tests skip without torch). The
end-to-end tests drive llama.cpp through CPU binaries built from a pinned runtime, and pass with
either pin's (the expectations that differ by pin follow the state driver's revision); set them
or those tests skip:

| Variable | Binary | Used by |
| --- | --- | --- |
| `KVPARK_STATE_DRIVER` | `kvpark-state-driver` (linear_transfer/state_driver.cpp) | test_families, test_dsv4 |
| `KVPARK_LLAMA_ARCHS` | `test-llama-archs` (a CPU build with `LLAMA_BUILD_TESTS`: `runtime-next/build-cpu` at a7b94df2c, or an 85d5703a3 test build) | test_families, test_dsv4 |
| `KVPARK_KV_PROBE` | `kvpark-kv-probe` (linear_transfer/kv_probe.cpp), built from the current source: it needs the MLA, DSA, GQA-family and deepseek4 rows | test_families' cross-family hub, test_dsv4's head fit |
| `KVPARK_VOCAB_TOOL` | `kvpark-vocab` (linear_transfer/vocab_tool.cpp): llama.cpp's own vocabulary | test_engine.RealVocabularyTest (every local GGUF, about 2 minutes) |
| `KVPARK_LLAMA_SERVER` | a llama-server (`runtime/build/bin/llama-server`, a7b94df2c; `runtime-85d5703a3/build/bin/llama-server`, 85d5703a3) | test_engine's llama-server comparisons (throwaway servers from port 18574, or `KVPARK_TEMPLATE_PORT`) |

Build them without a download, from the repository root. At a7b94df2c, against the CPU build
with tests in `runtime-next/` (the builds in `runtime-kvtools/` are made this way):

```bash
for target in kvpark-state-driver kvpark-kv-probe kvpark-vocab; do python3 experiments/linear_transfer/build_collector.py --llama-source runtime-next/src --llama-build runtime-next/build-cpu --target $target --output runtime-kvtools/$target --jobs 16; done
```

At 85d5703a3, build the CPU test build of `runtime-85d5703a3/source` as in
[MLA, DSA and K-only hybrid states](../linear_transfer/README.md#mla-dsa-and-k-only-hybrid-states)
(`/tmp/kvpark-build-tests`), then the tools against it with the 85d5703a3 headers and an explicit
revision (that build's info names kvpark's own commit, so `build_collector.py` cannot tell the pin):

```bash
for target in kvpark-state-driver kvpark-kv-probe kvpark-vocab; do python3 experiments/linear_transfer/build_collector.py --llama-source runtime-85d5703a3/source --llama-build /tmp/kvpark-build-tests --runtime-revision 85d5703a3b1b47243213a39059a6e3076c92733a --target $target --output /tmp/kvpark-85d5703a3/$target --jobs 16; done
```

[Tests on both pinned runtimes](../linear_transfer/README.md#tests-on-both-pinned-runtimes) has
the complete recipe. Then, per pin:

```bash
cd experiments/universal_kv
export KVPARK_STATE_DRIVER=$PWD/../../runtime-kvtools/kvpark-state-driver/kvpark-state-driver KVPARK_LLAMA_ARCHS=$PWD/../../runtime-next/build-cpu/bin/test-llama-archs KVPARK_KV_PROBE=$PWD/../../runtime-kvtools/kvpark-kv-probe/kvpark-kv-probe
../linear_transfer/.venv/bin/python -m unittest discover -q -p 'test_*.py'
~/venvs/rocm/bin/python -m unittest discover -q -p 'test_*.py'
```

(at 85d5703a3: `KVPARK_STATE_DRIVER=/tmp/kvpark-85d5703a3/kvpark-state-driver/kvpark-state-driver
KVPARK_LLAMA_ARCHS=/tmp/kvpark-build-tests/bin/test-llama-archs
KVPARK_KV_PROBE=/tmp/kvpark-85d5703a3/kvpark-kv-probe/kvpark-kv-probe`).

Opt-in parts:

- `test_dsv4.EngineBridgeTests` runs `kvpark.bridge.switch` in a `python3` that imports psutil
  (the product's registry needs it; the engine venvs lack it) and skips that part without one.
- `test_engine`'s llama-server comparisons need `KVPARK_LLAMA_SERVER` and, for the M-RoPE media
  slot, `KVPARK_QWEN_MMPROJ` (the Qwen3.8 mmproj).
- `test_engine.RealVocabularyTest` needs `KVPARK_VOCAB_TOOL` and reads the local GGUFs of
  `LOCAL_MODELS` (or `KVPARK_VOCAB_MODELS`, colon-separated).
- `test_translate` also compares with a snapshot of the pre-split translate.py when
  `KVPARK_TRANSLATE_BASELINE` names a directory holding one; the frozen output digests are
  always checked.

## Typical run

`campaign.py run --name NAME` does steps 1-5 for one new model against hub4-d4096 (next section);
by hand:

1. Capture K/V of each model on the hub corpus (`../linear_transfer/chat.py`, `collect_v2.py`;
   `--mode states` for models whose graphs name no pre-RoPE key: kimi-k3, bailingmoe3, hy_v4,
   hy_v3 and glm5-next; the states are read in the layout of the probe's pin; gemma4 on the CPU
   probe, since on Vulkan its capture run and a plain run differ by more than rounding).
2. `fit_hub.py --model NAME=CAPTURES@RENDERED.jsonl ... --output HUB --hub-dim 4096 --view-dim 4096`
   (`--source-global NAME` for sliding-window models, `--indexer-ridge NAME` for DSA models,
   `--heads NAME` for DeepSeek V4).
3. `convert_hub.py HUB HUB-v2` (about 25 s for hub4-d4096; the engine reads either format,
   format 2 much faster: a switch reads only its two sides).
4. For each hybrid target, once: `r0_experiments.py calibrate --model GGUF --name NAME --corpus
   CONVERSATIONS.jsonl ... --probe PROBE --output DIR` (a Vulkan probe build; about 18 min of
   GPU prefill for Qwen3.8) gives `DIR/recurrent-mean.npz`.
5. Optionally `token_tables.py --model ... --output HUB/../token-tables` (for `fill="table"`
   in the evaluations) and `pair_eval.py --hub HUB --source A=GGUF --target B=GGUF --corpus
   CONVERSATIONS.jsonl --probe PROBE --output DIR`.
6. Register servers and the engine with `kvpark bridge add` (with `--recurrent-mean` for
   hybrids) / `kvpark bridge engine`, then `kvpark bridge switch` (or `/kvpark switch` in Hermes).

Steps 2, 4 and 5 are heavy: run them as the next section says.

## Running heavy jobs safely

On October 5, 2026 at 22:02 the machine powered off. The CPU's internal thermal limit tripped
(AMD reset reason 0x00080a00). There was no OOM kill, but swap was 93% used. At the time, a joint
seven-model hub refit (numpy, BLAS on every core) and an onboarding diagnostic were running on the
CPU together, after hours of GPU captures. The CPU and the Radeon 8060S share one package and one
cooler.

Every heavy job now runs under `resource_guard.py` (standard library only):

| Guard | Default | Setting |
| --- | --- | --- |
| BLAS/OpenMP/MKL threads of heavy children and of fit_hub.py, campaign.py, r0_experiments.py (unless set; OpenBLAS alone would start 32) | 8; 4 for GPU evaluations | `threads`, `KVPARK_GUARD_THREADS` |
| llama.cpp probe `--threads` (captures, pair_eval, r0_experiments) | 8 (was 16); 4 for pair_eval | `threads` |
| start a heavy step only when the package (hottest of k10temp Tctl/Tdie/Tccd*; with `sensors=package` also the amdgpu edge, the rule until 2026-10-08, when 58% of the GPU evaluations' pauses came from the edge alone) is at most | 88 °C: `pause_c` − 2 (75 °C until 2026-10-08; for GPU evaluations 83 °C on that evening) | `start_c` |
| stop the job's process group (SIGSTOP) at | 90 °C, GPU evaluations too (85 °C for them until the evening of 2026-10-08), swap use ≥ 85% (while MemAvailable < 24 GiB), MemAvailable < 8 GiB | `pause_c`, `swap_pause`, `swap_ignore_gib`, `mem_floor_gib` |
| continue it (SIGCONT) at | ≤ 88 °C: `pause_c` − 2 (80 °C, and 65 °C for GPU evaluations, until 2026-10-08) with MemAvailable ≥ 12 GiB and swap use < 80% (or MemAvailable ≥ 24 GiB) | `resume_c`, `mem_resume_gib`, `swap_resume`, `swap_ignore_gib` |
| end it (SIGTERM + SIGCONT, SIGKILL after 30 s) at | 98 °C on 2 samples in a row (stopped at the first), no CPU temperature for 60 s, a pause longer than 30 min | `abort_c`, `abort_samples`, `sensor_loss_s`, `max_pause_s` |
| samples (temperatures, PPT power, NVMe, MemAvailable, swap, load, the job's RSS) | every 2 s, JSONL | `poll_s` |
| CPU-heavy numpy work: one job at a time, `/tmp/kvpark-campaign-cpu.lock`, always taken after the GPU/memory lock | | |
| a fit is refused when its estimated anonymous peak exceeds MemAvailable minus | 25 GiB | `fit_margin_gib` |
| end the job at the first amdgpu ring timeout, ring reset, GPU reset or "device wedged" line in the kernel log since the job started (`journalctl -k -f`, or `/dev/kmsg` when readable) | on | `gpu_watch` (auto, journal, kmsg, off) |

Tctl moves fast on this machine. In the guard's own logs, one nice'd AVX-512 thread took it to
89 °C (36 W PPT). Four threads took it to 91.5 °C (63 W): the guard stopped that job, and 2 s later
Tctl was at 62 °C. A job running 8 BLAS threads will therefore be paused regularly. This is also
why a single sample at 98 °C only stops a job, and a second one in a row ends it. The line was 95 °C
until 2026-10-08: work outside the locks kept the package at 95-97 °C while the job was already
stopped, and ending a stopped job cools nothing.

**Long GPU evaluations and other work.** A pair_eval run into Qwen3.8 draws 100-138 W (PPT) in bursts
and takes Tctl from about 65 to 90 °C within 2 s. On the night of October 6 the guard ended such runs
five times; at least three of the five came while the job was already stopped, with other work outside
the locks (another project's test sandboxes, up to 108 W) heating the package. A stopped job cannot cool
it, so evaluations now resume (`pair_eval.py --resume`, campaign.py's evaluate stage). Most of these
ends came from the run itself, though, with a lag: a 2 s burst at 125 W (GPU edge 89 °C) is followed, 4-8 s
into the pause and at 30 W, by a Tctl peak of 97-98 °C, as the heat reaches the CPU dies. The peak
follows the temperature the burst starts from. GPU evaluations (`resource_guard.GPU_EVALUATION`: pair_eval,
campaign.py's evaluate stage and baseline, `assemble --evaluate`, `campaign.py exec --gpu-evaluation`) run with
4 threads (the probe's CPU threads add heat and nothing to a Vulkan run: the scores are the same as with 16
threads), and their thresholds have changed twice:

- Until October 7 they stopped at 90 or 85 °C and continued at 80 or 75 °C (a job ended at 95 °C): with resume at
  75 °C a Qwen3.6 → Qwen3.8 run was ended after 2-4 minutes again and again.
- On October 7-8 they stopped at 85 °C and continued only at 65 °C, which let the lagged peak start lower: the
  twelve evaluation runs of October 7 all completed, but 69 of their 24,487 samples (0.28%) reached 95 °C or more,
  peaking at 97.9 °C, and the guard ended a run 7 times, each resumed (`--resume`), some of them while the job was
  already stopped (other work on the package).
- Since the evening of October 8 they stop at 90 °C and continue at 88 °C like every job, and a job ends at 98 °C:
  the user's choice, to keep the GPU busy and not wait for cool-downs. That trades the low resume temperature's
  margin for throughput: in the campaigns' guard logs to October 9, 1.9% of the GPU evaluations' 13,294 samples
  reached 95 °C or more (0.37% of 30,008 under 85/65 °C), 4 reached 98 °C or more (at most 98.4 °C), and 2 of
  their 17 runs were ended by the guard and resumed.

`KVPARK_GUARD_*` still override the defaults.

**Swap and stopped jobs.** A stopped job's memory is what the kernel swaps out first. On October 7 the
joint seven-model refit ran with the GPU-evaluation settings (85/65 °C, 4 threads) next to kvpark.service's
llama-server (47 GiB of GPU memory): it was stopped 70% of the time, its anonymous memory (22 GiB at the
GCCA) went to swap while MemAvailable stayed at 30-50 GiB, and swap reached 99%. The guard then cannot let
the job continue (that needed swap use below 80%, and the job's own pages only come back when it runs), so
it would have ended it after the 30-minute pause limit; it was ended by hand. An hour later a stopped
pair_eval (12 GiB swapped out) stalled the same way. Swap use now counts only while MemAvailable is below
24 GiB (`swap_ignore_gib`; 0 restores the old rule): with more available, full swap is pages swapped out
earlier, and the MemAvailable floor (8 GiB) still stops a job when memory is really short. Run CPU fits
with the CPU defaults (90/80 °C, 8 threads: shorter pauses), and the joint refit with the model servers
stopped.

**GPU hangs.** On October 6 at 09:53 a fit benchmark ran plain float32 torch products on the iGPU
(about 67 GFLOPS there, 8 to 33 s per kernel). The desktop draws on the same GPU: its graphics ring
timed out again and again (amdgpu `ring gfx_0.0.0 timeout`, charged to the Claude app, ChatGPT and
GNOME Shell), GNOME Shell aborted and took every window with it, and the driver reset the GPU. Every
Watch therefore follows the kernel log from its job's start and ends the job like an abort at the
first ring timeout, ring reset, GPU reset or wedged device (event `gpu-hang` with the line, and
`gpu_hang` in the summary and in `run-done`); a fit's own Guard does the same between chunks when it
runs GPU kernels in-process (`Guard(gpu=True)`), and its background thread ends the process when a
GPU call has not returned two polls after such a line. With the guard off the line is logged, not
acted on. GPU work itself keeps every kernel short: see Fit placement.

Settings come from `KVPARK_GUARD_<NAME>` variables (`KVPARK_GUARD_PAUSE_C=88`, ...) or
`--guard name=value` on every command line above. `KVPARK_GUARD=off` (or `--guard off`) prints a
loud warning and turns off the waiting, pausing, aborting and the fit refusal. Samples are still
logged, and the CPU lock and thread caps stay.

How each job is guarded:

- **campaign.py** does all of it for every command it runs: thread caps, its locks (GPU/memory,
  then CPU for fit_hub, token_tables, r0 calibrate, pair_eval and a capture on the CPU probe),
  `wait_cool` once the locks are held, and a Watch.
  - The Watch's samples go to `campaign/NAME/logs/<label>.guard.jsonl`. Pauses, resumes and aborts
    also go to `log.jsonl`.
  - `state.json` records `cool_wait_s`, `paused_s`, `max_c` and `guard_abort` for each command.
  - Pauses count neither toward a command's seconds nor its timeout. A guard abort fails the
    stage (`GuardStop`), and the run resumes as usual.
  - The lines in `commands.sh` go through `campaign.py exec ... --cpu-lock --guard-log FILE`, which
    guards the same way.
- **fit_hub.py** by hand:
  - A GPU fit (`--fit-device gpu`, or `auto` with torch: see Fit placement) takes the GPU/memory lock
    first; under `campaign.py exec` it finds the descriptor its parent passed down and holds the lock
    through it. Its Guard reads the kernel log for GPU hangs.
  - It takes the CPU lock. Under campaign.py it inherits the lock through `KVPARK_CPU_LOCK_FD`.
  - It logs its memory estimate (stage `memory`) and refuses a fit that does not fit
    (`--estimate` prints the estimate and exits).
  - It waits until the CPU is cool, and every fit loop calls the guard between chunks
    (`hub.guard_check`), with samples in `OUTPUT.guard.jsonl`.
  - A single LAPACK call cannot be paused from inside the process, so run a joint refit, whose
    largest eigendecomposition takes many minutes, under an outer Watch:

  ```bash
  ../linear_transfer/.venv/bin/python resource_guard.py run --cpu-lock --log ../../evidence/universal-kv/hub7.guard.jsonl -- \
      ../linear_transfer/.venv/bin/python fit_hub.py --model ... --output ../../evidence/universal-kv/hub7 --hub-dim 4096 --view-dim 4096
  ```
- **pair_eval.py and r0_experiments.py** run by hand put each probe run (each model load) under
  `wait_cool` and a Watch, with samples in `<output>/<step>.guard.jsonl` (pair_eval with the
  GPU-evaluation defaults). Inside a campaign command, the campaign's Watch covers their whole process
  group.
- **assemble --evaluate** (`campaign.py evaluate-onboarded`) runs as a child process on the GPU (the
  fit backend of `~/venvs/rocm/bin/python`) under the locks and a Watch with the GPU-evaluation defaults.
  Until October 7 it evaluated on the CPU inside campaign.py, where its Guard could pause only between
  chunks: the package went from 82 °C to 98 °C within one, and the guard ended it ("outside check()").
- **live_bridge.py** (live switches on throwaway llama-servers) runs under `campaign.py exec --memory-bytes N
  --gpu-evaluation --guard-log FILE -- python3 live_bridge.py ...`. Under that Watch it keeps its llama-servers
  and the engine daemon (which the bridge would start in a session of its own) in its process group, so the
  guard pauses and ends them with the run; the daemon gets an idle exit of a week (the run stops it: its default
  600 s counts on a clock that runs on while the guard holds the run), and every switch first checks that the
  daemon answering is still that one. When the run's command exits, campaign.py stops whatever it left in its group
  before releasing the locks. `hops --cool-c 65` waits for the package to cool before each timed step; the hop
  records carry the wall clock of each step, from which the guard's pauses inside it are known (a pause stops the
  processes while the GPU finishes its queue, so the step's own time is bounded by its wall time with and without
  them), and `--native-chunk 1024` prefills the native reference again in cooled requests. Use the ggml Vulkan
  engine there (`--engine vulkan=EXPERIMENT_PYTHON:vulkan`): on October 9
  the torch-ROCm daemon's busy runtime thread ([Fit placement](#fit-placement-fit_backendpy)) alone held Tctl
  at 93 °C (48 W, everything else idle), so the guard stopped and continued the run every few seconds, and
  run outside the group its first load drew 59 W while the stopped run's Tctl peaked at 98.2 °C and the guard
  ended it.
- **Any other heavy command**, a diagnostic for instance, runs under
  `resource_guard.py run [--cpu-lock] [--log FILE] -- COMMAND ...` (exit status 75 when the guard
  ended it). In Python, use `with resource_guard.cpu_lock(), resource_guard.Guard(log_path=...) as
  guard:`, then `guard.wait_cool()` and `guard.check()` between chunks.
- `resource_guard.py status` prints one sample, the settings, the holder of the CPU lock and whether
  the kernel log is readable (`kernel_log`).

**Fit memory** (`fit_hub.memory_estimate`): what a fit takes off MemAvailable at its peak. That is its
anonymous memory (its RSS is mostly memory-mapped captures, which the kernel can drop) and, for a GPU fit, the fit
backend's device memory: on this APU (2 GiB of VRAM carved out) torch allocates GTT, system RAM that MemAvailable
no longer counts and that no process's RssAnon shows (a 4 GiB torch allocation added nothing to RssAnon, 4.0 GiB
to GTT and took 4.01 GiB off MemAvailable). Until October 9 the estimates were calibrated on anonymous peaks
only, so the GPU fits' real peaks (anonymous + torch's reserved peak) exceeded the onboarding estimate in 7 of 9
onboardings by 1.0-3.9 GiB. The guard and campaign.py now record a command's GPU memory as the rise of the GPU's
GTT use since it started (`max_gpu_gib`) and the largest anonymous + GPU sum of one sample (`max_total_gib`, in
`state.json`, the guard summaries and the report's "Peak memory").

- **Onboarding.** This is campaign.py's model (`fit_hub_extra`): 5.5 GiB, plus 3.5 GiB on the GPU, plus the base
  hub, plus, for one kind, 28·W1² + 10·D·C + 12·chunk·C + 12·dev·pivots·D bytes. W1 is the new model's summed
  stage-1 ranks, C its K/V columns, D the hub dimension and chunk the rows fit_hub streams at a time (4,096). It is
  calibrated on the twelve onboardings: the three CPU fits' anonymous peaks (LFM2-2.6B 11.0 GiB, DeepSeek-V2-Lite
  16.7, Qwen3.6-35B-A3B 13.0) and the nine GPU fits' anonymous + reserved peaks (13.5-16.5 GiB for the six with at
  most 8,192 columns, GLM-4.7-Flash 31.3, Gemma-4-26B-A4B 24.2, Gemma-4-31B 56.4). At each onboarding's own dev
  spans the estimate lies 0.5-3.8 GiB above each (Kimi-Linear closest: 16.1 against 15.6); campaign.py's lock
  counts the base hub's 17,355 dev spans, which bound every model's (Gemma-4-31B: 60.3 GiB). Until October 9 it
  also counted 4·dev·pivots·C for dev rows tiled over the pivots, which the fit has streamed since October 6: that
  asked 98 GiB for Gemma-4-31B (225,280 columns), more than any lock could grant. A re-encode (`campaign.py
  reencode`) asks for the bundle, 5.5 + 2 GiB and the 3.5 GiB on the GPU (Granite-4.0-H-Tiny's: 9.6 GiB anonymous
  and 4.1 GiB reserved on the GPU, 4.7 GiB bundle).
- **Joint fit.** The GCCA holds the joint covariance and its whitened copy (2·P² float64, P the views' summed
  dimensions) plus what `fit_backend.top_eig` allocates on the path it takes (`fit_hub.eigh_bytes`, measured):
  with scipy, as in `~/venvs/rocm` where fits run, the float32 subset solver adds 1.3·P² float64 and the
  eigenvectors; numpy's float64 `eigh` would add 4.1·P² (the estimate until October 7 counted that path, 6.1·P²,
  37.4 GiB at P = 28,672). Seven views of 4,096 give P = 28,672: the GCCA phase is 21.5 GiB, so the decoder phase
  (26.6 GiB, growing with the dev spans) is the anonymous peak, with the fitted K adapters (3.1 GiB), the
  transforms and the base on top: 35.6 GiB of anonymous memory for the joint seven-model refit, which peaked at
  28.3 GiB. On the GPU the device adds 3.5 GiB and 18 bytes per P² cell (the joint covariance's float64 sums and
  their tiles there), calibrated on that refit, whose torch peak was 16.6 GiB reserved: 53 GiB in all for it, which
  the guard starts at MemAvailable ≥ 78 GiB. Without scipy (the experiment venv) the "float32" path adds 5.7·P²
  (measured: numpy.linalg computes a float32 matrix in float64, so the float32 copy and its promotion come on top
  of `eigh`, and it is slower, 50 s against 25 s at 8,192): run large fits with `~/venvs/rocm/bin/python`.
- **R² of an assembled hub** (`assemble --evaluate`, `campaign.py evaluate-onboarded`): the lock asks for 2.3
  times the hub's files. hub6-20261008's (21.6 GiB of files, fifteen models, on the GPU) peaked at 36.2 GiB
  anonymous and 12.2 GiB on the GPU, and MemAvailable fell by 46.9 GiB; hub6-20261009's, with the guard's GTT
  reading, 36.1 and 12.5 GiB, 46.6 GiB at the same time. Until October 9 the lock asked for the files alone.
- **Train rows.** No phase holds the train rows: every statistic is a float64 sum over 4,096-row
  chunks. An estimate proportional to models × train rows × view_dim cannot fit the recorded peaks.
  DeepSeek-V2-Lite had the fewest train spans (87,712, against 116,865 for LFM2 and 185,672 for
  Qwen3.6) and the largest peak.
- **Calibration.** The eigendecompositions are measured (`test_resource_guard.py` checks both the
  numpy and the scipy subset path, and the onboarding estimate against the twelve recorded peaks); the decoder
  phase of a joint fit has one recorded peak (RESULTS.md section 9).

## Fit placement (fit_backend.py)

Which parts of a hub fit run on the CPU and which on the GPU was decided by measurement on October 6,
2026. The runs used the thermal guard, 8 BLAS threads and one fit at a time. The raw results are in
`evidence/universal-kv/fitplace/runs/` and the scripts that made them in
`evidence/universal-kv/fitplace/scripts/` (`profile_fit.py` times fit_hub.main by category,
`analyze.py` takes the guard's pauses out and integrates its PPT samples).

**What a fit does.** Every fit streams 4,096-span chunks of capture rows through three kinds of work:

- projections: rows times a fixed weight (stage-1 bases, stage-2 bases, encoders, decoders);
- accumulations: float64 sums of each chunk's Gram `x'x` (`hub.Moments`) and cross products `a'b`;
- solves: `eigh`, `solve` and `svd` of the summed float64 matrices.

**Where it runs now.**

| Work | `--fit-device gpu` | `--fit-device cpu` |
| --- | --- | --- |
| projections | hipBLASLt float16 products (float32 accumulation, float16 out), rows and weight columns scaled by powers of two | numpy float32 |
| Grams and cross products | a Triton kernel: float16 operands, float32 accumulation, the tile unscaled and added to the float64 sum on the device | numpy float32, summed in float64 |
| eigh, solve, svd | float64 LAPACK on the CPU | the same |
| the top k eigenvectors of an n-square matrix, n ≥ 4,096 and k ≤ n/2 (a wide model's stage-2 PCA, the GCCA) | LAPACK in float32 on the CPU (`KVPARK_FIT_LARGE_EIGH=float64` keeps float64), scipy's subset solver (`dsyevr`) when scipy imports (`~/venvs/rocm`), else numpy | the same |

`auto`, the default, takes the GPU when torch sees one. The experiment venv has no torch, so run fits
with `~/venvs/rocm/bin/python` (it has numpy, scipy, torch and Triton). The manifest's settings record
the device (`fit_device`) and the precision of the large eigensolves (`large_eigh`).

Kernel measurements (one launch each, synchronized; under the guard and its GPU-hang stop):

| Product (4,096 rows) | Output | One launch | TFLOPS | Error vs float64 |
| --- | --- | --- | --- | --- |
| Gram, Triton float16 → float32 (BM 128, BN 64, BK 32), 4,096 columns | 4,096² | 0.009 s | 15.8 | 5e-5 |
| the same, 8,192 columns | 8,192² | 0.056 s | 9.8 | 6e-5 |
| Gram, hipBLASLt float16 → float16 (scaled) | 4,096² | 0.008 s | 18.0 | 5e-4 |
| Gram, torch.mm float16 → float32 (`out_dtype`, rocBLAS) | 4,096² | 0.025 s | 5.4 | 4e-5 |
| Gram, the same through hipBLASLt | 4,096² | 0.044 s | 3.1 | 5e-5 |
| Gram, bfloat16 → bfloat16 | 4,096² | 0.008 s | 16.9 | 4e-3 |
| Gram, torch float32 (the earlier run that hung the desktop) | 8,192² | 8.2 s | 0.07 | — |
| projection, hipBLASLt float16 → float16 | [4,096 × 18,432] · [18,432 × 4,096] | 0.028 s | 22.4 | 5e-4 |
| Gram, numpy float32, 8 threads, under the guard (earlier run) | 8,192² | 1.6 s | 0.34 | 4e-7 |
| the same | 16,384² | 20.5 s | 0.11 | |

Never use torch's float32 products or float16 products with float32 output on this GPU. They run at
0.07 to 5 TFLOPS, and the float32 path's 8 to 33 s kernels caused the GPU reset of October 6.

**Accuracy.** The inputs are float16, so the products sit at the float16 rounding floor. A Gram has a
relative error of about 5e-5 per entry against float64, against 4e-7 for numpy's float32. These
errors average out over the chunks and are far below the fits' statistical noise.

**The GPU RULE in the code.** Every GPU product is tiled so that one launch does at most the
calibrated rate times 0.04 s, and never more than 0.3 TFLOP. `TorchBackend.calibrate` measures the
rate at start, small sizes first. In practice that is 0.01 to 0.03 s per launch. Accumulation tiles
are 4,096 square because 8,192-square tiles run at half the rate. Inputs are scaled by powers of two
to RMS 1 per column (Grams) or to a norm of about 32 (projections), so no float16 value overflows.
The scales are computed in two steps, first to a maximum of 1: the GPU flushes float32 subnormals to
zero, and K rows have columns of 1e-26 whose squares would vanish and blow the scale up. A GPU fit
holds the GPU lock and runs under the guard's GPU-hang stop.

**A busy thread in every torch-ROCm process.** From its first GPU kernel on, a torch process on
this machine keeps one runtime thread busy at 100% of a core, also while the process sleeps. The
thread is created during `torch.cuda` initialization (`spinner.py`). None of these stopped it:
`OMP_WAIT_POLICY=PASSIVE`, `HSA_ENABLE_INTERRUPT=1`, `AMD_DIRECT_DISPATCH=0`,
`ROC_ACTIVE_WAIT_TIMEOUT=0`, `HIP_LAUNCH_BLOCKING=1`, `HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=1`,
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`. It used 152 of the 584 CPU-seconds in the GPU
onboarding below, and 996 of the 7,015 in the GPU joint fit. The engine daemon on the torch backend has the
same thread for as long as it stays resident, idle and after its idle unload too (`top -H`, October 9: one
thread at 100%, Tctl 93 °C at 48 W with the loaded llama-servers idle); the ggml Vulkan engine has none.

**Algebra that helps both devices.** These changes give the same numbers to float rounding
(`test_fit_backend`, `CpuBackendTest`):

- A decoder's cross moments with the pooled estimates of P sources used P copies of the target's
  rows (`np.tile`). They are now the cross moments of the sources' summed estimates with the rows,
  which costs P times less.
- Each source's estimates get their own Moments, and every target pools them by the parallel-axis
  theorem. A joint fit no longer needs M - 1 Grams per target.
- The dev R² of each ridge strength used to predict every dev row in float64 for every alpha. That
  was about 80 TFLOP for Qwen3-4B's K decoders. It now comes in closed form from the dev Gram and
  cross moments in the Gram's eigenbasis.
- `evaluate` reads each test chunk once and encodes each source once for all of its targets.

**Where the time goes.** These are profiles of fits run by the scratch profiler: exclusive time per
category, pauses taken out. For the onboarding of LFM2-2.6B into hub4-d4096 (K and V, four pivots,
116,865 train spans, `--evaluate new`):

| Running seconds | CPU, old code | CPU, new code | GPU |
| --- | --- | --- | --- |
| capture reads, and waits for them | 94 | 10 | 3 |
| projections | 109 | 88 | 11 |
| Gram and cross-product accumulation (GPU: the GPU's work and the waits for it) | 64 | 125 | 67 |
| fit-loop numpy and Python (row tiles, the dev sweep's predictions, means) | 85 | 8 | 10 |
| eigh | 31 | 31 | 32 |
| solve, decoder solve | 8 | 17 | 17 |
| load and save bundles | 9 | 9 | 9 |
| total | 400 | 289 | 151 |

The old code's reads look slower partly because it ran first, on a colder page cache (the five
models' captures exceed the 123 GiB of RAM). Its `evaluate` also read every pair's rows again, and
it read pivots one layer at a time.

For a reduced joint fit on the GPU (Qwen3-4B, Ornith and LFM2-2.6B; view dimension 4,096, hub
dimension 2,048, 30,000 fit rows, 119,004 train spans) with float64 eigendecompositions, the 994
running seconds break down as:

| Running seconds, K and V | GPU |
| --- | --- |
| eigh | 821 |
| Gram and cross-product accumulation (309 TFLOP on the GPU, and the waits for it) | 109 |
| solve, decoder solve | 23 |
| fit-loop numpy and Python | 21 |
| projections | 14 |
| capture reads, bundles | 2 |

Two calls account for most of the eigh time:

- Qwen3-4B's stage-2 PCA, the top 4,096 eigenvectors of an 18,432-square matrix: about 280 s per
  kind with scipy's subset solver;
- the GCCA, the top 2,048 of a 12,288-square matrix: about 90 s per kind.

LAPACK on 8 heat-limited threads runs at about 0.1 to 0.2 TFLOPS in float64. A seven-model refit's GCCA
is 28,672 square, about 12 times this one's eigendecomposition: some 20 minutes per kind in float64.
Once the GPU does the products, these eigendecompositions are what a joint fit costs. In float32 the
same fit spent 568 s instead of 821 s in eigh.

The eigendecompositions stay on the CPU. The GPU's float64 rate is a small fraction of its float32
rate, and torch's float32 products, which a GPU eigensolver would use, run at 0.07 TFLOPS here and
broke the GPU RULE. Two changes make them cheaper. The first is subset solvers for the top
eigenpairs. The second is float32 for the large ones. Both were measured with 8 threads under the
guard, for the top quarter of the eigenpairs (`bench_eigh.py`, `bench_eigh32.py`):

| n | numpy `eigh`, all, float64 | scipy subset (dsyevr), float64 | scipy subset, float32 | scipy dsyevd, all, float64 |
| --- | --- | --- | --- | --- |
| 4,096 | 6.8 s, 37 CPU-s | 4.6 s, 18 CPU-s | | 6.8 s, 42 CPU-s |
| 8,192 | 57.7 s, 267 CPU-s, 3.1 kJ | 40.1 s, 175 CPU-s, 1.0 kJ | 22.3 s, 94 CPU-s, 0.6 kJ | 52.4 s, 317 CPU-s |

The float64 subset solver's eigenvalues agree with numpy's to 2e-15, and its eigenvectors agree up to
sign. In float32 the eigenvalues are within 1.2e-7 and the subspace within a cosine of 6e-6. The
subset solver is used from n = 4,096 on, because at 2,048 the full eigh is faster. Float32 is used for
subset problems only: the full eigendecompositions of the GCCA's per-view whitening, the decoders and
the onboarding encoders stay float64. The reduced joint fit below validates the float32 default.

**Before and after.**

Onboarding LFM2-2.6B into hub4-d4096 (both kinds, `--evaluate new`). "CPU, old" is the code before
this change (8 threads); the pilot of October 5 used 16 threads and gave the same numbers:

| | CPU, old | CPU, new | GPU, new |
| --- | --- | --- | --- |
| wall time (with the guard's pauses) | 552 s | 386 s | 302 s |
| running time (pauses taken out) | 400 s | 289 s | 153 s |
| process CPU time | 2,152 s | 1,661 s | 584 s |
| guard pauses | 57, 152 s | 46, 97 s | 74, 150 s |
| package power while running (PPT, mean) | 53.8 W | 53.0 W | 50.4 W |
| energy while running (PPT × running time) | 21.1 kJ | 15.2 kJ | 7.7 kJ |
| energy over the wall time | 24.5 kJ | 16.8 kJ | 9.8 kJ |
| peak Tctl (a 2 s sample) | 95.9 °C | 96.6 °C | 95.3 °C |
| device memory (torch peak) | | | 3.9 GiB allocated, 4.5 GiB reserved |
| held-out R², largest difference from the pilot (9 pairs, K and V) | 0 | 0 | 8e-6 |
| alphas | the pilot's | the pilot's | the pilot's |
| the new model's decoders / encoder action, relative to the pilot | 0 / 0 | 1e-6 / 0 | 8e-4 / 1e-3 |

A first GPU run gave the same running time (151 s), CPU time (579 s) and results. Its wall time was
710 s: other jobs kept the package at 80 to 86 °C while the fit was stopped, so three pauses waited
55 to 184 s for 80 °C.

Reduced joint fit (Qwen3-4B, Ornith, LFM2-2.6B; view dimension 4,096, hub dimension 2,048):

| | float64 eigh | float32 eigh (the default) |
| --- | --- | --- |
| wall time (with the guard's pauses) | 1,837 s | 1,245 s |
| running time | 994 s | 716 s |
| of which eigh | 821 s | 568 s |
| process CPU time | 7,015 s | 5,205 s |
| guard pauses | 408, 843 s | 257, 529 s |
| package power while running (mean) | 57.0 W | 61.1 W |
| energy while running | 55.6 kJ | 43.0 kJ |
| energy over the wall time | 68.6 kJ | 50.9 kJ |
| peak Tctl | 97.0 °C | 96.8 °C |
| GPU products | 309 TFLOP, 14,572 launches, the longest timed one 0.01 s | the same |
| device memory (torch peak) | | 6.7 GiB allocated, 8.7 GiB reserved |
| held-out R², largest difference from the float64 fit (9 pairs) | | K 2.6e-5, V 1.6e-5 |
| translations, largest difference from the float64 fit (2,048 test spans, per target layer, relative to its spread) | | K 1.3%, V 1.1% |

The translations move by up to 1.3% of a layer's spread while R² stays put. Float32 resolves the
stage-2 PCA's smallest kept components, near its 4,096th, less sharply, and the hub follows. Those
components carry almost no variance, which is why R² does not move. To compare against
fits made before October 6 to the last digit, set `KVPARK_FIT_LARGE_EIGH=float64`.

The K kind alone, against the old CPU code:

| K only | CPU, old | GPU, float64 eigh | GPU, float32 eigh |
| --- | --- | --- | --- |
| wall time (with the guard's pauses) | 895 s | 863 s | 617 s |
| running time | 697 s | 487 s | 354 s |
| of which eigh | 376 s | 404 s | 285 s |
| process CPU time | 4,383 s | 3,451 s | 2,620 s |
| guard pauses | 97, 199 s | 182, 376 s | 127, 263 s |
| energy while running | 40.4 kJ | 28.1 kJ | 21.5 kJ |
| energy over the wall time | 43.8 kJ | 34.0 kJ | 25.4 kJ |
| peak Tctl | 96.8 °C | 96.5 °C | 95.5 °C |
| held-out R², largest difference from the old CPU fit (9 pairs) | | 7e-6 | 1.9e-5 |

The old code spent its other 321 running seconds on the products and the dev sweep's float64
predictions. In the GPU runs those take about 80 s. A first run of the old code gave 947 s, 708 s and
47.3 kJ, with the same results. The old code's float64 eigh (numpy's full
`eigh`, the experiment venv's OpenBLAS 0.3.34) was no slower than the GPU runs' float64 subset solver
(scipy, `~/venvs/rocm`'s OpenBLAS 0.3.31). During the GPU runs the package also ran hotter while
running (Tctl 91 against 85 °C on average), with the busy runtime thread on top, so the bench's
1.4-fold advantage did not show. Float32 is what cut the eigendecompositions.

Every run reached a Tctl peak of 95 to 97 °C. On this machine Tctl spikes within one 2 s sample,
whatever runs, and the guard stops the job at that first sample, so the peak does not tell the runs
apart. No GPU run logged a ring timeout or reset: the guard's GPU-hang stop followed the kernel log
through every run, and `journalctl -k` showed nothing afterwards either.

**Choice.** `--fit-device auto`, the default, takes the GPU whenever torch sees one. For the
onboarding it takes 1.8 times less wall time, 2.6 times less running time, 2.5 times less energy and
3.7 times less CPU time than the old code, and the numbers are the same: held-out R² within 1e-5.
For a joint fit the GPU leaves only the eigendecompositions, and float32 for the large subset
problems cuts those by 30% at R² within 3e-5. For the reduced joint fit's K kind that is 1.5 times
less wall time, 2 times less running time and 1.7 times less energy than the old code. The CPU path
stays as the fallback, for the experiment venv and machines without a GPU. It got 1.4 times faster
and uses 1.4 times less energy than the old code, from the algebra alone, at identical results.

A joint fit's next lever is a top-k eigensolver whose products run on the GPU, for example subspace
iteration with split float16 products for float32 accuracy, followed by Rayleigh-Ritz. It would need
the same validation. Caching each model's stage 1 and 2 across refits would also help, since they
depend only on its own captures.

**How Track A runs fits.**

```bash
cd experiments/universal_kv
../linear_transfer/.venv/bin/python campaign.py exec --memory-bytes N --cpu-lock --guard-log OUT.guard.jsonl -- \
    ~/venvs/rocm/bin/python fit_hub.py --base ... --model ... --output OUT --fit-device gpu
```

`campaign.py exec` takes the GPU/memory lock and then the CPU lock, and watches the job, including
for GPU hangs. fit_hub.py finds the GPU lock its parent passed down and inherits the CPU lock. With
`--fit-device gpu` a missing GPU is an error. `auto` falls back to the CPU instead. A fit started by
hand without campaign.py takes both locks itself. For a joint refit, keep the outer Watch: its largest
eigendecompositions run for minutes in one LAPACK call, and only the Watch can pause them.
`campaign.py run` still starts fit_hub with the experiment venv, so its fits run on the CPU.

## Onboarding campaign (campaign.py)

One model per invocation, every stage resumable; the state, logs and outputs live in
`evidence/universal-kv/campaign/NAME/` (gitignored). NAME is the model's hub name and, unless
`--catalog-id` names another entry, its id in `evidence/models/catalog.json`, which gives the GGUF,
the architecture and the smoke prefill speed (`--gguf` and `--capture-tps` for a model the catalog
lacks; a refused architecture, deepseek4, stops `plan` and `run` with one line):

```bash
cd experiments/universal_kv
../linear_transfer/.venv/bin/python campaign.py plan --name lfm2-2.6b       # what a run would do; loads no model
../linear_transfer/.venv/bin/python campaign.py run --name lfm2-2.6b        # every stage not done yet
../linear_transfer/.venv/bin/python campaign.py run --name lfm2-2.6b --stages render,capture
../linear_transfer/.venv/bin/python campaign.py run --name lfm2-2.6b --redo onboard   # and every stage after it
../linear_transfer/.venv/bin/python campaign.py status                      # and assembled hubs it outdated
../linear_transfer/.venv/bin/python campaign.py run --name deepseek-v2-lite --catalog-id deepseek-v2-lite-chat
```

| Stage | What it does | Output |
| --- | --- | --- |
| render | the hub corpus (corpus v2 + v3-train, the 248 conversations of hub4-d4096) with the model's own template (chat.py corpus mode); chat.py against llama-server's `/apply-template` on a tokenizer twin (template_check server mode, ports 18690-18699; `KVPARK_TWIN_PORT` moves the ten) for two conversations per source, as prompts under every option set and as whole conversations the way the records render them (no generation prompt, `--no-prefill-assistant`); the probe tokenizes every record | `render/` (`parity-*.json`, `records-differing.json`) |
| capture | the forecast (`capture-plan.json`), then `collect_v2.py` chunk by chunk: the dev and test documents, then train chunks while the measured speed and size allow (Vulkan probe; `--mode states` on the CPU probe for kimi-k3, bailingmoe3, hy_v4, hy_v3 and glm5-next; collect on the CPU probe for gemma4, whose Vulkan capture run differs from a plain run: `CPU_CAPTURE_ARCHITECTURES`) | `capture/chunk-NN/`, `capture-docs.jsonl` |
| onboard | `fit_hub.py --base hub4-d4096 --link-base --evaluate new --onboard-encoder reverse --fit-device auto` with all four base models as pivots (`--source-global` for a sliding-window model, `--indexer-ridge` for a DSA model), run with `~/venvs/rocm/bin/python` when it exists (the fit's products on the GPU; `$KVPARK_FIT_PYTHON` names another, else the experiment venv fits on the CPU), then `convert_hub.py --reuse hub4-d4096-v2`; the sha256 of the model's own files and its encoder go to `state.json` | `hub/` (held-out R² per kind against each pivot in `hub/evaluation.json`), `hub-v2/` |
| tables | `token_tables.py --only NAME` on the onboarding's spans | `token-tables/NAME.npz` |
| recurrent | hybrids whose recurrent cell holds a state (SSM, delta rule, WKV, lightning): `r0_experiments.py calibrate` on the training conversations of corpus v2 and v3-train. Skipped, with the reason in `state.json`, for a cell of convolution taps only (lfm2: the last two inputs, which the native tail replaces after two tokens; the zero cell is restored) | `recurrent/recurrent-mean.npz` |
| evaluate | `pair_eval.py` Qwen3.8 → NAME and NAME → Qwen3.8 through `hub-v2/` on RESULTS.md section 2's eight held-out chats (`--ids`; where the target renders fewer than six of them 4,096-16,384 tokens before the final message, pair_eval's own pick from the same corpus and sources), with the token tables and, into a hybrid, its recurrent cell (the mean, or zero for a conv-only cell). A direction whose target renders some of the eight too short also runs the longer set (`-long`: every chat it renders 6,144-16,384 tokens long, at least 4,096 translated). Then Qwen3-4B → Qwen3.8 with the settings of NAME → Qwen3.8, once for all campaigns (`campaign/_baseline/`). When the render stage found that the model's server renders conversations other than its template's defaults (Qwen3.6's keeps earlier reasoning), both directions run again with both models rendered as their servers render them (`pair_eval.py --server-render`, `-server`): what a live switch holds | `eval/SOURCE-to-TARGET[-server][-long]/report.json` and `.command.json` |
| report | wall time, the command time of every attempt since the stage last started over (log.jsonl: a resumed stage keeps finished chunks and evaluations), lock waits, disk and peak RSS per stage; the capture; R²; per run ΔNLL / KL / top-1 with document-bootstrap 95% CIs for every condition (hub, floor, hub_prefix, floor_prefix, hub_table_prefix); per direction (both sets pooled) paired differences with CIs, the share of the floor's KL removed and how much of the floor one chat carries; NAME → Qwen3.8 against Qwen3-4B → Qwen3.8 chat by chat; the `-server` runs against the templates' rendering chat by chat; the `kvpark bridge add` settings (with `--draft-window` from the sliding window of a DFlash draft the catalog lists) | `report.json`, `report.md` |

`state.json` records per stage its status, wall time, lock waits, disk (`bytes` the stage's own
files, `linked_bytes` hard links to the base hub), peak RSS, the commands it ran and its result;
every command entry carries git HEAD, the files changed under `experiments/` and `src/` and the
sha256 of the scripts and the probe it ran. `commands.sh` holds every command as it ran (a
model-loading one through `campaign.py exec --memory-bytes N --`, which takes the lock and checks
memory exactly as the run does), `logs/` their output and `log.jsonl` the events, lock and memory
waits included.

- **Template parity.** Every sampled whole conversation must match the server's bytes with no
  server error, and every prompt case too except the server's refusals of option sets named with
  `--parity-tolerate` (none by default); at least one prompt must match, so a twin that refuses
  everything fails. A record whose bytes are not the server's own rendering of the conversation is
  accepted and recorded in `render/records-differing.json` with the server context that reproduces
  the server's bytes, the variables in which it differs and the first difference: Qwen3.6's server
  keeps earlier reasoning (`preserve_reasoning`, the server's default), the corpus record drops it.
  The capture then holds context a live server would not; the onboarding fits per-token maps on the
  spans both renderings share, so that only removes those reasoning tokens from the fit, and the
  bridge's engine renders a live conversation the way the server does. The evaluation of such a
  model runs both ways: with the templates' defaults, comparable with RESULTS.md section 2 and the
  other campaigns, and as the servers render (`-server`), which is what a live switch holds.
- **Capture.** The forecast takes every dev and test document (the onboarding's alpha choice and
  held-out R², the same documents for every model), then training documents round-robin over the
  corpus sources within 60 GB (float32 K and V of every attention layer, plus the DSA indexer
  keys) and one hour (`--budget-gb`, `--budget-minutes`) at 0.4 of the catalog's smoke prefill
  speed (Ornith ran at 0.49, Qwen3-4B at 0.33, LFM2 at 0.67-0.72). The capture itself runs in
  chunks: the held-out documents, then train chunks of a sixth of the forecast budget (at least
  16,384 tokens) in the forecast's order, each started only when the speed measured so far (rows
  per probe second, lock waits excluded, model loads included) and the bytes so far say it fits; a
  `--mode states` chunk counts twice, since its state files and its rows exist together. Every
  chunk runs with the time the capture has left of 1.5 times its budget as a hard timeout: a train
  chunk that times out ends the capture there, the held-out chunk fails the stage.
- **GPU/memory lock.** Every command that loads a model or a hub (capture, onboard: the base hub,
  tables: the captures, recurrent, evaluate, `assemble --evaluate`) holds an exclusive `flock` on
  `/tmp/kvpark-campaign-gpu.lock` and starts only once MemAvailable is at least what it loads plus
  25 GiB (else it releases the lock, waits 60 s and retries, logging each wait). What it loads:
  the model; for pair_eval the larger model, the two hub sides it reads and 4 GiB for the probe's buffers and its
  states (Nemotron-3.5 → Qwen3.8 took 34.0 GiB, 30.5 of them on the GPU, where the model and sides came to 33.2); for fit_hub the
  base hub plus an estimate that grows with the new model's K/V width and the dev spans
  (`fit_hub_extra`: its stage-1 covariance, its decoder's statistics and chunk rows, per dev span
  and pivot what does not grow with the model, counted at the base hub's 17,355 dev spans, which
  bound every model's, and on the GPU the device's memory; with the base hub, for a GPU fit, 17.5 GiB
  for LFM2-2.6B, 34.1 GiB for GLM-4.7-Flash, measured 31.3 with the GPU's, and 60.3 GiB for
  Gemma-4-31B, measured 56.4: "Fit memory" above). The command
  inherits the locked descriptor, so the lock stays held as long as it runs even if campaign.py is
  killed; it runs in its own process group, which campaign.py stops on SIGINT, SIGTERM, SIGHUP or
  a timeout. render loads no model. The CPU-heavy commands also hold the CPU lock, after this one,
  and every command runs under the resource guard ([Running heavy jobs safely](#running-heavy-jobs-safely)).
- **Resume.** A stage runs when it is not done; `--redo` runs a stage again with every stage that
  depends on it (they are marked stale first). A failed or interrupted stage is cleaned and run
  again, but each published capture chunk is adopted when its manifest names the same documents,
  model, probe build, collect_v2.py, capture spec and settings, and an evaluation run is kept when
  its `.command.json` holds the same command. One run per model at a time.
- **Recommended settings.** Into the Qwen family the native prompt start is on (RESULTS.md section
  2). Into another target it is on when the paired 95% CI of hub_prefix − hub KL lies below 0,
  off when it lies above, and on when it spans 0: the native prefix is the target's own values
  from a 14-30 token prefill, and the attention-sink collapse it prevents (Qwen3-4B, KL 0.77
  without it) is far larger than its cost where it hurts (Muse, +0.05 nats). A model whose catalog
  entry lists a DFlash draft gets `--draft-window` from the draft's
  `dflash.attention.sliding_window` (Qwen3.6-35B-A3B's is 4,096, the bridge assumes 2,048): with
  the draft on its server a switch keeps a native tail of the window minus one token, so that the
  draft sees native context only.
- **Held-out chats: a selection set now.** Every evaluation uses the long-chat corpus's test split
  (`corpus-long-chat`), kept out of the hub corpus by id. By October 9 all 18 of its chat conversations had been
  scored, the eight fixed ones 48-88 times each, and the `reverse` encoder, its variance cap and the native-prefix
  rule were chosen on those eight: the campaigns' numbers are selection-set results (RESULTS.md section 9,
  "Selection set"), and those choices are frozen. A fresh set for the next choice is built and kept unscored:
  `evidence/linear-transfer/corpus-heldout-20261009/`, 30 multi-turn chats (24 primary, 6 reserve) of 6,631-14,746
  Muse tokens before the final message, drawn by `build_corpus.py` from the local SmolTalk2 file of
  `multi_turn_reasoning_if_think` (no download), with every earlier corpus excluded by id and by user turns (no
  conversation in common with any of them). Point `pair_eval.py --corpus` at it only for a decision not yet
  taken.
- **Floors.** Into Qwen3.8 the campaign starts the recurrent cell from Qwen3.8's mean on the new
  pin; RESULTS.md section 2 started it from zero on the old pin, so those floors and recovered
  shares are not comparable with a campaign's. The Qwen3-4B → Qwen3.8 baseline runs under the
  campaign's settings: same chats, same floors, only the source differs.
- **How the hub grows.** Every model is onboarded against the same base, hub4-d4096 with its four
  models as pivots, so its adapters depend on the base and its own captures only, and no
  onboarding refits anything. A model's `hub/` and `hub-v2/` are the base's files (read-only hard
  links, checked against the arrays fit_hub holds before they are linked) plus its own. One hub
  for the bridge is assembled from the base and any onboarded models, in both formats, without
  refitting or copying (`hub.merge_bundles`: hard links on the members' filesystem, in any order;
  models added twice, bundles not grown from the base, a member being onboarded or whose bundles
  no longer hold its recorded files are refused):

  ```bash
  ../linear_transfer/.venv/bin/python campaign.py assemble --output ../../evidence/universal-kv/hub5-YYYYMMDD [--evaluate] lfm2-2.6b [NAME ...]
  ```

  gives `hub5-YYYYMMDD/` (format 1) and `hub5-YYYYMMDD-v2/` (format 2, for `kvpark bridge engine
  --hub`), recorded in each model's `assembled.json`; `--evaluate` adds held-out R² between the
  onboarded models (`evaluation-onboarded.json`). Adding a model later assembles a new directory
  with one more name; the bridge is pointed at it, and a hub it reads is never modified. After
  `--redo onboard` or `reencode` an assembled hub keeps the old adapters, and `campaign.py status`
  flags it.
- **The onboarded encoder.** An onboarded model's decoder is fitted like a jointly fitted one's
  (from the pooled pivot estimates), but its encoder cannot be: the hub latent was fitted without it.
  Since October 6 it is `reverse` (`ONBOARD_ENCODER`, `hub.DEFAULT_ONBOARD_ENCODER`): the regression
  onto the pivots' mean estimate, rescaled per dimension to be unbiased given it instead of shrunk
  toward it. The pilots were onboarded with `ridge`, whose encodings spread half as far as a jointly
  fitted model's; with `reverse` DeepSeek-V2-Lite, LFM2 and Qwen3.6 remove 17%, 24% and 41% of the
  floor's KL into Qwen3.8 instead of 3%, 4% and 11% (RESULTS.md section 9; Qwen3-4B, fitted jointly:
  30%); on the corpus's eleven chats in range 20%, 25% and 40% against ridge's 5%, 6% and 12%. Rescaling K
  alone gives about 70% of that gain and V the rest; CCA onboarding, at unit gain or scaled like the base
  views' estimates, did worse in both runs. They were re-encoded with it on October 7. Its scale is about
  1 / R² per hub dimension, so a dimension the new view barely predicts would be inflated without bound
  (the layer-subset encoder of a sliding-window model is the likely case); each dimension's variance is
  therefore capped at 4 times the pooled pivot estimates' (`hub.ONBOARD_VARIANCE_CAP`, `fit_hub.py
  --onboard-variance-cap`, 0 for none),
  the largest the end-to-end comparison covered (the pilots reached at most 3.75 on their train spans;
  `reverse_z`, median 8-10 and up to 41, lost on LFM2). The cap bound on none of the first four models
  onboarded, nor on Falcon-H1, Kimi-Linear or the two Gemma 4 models. It bound on Ling-3.0-tiny (1,509 of
  4,096 K dimensions, smallest per-dimension R² 0.061), which still translates well, and on three Mamba hybrids
  where a dimension's R² was also below 0.05: Granite-4.0-H-Tiny (2,561; R² 0.044), Nemotron-3.5 (2,915; 0.028)
  and Jamba (3,926; 0.003), whose `reverse` encoders did no good or harm
  ([hub6](#hub6-and-onboarding-results-by-family)). The fit logs each encoder's smallest per-dimension R², its
  variance ratios and the dimensions capped (stage `onboard-encoder`), and warns (stage `onboard-warning`,
  stderr, the campaign report) when the cap binds or a dimension's R² is below 0.05: when both fire on about
  half of the dimensions or more, evaluate the direction out of the model before using its `reverse` encoder,
  and re-encode it with `ridge` where it does not help (Granite and Nemotron, October 9). Another method for
  models already onboarded:

  ```bash
  ../linear_transfer/.venv/bin/python campaign.py reencode [--encoder METHOD] NAME [NAME ...]
  ../linear_transfer/.venv/bin/python campaign.py run --name NAME      # evaluates the directions out of NAME again, reports
  ```

  `reencode` fits only the encoder again (`fit_hub.py --reencode`: one pass over the train spans, a few
  minutes on the GPU); the transforms and decoders stay bit-identical, so the evaluation runs into the
  model are kept and only those out of it run again. The previous bundles, fit log and runs move to
  `history/onboard-<encoder>-<time>/`.
- **Stopped evaluations resume.** On this machine the guard ends long GPU evaluations now and then
  (other work on the shared package, outside the locks, can push Tctl past 95 °C while the job is
  stopped). An unfinished evaluation of the same command continues: `pair_eval.py --resume` keeps its
  saved states and every document it has scored (one probe run per document), and translates again only
  the documents still to score. A document's scores are kept only when its translations would be built from
  the same inputs (`scores/<document>.built.json`: the sha256 of every hub file of the source and the
  target in `--hub` and each `--extra-hub`, the token table, the recurrent options and the scoring input),
  so an `--extra-hub` label pointing at a bundle re-encoded since is scored again. The finished runs of the
  pilots match pair_eval's single-run scores to the last digit.

`test_campaign.py` covers the planning and resume logic, the capture forecast and its chunks,
the parity rules, the lock and the commands under it, the statistics and the reports without a
model; `test_hub.py` the linked onboarding bundles (read-only, the array check, no cross-filesystem
link), `--reuse` and the assembly.

Pilot, October 5, 2026 (run again after the review of the first run, whose report is kept in
`campaign/lfm2-2.6b/history/pilot1-20261005/`): LFM2-2.6B (`lfm2`, 8 attention layers among 30,
short-convolution recurrent rows) went through every stage in 71 minutes of wall time, 64 of them
the evaluations (21 of those the Qwen3-4B → Qwen3.8 baseline, which later campaigns reuse), and 7.2
GB of new disk (`evidence/universal-kv/campaign/lfm2-2.6b/report.md`). Template parity held on 96/96
prompts and 12/12 whole conversations, with records byte-identical to the server's; the whole hub
corpus fits the budget (181,445 tokens, 5.95 GB, two chunks at 2,519 tokens/s, 0.73 of the smoke
prefill); onboarding gives K R² 0.70-0.76 into it and 0.58-0.71 out of it (V 0.50-0.56 / 0.45-0.53),
with fit_hub needing 11 GiB of anonymous memory (62 GiB RSS, mostly the memory-mapped captures).
Its recurrent cell holds the last two inputs, so no mean is calibrated: on the six fixed chats into
LFM2 the zero cell and the first run's mean differ by at most 0.0012 KL per chat.

- **Qwen3.8 → LFM2** (13 distinct chats: six of the fixed set, LFM2 renders the other two too short,
  and the ten-chat longer set): hub_prefix against floor_prefix KL −0.092 [−0.265, −0.005], lower on
  13/13, 37% [33%, 55%] of the floor's KL removed. One chat carries 92% of the floor's KL (a5add943,
  T0 10,240, a reply that rewrites the whole earlier answer: floor KL 2.39, hub_prefix 1.51); without
  it the KL is 0.0121 against 0.0219 (45% removed). So a switch into LFM2 does no harm and removes
  about half of a small floor on ordinary chat, but recovers only about a third (37%) of what a reply
  that copies from the old context loses.
  hub_prefix − hub is +0.002 [−0.003, +0.011]: the native prefix stays on by default.
- **LFM2 → Qwen3.8** (the eight fixed chats): hub_prefix against floor_prefix KL −0.0027 [−0.0046,
  −0.0009], 4% of the floor's KL. Qwen3-4B → Qwen3.8 under the same settings (identical floors)
  removes 30%: LFM2 does worse by +0.0175 [+0.0052, +0.0332] KL, on 7 of 8 chats, with nearly the
  same share of mapped cells chat by chat (from 10% to 85-88%: both templates drop the reasoning
  Qwen3.8 renders). The source's cache limits this direction (K/V R² into Qwen3.8 0.58/0.45 from LFM2,
  0.69/0.55 from Qwen3-4B), not its template.

`hub5-20261005b` (and `-v2`) is hub4-d4096 plus this onboarding of LFM2-2.6B; `hub5-20261005`,
assembled from the first run, holds older LFM2 adapters, which `campaign.py status` flags.

Second campaign, October 5, 2026: DeepSeek-V2-Lite-Chat Q8_0 (`deepseek2`, MLA, 27 K-only layers of
[c_kv 512 | k_pe 64], YaRN x40; `--name deepseek-v2-lite --catalog-id deepseek-v2-lite-chat`) went
through every stage in 61 minutes, 48 of them the evaluations (the baseline was reused), and 18.5
GB of new disk (`evidence/universal-kv/campaign/deepseek-v2-lite/report.md`; diagnostics in its
`checks/`). Parity held on 96/96 prompts and 12/12 whole conversations; the corpus renders to
127,354 tokens (the template drops reasoning and tools) and was captured whole in three chunks at
1,098 tokens/s (0.69 of the smoke prefill, 14.97 GB). The pre-RoPE rows that kvstate recovers from
f32 state saves (k_pe de-rotated with the real YaRN settings) match the probe's named nodes as
closely in the rope columns as in the latent ones (relative RMS 2e-4 at layer 0). Onboarding gives
K R² 0.47-0.53 into it and 0.60-0.71 out of it (V 0.44-0.49 / 0.43-0.53); fit_hub needed 16.7 GiB
of anonymous memory (a CPU fit), which the lock's estimate covers (`fit_hub_extra`).

- **Qwen3.8 → DeepSeek-V2-Lite** (13 distinct chats: six fixed, two render too short, and nine of
  the longer set): hub_prefix against floor_prefix KL −0.130 [−0.384, −0.004], lower on 13/13, 47%
  [28%, 49%] of the floor's KL removed; a5add943 carries 93% of the floor's KL, and without it the
  switch removes 30% (−0.0066 [−0.0116, −0.0033]). The native prompt start (BOS and the system
  message, 11 tokens) decides this target: Qwen3.8 has no BOS, so without it the hub does worse than
  the floor on 10 of 12 chats (+0.061 [+0.023, +0.099] KL without a5add943), and with it better on
  12 of 12. On a5add943, the reply that copies the earlier answer, it is the other way round (KL 0.49
  without the prefix, 1.29 with it, floor 2.74), which makes the pooled hub_prefix − hub CI span 0
  (+0.013 [−0.107, +0.200]): on by default.
- **DeepSeek-V2-Lite → Qwen3.8** (the eight fixed chats): −0.0020 [−0.0037, −0.0003], 3% of the
  floor's KL, against 30% for Qwen3-4B under the same settings (+0.0182 [+0.0058, +0.0334] KL, worse
  on 7 of 8, the same share of mapped cells per chat). Like LFM2's 4%. Both onboarded encoders give
  hub latents half as spread as the jointly fitted models' (0.48-0.50 of Qwen3.8's own, Qwen3-4B
  0.97-1.01) at the same cosine to Qwen3.8's encoding; the R² gap (K 0.60 vs 0.70, V 0.45 vs 0.56)
  looks small for a tenfold gap in KL removed, so whether the onboarded encoder rather than the
  source model limits this direction was open: it was the encoder (RESULTS.md section 9).

Third campaign, October 5, 2026: Qwen3.6-35B-A3B Q8_0 (`qwen35moe`, 40 layers: 10 full-attention
layers of 2 x 256 KV and 30 gated-DeltaNet layers; 36.9 GB read from the Windows SSD;
`--name qwen3.6-35b-a3b`) went through every stage in 116 minutes of wall time: 25 minutes to
the evaluation, then 91 minutes of evaluations (five runs; the baseline was reused), and 13.0 GB
of new disk (`evidence/universal-kv/campaign/qwen3.6-35b-a3b/report.md`; checks in its `checks/`). The
evaluation was interrupted once and resumed to add the `-server` runs; the two runs already done
were kept. Parity held on 96/96 prompts and 12/12 whole conversations. Four of the twelve sampled
records differ from the server's bytes: the server keeps earlier reasoning (`preserve_thinking`) and
the template's defaults drop it, and the server context reproduces all four. The corpus renders to
280,371 tokens and was captured whole in two chunks (596 tokens/s, 0.57 of the smoke prefill, 11.49
GB). The first chunk's three minutes are mostly the model load from the NTFS SSD at about 180 MB/s.

Its tokenizer keeps every span of the base (train 185,672, dev 17,355, test 18,415), so its R² is
directly comparable with section 1 of RESULTS.md: K 0.70-0.75 / V 0.58-0.65 into it (Qwen3.8 → it
0.754 / 0.655), K 0.69-0.75 / V 0.48-0.58 out of it. fit_hub needed 13.0 GiB of anonymous memory.
The estimate by width alone had asked for 12.1 GiB, so `fit_hub_extra` now counts the dev spans. The
recurrent mean of 121 saves (433 s) explains 0.83 of the delta-rule state's energy.

- **Qwen3.8 → Qwen3.6 with the templates' rendering** (12 chats: six fixed, two of which render too
  short because the template drops earlier reasoning, plus the longer set): hub_prefix against
  floor_prefix KL −0.0150 [−0.0305, −0.0036], lower on 11/12, 39% [26%, 49%] of the floor's KL
  removed. a5add943 carries 47% of the floor's KL; without it 42% is removed.
- **Qwen3.8 → Qwen3.6 as the servers render** (the eight fixed chats, all long enough, 99% of the
  cells mapped): −0.0302 [−0.0475, −0.0149], lower on 8/8. This removes 45% [37%, 53%] of the
  floor's KL and 47% of its ΔNLL (hub_prefix +0.035 [+0.020, +0.048] against floor_prefix +0.065),
  like Qwen3.8 → Ornith in RESULTS.md section 2.
- **Qwen3.6 → Qwen3.8 with the templates' rendering** (eight fixed chats): −0.0076 [−0.0136,
  −0.0026], lower on 8/8, 11% [7%, 14%] of the floor's KL removed. Qwen3-4B removes 30% under the
  same settings, so Qwen3.6 is worse by +0.0126 [+0.0040, +0.0241] KL, on 7 of 8 chats, with the
  same mapped share chat by chat.
- **Qwen3.6 → Qwen3.8 as the servers render**: the source now holds the earlier reasoning, so 98%
  of the cells are mapped instead of 45%. The switch removes 18% [7%, 24%] of the floor's KL
  (−0.0119 [−0.0236, −0.0023], lower on 7/8) and 25% of its ΔNLL.
- **What limits the direction into Qwen3.8.** The encoder spread check (`checks/`, the
  DS-V2-Lite campaign's script) compares Qwen3.6's onboarded encoder with Qwen3-4B's:
  - Spread: Qwen3.6's hub latents spread 0.50 (V) / 0.51 (K) as far as Qwen3.8's own, like LFM2's
    and DS-V2-Lite's (0.48-0.50); the jointly fitted models reach 0.98-1.00.
  - Cosine: Qwen3.6's latents have the highest cosine to Qwen3.8's encoding of any source (0.53,
    against 0.44-0.52).
  - R²: the rows they decode into Qwen3.8 match Qwen3-4B's (K 0.694 vs 0.689, V 0.537 vs 0.546).
  - Variance: those rows carry less variance (K 0.61 vs 0.72, V 0.41 vs 0.55 of the native).

  Three very different onboarded sources show the same half-spread latents, and a strong
  same-family source recovers a third of what Qwen3-4B does. That points to the onboarded encoder
  (a ridge regression onto the mean of the pivots' estimates) more than to the source model. It was
  the encoder: with `reverse`, the default since (RESULTS.md section 9), DeepSeek-V2-Lite, LFM2 and
  Qwen3.6 remove 17%, 24% and 41% (49.8% as the servers render) of the floor's KL into Qwen3.8
  instead of 3%, 4% and 11% (18%), and the three campaigns were re-encoded with it.

Bridge settings: `--hub-name qwen3.6-35b-a3b --native-prefix --recurrent-mean
.../campaign/qwen3.6-35b-a3b/recurrent/recurrent-mean.npz`. With its DFlash draft on the server
also add `--draft-window 4096`: the draft's window is 4,096 tokens and the bridge assumes 2,048.

Fourth campaign, October 6-7, 2026, the first with the `reverse` encoder: GLM-4.7-Flash Q5_K_M
(`deepseek2`, MLA, 47 layers of [c_kv 512 | k_pe 64]; `--name glm-4.7-flash`; captured on October 5,
before the shutdown: the whole corpus, 260,546 tokens, 53.3 GB, 318 tokens/s, 0.50 of the smoke
prefill). Parity held on 96/96 prompts and 12/12 whole conversations; like Qwen3.6, its server keeps
earlier reasoning, so the evaluation also ran as the servers render. Its template keeps reasoning, so
its spans are nearly the base's (train 185,663). Onboarding (GPU fit, `reverse`) took 21 minutes of
running time, 19.2 GiB of anonymous memory and 12.1 GiB on the GPU (estimated then 33.0 in all; 34.1 now); held-out K R² 0.60-0.65 into it and
0.66-0.75 out of it (V 0.53-0.58 / 0.50-0.60).

- **Qwen3.8 → GLM-4.7-Flash** (13 chats: six fixed and the longer set): the native prompt start
  decides this target too. Without it the translation collapses (KL +2.2 above hub_prefix: GLM's
  `[gMASK]` start token is its attention sink); with it 78.5% [74.9%, 82.4%] of the floor's KL is
  removed, lower on 13/13, and 81.0% [77.6%, 84.7%] as the servers render (8 chats).
- **GLM-4.7-Flash → Qwen3.8** (eight fixed chats): 38.1% [22.7%, 46.3%] of the floor's KL removed,
  lower on 8/8, better than Qwen3-4B under the same settings (−0.0056 [−0.0107, −0.0014] KL, on 7 of
  8 chats); 48.1% [35.4%, 55.4%] as the servers render.

Bridge settings: `--hub-name glm-4.7-flash --native-prefix`.

**hub5-20261006** (and `-v2`) is hub4-d4096 plus the four onboarded models with `reverse` encoders:
qwen38, muse, qwen3-4b, ornith, lfm2-2.6b, deepseek-v2-lite, qwen3.6-35b-a3b, glm-4.7-flash. Held-out R²
between the onboarded models (`evaluation-onboarded.json`, 8,973 test spans all eight share): K
0.43-0.71, V 0.41-0.61; into DeepSeek-V2-Lite is the weakest (K 0.43-0.49), as into it from the base
models. `hub5-20261005` and `hub5-20261005b` hold the older LFM2 adapters (`campaign.py status` flags
them). To add another model:

```bash
cd experiments/universal_kv
../linear_transfer/.venv/bin/python campaign.py plan --name NAME                 # [--catalog-id ID]; loads no model
../linear_transfer/.venv/bin/python campaign.py run --name NAME                  # again after a guard stop: it resumes
../linear_transfer/.venv/bin/python campaign.py assemble --output ../../evidence/universal-kv/hub7-YYYYMMDD --evaluate \
    lfm2-2.6b deepseek-v2-lite qwen3.6-35b-a3b glm-4.7-flash granite-4.0-h-tiny falcon-h1-3b \
    nemotron-3.5-lightning-30b-a3b kimi-linear-48b-a3b ling-3.0-tiny gemma-4-26b-a4b gemma-4-31b NAME
# only if the guard ended the R^2 evaluation of an assembled hub (the directory is kept):
../linear_transfer/.venv/bin/python campaign.py evaluate-onboarded --hub ../../evidence/universal-kv/hub7-YYYYMMDD \
    lfm2-2.6b deepseek-v2-lite ... NAME
```

hub6 ([below](#hub6-and-onboarding-results-by-family)) holds every model onboarded so far but Jamba.

The joint refit of the seven models (Track A's option (c), RESULTS.md section 9) ran to the end on its fifth
try (October 8-9: 1 h 44 min wall, 235 kJ). Fitted on the seven models' 87,426 common train spans, it did worse
end to end than onboarding with `reverse` out of every model into Qwen3.8 (DS-V2-Lite 6% against 17%, LFM2 7%
against 24%, Qwen3.6 23% against 41% of the floor's KL removed, and even Qwen3-4B 23% against 30%; individually
significant, exact sign-flip p 0.023, for LFM2 and Qwen3.6) and into LFM2 and Qwen3.6, although its K R² was
higher (+0.03 to +0.07 out of the onboarded models). It differs from onboarding in three ways at once (joint, on
fewer spans, every decoder refitted), so this rules out that refit as a replacement, not joint refits in general;
keep onboarding. The earlier tries: on
October 8, with kvpark.service stopped (MemAvailable 104-112 GiB), it ran four times and the guard ended
it each time (after 7, 20, 91 and 17 minutes), always on two samples at 95 °C or more taken while the job
was already stopped: other projects' jobs outside the locks (54-122 W) heated the package. fit_hub.py
cannot resume, so each try started over, after waiting 17-67 minutes for the GPU/memory lock behind the
other campaigns. A whole refit needs about 1 h 50 min of wall time, half of it stopped, with the package
free of work outside the locks. To run it again:

```bash
cd evidence/universal-kv/onboard-encoders
../../../experiments/linear_transfer/.venv/bin/python window_c.py --steps joint,e2e,compare   # fit, convert, end to end, tables
```

`compare` writes `joint7/comparison.json`: held-out R² of every pair against hub4 plus the reverse pilots
on the same 8,884 test spans, then the joint hub against the onboarded hub end to end on the campaigns'
own commands (`JOINT_RUNS`, `e2e-joint/`), then the cost from the logs.

The evaluations run with the GPU-evaluation guard defaults (90 °C / 88 °C / 4 threads; 85 / 65 °C until
October 8) without environment variables, and `assemble --evaluate` runs on the GPU as a watched child process
([Running heavy jobs safely](#running-heavy-jobs-safely)). hub5-20261006's evaluation predates that: the
guard ended the in-process CPU evaluation once (98 °C), and it ran with `onboard_encoders.py evaluate` (GPU
products) under `campaign.py exec`. Read the report's encoder line: an onboarding whose `reverse` encoder
hit the variance cap, or a dimension the view barely predicts, carries a warning there (state.json
`onboard.result.warnings`).

### hub6 and onboarding results by family

**hub6-20261009** (format 1) and **hub6-20261009-v2** (format 2, for `kvpark bridge engine --hub`) hold fifteen
models: hub4-d4096's four (qwen38, muse, qwen3-4b, ornith) and eleven onboarded (lfm2-2.6b, deepseek-v2-lite,
qwen3.6-35b-a3b, glm-4.7-flash, granite-4.0-h-tiny, falcon-h1-3b, nemotron-3.5-lightning-30b-a3b,
kimi-linear-48b-a3b, ling-3.0-tiny, gemma-4-26b-a4b, gemma-4-31b), all with `reverse` encoders but Granite and
Nemotron, which carry `ridge` encoders (`campaign.py reencode --encoder ridge`, October 9). Jamba, whose onboarding
fails as a source, is left out. It was assembled on October 9 with hard links (no new disk), like
**hub6-20261008** (`-v2`), which it replaces: that one carried the `reverse` encoders of Granite and Nemotron, so a
switch out of Nemotron did worse than no translation (`campaign.py status` flags it as built from an older
onboarding). Held-out R² between the eleven onboarded models (8,699 test spans they all share): K 0.35-0.75, V 0.28-0.63; the
per-model ranges are in RESULTS.md section 10. Register each model with its campaign report's settings (`report.md`,
"Bridge settings": `--hub-name NAME --native-prefix`, `--recurrent-mean .../campaign/NAME/recurrent/recurrent-mean.npz`
for a hybrid whose cell holds a state, and `--no-source` where a switch out of the model did worse than the floor:
Nemotron-3.5).

Share of the floor's KL a switch removes, with Qwen3.8 (RESULTS.md section 10 has every interval and p). Into the
model: on the six fixed chats every model's runs share, and in brackets the reports' pooled figure over every chat
its runs scored (the longer set adds a5add943, a reply that rewrites the earlier answer, whose floor is 10-100 times
the others' and dominates some pooled figures). Out of it: the eight fixed chats. "n.s.": no measurable difference
(exact sign-flip p above 0.05). Qwen3-4B → Qwen3.8 under the same settings: 30%. These are selection-set numbers
([Onboarding campaign](#onboarding-campaign-campaignpy), "Held-out chats").

| Family | Models (llama.cpp arch) | Qwen3.8 → model, six chats (pooled) | model → Qwen3.8 | In hub6 |
| --- | --- | --- | --- | --- |
| Gated DeltaNet hybrid | Qwen3.6-35B-A3B (`qwen35moe`) | 50% (39%) | 41% (50% as the servers render) | both directions |
| Short-convolution hybrid | LFM2-2.6B (`lfm2`) | 51% (37%) | 24% | both directions |
| MLA | DeepSeek-V2-Lite, GLM-4.7-Flash (`deepseek2`) | 31% (47%), 83% (79%) | 17%, 38% (48%) | both directions; the native prompt start is essential into them |
| MLA + linear-attention hybrid | Kimi-Linear-48B-A3B (`kimi-linear`), Ling-3.0-tiny (`bailingmoe3`) | 21% (70%), 71% (72%) | 32%, 42% | both directions |
| Gemma 4 iSWA | Gemma-4-26B-A4B, Gemma-4-31B (`gemma4`) | 45% (44%), 35% (27%) | 37% (35%), 33% (34%) | both directions; the source side reads only the full-attention layers |
| Mamba2 + attention in every layer | Falcon-H1-3B (`falcon-h1`) | 9.5% n.s. (51%, nearly all of it a5add943) | 25% (p 0.055) | both directions |
| Mamba hybrids with few attention layers | Granite-4.0-H-Tiny (`granitehybrid`, 4), Nemotron-3.5-Lightning-30B-A3B (`nemotron_h_moe`, 6), Jamba-Reasoning-3B (`jamba`, 2) | 6% n.s. (42%, nearly all a5add943), 16% (23%), 7% n.s. (4%) | `reverse`: 5% n.s., −16% n.s. (−115% as the servers render), −1031%; `ridge`: 1.3%, 4.7% (−44% as the servers render) | Granite with its `ridge` encoder; Nemotron with its `ridge` encoder, registered with `--no-source` (switches out of it skipped); Jamba left out |

What decides out of a model is not the number of attention layers (Gemma-4-26B's source reads 5 full-attention
layers and removes 37%, Ling with 6, as many as Nemotron, 42%) but whether the onboarding's encoder warnings fire:
a hub dimension the model's attention cache predicts with train R² below 0.05, together with the variance cap
binding on about half of the dimensions or more (Granite 0.044 and 49-63%, Nemotron 0.028 and 71-72%, Jamba 0.003
and 95-96%; Ling, 0.061 and 37%, works). Where they fire, `reverse` scales each dimension by about 1 / R² and inflates
what the view does not carry: no gain (Granite), harm as the servers render (Nemotron), great harm (Jamba). A `ridge`
encoder keeps about as little as the floor and is harmless out of Granite, which is what hub6 now carries; out of
Nemotron as the servers render (what a live switch holds) it still does worse than the floor (−44%, lower on 1 of 8
chats), so register Nemotron with `kvpark bridge add --no-source`: switches out of it are skipped and the next model
prefills natively (the campaign report's "Bridge settings" add the flag when a switch out of the model did worse
than the floor). When the earlier conversation matters, a native prefill is the better choice out of Granite too. The warnings come from the onboarding
itself, before any evaluation (state.json `onboard.result.warnings`, the report's encoder line): for a new model,
evaluate the direction out of it before shipping its `reverse` encoder, and re-encode with `ridge` where it does not
help.

Into a model the decoders, which do not depend on the encoder, removed 16-83% of the floor on the six shared chats for
nine of the twelve, and nothing measurable into Granite, Falcon-H1 and Jamba (6-10%, sign-flip p 0.09-0.25); into
Granite, Falcon and Kimi-Linear one chat (a5add943) carries 93-95% of the pooled floor, so their pooled figures are
mostly that chat.

Live with hub6 (October 9; RESULTS.md section 10, `evidence/universal-kv/live-hub6-20261009b/`): ten switches on
throwaway a7b94df2c ROCm servers, two models per run under `campaign.py exec --gpu-evaluation`, the Vulkan engine:
Qwen3.8 → Granite-4.0-H-Tiny, Qwen3.8 ↔ Gemma-4-26B-A4B, Kimi-Linear → Qwen3.8 and Kimi-Linear → Gemma-4-26B-A4B
(two onboarded models). Every switch reused exactly T0. Into Qwen3.8 (12-15K tokens) a switch took 9.7-12.5 s of wall
time against 48-52 s of native prefill: 3.3-6.8 times faster where the native prefill met few guard pauses (a pause
stops the job while the GPU finishes its queue, so a timing with pauses inside is only bounded, and the first series'
pause-corrected "3.6-5.9 times" is withdrawn). Into Gemma-4-26B 2.7-3.5 s against 6.6-6.8 s without a pause (2.0-3.1
times), into Granite 1.8-2.1 s against 3.8-5.3 s (it prefills fast). The first 32 generated tokens matched the native
run's in all four switches into Gemma and in one of two into Granite, which mostly reflects the 2,048-token native
tail; into Qwen3.8 they part early but restate the request correctly. To repeat them:
`live-hub6-20261009b/scripts/live.sh HUB NAME=A,B@PORT ...`; `live_bridge.py hops --native-chunk 1024` also prefills each
native reference in cooled 1,024-token requests (an upper bound: each request adds its own overhead).

## The engine

`engine.py JOB.json` runs one translation and exits (its docstring documents the job);
`engine_daemon.py --bridge-dir DIR` keeps the hub sides (and the GPU) warm and serves jobs over
`DIR/engine.sock`:

```
python engine_daemon.py --bridge-dir ~/.local/share/kvpark/bridge --device auto \
    --idle-unload 120 --idle-exit 600
```

The bridge client starts it on demand (with the engine Python of `kvpark bridge engine`), so
running it by hand is only for debugging; its log is `DIR/engine.log` (timings and sizes, no
conversation text). `--keep-cache` keeps the GPU allocator's cached blocks between switches
(they are returned by default, since on a unified-memory machine the llama-servers share them)
and `--keep-host` keeps freed host memory until the idle unload. The device comes from the
job's `"device"`, else `KVPARK_ENGINE_DEVICE`, else `auto` (torch on a GPU when it imports,
numpy otherwise); DeepSeek V4 targets always run on numpy. The engine Python needs numpy with
an optimized BLAS: the report carries `blas` and a warning for the reference BLAS (10-100x
slower, the cause of a 350 s switch in the first live run). For the GPU it needs torch with a
ROCm or CUDA build for the GPU (the local `~/venvs/rocm` has torch 2.12 for gfx1151).

Design: [ARCHITECTURE.md, section 6](ARCHITECTURE.md#6-the-engine-gpu-backends-and-a-resident-daemon).
Numbers: [RESULTS.md](RESULTS.md) sections 6 and 8 (a warm 13K-token switch takes 0.45-0.78 s on
the iGPU and 2.1-2.8 s on the CPU with 16 BLAS threads; the one-shot numpy engine it replaced
took 13.3 s).
