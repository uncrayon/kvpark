# Universal KV translator (experiment)

One shared per-token latent ("hub") that translates the attention caches of different open
models into each other, so a conversation can move between models without a full prefill.
The product side is `kvpark bridge` ([docs/bridge.md](../../docs/bridge.md)).

- [ARCHITECTURE.md](ARCHITECTURE.md): design, the state families of the top-20 open models,
  what is supported for each llama.cpp architecture and on what it was validated, the engine
  and its daemon, recurrent state, images, draft models, DeepSeek V4, open problems.
- [RESULTS.md](RESULTS.md): measurements on four local models (Qwen3.8-27B, Ornith-1.5-9B,
  Qwen3-4B, Muse-Glimmer-30B) and live switches on real llama-servers; a summary table at the
  top.
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
| GQA family | `llama`, `mistral3`, `qwen3moe`, `hy_v3`, `phi3` (none of them LongRoPE), `gemma3`, `gemma4` (real Gemma-4 saves), `gpt-oss`, `laguna`, `mimo2` |
| Hybrids with a recurrent cell | `qwen35` (real models, live), `qwen35moe`, `qwen3next`, `qwen4exp`, `minimax-01`, `jamba`, `falcon-h1`, `granitehybrid`, `nemotron_h`, `nemotron_h_moe`, `plamo2` (cache only: its PLaMo-2 vocabulary is not read, so the bridge skips it as `tokenizer`), `lfm2`, `lfm2moe` |
| MLA, DSA, K-only hybrids | `deepseek2`, `mistral4`, `hy_v4`, `deepseek32`, `glm-dsa`, `dots3note`, `kimi-linear`, `kimi-k3`, `bailingmoe3`, `glm5-next` (a7b94df2c only) |
| Pooled latent (target only) | `deepseek4` |

Draft models: DFlash/DSpark (an aligned dummy), MTP (an empty head cache) and the `shared` kind
(Gemma 4's MTP head, which writes no cells of its own: its 8-byte draft blob is kept as it is).
Vocabularies: GPT-2 byte-level BPE, Gemma 4's BPE and SentencePiece (from the engine's
`vocabularies` feature). Chat templates render as llama-server renders them (`chat.py`), checked
by `template_check.py` against `/apply-template` on every local model.

## Code

| File | Purpose |
| --- | --- |
| hub.py | hub fitting (PCA views, multi-view CCA, ridge encoders/decoders, global-layer encoders for sliding-window sources), onboarding a model without refitting, evaluation (every pair, or those `involving` one model); DSA indexer ridges (`IndexerRidge`) and DeepSeek V4 head kinds (`HeadKind`: decoders of the compressor inputs of each CSA, HCA and LID layer); translation sides (`EncoderSide` with the fused encoder `(basis2 * scale) @ encoder`, `DecoderSide` with per-section decoders); bundles: `save_hub`/`load_hub` (format 1 and 2; `link_base`: an onboarding's bundle hard-links the base's files), `open_hub` (lazy, side by side, sha256 checked once per file identity), `merge_bundles` (one bundle of a base and models onboarded into it separately, hard links) |
| fit_hub.py | CLI: fit a hub on captured models, or onboard one (`--base`; `--link-base` links the base's files, `--evaluate new` tests only the pairs with the new model); `--indexer-ridge NAME` for a DSA model, `--heads NAME` for a DeepSeek V4 model; compares with a pairwise mapper (`--direct`) |
| convert_hub.py | format 1 bundle → format 2: one flat C-contiguous `.npy` per (model, kind, role), fused encoders, float16 copies, sha256 manifest; indexer ridges and head kinds carried over; `--reuse V2` links the sides of models an earlier conversion holds (same files) |
| token_tables.py | per-model token tables (mean K/V per token id over the cells the hub does not cover); `--only NAME` for an onboarded model on its onboarding's spans |
| campaign.py | onboards one model end to end, resumable stage by stage (render with strict template parity, a chunked capture within a size and time budget, onboard, token tables, the recurrent mean where the cell holds a state, evaluation with Qwen3.8 and the Qwen3-4B baseline, a report with paired CIs), every command that loads a model or a hub under the GPU/memory lock, which the command itself holds; `exec` runs one command under it, `assemble` builds one hub of the base and the onboarded models, `status` flags assembled hubs a later onboarding outdated ([Onboarding campaign](#onboarding-campaign-campaignpy)) |
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
| pair_eval.py | end-to-end evaluation of a model pair (render, tokenize, save states, translate, seed-score); `--recurrent`, `--recurrent-mean`, `--recurrent-group` for hybrid targets; `--ids` (with `--min-documents`) a fixed set of conversations, such as RESULTS.md section 2's; `--server-render` renders both models as llama-server does (its default template kwargs, a fixed clock) instead of with the templates' defaults |
| score_translate.py | hub vs a pairwise mapper on the documents of a linear_transfer long run; the same recurrent options |
| recurrent_reset.py | the cost of resetting a hybrid model's recurrent state (R0 alone; superseded by `r0_experiments.py run --strategies zero`) |
| live_bridge.py | live tests on throwaway llama-servers: `hops`, `drafts`, `media`, `drafts-control`, `media-control`, `options` (request template options, the recurrent mean); a memory guard before each model start |
| pair_table.py | markdown table of pair_eval reports |
| fixtures_families.py | test helpers: tiny MLA/DSA, K-only hybrid and GQA-family models (test-llama-archs and gguf-py Gemma 3/4), the state driver and probe wrappers, exact identity hubs |
| fixtures_bridge/ | `make_fixtures.py` (tiny Qwen3.5 and Qwen3 targets with DFlash and MTP drafts, and a 5120-wide Qwen3.5 the Qwen3.8 mmproj can feed), `live_resume_check.py` (the bridge's `.resume` sidecars on a kvpark-patched server), `stub_engine.py` (a same-model stand-in engine for live switch checks) |
| test_*.py | `test_hub`, `test_campaign` (campaign.py's planning and resume, capture chunks, parity rules, the lock and process groups, statistics and reports, no model), `test_pair_eval` (its renderings, no model), `test_translate` (every cache family on synthetic models with hand-built hubs), `test_backends`, `test_ggml_backend`, `test_state_writer`, `test_engine` (rendering, media sources, features, vocabularies, specs and pins), `test_engine_daemon`, `test_recurrent_init`, `test_families` (MLA/DSA/hybrids and the GQA family through llama.cpp), `test_dsv4` (DeepSeek V4 through llama.cpp, the engine and `kvpark.bridge.switch`) |

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
   hy_v3 and glm5-next; the states are read in the layout of the probe's pin).
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
| render | the hub corpus (corpus v2 + v3-train, the 248 conversations of hub4-d4096) with the model's own template (chat.py corpus mode); chat.py against llama-server's `/apply-template` on a tokenizer twin (template_check server mode, ports 18690-18699) for two conversations per source, as prompts under every option set and as whole conversations the way the records render them (no generation prompt, `--no-prefill-assistant`); the probe tokenizes every record | `render/` (`parity-*.json`, `records-differing.json`) |
| capture | the forecast (`capture-plan.json`), then `collect_v2.py` chunk by chunk: the dev and test documents, then train chunks while the measured speed and size allow (Vulkan probe; `--mode states` on the CPU probe for kimi-k3, bailingmoe3, hy_v4, hy_v3 and glm5-next) | `capture/chunk-NN/`, `capture-docs.jsonl` |
| onboard | `fit_hub.py --base hub4-d4096 --link-base --evaluate new` with all four base models as pivots (`--source-global` for a sliding-window model, `--indexer-ridge` for a DSA model), then `convert_hub.py --reuse hub4-d4096-v2`; the sha256 of the model's own files go to `state.json` | `hub/` (held-out R² per kind against each pivot in `hub/evaluation.json`), `hub-v2/` |
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
  the model; for pair_eval the larger model and the hub bundle it holds whole; for fit_hub the
  base hub plus an estimate that grows with the new model's K/V width and the dev spans
  (`fit_hub_extra`: its stage-1 covariance, decoder moments and the dev spans tiled over the
  pivots, counted at the base hub's 17,355 dev spans, which bound every model's; 13.8 GiB for
  LFM2-2.6B, measured 11.0, 21.0 GiB for DeepSeek-V2-Lite, measured 16.7, and 14.3 GiB for
  Qwen3.6-35B-A3B, measured 13.0, where the estimate by width alone had asked for 12.1: its
  tokenizer keeps every dev span of the base, twice LFM2's). The command
  inherits the locked descriptor, so the lock stays held as long as it runs even if campaign.py is
  killed; it runs in its own process group, which campaign.py stops on SIGINT, SIGTERM, SIGHUP or
  a timeout. render loads no model.
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
  `--redo onboard` an assembled hub keeps the old adapters, and `campaign.py status` flags it.

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
of anonymous memory, which the lock's estimate now follows (`fit_hub_extra`).

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
  source model limits this direction is open (onboarding a base model against the other three
  pivots would tell).

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
  (a ridge regression onto the mean of the pivots' estimates) more than to the source model. The
  next test is an end-to-end run with the latents rescaled, or onboarding a base model against the
  other three pivots.

Bridge settings: `--hub-name qwen3.6-35b-a3b --native-prefix --recurrent-mean
.../campaign/qwen3.6-35b-a3b/recurrent/recurrent-mean.npz`. With its DFlash draft on the server
also add `--draft-window 4096`: the draft's window is 4,096 tokens and the bridge assumes 2,048.

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
