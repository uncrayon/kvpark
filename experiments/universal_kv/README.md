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
| hub.py | hub fitting (PCA views, multi-view CCA, ridge encoders/decoders, global-layer encoders for sliding-window sources), onboarding a model without refitting, evaluation; DSA indexer ridges (`IndexerRidge`) and DeepSeek V4 head kinds (`HeadKind`: decoders of the compressor inputs of each CSA, HCA and LID layer); translation sides (`EncoderSide` with the fused encoder `(basis2 * scale) @ encoder`, `DecoderSide` with per-section decoders); bundles: `save_hub`/`load_hub` (format 1 and 2), `open_hub` (lazy, side by side, sha256 checked once per file identity) |
| fit_hub.py | CLI: fit a hub on captured models, or onboard one (`--base`); `--indexer-ridge NAME` for a DSA model, `--heads NAME` for a DeepSeek V4 model; compares with a pairwise mapper (`--direct`) |
| convert_hub.py | format 1 bundle → format 2: one flat C-contiguous `.npy` per (model, kind, role), fused encoders, float16 copies, sha256 manifest; indexer ridges and head kinds carried over |
| token_tables.py | per-model token tables (mean K/V per token id over the cells the hub does not cover) |
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
| pair_eval.py | end-to-end evaluation of a model pair (render, tokenize, save states, translate, seed-score); `--recurrent`, `--recurrent-mean`, `--recurrent-group` for hybrid targets |
| score_translate.py | hub vs a pairwise mapper on the documents of a linear_transfer long run; the same recurrent options |
| recurrent_reset.py | the cost of resetting a hybrid model's recurrent state (R0 alone; superseded by `r0_experiments.py run --strategies zero`) |
| live_bridge.py | live tests on throwaway llama-servers: `hops`, `drafts`, `media`, `drafts-control`, `media-control`, `options` (request template options, the recurrent mean); a memory guard before each model start |
| pair_table.py | markdown table of pair_eval reports |
| fixtures_families.py | test helpers: tiny MLA/DSA, K-only hybrid and GQA-family models (test-llama-archs and gguf-py Gemma 3/4), the state driver and probe wrappers, exact identity hubs |
| fixtures_bridge/ | `make_fixtures.py` (tiny Qwen3.5 and Qwen3 targets with DFlash and MTP drafts, and a 5120-wide Qwen3.5 the Qwen3.8 mmproj can feed), `live_resume_check.py` (the bridge's `.resume` sidecars on a kvpark-patched server), `stub_engine.py` (a same-model stand-in engine for live switch checks) |
| test_*.py | `test_hub`, `test_translate` (every cache family on synthetic models with hand-built hubs), `test_backends`, `test_ggml_backend`, `test_state_writer`, `test_engine` (rendering, media sources, features, vocabularies, specs and pins), `test_engine_daemon`, `test_recurrent_init`, `test_families` (MLA/DSA/hybrids and the GQA family through llama.cpp), `test_dsv4` (DeepSeek V4 through llama.cpp, the engine and `kvpark.bridge.switch`) |

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
