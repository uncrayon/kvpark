# Qwen → Muse value-cache probe

Start with an offline ridge regression inspired by
[NVIDIA's linear mapping paper](https://arxiv.org/html/2608.03893v1).
The paper fits affine maps from paired cache observations without backpropagation.
Its full mapper uses multiple source layers, separate key/value maps, and
position-free keys. This experiment implements only the first diagnostic: can
the values from one source layer predict one target layer on unseen text?
All source heads are inputs; each target output coordinate has its own weights.
Solving the outputs together is equivalent to independent per-head regressions
with the same inputs and regularization.

This is the easiest fitting mechanism to start with. In comparison,
[C2C](https://arxiv.org/html/2510.03215v2) trains a neural fuser through a frozen
receiver's language-model loss and uses the receiver's own context processing.
It targets semantic communication, with a more involved training pipeline.

**Status: the v2 pipeline has run end to end on the real Q8 checkpoints.**
It fits multi-layer K/V maps for all 52 Muse layers from chat-formatted
captures, then injects the mapped prefix into Muse and scores the continuation.
On long transferred prefixes, next-token loss rose by 0.11 nats/token, recovering
about 83% of the gap to an information-free cache (5 documents). See
[v2 results](RESULTS_v2.md) and the [v2 pipeline](#v2-pipeline) below. The first
prototype's one-layer value mapper, with R² 0.397568 (0.451 once λ is chosen by
CV), is described in the [v1 results](RESULTS.md).
Later runs test the shippable handoff (a mapped Muse slot file plus a native tail):
[v3](RESULTS_v3.md) (meets the 0.05 nats/token target on 4–8K chats and fails on long
documents), [v3b](RESULTS_v3b.md) (chats still meet it at 8K+, and agent traces fail,
mostly in table-filled template cells), and the [live server test](RESULTS_live.md) (a
real llama-server reuses the built file; handing over takes 21–28 s at any length).
Switching live conversations between models is `kvpark bridge`
([docs/bridge.md](../../docs/bridge.md)), built on the many-model hub in
[../universal_kv](../universal_kv/README.md), which uses this directory's probe, state
codec and template code. The collector runs its
own model process; fitting and application work on saved arrays. Both tools are
separate from kvpark's installed package and its exact snapshot restore path.
Tests also include synthetic arrays and real CPU execution of tiny, randomly
initialized Qwen and Muse architectures; those fixtures test implementation,
not language quality.

For this pair, Qwen → Muse is our first research direction. The
[Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
mixes recurrent and full-attention layers; the
[Muse configuration](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json)
mixes sliding and full attention. Starting with Muse as receiver avoids having
to construct Qwen's recurrent state. This is an engineering choice, not a
validated transfer result. The NVIDIA experiments cover within-family pairs
with shared tokenizers, matched KV geometry, and full attention; this pair is
outside that validation.

## Build and collect

The first collector supports Linux with a linked CPU llama.cpp build
(`GGML_BACKEND_DL=OFF`). It uses the existing evaluation callback; it requires
no engine patch or change to a running server. Use matching headers and compiled
libraries from kvpark's pinned runtime:

```bash
python experiments/linear_transfer/build_collector.py \
  --llama-source runtime/source \
  --llama-build runtime/build \
  --output evidence/linear-transfer/collector-build
```

The build directory must be new. Keep the resulting executable and its
`.build.json` manifest together. The manifest fingerprints the executable and
llama/ggml libraries; collection refuses changed artifacts. The runtime
libraries remain where they were built. If using a different runtime revision,
pass `--runtime-revision` and validate compatibility yourself. Initial support
is for the pinned Qwen35 and Muse Glimmer graph implementations.

Create UTF-8 JSONL corpus files with one independent raw document per line:

```json
{"text": "The small green notebook contains the number 47. Its owner is Ada."}
{"text": "A separate document with different facts, accents like café, and code."}
```

Those examples only illustrate the format. Prepare independent training and
evaluation documents before collecting. Preserve the exact same text files for
both models. Collection rejects duplicates, empty documents, documents over
1 MiB, and prompts exceeding the context limit; it does not silently truncate.

With NumPy installed as described below, collect one side at a time:

```bash
python experiments/linear_transfer/collect.py \
  --collector evidence/linear-transfer/collector-build/kvpark-collect-values \
  --model /path/to/Qwen3.8-27B-Q8_0.gguf \
  --model-id Qwen/Qwen3.8-27B --layer 31 \
  --corpus evidence/linear-transfer/train.jsonl \
  --output evidence/linear-transfer/qwen-train.npz

python experiments/linear_transfer/collect.py \
  --collector evidence/linear-transfer/collector-build/kvpark-collect-values \
  --model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --model-id meta-models/Muse-Glimmer-30B --layer 27 \
  --corpus evidence/linear-transfer/train.jsonl \
  --output evidence/linear-transfer/muse-train.npz
```

Repeat with `eval.jsonl` and distinct `qwen-eval.npz` / `muse-eval.npz` outputs.
The example middle-layer pair is a starting point, not a selected optimum.
Layer indices are zero-based and checked against the GGUF's attention pattern;
Qwen recurrent layers are rejected. The initial collector accepts only
single-file GGUFs. It hashes the entire model for an immutable identity, which
adds disk reads for large checkpoints.

Defaults are `--ctx 1024 --batch 128 --threads 4`. It runs all model layers on
CPU with memory-mapped weights, weight repacking disabled, F32 K/V storage,
flash attention disabled, and no speculative or MTP model. `Vcur` is captured
before cache storage; with these F32 settings it has the same value coordinates
as the stored cache. This differs from the live servers' F16 caches and needs
its own quality evaluation. Copying respects tensor strides, including Muse's
fused QKV view. State is cleared before each document, including Qwen recurrent
state. No chat template or special tokens are added.

The extractor does not discover dynamic backend plugins. The wrapper clears
`GGML_*` overrides and fixes library lookup to the fingerprinted artifact
directories. Record new artifacts after rebuilding instead of editing a
manifest to accept changed binaries.

Memory mapping does not eliminate memory or I/O demand: this runtime eagerly
reads mapped files, and every layer executes. The collector is a correctness
baseline, not a fast production extraction path. If models already fill RAM,
schedule collection separately or impose process limits appropriate to the
machine. It does not stop, unload, or operate on existing server processes.

For a tokenizer-only preflight without loading weights, the executable also
accepts `--tokenize-only`. Its input JSONL additionally needs each document's
`document_sha256`, and it writes a raw report rather than a probe NPZ. A
tokenizer-only report does not contain tensor observations.

## Run the fitting code

NumPy is an experiment-only dependency. From the repository root, use a research
environment with `python -m pip install -r experiments/linear_transfer/requirements.txt`.

For the complete first prototype, use the supplied [synthetic corpus](corpus_v1.md).
This runner collects each model once, separates whole documents into 32 training
and 8 held-out examples, and fits the one-layer value mapper:

```bash
OPENBLAS_NUM_THREADS=4 python experiments/linear_transfer/run_prototype.py \
  --collector evidence/linear-transfer/collector-build/kvpark-collect-values \
  --source-model /path/to/Qwen3.8-27B-Q8_0.gguf \
  --source-model-id Qwen/Qwen3.8-27B --source-layer 31 \
  --target-model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --target-model-id meta-models/Muse-Glimmer-30B --target-layer 27 \
  --corpus experiments/linear_transfer/corpus_v1.jsonl \
  --ctx 512 --batch 256 --threads 12 \
  --output evidence/linear-transfer/prototype-v1
```

Choose a new output directory, or append `--resume` to the same command after an
interruption. Resume verifies the corpus, models, runtime, settings, source
code, and completed artifacts before reusing work. Do not edit these inputs
mid-run. `source.log` and `target.log` show collection progress; `summary.json`
and `fit/report.json` contain results. The detailed fit report includes metrics
for each held-out document and target head, alongside the aggregate and
training-mean baseline. Negative held-out R² is a valid failure result.

Apply a fitted mapper with neither model loaded:

```bash
OPENBLAS_NUM_THREADS=4 python experiments/linear_transfer/apply_mapper.py \
  --mapper evidence/linear-transfer/prototype-v1/fit/mapper.npz \
  --source evidence/linear-transfer/prototype-v1/source-eval.npz \
  --output evidence/linear-transfer/prototype-v1/predictions.npz
```

This produces `predicted_values` on the **source token-end grid**. The source
identity must match the fitted direction. The file is deliberately not a native
cache or a measured target capture; unmatched target tokens, keys, other layers,
and receiver state are still missing. Reported matrix time excludes loading,
I/O, and model execution and must not be interpreted as handoff latency.

Then run the Python checks (native checks skip unless a collector is supplied):

```bash
python -m unittest discover -s experiments/linear_transfer -p 'test_*.py' -v
```

Run the native checks against the compiled collector as well:

```bash
KVPARK_COLLECTOR="$PWD/evidence/linear-transfer/collector-build/kvpark-collect-values" \
  python -m unittest discover -s experiments/linear_transfer -p 'test_*.py' -v
```

The miniature GGUF fixtures are generated locally with the pinned runtime's
`gguf-py`; no model download is required. Checks compare chunked and unchunked
execution, document resets, a NumPy projection reference, and separate versus
fused QKV layouts.

A local smoke run on 2026-09-27 also extracted real Q8 checkpoint tensors on
CPU: Qwen3.8-27B layer 31 produced `[37, 4, 256]` values and Muse Glimmer-30B
layer 27 produced `[35, 2, 128]` from the same 168-byte document. All values were
finite and nonconstant; 35 token-end boundaries aligned. Collection used
`--ctx 128 --batch 64 --threads 2`, F32 K/V, and sequential isolated processes.
This verifies extraction for that sample, not alignment coverage on other text,
an adapter fit, transfer quality, or compatibility with live F16 GPU caches.
The runs used process-local 3 GiB data-memory and 180-second wall-time limits;
the data limit is not a cap on resident file-backed pages.

Once real captures exist, run:

```bash
python experiments/linear_transfer/probe.py \
  --train-source evidence/linear-transfer/qwen-train.npz \
  --train-target evidence/linear-transfer/muse-train.npz \
  --eval-source evidence/linear-transfer/qwen-eval.npz \
  --eval-target evidence/linear-transfer/muse-eval.npz \
  --output evidence/linear-transfer/probe-001
```

Those input paths are examples; no real captures ship with this experiment.
The output directory must not already exist. It receives `mapper.npz`
(float64 weight matrix, bias, and JSON provenance) and `report.json`
(alignment coverage and held-out reconstruction metrics). The artifact records
its direction and is explicitly marked as a reconstruction probe.

The objective is summed squared error plus `lambda * ||W||²`, with a free bias.
`--ridge` defaults to `auto`: lambda is chosen by 8-fold cross-validation grouped by
training document, over logspace(-2, 7). `report.json` records the CV curve
(`ridge_selection`), in-sample training metrics, and document-bootstrap intervals.
A fixed value such as `--ridge 0.01` reproduces the original fit, which is effectively
unregularized on the v1 data. Changing the number of rows changes the relative
strength of this regularizer. Centering uses training observations only.
The solver prints matrix allocation estimates before fitting; these are not
peak RAM estimates. Captures and selected rows are loaded into memory. Start
with one layer and a small corpus rather than whole-model dumps.

## Capture format

Each NPZ contains one model layer, with `allow_pickle=False`-compatible arrays:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `values` | `[N, kv_heads, head_dim]`, floating point | Actual attention-cache values, before head repetition |
| `document_sha256` | `[N]`, Unicode strings | Lowercase SHA-256 of the complete raw UTF-8 document |
| `byte_end` | `[N]`, positive integers | Exclusive end of the token in that original byte string |
| `token_id` | `[N]`, integers | Original model-specific token IDs, for capture auditing |
| `metadata_json` | Scalar Unicode string | JSON object below |

For example, metadata for a source capture has this structure. Replace the
revision placeholders with immutable checkpoint and runtime identities:

```json
{
  "format_version": 1,
  "model_id": "Qwen/Qwen3.8-27B",
  "model_revision": "checkpoint digest or immutable revision",
  "runtime_revision": "exact engine commit and extraction revision",
  "weight_quantization": "Q8_0",
  "layer": 31,
  "attention_type": "full",
  "cache_kind": "value",
  "text_mode": "raw_utf8_no_special_tokens"
}
```

Layer indices are zero-based. The target has its own identity and layer index.
The collector also records model/library SHA-256 identities, stored GGUF file
type, cache dtype, tokenization options, and execution settings. The GGUF file
type is a numeric enum string rather than a human quantization label. Train and evaluation captures for a given side
must have identical metadata (including extra fields), head shapes, and stored
array dtypes. Keep split labels out of metadata; it describes the extraction
configuration, not the dataset split.

Any alternative collector must start each document from an empty model state, use exactly
the same raw bytes for both models, and disable chat templates, special-token
insertion, and speculative decoding for this initial probe. Do not produce
`byte_end` by assuming token indices match or by multiplying character offsets;
derive it from exact tokenizer byte spans and verify that the token pieces
reconstruct the original bytes. Reject unsupported normalization or ambiguous
spans instead of inventing offsets. The probe checks the supplied structure but
cannot verify that an extractor faithfully followed this contract.

Alignment uses only identical `(document_sha256, byte_end)` pairs. Thus both
observations see the same text prefix, even with different token counts. Source
states that have seen text beyond a target boundary are not valid substitutes.
Duplicate keys are rejected. Shared boundaries omit some target tokens: this
alignment is sufficient for a diagnostic, **not for constructing a complete
target cache**. Solving missing boundaries remains part of the transfer research.

Split complete independent documents before collection. The probe rejects
identical document hashes across train and evaluation. It cannot detect near
duplicates, overlapping excerpts, or two documents from the same conversation;
keep those in the same split yourself. Use synthetic or explicitly chosen
calibration text, not resident user conversations.

## Prototype stages and next milestones

1. **Completed:** collect one full-attention layer from each model.
   The standard chat API and kvpark's opaque `.bin`/`.resume` snapshots are
   insufficient. This initial run used 32 training and 8 held-out evaluation documents, each
   below 512 tokens under both tokenizers. This corpus size is a smoke
   experiment, not a quality guarantee.
2. **Completed:** collect Qwen and Muse separately from the same corpus. Only one
   collector model needs to run at a time. Once both sets of arrays are saved, fitting needs neither
   model loaded. Running collection on isolated processes requires enough RAM;
   otherwise schedule sequential collection after preserving live sessions.
3. **Completed for one pair:** compare held-out value MSE against the
   training-mean baseline and inspect boundary coverage. Next compare layer
   pairs on a separate development split.
   A good reconstruction score is only a reason to continue the experiment.
4. Extend to source-layer selection, keys with model-specific positional
   treatment, and all receiver layers. Resolve unmatched target token
   boundaries and sliding-window state. Then implement cache import and measure
   receiver generation against normal prefill on held-out tasks.
5. Only after quality validation, measure complete handoff latency, including
   disk I/O, model/mapper loading, state conversion and import, and first-token
   time. Integrate validated adapters into kvpark as a separate capability with
   ordinary prefill as fallback. Keep native model-identity checks intact.

`variance_weighted_r2` is undefined for constant held-out targets and is reported
as JSON `null`; the baseline ratio is likewise `null` when baseline error is
zero. Neither metric establishes usable continuation, speedup, or a successful
Qwen → Muse transfer. Reverse transfer requires a separate experiment and must
also address Qwen's recurrent state.

## v2 pipeline

v2 adds new files; the v1 scripts above still work with the v1 collector.

| File | Role |
| --- | --- |
| `kv_probe.cpp` (`kvpark-kv-probe`) | One pass per model with modes `tokenize` (vocabulary only), `collect` (K, V and optionally Q, gate and attention output for chosen layers), `patch-score` (overwrite K/V at given positions, then score NLL, KL and top-1 against native), `save-state` (prefill, then write `llama_state_seq_save_file` files at chosen token counts) and `seed-score` (restore a state file, decode the rest, score like `patch-score`). `--cache-type f32\|f16`, `--flash-attn off\|on` and `--swa-full on\|off` select the cache layout; the defaults are the v2 settings. Supported GGUF architectures come from a table in the source: `qwen35` (Qwen3.5/3.8, Ornith), `muse-glimmer`, the dense `qwen3`, the GQA hybrids `qwen3next`, `qwen35moe`, `qwen4exp`, `plamo2`, `minimax-01`, `jamba`, `granitehybrid`, `nemotron_h`, `nemotron_h_moe`, `falcon-h1`, `lfm2`, `lfm2moe`, the MLA/DSA family `deepseek2`, `mistral4`, `deepseek32`, `glm-dsa`, `dots3note`, `kimi-linear` (an MLA K row is captured as its latent plus the pre-RoPE `k_pe`, DSA indexer keys as key `ik`) and `deepseek4` (keys `k`, `csa`, `hca`, `lid`); `kimi-k3`, `bailingmoe3` and `hy_v4` name no pre-RoPE key in their graphs and get captures from f32 state saves (`collect_v2.py --mode states`) |
| `kvstate.py` | Parses and writes llama.cpp sequence states (slot files and `llama_state_seq_get_data` blobs) for Muse's iSWA cache, Qwen3.5's hybrid memory, a dense model's plain KV cache (`qwen3`), the MLA, DSA and K-only hybrid layouts listed under [MLA, DSA and K-only hybrid states](#mla-dsa-and-k-only-hybrid-states), every recurrent family under [Recurrent families and media slots](#recurrent-families-and-media-slots), including slot files that hold images or audio, and DeepSeek V4's pooled caches under [DeepSeek V4 pooled caches](#deepseek-v4-pooled-caches) (f16, f32 and q8_0); builds Muse, dense, MLA and DSA slot files from per-layer K/V (K only for MLA and indexer rows); cuts states at a token index (`truncate_state`); RoPE (YaRN and frequency factors included) and de-rotation helpers matching the CPU runtime |
| `state_driver.cpp` (`kvpark-state-driver`) | Loads any GGUF on the CPU and drives the sequence-state API with token ids: `save` (decode, `llama_state_seq_save_file`, optionally decode a continuation and dump its logits), `restore-decode` (load a state into a fresh context, decode a continuation, dump float32 logits) and `capture` (dump named graph tensors through the eval callback; with `--state` it also saves that run's state, whose cache rows are the captured graph's). `--parallel N` (`-np`), `--seq S` (the slot every mode uses), `--rs-seq N` (rollback snapshots), `--cache-type f16\|f32\|q8_0` and `save --partial NEW_BLOB` (the PARTIAL_ONLY blob a server checkpoint keeps). No architecture table |
| `fixtures_media.py` | Media fixtures: a stdlib PNG writer, the 5120-wide tiny Qwen3.5 the local Qwen3.8 mmproj can feed, 64-wide qwen3/muse-glimmer models with vision tokens, a random tiny `qwen3vl_merger` projector, and a throwaway `llama-server` wrapper (slot save/restore, `/completion` with images) |
| `chat_apply.cpp` (`kvpark-chat-apply`), `template_check.py` | llama.cpp's own chat-template application (`common_chat_templates_apply`, as `llama-server --jinja`) and its comparison with `chat.py` renders, as text and as probe tokens |
| `check_model_support.py` | One-document end-to-end check of a model: tokenize, collect, save-state, seed-score (native, saved and rebuilt states) and `kvstate` round trip and de-rotation |
| `fetch_dataset.py`, `build_corpus.py`, [`corpus_v2.md`](corpus_v2.md) | Pinned SmolTalk2 subsets turned into the chat-format calibration corpus |
| `chat.py` | Renders conversations with each GGUF's own chat template and records the byte span of every message segment |
| `captures.py`, `collect_v2.py`, `align.py` | Capture format and collection wrapper; cross-model token alignment within message segments |
| `ridge.py`, `fit_layers.py` | Grouped-CV ridge; top-k Qwen layers to each Muse layer, separately for K and V |
| `attention_eval.py`, `inject.py` | Offline attention-output metrics; prefix injection conditions for `patch-score` |
| `run_v2.py` | Resumable orchestrator: prepare, smoke, collect-source, collect-target, align, fit, inject, attention, report |
| `reanalyze_v1.py`, `gguf_tensors.py` | Step-1 re-analysis of v1 with receiver-side and output-space baselines |

Limits of the sequence-state tools. `seed-score` checks a state file's header
and, after the restore, that `llama_memory_seq_pos_max` equals the seed length
minus one. For Muse that is the SWA cache's end and for Qwen the smaller of the
attention and recurrent ends, so it cannot see a base (Muse) or attention (Qwen)
section that reaches further. `kvstate.StateFile.to_bytes` and `write_file`
therefore refuse any state whose sections do not all end at the last token
(`check_positions=False` exists only for deliberately broken test files).
`seed-score` runs with one sequence and one stream, so a server archive saved
with more streams must be rewritten with `kvstate` before it can be restored
there. A state saved at a length off the `--batch` grid can differ from the
native pass by CPU summation-order noise (KL up to about 1e-14 on the
fixtures); on the grid the scores are identical. The forward RoPE in `kvstate`
matches the pinned CPU build bit for bit (libm `powf`/`cosf`/`sinf`, fused
multiply-adds); GPU backends round differently.

### More architectures (universal translator)

The probe, `kvstate` and `captures` also support the dense GGUF architecture `qwen3`
(Qwen3-4B-Instruct-2507: every layer full attention, 8 KV heads of 128, NEOX RoPE over the
whole head, one plain KV section in its slot files); Ornith-1.5-9B runs through the `qwen35`
path (32 trunk layers after its nextn block, 8 attention layers, IMROPE 64 of 256). Build
the probe against the Vulkan runtime, and the template reference against a runtime with
`libllama-common` (the 85d5703a3 `llama-server` build):

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/build_collector.py --llama-build evidence/linear-transfer/runtime-vulkan-t1 --target kvpark-kv-probe --output evidence/linear-transfer/kv-probe-build-u1
```

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/build_collector.py --llama-build evidence/linear-transfer/muse-server-85d5703a3 --target kvpark-chat-apply --output evidence/linear-transfer/chat-apply-build-u1
```

Render a corpus with a model's template, compare it with llama.cpp's application, and check
the model end to end on one document (GPU; the summary lands in `summary.json`):

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/chat.py --conversations evidence/linear-transfer/corpus-v2/conversations.jsonl --model-name qwen3-4b --gguf /srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf --output evidence/linear-transfer/model-support-u1/rendered-qwen3-4b.jsonl --template-output evidence/linear-transfer/model-support-u1/qwen3-4b.jinja
```

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/template_check.py --model /srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf --conversations evidence/linear-transfer/corpus-v2/conversations.jsonl --chat-apply evidence/linear-transfer/chat-apply-build-u1/kvpark-chat-apply --probe evidence/linear-transfer/kv-probe-build-u1/kvpark-kv-probe --output evidence/linear-transfer/model-support-u1/template-check-qwen3-4b.json
```

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/check_model_support.py --probe evidence/linear-transfer/kv-probe-build-u1/kvpark-kv-probe --model /srv/models/qwen3-4b-instruct/Qwen3-4B-Instruct-2507-Q8_0.gguf --docs evidence/linear-transfer/model-support-u1/rendered-qwen3-4b.jsonl --output evidence/linear-transfer/model-support-u1/check-qwen3-4b --device vulkan
```

On one device a saved state seeds a run whose scores equal the native ones exactly. On
the CPU a state rebuilt from collected `Kcur_normed`/`Vcur` with `kvstate.cache_keys` is byte
for byte the native one. The Vulkan cache rotates keys with `pow` and GPU `sin`/`cos`, so
0.2 to 0.35 % of the rebuilt f16 keys differ from Vulkan's by one unit in the last place.
On the 443/564-token check document that alone costs KL about 3.6e-3 nats per token.
Qwen3-4B's template serializes tool-call arguments with `tojson`; `chat.render` refuses a
string argument that JSON escaping would change (quotes, backslashes, control characters)
instead of locating a wrong span. No corpus v2 or long-chat argument needs escaping.

Create the research environment and fetch the data (network access required):

```bash
python3 -m venv experiments/linear_transfer/.venv
```

```bash
experiments/linear_transfer/.venv/bin/pip install -r experiments/linear_transfer/requirements.txt
```

```bash
python3 experiments/linear_transfer/fetch_dataset.py --output evidence/linear-transfer/datasets/smoltalk2
```

Build the probe against the pinned runtime, then build the corpus as described in
[corpus_v2.md](corpus_v2.md):

```bash
python3 experiments/linear_transfer/build_collector.py --target kvpark-kv-probe --output evidence/linear-transfer/kv-probe-build
```

Collection and injection load each 29 GB model on CPU and need both live model
servers stopped. The run refuses to start when available memory is below the model
size plus 6 GiB. One command runs every stage and can be rerun with `--resume` after
an interruption:

```bash
experiments/linear_transfer/.venv/bin/python experiments/linear_transfer/run_v2.py --output evidence/linear-transfer/run-v2 --probe evidence/linear-transfer/kv-probe-build/kvpark-kv-probe --corpus evidence/linear-transfer/corpus-v2/conversations.jsonl --source-model /path/to/Qwen3.8-27B-Q8_0.gguf --source-model-id Qwen/Qwen3.8-27B --target-model /path/to/Muse-Glimmer-30B-Q8_0.gguf --target-model-id meta-models/Muse-Glimmer-30B --ctx 5120 --batch 512 --threads 16 --fit-threads 16 --inject-docs 12 --max-scored 256
```

The September 28 run took about 4.9 hours on a Ryzen AI Max+ 395 with 16 threads:
3.3 h of collection, 17 min of fitting, 63 min of injection and 12 min of attention
evaluation. It wrote about 62 GB. Stages report progress as JSON lines. `--adopt`
records collection outputs produced outside `run_v2.py` after validating them.
`summary.md` in the run directory holds every table.

Run the checks from the experiment directory. Native checks run when the binaries
are supplied:

```bash
KVPARK_KV_PROBE="$PWD/evidence/linear-transfer/kv-probe-build/kvpark-kv-probe" python3 -m unittest discover -s experiments/linear_transfer -p 'test_*.py'
```

### MLA, DSA and K-only hybrid states

`kvstate` derives every layout below from GGUF metadata, mirroring the loader and
`llama_model::create_memory` of the pinned runtime, and checks it against a save of the
target (any native save; one token is enough): parsing refuses any row size, layer count,
section or `v_trans` it does not expect, and `kvstate.template_layout` reports what a
translated state has to copy (sections, layers, stream, cache type, `v_trans`, nextn layers,
recurrent row sizes).

| Memory | GGUF architectures | Sections (flags 0) | Notes |
| --- | --- | --- | --- |
| `kv` | `deepseek2` (DeepSeek V2/V3, Kimi K2.x, GLM-4.7-Flash), `mistral4`, `hy_v4` without an indexer | `kv` | K-only MLA rows `[RMSNorm(c_kv)·g (kv_lora) \| RoPE(k_pe)]`; no V blocks, but `v_trans` (= no flash attention) is still written and checked; deepseek2/mistral4 nextn layers stay in the section as zero rows |
| `dsa` | `deepseek32`, `glm-dsa`, `hy_v4` with an indexer | `kv`, `indexer` | the indexer section holds one K-only row per cell, `H·RoPE(LayerNorm(W_k x))`, on the full-indexer layers; PARTIAL_ONLY writes both |
| `dsa_iswa` | `dots3note` | `base`, `indexer`, `swa` | sliding layers have their own latent width and RoPE base; PARTIAL_ONLY writes `swa` only |
| `hybrid` | `kimi-k3`, `kimi-linear` (NoPE MLA), `bailingmoe3` | `attn`, `recurrent` | K-only attention rows and KDA recurrent rows; nextn layers stay in `attn` as zero rows (inferred from the default hybrid filter: no fixture has them) |

MLA specs never read `attention.value_length`: the cache stores no V and attention reads the
c_kv view of K as V, so the logical V width is `kv_lora` (`kv_lora_rank_swa` on sliding
layers). This matters for real GGUFs: the kimi-linear and bailingmoe3 converters leave the
base class's `head_dim` (or nothing) there, and llama.cpp saves byte-identical states either
way (`test_converter_value_length_plays_no_part`). The dots3note nextn/MTP layers are untested
too (test-llama-archs skips dots3note).

The existing `qwen3`, `muse-glimmer` and `qwen35` layouts are unchanged and still refuse RoPE
scaling. For the MLA family `ModelSpec.from_metadata(meta, runtime=...)` accepts
`rope.scaling.type` none, linear and yarn and takes the receiving server's settings
(`rope_scaling_type`, `rope_freq_base`, `rope_freq_scale` = 1 / `--rope-scale`,
`yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`, `yarn_orig_ctx`;
`None` means unset, anything else must be a finite number). `kvstate.rope_runtime` mirrors
`llama_context`: beta is 32/1 for these loaders, and with YaRN the attention factor is
overwritten, so `--yarn-attn-factor` is ignored. The k_pe magnitude `Rope.magnitude()` is 1
(up to float rounding) for arch `deepseek2` and whenever `mscale_all_dim` is 1, `(1 + 0.1 ln
f) / (1 + 0.1·m_all ln f)` for `mistral4`/`deepseek32` otherwise, and `1 + 0.1 ln f` for
loaders that do not read `yarn_log_multiplier` (`glm-dsa`, `hy_v4`, `bailingmoe3`,
`dots3note`). `cache_keys`/`pre_rope_keys` rotate only the k_pe tail of an MLA row and, with
`role="indexer"`, apply or undo RoPE plus the Hadamard rotation of the indexer keys (NEOX over
the leading dims for deepseek32/dots3note, NORM for glm-dsa, NORM over the trailing dims
without Hadamard for hy_v4). The forward rotation reproduces the CPU build bit for bit, YaRN
included (the build fuses the YaRN mix and magnitude into FMAs, which `kvstate` emulates). The
inverse always divides by cos² + sin² of the float32 cache (`universal_kv`'s backends mirror
it), which also undoes the rounding of `cosf`/`sinf` when the magnitude is 1: with f32 caches
`cache_keys(pre_rope_keys(row))` changes the last bit of 12-13% of the rotated elements
(measured on deepseek2, deepseek32, mistral4-yarn and the qwen3next, qwen35moe, falcon-h1 and
lfm2moe fixtures; 34-35% on the magnitude-1 models with the earlier plain transposed rotation)
and 38% of the indexer elements (the Hadamard product rounds too); at most ~1e-7, and logits
stay within 5e-7. Other MLA helpers: `mla_split`, `value_rows` (V of an MLA layer is its c_kv),
`mla_renormalize_latent` (divide by `attn_kv_a_norm`, rescale to RMS 1, multiply back; pass
the model's `attention.layer_norm_rms_epsilon` as `eps`; elements where the weight is 0 stay 0
and the RMS is taken over the rest), `nextn_layers`, `build_kv_stream(..., role=...)` (K-only
layers take no V, missing nextn rows are zero-filled) and `build_dense_state(...,
indexer_k=...)`.

Recurrent sections are generic: `recurrent_row_floats(meta, family)` gives the R/S row sizes
of the `ssm`, `kda`, `rwkv`, `shortconv` and `lightning` families (0-byte rows included) and
`ple_rows_from_metadata` the qwen4exp PLE row, which kvstate parses after its layer's R rows;
`recurrent_reset` zeroes every row of one cell (R0). The plain-attention hybrids and the
recurrent-only models are described under [Recurrent families and media
slots](#recurrent-families-and-media-slots). Refused: quantized caches, the GLM-5.2 built-in indexer
pattern (GGUFs without `attention.indexer.types` and a 1M context), legacy deepseek2 GGUFs
without `key_length_mla` (expanded MHA cache) and `hy_v4` with nextn layers. Write indexer
rows at their final positions: llama.cpp's K-shift of DSA indexer keys looks wrong (inferred
from the source, not run). Indexer keys change outputs only once a query sees more than
`indexer.top_k` unmasked cells; up to that any finite values (`indexer_k="zeros"`) give
bit-identical logits. Every DSA graph adds the KV mask to the scores before `ggml_top_k`, so
the KV view's padding to 256 cells does not count (measured: the driver's view is always 256
cells wide) and neither, by the same mask, do other sequences' cells in a unified cache
(inferred from the source; the driver runs one sequence).

Build a CPU test build of the pinned runtime out of tree (no download; the UI, server and
tools stay off) with the CPU flags of `runtime/build`, then the state driver against it:

```bash
cmake -S runtime/source -B /tmp/kvpark-build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_OPENMP=ON -DGGML_BACKEND_DL=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_CUDA=OFF
cmake --build /tmp/kvpark-build-tests -j 24 --target test-llama-archs llama ggml llama-common
cmake -S experiments/linear_transfer -B /tmp/kvpark-state-driver -DLLAMA_SOURCE=$PWD/runtime/source -DLLAMA_LIBRARY_DIR=/tmp/kvpark-build-tests/bin -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/kvpark-state-driver --target kvpark-state-driver
```

Fixtures: `tiny_fixture.create_mla_fixture` (or `experiments/linear_transfer/.venv/bin/python
experiments/linear_transfer/tiny_fixture.py deepseek2 OUT.gguf [--q-lora 0] [--yarn] [--nextn 1]`) writes byte-vocabulary `deepseek2`,
`mistral4`, `deepseek32`, `glm-dsa` and `dots3note` models (kv_lora 32, k_pe 16, indexer 2×64,
top_k 8); `test-llama-archs -a ARCH -s 7 -o DIR` writes random 576/512 models for every other
architecture (no vocabulary: drive them with token ids). Run the checks with both binaries:

```bash
KVPARK_STATE_DRIVER=/tmp/kvpark-state-driver/kvpark-state-driver KVPARK_LLAMA_ARCHS=/tmp/kvpark-build-tests/bin/test-llama-archs experiments/linear_transfer/.venv/bin/python -m unittest discover -s experiments/linear_transfer -p 'test_layouts.py'
```

`test_layouts.py` saves every fixture with f16/f32 caches and flash attention off/on,
re-serializes byte for byte, compares the k_pe and indexer RoPE with `capture` dumps of the
CPU build bit for bit (GGUF YaRN with all three magnitude rules, runtime YaRN settings, the
dots3note sliding base, the hy_v4 tail rotation), restores states whose K rows went through
`pre_rope_keys`/`cache_keys` and states rebuilt with `build_dense_state` (logits equal to the
native continuation within 2e-6 of their scale), restores states moved to positions + 37
(within 2e-6 with f32 and 2e-3 with f16 caches; this, unlike a same-position re-rotation,
catches wrong RoPE parameters), checks the bailingmoe3 and hy_v4 k_pe RoPE (± runtime YaRN)
and the raw kimi k_pe against the `attn_norm` capture times the GGUF's `attn_kv_a_mqa` weight
(these graphs name no k_pe node before RoPE; agreement within 2e-7 of the scale, a 7× base
misses by about 1), loads converter-shaped kimi-linear/bailingmoe3 GGUFs (`value_length`
missing or the KDA head size, written with `tiny_fixture.copy_with_metadata`), shows the
indexer is irrelevant at or below `top_k` cells and decisive above, and checks the RMS shell
of the latents. `build_collector.py --target kvpark-state-driver` builds the driver against
existing runtime libraries with a manifest, like the probe. The driver writes
every output through `NAME.partial` and hard-links it into place, so it never replaces a file
and leaves no `.partial` behind on failure; it refuses to start over an existing `.partial`.

### Recurrent families and media slots

Every recurrent family of the pinned runtime parses and writes back byte for byte (`deepseek4`,
hybrid in `llm_arch_is_hybrid` but with its own dsv4 cache, has [its own section](#deepseek-v4-pooled-caches)), and
every layout `kvstate` knows is listed here (sections at flags 0 / with PARTIAL_ONLY, the
server's checkpoint blobs). Row sizes are F32 bytes per layer and cell.

| Memory | GGUF architectures | Sections | Attention rows | Recurrent rows (R / S) |
| --- | --- | --- | --- | --- |
| `kv` | `qwen3`; `deepseek2`, `mistral4`, `hy_v4` without an indexer | `kv` / `kv` | GQA, NEOX over the head (qwen3); K-only MLA (above) | none |
| `iswa` | `muse-glimmer` | `base`, `swa` / `swa` | NoPE full layers, NORM sliding layers | none |
| `dsa`, `dsa_iswa` | `deepseek32`, `glm-dsa`, `hy_v4` with an indexer; `dots3note` | above | K-only MLA plus indexer keys | none |
| `hybrid` | `qwen35`, `qwen35moe` | `attn`, `recurrent` / `recurrent` | IMROPE; cells carry `ext` (x, y, token) | GDN: (d_conv−1)(d_inner + 2·n_group·d_state) / d_state·d_inner |
| `hybrid` | `qwen3next` | same | NEOX | GDN |
| `hybrid` | `minimax-01` | same | NEOX over `rope.dimension_count` | lightning: 0 / head_dim²·n_head |
| `hybrid` | `jamba` | same | none | Mamba-1 (n_group 0) |
| `hybrid` | `falcon-h1` | same; every layer in both sections (kind `parallel`) | NEOX | Mamba-2 |
| `hybrid` | `granitehybrid` | same | NORM with the GGUF's optional `rope_freqs.weight` frequency factors; none without `rope.scaling.finetuned` | Mamba-2 |
| `hybrid` | `nemotron_h`, `nemotron_h_moe` | same; MLP-only layers (kind `mlp`) and the converter's MTP block in neither | none | Mamba-2 |
| `hybrid` | `plamo2` | same | NEOX (none when layer 0 is Mamba: n_rot comes out 0) | SSM (the `ssm` formula) |
| `hybrid` | `lfm2moe`, `lfm2` without a sliding window | same | NEOX | short conv: n_embd·(l_cache−1) / 0 |
| `hybrid_iswa` | `lfm2` with a sliding window | `base` (no layers), `swa`, `recurrent` / `swa`, `recurrent` | NEOX | short conv |
| `hybrid_idx` | `qwen4exp` | `attn`, `recurrent`, `indexer` / `recurrent` | IMROPE; the indexer holds raw keys plus V rows nothing writes (zeros); with PLE all cells carry `ext` | GDN, plus the PLE row after its layer's R rows |
| `hybrid` | `kimi-k3`, `kimi-linear`, `bailingmoe3` | above | K-only MLA | KDA: 3·(d_conv−1)·n_head·head_dim / head_dim²·n_head |
| `recurrent` | `mamba`, `mamba2`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` | `recurrent` / `recurrent` | none: parse only, the bridge re-prefills these | Mamba, Mamba-2; RWKV: token_shift_count·n_embd / n_embd·head_size |
| `dsv4` | `deepseek4` | header, `base`, `swa`, `csa`, `hca`, `lid`, `csa_state`, `hca_state`, `lid_state` / header, `swa`, the three rings ([below](#deepseek-v4-pooled-caches)) | K-only latents of the last n_swa tokens, pooled CSA/HCA/LID rows; f16, f32 or q8_0 | none (F32 compressor rings instead) |

The bridge's engine ([../universal_kv](../universal_kv/ARCHITECTURE.md#what-the-translator-supports-today))
translates every layout here except `recurrent`, and `deepseek4` as a target only. Real models
have been run for `qwen3`, `muse-glimmer` and `qwen35`; the others are checked on tiny fixtures
only (this section, the MLA section above and DeepSeek V4's below).

Specs mirror the loaders: per-layer head counts are a scalar or an array of exactly
`block_count` values, head sizes and `n_rot` follow layer 0's head count (a layer 0 without
heads leaves them 0; `plamo2` reads the head sizes again, so a converted PLaMo-2 has no RoPE at
all, inferred), every layer rotates with `n_rot(0)`, and the cell `ext` is present exactly where
`llama_kv_cache::has_cell_ext` says (`ModelSpec.has_cell_ext(role)`, `SectionLayout.ext`). The
nextn layers of `qwen35moe`, `qwen3next`, `qwen4exp` and `minimax-01` sit in neither section,
and so does the MTP block the converter appends to a Nemotron-3 MoE (one block index listed as
an attention and an MLP layer, so `feed_forward_length` is the MoE width and nemotron's
attention filter, n_ff = 0, skips it; checked live on a test-llama-archs `nemotron_h_moe` given
MoE experts and that block by `tiny_fixture.create_nemotron_mtp_fixture`). A nemotron nextn
layer with n_ff = 0, and the nextn layers of the other hybrids, would sit in the attention cache
(seen live for nemotron): not modelled and refused. Their RoPE takes the receiving server's
settings like the MLA family (`runtime=`). `rope_freqs.weight` is an optional tensor, so only
`ModelSpec.from_gguf` knows granitehybrid's frequency factors: a rotating granitehybrid spec from
`from_metadata` refuses `rope()` (its layouts still parse) unless the caller passes
`rope_freq_factors=` (the tensor's values, or `()` when the GGUF has none). The forward
rotation reproduces the CPU cache rows bit for bit for every hybrid, with the GGUF's RoPE and
with runtime YaRN (frequency factors included); in a mutation run a 1.0001× base, NORM pairs for
NEOX or dropped frequency factors each fail it.

R0 and cutting. `recurrent_reset(part, position)` is llama.cpp's fresh-sequence state for every
family (R, S and PLE rows, 0-byte rows included). `truncate_state(state_file, cut, *,
recurrent=None, sliding="native")` cuts a state to its first `cut` token-list entries as a
native save of that prefix: KV cells at positions ≥ `MediaTokens.pos_next(cut)` go, sliding
sections keep exactly their native window, and the token words keep what lies before the cut.
A recurrent section cannot be cut: `recurrent=None` refuses ("needs R0"), `recurrent="reset"`
puts an R0 cell at the last kept position. A cut at the end (`cut == len(tokens)`, e.g.
`media_cut` of a state without media) is no cut and leaves every section as saved, recurrent
rows included. Kept cells that form one run (the usual cut, a prefix) are sliced, so their rows
stay views of the input (of a memory-mapped file: nothing is copied). A saved SWA section holds
only the last n_swa positions before its own end, so a real cut usually lacks part of the
window a native save would hold; `sliding="native"` refuses it and `sliding="keep"` keeps
whatever cells lie before the cut (for reading source rows; the result is not position-checked
and may not restore).

Media slots. With an mmproj, llama-server's slot files keep media in their token words: one
`LLAMA_TOKEN_NULL` entry per media cell, then the media start indices and each chunk as
`mtmd_input_chunk_save` writes it (type, grid nx × ny, position type, temporal merge, the chunk
id, which is the sha256 of the uploaded file, and per-entry sizes; never pixels or audio).

- `parse_media_tokens(words)` returns `MediaTokens(tokens, media, server_packed)` with one
  `MediaChunk(start, n_tokens, n_pos, modality, id, pos_type, nx, ny, blob)` per medium, after
  llama-server's own checks (every `-1` run is exactly one chunk; n_tokens and n_pos positive;
  only zero padding after the last chunk). `parse_media_chunk(blob, start)` reads one chunk.
- `MediaTokens.pos_next(n)` mirrors `server_tokens::pos_next`; `cell_positions()` gives each
  entry's cell position (all cells of an M-RoPE image share the image's first position; NORMAL
  and HunyuanVL images and audio count up); `to_words()` re-encodes byte for byte (chunk blobs
  verbatim); `truncated(cut)` keeps whole chunks only, like `server_tokens::keep_first`.
- `StateFile.media` is that table; `StateFile.tokens` and `decode_token_words` stay text-only
  and refuse media. `check_positions` ends every section at the last entry's cell position.
- `media_cut(state_or_media, before=img_beg_ids)` is the token index where the first medium's
  span begins, moved back over the projector's img_beg tokens (`<|vision_start|>` for Qwen-VL),
  which mtmd appends to the preceding text chunk.
- `handoff._stream_index(stream, what, pos_cut=None)` tolerates cells that share a position:
  it leaves them out of the index and `_cells` refuses only a lookup of such a position;
  `pos_cut` (a position, not a token index) leaves out every cell at or after it. The cells
  `truncate_state(state, cut)` keeps are those before `MediaTokens.pos_next(cut)`: after an
  M-RoPE image the two numbers differ.

Measured on throwaway CPU servers (`runtime/build/bin/llama-server`, ports 18540-18549, a
random 128 × 96 PNG, `--image-max-tokens 16`):

- The 5120-wide tiny Qwen3.5 with the local Qwen3.8 mmproj saves one image chunk of 4 × 3 = 12
  cells with n_pos 4 and the PNG's sha256 as id; the 12 cells sit at position 15 with `ext`
  (15 + column, 15 + row, −1), the text after the image continues at 19, the recurrent cell sits
  at the last position, and `cell_positions()` equals the saved positions.
- Cut before `<|vision_start|>`, its attention section and token words are identical to a native
  save of that prefix (the recurrent rows are R0). Restored, the same image request reuses
  exactly the cut. Cut right after the image, the restore needs the mmproj, keeps the chunk, and
  the same image request reuses every kept cell (the chunk matches by id and n_tokens).
- The 64-wide `qwen3` and `muse-glimmer` (window 64) fixtures with the random projector store
  the image at NORMAL positions (nx = 12, ny = 1, no `ext`); the state cut before the image is
  the native save of the prefix byte for byte. llama-server reprocesses an iSWA slot restored
  without checkpoints whose SWA cells do not reach back more than n_swa (pos_min ≥ pos_next −
  n_swa, `tools/server/server-context.cpp:3366-3445`); the native save restored with the same
  empty `.resume` behaves the same, while with the server's own `.resume` (it saves checkpoints)
  it reuses 9 cells.

Caveats. The test-llama-archs models' outputs barely depend on their recurrent state (an R0
`mamba`/`mamba2` state continues exactly like a fresh sequence, but so would a wrong one), so
those tests check layouts, acceptance and finite logits, not quality. The random `rwkv7` and
`arwkv7` fixtures move their logits by about 1e-2 with the thread count or the ubatch split, so
their R0 restores are only checked to be accepted and finite (`rwkv6`/`rwkv6qwen2`: within
1e-6 of a fresh sequence); the pinned `rwkv7` also crashes in `llama_decode` when the driver
runs it with `--batch` 1 or 2 (not investigated; test-llama-archs skips RWKV as "hanging").
The live family tests check that R0 and truncated states hold zero R, S and PLE rows of the
native sizes; in a mutation run a reset that zeroes only the R rows fails 32 of their subtests
(every family except `lfm2`/`lfm2moe`, whose S rows are 0 bytes) and a reset that zeroes
nothing fails all 36.
Not modelled: nextn layers of the default-filter hybrids (and a nemotron nextn layer with
n_ff = 0), quantized caches (except DeepSeek V4's q8_0), `ON_DEVICE` blobs.

### DeepSeek V4 pooled caches

`deepseek4` (DeepSeek V4, V4-Pro, V4-Flash; `src/llama-kv-cache-dsv4.cpp`) is memory `dsv4`.
Every trunk layer slides: a K-only raw iSWA cache keeps one latent per token (RoPE'd on its
last n_rot dims, read as both K and V) for the last n_swa tokens. Layers of compress ratio 4
(CSA, with a lightning indexer, LID) or 128 (HCA) also keep one pooled row per completed block,
and an F32 ring of the per-token compressor inputs (a, score) of the unfinished block
(`kvstate.CompressSpec` has the formulas and source references; `ModelSpec.pools()` the
geometry). Sections:

| Flags | Sections, in order |
| --- | --- |
| 0 | header `DSV4` (version 1, mode 0), `base` (cell metadata of every position, no layers), `swa` (K-only rows of every layer for the last n_swa cells, nextn/MTP layers included as zero rows), `csa`, `hca`, `lid` (`PooledRows`: n_rows = floor((pos_max + 1) / ratio)), `csa_state`, `hca_state`, `lid_state` (`CompressorRing`: 8 / 128 / 8 rows of F32 a and score) |
| PARTIAL_ONLY | header (mode 1), `swa`, the three rings (the 8-byte io prefix comes from `StateBlob`) |

The raw sections write n_stream = `-np` (the raw cache is never unified) with the slot's stream
non-empty; the row and ring sections write ns = 1 for a sequence. Rows are f16, f32 or q8_0
(`Q8_0_BLOCK`, structured numpy blocks; only these sections accept them); a quantized cache
stores raw, CSA and HCA rows Hadamard-rotated, LID rows are rotated in every cache. RoPE
(`ModelSpec.rope(layer, role)`): window-only layers rotate with `rope.freq_base` and no YaRN;
compressed layers rotate their latents ("attention"), pooled CSA/HCA rows ("pooled") and LID rows
("lid") with `attention.compress_rope_freq_base`, the context's YaRN and an attn_factor
(`dsv4_rope_attn_factor`) that cancels ggml's YaRN magnitude. `quantize_q8_0` reproduces the x86
CPU build's quantizer (a CUDA/HIP cache rounds differently, so its q8_0 rows can differ by one
step). `truncate_state` refuses these states (their pools hold every token).
`tiny_fixture.create_dsv4_mtp_fixture` adds an MTP block to test-llama-archs' deepseek4 model.

Checked (`test_kvstate.Dsv4*`, `test_layouts.Dsv4LayoutTests`, on test-llama-archs' random
deepseek4 model: layers 0-1 window only, 2 CSA, 3 HCA, window 32, head 64, n_rot 32):

- Full saves and PARTIAL_ONLY blobs parse and write back byte for byte with f16 and f32 caches
  (flash attention off and on), q8_0 (flash attention on; llama.cpp refuses a quantized V cache
  without it), `-np 2` from slot 1, `--rs-seq 3` (the layout does not change), and with the MTP
  layer, whose raw rows stay zero (the main context's raw cache keeps the nextn layer, as
  inferred from the source); also the research fixture's 31 saves and 11 blobs.
- Raw rows are f16(RoPE(kv_norm)) bit for bit, runtime YaRN included (`--rope-scaling yarn
  --rope-freq-scale 0.25 --yarn-orig-ctx 64`), and the q8_0 rows are `quantize_q8_0` of the
  Hadamard-rotated rows block for block; the rings hold the last tokens' a and APE-added score
  at row pos % state_size bit for bit.
- `check_model_support.py` runs end to end on the model given a byte vocabulary
  (`test_kv_probe_integration.KvProbeDsv4Tests`; captures.py takes `deepseek4`, the keys `csa`,
  `hca`, `lid` and the probe report's per-layer `compress`): rings exact, de-rotated latents
  within 1 ulp, and the rebuilt state (raw window from the collected latents; pooled rows stay
  native) scores exactly as the saved one.

How pooled rows are synthesized, and the restore tests against native states, are in
`experiments/universal_kv` (`dsv4_pool.py`, `dsv4_target.py`, `test_dsv4.py`).

Run the checks (from the repository root; `test_media_state.MediaSlotTests` starts
`llama-server`, the rest of the file needs nothing):

```bash
KVPARK_STATE_DRIVER=/tmp/kvpark-state-driver/kvpark-state-driver KVPARK_LLAMA_ARCHS=/tmp/kvpark-build-tests/bin/test-llama-archs experiments/linear_transfer/.venv/bin/python -m unittest discover -s experiments/linear_transfer -p 'test_layouts.py'
```

```bash
KVPARK_LLAMA_SERVER=$PWD/runtime/build/bin/llama-server KVPARK_QWEN_MMPROJ=/srv/models/qwen3.8-27b-q8/mmproj-Qwen3.8-27B-F16.gguf experiments/linear_transfer/.venv/bin/python -m unittest discover -s experiments/linear_transfer -p 'test_media_state.py'
```
