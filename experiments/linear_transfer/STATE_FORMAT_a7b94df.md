# llama.cpp state formats: 85d5703a3 → a7b94df2c

Scope: what changed between kvpark's old pin `85d5703a3b1b47243213a39059a6e3076c92733a` and the new
pin `a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` (583 upstream commits, 2026-09-05 → 2026-10-04) in
everything kvpark and the universal KV translator read or write: slot files, sequence-state blobs,
memory sections, the `.resume` sidecar, server token lists, mtmd chunks, speculative state, per-
architecture cache layouts and RoPE. It ends with the exact changes `kvstate.py`, `kv_probe.cpp`,
`state_driver.cpp` and `src/kvpark/bridge/resume.py` need. Nothing under `experiments/` was changed
in that stage; [Status](#status-october-4-2026) at the end says what the next stage changed.

Line numbers below refer to `runtime-next/src` (the new pin plus `patches/llama-slot-resume.patch`).

## How this was checked

1. Source diff of every file named below (`git diff 85d5703a3 a7b94df2c`, in `runtime-next/checkout`).
2. Live, on the new CPU build (`runtime-next/build-cpu`, `runtime-next/product`):
   * `scripts/acceptance.py` with Muse-Glimmer-30B Q8_0 (same settings as the September 27 run):
     the parked `.bin` is 114,788,284 bytes and the `.bin.resume` 164,988,268 bytes, **byte-for-byte
     the same sizes as on the old pin**; continuation reused 2,462 tokens as before.
   * `experiments/universal_kv/fixtures_bridge/live_resume_check.py` (all 7 cases pass: no draft,
     DFlash M-RoPE, DFlash NEOX, MTP, media, bridge switch into DFlash and MTP targets).
   * `evidence/llamacpp-upgrade/tools/dflash_image_check.py` (DFlash draft positions, see below).
3. The experiment suites run against new-pin binaries (`test-llama-archs` from `runtime-next/build-cpu`,
   `kvpark-kv-probe` and `kvpark-state-driver` rebuilt with `build_collector.py --runtime-revision
   a7b94df2c…` into the scratchpad). Logs: `compat-linear_transfer-newbins.log` (407 tests: 4
   failures, 34 errors, 21 skipped) and `compat-universal_kv-newbins.log` (152 tests: 10 failures, 9
   errors, 14 skipped). With the old binaries both suites pass unchanged. Every failure maps to one
   of the causes listed under "Changed" or "Test fixtures" below.

## Unchanged (byte formats)

| Item | New-pin location | Status |
| --- | --- | --- |
| Slot file: `ggsq` magic, `LLAMA_STATE_SEQ_VERSION 3`, token words, payload | `include/llama.h:48-49`; `src/llama-context.cpp:3324-3442` | unchanged |
| Sequence blob: io magic `0xaf143cd8`, `i32 seq_id`, payload | `src/llama-context.cpp:3190-3260` | unchanged (a failed `set_data` now calls `io.discard()`) |
| State flags | `include/llama.h:919-929` | unchanged; `LLAMA_STATE_SEQ_FLAGS_SWA_ONLY` added as an alias of `PARTIAL_ONLY` (both 1) |
| KV section (meta, data, `v_trans`, row headers, SWA-masked cells skipped) | `src/llama-kv-cache.cpp:2065-2131, 2211-2341` | unchanged |
| `llama_kv_cell_ext` (x, y, tok) and when it is present (`n_pos_per_embd() > 1` or PLE) | `src/llama-kv-cells.h`, `src/llama-hparams.cpp:285-288` | unchanged (`GGML_MROPE_SECTIONS` = 4) |
| Recurrent section | `src/llama-memory-recurrent.cpp:784-…` | unchanged |
| Section order: iSWA, hybrid, hybrid-iSWA, DSA, DSA-iSWA, DSV4 (`'DSV4'` v1), hybrid-idx | `state_write` in the respective files | unchanged |
| `server_tokens::serialize`, `SERVER_TOKENS_STATE_VERSION 1` | `tools/server/server-common.cpp:286, 570-640` | unchanged |
| mtmd chunk serialization, `MTMD_SERIALIZATION_VERSION 2` | `tools/mtmd/mtmd.cpp:30, 2413-2460` | unchanged |
| EAGLE3 speculative state (`pos`, `g_embd` row) | `common/speculative.cpp:944-978` | unchanged |
| `.resume` sidecar (magic `0x51534c54`, version 1, draft blob, speculative blob, checkpoints) | `tools/server/server-context.cpp:42-122` (patch) | **unchanged**: the rebased patch produces a source file identical to applying the old patch with offsets; the server-side structs it serializes (`common_prompt_checkpoint`) did not change |
| DFlash draft memory (iSWA, empty base, every layer sliding) and MTP draft memory for qwen35 (one full-attention layer) | live check | unchanged (`classify_draft` gives `dflash` and `mtp` on the new server) |

## Changed

### 1. NextN (MTP) layers leave the trunk cache of every non-hybrid architecture

`llama_model::create_memory` (`src/llama-model.cpp`, the plain-KV / iSWA branch): the filter
`il < n_layer()` used to apply only to `step35`, `hy-v3`, `glm-dsa`, `mimo2` and `deepseek32`. It
now applies whenever `n_layer_nextn > 0 && n_layer() > 0 && router_layer < 0`. The main context
of `deepseek2`, `mistral4`, `glm4`, `glm4-moe`, `bailingmoe2`, `exaone4`, `exaone-moe`,
`cohere2moe`, `openai-moe` and every other architecture with nextn layers therefore **no longer
writes the zero nextn rows** in its KV section; an MTP context holds only the nextn layers, as
before for the listed five.

Symptom with new binaries: `kvstate.StateFormatError: kv stream 0 holds 2 layers; the model's kv
cache has 3` (deepseek2-nextn fixtures; 15 tests across both suites).

### 2. qwen4exp indexer cache: K-only rows, wider with k-pooling; GLM5-Next uses the same memory

`llama_memory_hybrid_idx` (`src/llama-memory-hybrid-idx.cpp:44-60`):
* the indexer cache is now created as an MLA-style cache (`n_embd_head_k_mla_impl` /
  `n_embd_head_v_mla_impl` set), so it holds **no V rows** (upstream 311d4211b);
* with k-pooling (`attention.indexer.kpool > 0`), each cached row holds `indexer_kpool_row`
  head-size slots side by side: qwen4exp `key | pooled`, GLM5-Next `key | gate | pooled`
  (`hparams.indexer_kpool_row`, default 3; `indexer_kpool_select_tail`, `indexer_kpool_by_order`);
* section order is unchanged: `[attention KV] recurrent [indexer]`, the indexer gated by
  `PARTIAL_ONLY` like the attention cache; restore adopts the attention cache's cell layout.
* The qwen4exp **MTP draft context** now holds the MTP block's attention and indexer and no
  recurrent layer (`create_memory`, the `LLAMA_CONTEXT_TYPE_MTP` branch for `QWEN4EXP`): its draft
  blob is a hybrid-idx blob with an empty recurrent part, a new shape for the bridge.
* New architecture `glm5-next` (GLM-5.3-Flash) uses `llama_memory_hybrid_idx` with the same
  filters (and MTP via its nextn block).

Symptoms: `indexer stream 0 layer 1 K: row size 1024 does not match 512` (f32) / `512 … 256` (f16),
`payload does not walk` (qwen4exp row sizes), a truncated qwen4exp state the driver can no longer
load.

### 3. bailingmoe3 uses M-RoPE when the GGUF carries rope sections

`llama_model_rope_type`: `BAILINGMOE3` returns `MROPE` when `hparams.use_mrope()` (Ling 3.0 VL
files carry `[t, h, w]` sections), `NORM` otherwise. M-RoPE gives cells the
`llama_kv_cell_ext`, so the attention section of such a state has 3 extra `i32` per cell, and
text positions are broadcast over the sections like Qwen's. Symptom (the new `test-llama-archs`
fixture has sections `[8, 12, 12, 0]`): `attn stream 0 cell 1 belongs to N sequences`, i.e. the
reader does not expect the extension (16 tests).

### 4. Gated DeltaNet q/k normalization (numerics, not format)

Upstream 5fdfa6282 replaces `ggml_l2_norm` (x / max(‖x‖, eps)) with `x · rsqrt(Σx² + eps)` for
qwen35, qwen35moe, qwen3next, qwen4exp, kimi-k3, kimi-linear and bailingmoe3. Recurrent states and
outputs of these models differ from the old pin by rounding-level amounts. Saved states keep their
format; R2 replay references must put eps inside the root.

### 5. Restore failure handling and checks

* `state_read_meta` (KV and recurrent) refuses a single-sequence restore whose `cell_count`
  exceeds the cache size instead of failing later.
* A failed restore now zeroes the cells it had written (`llama_kv_cache::state_clear`,
  `llama_memory_recurrent::state_clear`), and DSA / hybrid / hybrid-idx memories undo the parts
  already restored when a later part fails. Readers and writers are unaffected; a bad translated
  state can no longer leave stale K/V data in free cells.
* A recurrent memory whose filter keeps no layer (an MTP draft of a hybrid model) disables
  rollback snapshots, and a partial `seq_rm` only moves its position.

### 6. Speculative decoding (DFlash) positions

* `common/speculative.cpp:1157-1240` (`process`): target features are injected into the draft at
  the **target's own positions** (`pos[0]` of the batch, broadcast to the four M-RoPE sections),
  and an embedding batch whose rows all share one position (an M-RoPE image) is **not injected**
  (upstream fa6769818). The draft's cache therefore has a gap of one image's positions after
  each image; text after the image continues at the target position.
* The server passes `pos0 = slot.prompt.tokens.pos_next()` (a position, not a token count) to every
  drafter (upstream b0dcb8192; `server-context.cpp`), so drafting after an image starts at the right
  position.
* The production-only `llama-dflash-contiguous-positions.patch` (draft-local counter) is
  superseded and must not be ported: on the old production binary an empty draft after a restore
  restarts the counter (measured: target at position 2,689, draft cells at 0-169), while the new
  pin keeps the draft at the target's positions with an empty draft, the aligned dummy, or a
  native draft (`dflash-fixture/*.json`).

### 7. API and server surface (no format change)

* `llama_batch_ext` / `llama_process()` were added and the server, mtmd and speculative code moved
  to them. `llama_batch_init`, `llama_decode` and `llama_batch` still work (a compat wrapper);
  `kv_probe.cpp`, `state_driver.cpp`, `collector.cpp` and `chat_apply.cpp` compile unchanged
  against the new headers and libraries.
* `mtmd_helper_post_decode_callback` now receives a `const mtmd_helper_embd_batch *` (n_tokens,
  embd, n_embd, section-major pos, n_pos, seq_id) instead of a `llama_batch`.
* Removed CLI flags: `--mmap`, `--no-mmap`, `--mlock`, `-dio/--direct-io`, `-ndio/--no-direct-io`
  (use `--load-mode`). kvpark already uses `--load-mode`.
* `/props` reports `modalities.video` for a Qwen3-VL-style mmproj (mtmd video via ffmpeg frames,
  each frame id `{sha256}+{frame}`); the bridge's capability check sees `vision: true, video: true,
  audio: false` for the Qwen3.8 projector.
* New server features that do not touch slot state: `/v1/systemone`, decision tasks
  (`server-decision.cpp`), shared prompt prefixes for child tasks, probabilistic drafting
  (`--spec-draft-sampling probabilistic`, temperature > 0 only), checkpoint de-duplication at the
  same `n_tokens`.
* `llama-server --version` reports the enclosing Git repository's commit when the source is a
  plain export inside another repository (pre-existing; `runtime-next/product` reports
  `b34-b14e742`, the worktree builds report `b11398-a7b94df2c`).

### 8. Architectures

| Architecture | Change relevant to states |
| --- | --- |
| qwen35, qwen35moe, qwen3next | GDN norm (§4); optional fused QKV (`build_qkv`, the `Kcur`/`Vcur` names stay on the projection outputs for unfused GGUFs); optional `cls_out` embedding projection. Cache layout unchanged. |
| qwen4exp | §2 (indexer K-only, k-pool rows), MTP added, QSA rewritten (pooled top-k over kpool blocks). |
| glm5-next (new) | hybrid-idx memory, NONE rope type for the trunk, MTP. |
| deepseek2, mistral4 | §1 (no nextn rows in the trunk); OCR path uses `build_qkv`. |
| deepseek4 | `build_hc_mean`; `dsv4_hc_pre/post` kernels gained a gated mode and identity mixing (defaults unchanged); cache layout unchanged. DeepSeek V4.1 (`deepseek41`) is still not in llama.cpp. |
| bailingmoe3 | §3 (M-RoPE with sections), §4. |
| kimi-k3, kimi-linear | §4; Kimi-K3 recurrent rollback snapshots (no format change). |
| mimo2 | DFlash feature extraction (`t_layer_inp`), MTP-only GGUFs, SWA pattern must be an array. |
| gemma4 | SWA pattern must be an array (`get_arr`); prefetch rows; layout unchanged. |
| muse-glimmer, laguna | `load_swa_pattern` (array, else scalar period 4): same result for existing GGUFs. |
| minimax-m3 | precision policy only. |
| New: clef (IMROPE), maple, hrm_text, spark2_5 (NEOX) | not handled by kvstate; clef is Qwen-like (IMROPE). |

RoPE: no change in the CPU rope kernels (`ggml/src/ggml-cpu/ops.cpp`), `ggml_rope_yarn_corr_dims`
or the rope types of the models kvstate supports, except bailingmoe3 (§3).

### 9. Test fixtures (`tests/test-llama-archs.cpp`)

* DSA fixtures: `attention.indexer.top_k` is now **131072** (was 8), so "indexer keys matter only
  above top_k" no longer has an above-top_k regime at fixture sizes.
* qwen4exp / glm5-next fixtures: `indexer.kpool = 4`, `kpool_select_tail = true`.
* bailingmoe3 fixtures: rope sections `[8, 12, 12, 0]` (M-RoPE).
* Tensor initialization stdev is a parameter (`-d/--stdev`, default 0.1).

Numerical knock-on effects seen with the new binaries: the DeepSeek V4 oracle tests'
synthesized-vs-native continuation error is 2.5e-6 to 1.4e-5 against a 1e-6 relative tolerance,
and the "zeroing LID rows/ring must move the logits" control no longer moves them (consistent with
the fixture changes above); the Mamba R0 restore equals native up to 1 float32 ulp in 31 of 384
logits instead of exactly; the "attention-insensitive fixtures" set in `test_families.py`
changes. These need re-baselining, not format work.

## What must change, per file

### `experiments/linear_transfer/kvstate.py`

1. **NextN filter** (§1): drop `"nextn"` from `LAYER_KINDS` for `deepseek2` and `mistral4` (and
   for any plain-KV/iSWA architecture added later) so that their KV section holds `n_layer()`
   trunk layers; extend `NEXTN_FILTERED` to "every non-hybrid architecture with
   `nextn_predict_layers > 0`, no router layer and a trunk". Keep `nextn` for the KDA hybrids
   (`kimi-k3`, `kimi-linear`, `bailingmoe3`): their attention filter still uses the hybrid
   defaults (the new suites pass for kimi). For `deepseek4` re-check the raw K-only cache: its
   tests still parse, but the nextn handling should be read again in `llama-kv-cache-dsv4.cpp`.
   Update the docstring ("deepseek2 and mistral4 keep a zero nextn layer…").
2. **hybrid_idx indexer** (§2): `IndexerSpec` for qwen4exp gets `value_size=0` (no V rows: read and
   write the indexer section like an MLA K-only section) and a key row width of
   `indexer.key_length × (kpool_row if kpool > 0 else 1)` with `kpool_row` 2 for qwen4exp
   (`key | pooled`) and 3 for glm5-next (`key | gate | pooled`); read `attention.indexer.kpool`,
   `kpool_select_tail` and `kpool_by_order` from the GGUF. Add `glm5-next` to `MEMORY`
   (`hybrid_idx`), `ROPE_TYPES` (`none`) and `LAYER_KINDS` once a fixture exists.
3. **bailingmoe3 rope** (§3): `ROPE_TYPES["bailingmoe3"]` becomes "mrope when
   `rope.dimension_sections` is present, else norm", and the cell-extension rule follows it.
4. **R0 / recurrent numerics** (§4): no format change; any bit-exact reference that recomputes
   GDN must use `rsqrt(Σx² + eps)`.
5. Update the source line references in the module docstring (all paths now under the new pin; the
   numbers in the "Unchanged" table above are a starting point) and the header line
   "pinned runtime (85d5703a3)".

### `experiments/linear_transfer/kv_probe.cpp`

* Compiles and links unchanged against the new pin (`experiments-probes-build.log`); the capture
  tests for qwen3, qwen35, muse-glimmer, deepseek4 and the recurrent families pass with the new
  binaries.
* Needs: the qwen4exp indexer capture and save-state path must follow §2 (K-only rows, k-pool row
  width); the DSA/MLA "nextn" rows expectation must follow §1; bailingmoe3 cell extension (§3).
* For fused-QKV GGUFs (new `convert --fuse-qkv`) `Kcur`/`Vcur` become `cont` views of `wqkv`
  instead of `MUL_MAT` outputs; the op filters only need extending if such GGUFs are used.
* `experiments/linear_transfer/build_collector.py` defaults `--runtime-revision` to
  `85d5703a3…` and `--llama-source/--llama-build` to `runtime/source`/`runtime/build`; update
  both when `runtime/` is rebuilt at the new pin. (Done: the revision comes from the build info,
  and since October 5, 2026 `runtime/` is a `scripts/build_runtime.py` build of a7b94df2c; the
  85d5703a3 runtime is kept at `runtime-85d5703a3/`.)

### `experiments/linear_transfer/state_driver.cpp`

* Compiles unchanged; save / restore-decode / capture work with the new library (its failures in
  the compat run are all kvstate parse errors of §1-3, plus one qwen4exp truncated state that the
  new library refuses to load because the state kvstate wrote has the old indexer layout).
* Optional: switch to `llama_batch_ext` / `llama_process` (the old batch API is a compat path now).

### `src/kvpark/bridge/resume.py`

* Format code needs **no change**: the `.resume` layout, the DFlash (iSWA, empty base) and the qwen35
  MTP (one-layer KV) draft blobs are the same, and the live switch checks pass on the new server.
* Docstring: update `server-context.cpp:41-121, save 2650, restore 2715` to `42-122, save 2805,
  restore 2870`; the remark "the production build counts draft positions from the draft's own last
  position" applies only to the old production binary. On the new pin the draft uses the
  target's positions, so the aligned dummy still lines up and an empty marker would also be
  correct for DFlash (measured on the fixture), but the aligned dummy remains the safe default
  for NEOX (non-M-RoPE) drafts, which require `next position = last + 1`.
* `classify_draft`: a qwen4exp MTP draft now produces a hybrid-idx blob (KV + empty recurrent +
  indexer), so it would be classified `other` and the bridge would skip such targets (inferred from
  the source; no qwen4exp MTP draft was run). Teach it this layout if qwen4exp MTP targets are wanted.
* A DFlash draft now has a position gap after each M-RoPE image (image rows are not injected).
  `_fits` in `parse_state_blob` assumes the draft holds positions `0..n_tokens-1`; for a save with
  images that is no longer true, so `n_tokens` cannot settle an ambiguous reading of such a blob
  (the parse still succeeds when only one reading fits, as on the fixture). Use the target's
  `pos_next` there instead of the token count.

### `experiments/universal_kv` (later stage)

* `fixtures_bridge/make_fixtures.py` and `live_resume_check.py` import gguf-py and the test image
  from `runtime/source`; both work with the new tree (`runtime-next/src`) as well.
* Re-baseline the DSV4 oracle tolerances and the attention-insensitivity set in
  `test_families.py` after rebuilding fixtures with the new `test-llama-archs` (§9).

## Status (October 4, 2026)

The changes above, as made (both pins' states read and written; old-pin states parse and
re-serialize byte for byte; README.md, [Tests on both pinned runtimes](README.md#tests-on-both-pinned-runtimes)):

- `kvstate.py`: `LLAMA_REVISIONS`, `ModelSpec.llama_revision` (default a7b94df2c; `from_metadata` /
  `from_gguf(..., llama_revision=)` also take a commit or llama-server's `build_info`,
  `llama_revision_of`), `for_revision`, `layout_variants`. The parsers read a state with the spec's
  revision, else with the other one where the model's layouts differ, and the parsed state's spec is
  the reading that fits (`strict=True`: the spec's reading only). §1: `ModelSpec.nextn_cached` (deepseek2/mistral4's plain cache keeps the
  nextn layers on 85d5703a3 only; the KDA hybrids and DeepSeek V4's raw cache on both). §2:
  `IndexerSpec.kpool`/`kpool_row` (from qwen4exp's `attention.compress_ratios`),
  `ModelSpec.indexer_row_slots`; the a7b94df2c indexer section is K-only with `[key | pooled]` rows,
  `cache_keys` writes zero pooled slots (a restore re-pools every block) and `pre_rope_keys` drops
  them. §3: `model_rope_type` (bailingmoe3: `mrope` on a7b94df2c with rope sections), `Rope` mode
  `mrope` (NEOX pairs, sections in runs, `rope_components`), M-RoPE cell extensions. §4: no code
  recomputes GatedDeltaNet. §5: documented. Docstring line references are a7b94df2c's.
- `translate.py`: the template's reading gives the target layout and the source state's the RoPE
  (`state_side`); torch and ggml backends rotate `mrope`.
- `kv_probe.cpp`: the GPU report's `graph_splits` also reads a7b94df2c's graph summary line
  (`graph: nodes = ..., splits = N`; 85d5703a3 printed `graph splits = N`); otherwise comments only.
  `state_driver.cpp`: comments only. Both build against both pins; the a7b94df2c builds (CPU probe,
  driver and value collector, Vulkan probe) are in `runtime-kvtools/`. `build_collector.py` takes
  the revision from the build info when it names a known pin, refuses a `--runtime-revision`
  naming the other one, and requires `--runtime-revision` when the build info names none (a
  runtime built before `scripts/build_runtime.py` set `LLAMA_BUILD_COMMIT`, such as the 85d5703a3
  one at `runtime-85d5703a3/`); the manifest records `runtime_build_info`.
- Tests: fixtures from `tiny_fixture.llama_archs_model` (N(0, 0.01) weights and indexer top_k 8 on
  both pins), the driver's revision from `tiny_fixture.driver_revision` (the layout of a save,
  checked against a manifest), per-revision expectations for the nextn layer, qwen4exp's indexer
  rows and the attention-insensitive hybrids. A bailingmoe3 copy with rope sections `[4, 6, 3, 5]`
  checks a7b94df2c's M-RoPE past the w run (unrotated fourth-component pairs, sections wrapping
  around their sum) against llama.cpp's k_pe rows, and gives an 85d5703a3 save of a model with
  rope sections (NORM, read as 85d5703a3 with a spec of either revision). With those
  fixtures the DeepSeek V4 oracle tolerances and the bit-exact Mamba R0 restore hold unchanged on
  a7b94df2c (the misses of the compat run above came with N(0, 0.1) weights and top_k 131072).
- `src/kvpark/bridge/resume.py`: docstring (line references, the draft-position remark, the qwen4exp
  MTP draft) and an unreadable draft refused with ValueError in `translated_draft`.

Not done in that stage: qwen4exp MTP drafts (`classify_draft` calls them "other"; inferred, no
model), `_fits` with an image's position gap (templates are text-only), fused-QKV op filters in
the probe, the `llama_batch_ext` API in the C++ tools (the old API keeps them building against
85d5703a3).

### New cache architectures (stream 1b)

No new difference between the pins: the architectures below write the same sections and rows on
both (where 85d5703a3 has them), so `layout_variants` gives one reading (one exception refused,
below).

- The GQA family (`kvstate.GQA_ARCHITECTURES`, README "GQA family, Gemma 4 and GLM5-Next"):
  `llama`, `mistral3`, `qwen3moe`, `hy_v3`, `phi3` (plain cache) and `gemma3`, `gemma4`,
  `gpt-oss`, `laguna`, `mimo2` (iSWA with a window). Per-layer KV heads and head sizes,
  per-layer-kind RoPE (`GQA_CONVENTIONS`, `ModelSpec.n_rot_swa`), frequency factors
  (`ROPE_FREQ_FACTOR_ARCHITECTURES`: gemma4's proportional RoPE on its full layers; read by
  `gguf_rope_freq_factors`, which a spec built from metadata alone needs, phi3's included, since
  `ROPE_FACTOR_TENSOR_ARCHITECTURES` refuse `rope()` with unknown factors), gemma4's shared KV
  layers in no section. Pin differences that leave states alone: a7b94df2c reads gemma4's
  and mimo2's `sliding_window_pattern` as flags only (85d5703a3 also took one value for every
  layer); its plain/iSWA trunk filter would drop nextn layers that 85d5703a3 keeps (refused for
  these models except mimo2 and hy_v3, which both pins filter). One that would not: a7b94df2c
  reads gemma3's, gpt-oss's and laguna's pattern as flags or a period, 85d5703a3 as a period only
  (flags leave its default period). Their converters write no pattern, so `from_metadata` with
  85d5703a3 refuses flags that disagree with the default period instead of modelling the
  difference (`for_revision` keeps the layer kinds of the spec it is given).
- `glm5-next` (a7b94df2c only, `ARCHITECTURE_SINCE`): `hybrid_idx` memory, NoPE MLA latent rows
  without k_pe, KDA rows, K-only `[key | gate | pooled]` indexer rows (`indexer_key_slots`: key and
  gate are per-token data, the pooled slot is re-pooled on restore).
- Gemma 4's MTP head (`gemma4-assistant`): its draft cache views the target's last sliding and full
  layers and writes nothing (`src/llama-kv-cache.cpp:2066-2069`), so a `.resume` draft blob is 8
  bytes; `kvpark.bridge.resume` classifies it `shared` (measured on the real Gemma-4 26B-A4B and
  31B servers).
- `kv_probe.cpp`: rows for these architectures (gemma4's V is `Vcur_normed`, a bare RMS norm),
  per-layer head sizes on sliding layers, the `shared` layer kind, and a `collect` option to save
  the capture run's own state (`"save_at": [token count]`). With a fused `attn_qkv` (Phi-3 and
  MiMo-V2.5 GGUFs), build_qkv's K and V are views the existing op filters take (llama and phi3
  fused copies checked on both pins), and mimo2's Q, K and V, unnamed views, are read as columns
  of the projection `wqkv` (`arch_def::fused_keys`; several keys share one node). The tools now
  know the pin they are built against (`KVPARK_LLAMA_REVISION`, passed by `build_collector.py`):
  an 85d5703a3 probe refuses gemma3/gpt-oss/laguna pattern flags that disagree with the default
  period, as kvstate does.
- `kvpark.bridge.capabilities`: the GQA family is translatable but for `ROPE_FACTOR_GQA` (llama,
  mistral3, gemma4, phi3): `experiments/universal_kv/engine.py` builds its specs from metadata and
  does not read their factor tensors yet, so their jobs would fail in the engine after the source
  slot is saved; they join once the engine passes `kvstate.gguf_rope_freq_factors`.

Still not done: `minimax-m3` (its MSA memory, `llama_kv_cache_msa`: a GQA cache plus an indexer
cache with V rows), qwen4exp MTP drafts.
