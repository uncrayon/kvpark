# First Qwen → Muse prototype results

The offline prototype completed on September 27, 2026 using the real local Q8
checkpoints. A one-layer affine value mapper reduced held-out squared error by
**39.96% relative to a training-mean baseline**. This establishes a measurable
cross-model reconstruction signal on this corpus. It does not establish useful
generation from a transferred cache or a prefill speedup.

Separately, kvpark passed a real Muse native disk park/resume test on the pinned
CPU runtime. The mapper and native snapshot restoration are separate paths;
kvpark's existing model-identity checks remain intact.

## Calibration and evaluation

The run used the fixed [version 1 corpus](corpus_v1.md): 32 training documents
and eight held-out documents. Both models processed the exact raw UTF-8 text
without chat templates or special tokens, with state cleared between documents.
Only equal document hashes and exact token-end byte offsets were paired.
The layer pair and ridge coefficient were fixed before evaluating this run.

| Setting | Source | Target |
| --- | --- | --- |
| Model | Qwen3.8-27B Q8_0 | Muse Glimmer-30B Q8_0 |
| Architecture | Qwen35 hybrid recurrent/full attention | Muse sliding/full attention |
| Selected layer, zero-based | 31, full attention | 27, full attention |
| Value shape per token | 4 heads × 256 dimensions | 2 heads × 128 dimensions |
| Captured tokens | 10,138 | 9,924 |
| Training tokens | 8,026 | 7,862 |
| Evaluation tokens | 2,112 | 2,062 |

The temporary collectors ran sequentially on CPU with context 512, batch 256,
12 threads, memory-mapped weights, no weight repacking, F32 K/V storage, no flash
attention, and no draft model. They used the pinned llama.cpp commit
`85d5703a3b1b47243213a39059a6e3076c92733a`. The run manifest and captures include
checkpoint, executable/library, corpus, and source-code fingerprints.

The fitter used 7,810 aligned training rows, all 1,024 source value coordinates,
256 target coordinates, a free bias, and ridge coefficient 0.01 with the summed
squared-error objective. It solved in float64. The resulting mapper file is
2,125,416 bytes, about 2.03 MiB.

| Held-out measurement | Result |
| --- | ---: |
| Aligned token boundaries | 2,050 |
| Target-token coverage | 99.418% |
| Mapper MSE | 0.155019 |
| Training-mean baseline MSE | 0.258188 |
| Mapper MSE / baseline MSE | 0.600411 |
| Variance-weighted R² | 0.397568 |

Both target heads improved over their respective training-mean baselines:

| Target head | MSE | MSE / baseline | R² |
| --- | ---: | ---: | ---: |
| 0 | 0.153021 | 0.513348 | 0.485326 |
| 1 | 0.157018 | 0.719298 | 0.277510 |

Every held-out document improved over baseline. The Spanish oral-history
document was the weakest case by relative error and R².

| Document | Aligned rows | MSE / baseline | R² |
| --- | ---: | ---: | ---: |
| eval01: DNS diagnostics | 247 | 0.618970 | 0.355359 |
| eval02: build scheduling | 259 | 0.576825 | 0.404289 |
| eval03: balloon telemetry | 258 | 0.574159 | 0.403711 |
| eval04: keyboard navigation | 258 | 0.560675 | 0.425475 |
| eval05: Spanish oral history | 263 | 0.675923 | 0.293055 |
| eval06: linear-light compositing | 258 | 0.617968 | 0.360957 |
| eval07: musical rhythm | 257 | 0.619372 | 0.357460 |
| eval08: council grants | 250 | 0.569567 | 0.413678 |

## Application without loading models

`apply_mapper.py` loaded only the saved mapper and source evaluation capture.
It produced `[2112, 2, 128]` predicted values on Qwen's token-end grid. An
independent comparison of the saved predictions with Muse's measured values
reproduced the fit report's MSE on all 2,050 shared boundaries. The run's pinned
source-code hashes were also unchanged.

One matrix application took 66.8 ms with four OpenBLAS threads. This is a single
observation covering matrix multiplication and bias only, excluding input
conversion, loading, saving, cache import, and inference. It is not an end-to-end
latency benchmark. Preparing calibration observations required running both
models; applying an existing mapper to saved source values does not load either model.

## Native Muse park/resume

The separate CPU acceptance used F16 K/V and a 4,096-token context. Its cold
prompt contained 2,467 tokens, crossing Muse's 2,048-token sliding window.
After parking and restarting both private backend and proxy processes with
RAM prompt caching disabled, the continued transcript reused 2,462 tokens and
processed 38. Its full returned reasoning message, under an eight-token output
budget, exactly matched the resident-cache continuation. Both `.bin` and
`.bin.resume` files survived.

See [validation](../../docs/validation.md#muse-cpu-snapshot-acceptance) for the
command, timing scope, and limits. This did not validate or configure the
existing Vulkan/DFlash deployment, and the short output budget did not test
final-answer accuracy.

## Artifacts and checks

Local generated evidence is under `evidence/linear-transfer/` and is not shipped
with the package:

- `prototype-v1/run.json`: immutable input fingerprints and collection settings.
- `prototype-v1/source-{all,train,eval}.npz` and corresponding `target-*.npz`:
  measured captures, including original token IDs and byte boundaries.
- `prototype-v1/fit/mapper.npz` and `fit/report.json`: mapper and full metrics.
- `prototype-v1/predictions.npz` and `application-verification.json`: standalone
  application and independent numerical comparison.
- `muse-real-acceptance-v2.json`: successful real Muse native restoration.
- `prototype-tests.log`: all 37 experiment tests passed, including four native
  collector checks. The core suite separately ran 163 tests, with one skip and
  no failures. Portability and whitespace checks passed.

The [README](README.md) contains build, collection, fit, and application commands.

## What this result supports next

Continue with affine mapping before investing in C2C training: this simple fit
already beats its baseline across all eight held-out documents. First expand
the calibration corpus, establish a fresh test split, and compare candidate
source layers using a separate development split. Reusing these eight documents
to select layers or regularization would make them development data.

Then evaluate keys with the correct model-specific positional treatment and
cover all receiver layers, including sliding-window state and missing target
token boundaries. Only a complete imported cache followed by scored receiver
generation can establish usable transfer. Compare that generation with normal
prefill before measuring handoff latency or integrating a production switch.

This run covers short synthetic prose and one direction, Qwen → Muse. Twelve
held-out Muse token boundaries remain unmatched. Other layers, keys, long
contexts, live F16 GPU caches, generation quality, and runtime import remain
untested by the mapper. Reverse transfer also needs Qwen's recurrent state.
