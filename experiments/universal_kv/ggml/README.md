# libkvbridge: the translation engine on ggml (Vulkan)

A small C library that runs the universal KV translator's products as ggml graphs, and the
engine backend that loads it: [`backends/ggml_backend.py`](../backends/ggml_backend.py)
(`GgmlBackend`, name `ggml`). It needs no torch and no ROCm or CUDA. ggml's Vulkan backend runs
on AMD, NVIDIA and Intel GPUs, so the bridge's GPU engine should work anywhere a Vulkan driver
does (tested on RADV only; see Limits). Without a Vulkan device the library falls back to ggml's
CPU backend, but only when asked to (`KVPARK_GGML_DEVICE=cpu` with device `vulkan`): on a CPU the
engine uses numpy.

| File | Purpose |
| --- | --- |
| kvbridge.cpp | the library: devices, device arrays and weights, `kvb_encode`, `kvb_section`, `kvb_cache_rows`, `kvb_rotate` |
| CMakeLists.txt | builds `libkvbridge` with ggml from source (static, symbols hidden) or against an installed ggml |
| ../backends/ggml_backend.py | the engine backend (NumpyBackend's interface) on the library, through ctypes |
| ../test_ggml_backend.py | the backend against kvstate and numpy; skipped when the library is not built |

## Build

ggml comes from upstream llama.cpp at the commit the rest of the repository is moving to,
a7b94df2c616bc1f62a73b964b4a71cb0dcc488e. Everything lands in `runtime-ggml/`, which is gitignored.
You need CMake ≥ 3.18, a C++17 compiler, the Vulkan headers and loader, and `glslc` (Ubuntu:
`libvulkan-dev glslc`).

```sh
cd kvpark
git init -q runtime-ggml/llama.cpp
git -C runtime-ggml/llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
git -C runtime-ggml/llama.cpp checkout -q FETCH_HEAD
cmake -S experiments/universal_kv/ggml -B runtime-ggml/build -DCMAKE_BUILD_TYPE=Release \
      -DKVBRIDGE_GGML_SOURCE=$PWD/runtime-ggml/llama.cpp/ggml
cmake --build runtime-ggml/build --target kvbridge -j
```

The build makes `runtime-ggml/build/libkvbridge.so`, about 47 MB because the Vulkan shaders are
compiled in. On this machine it took 66 s with ccache warm and 24 jobs. ggml is linked in
statically and its symbols are hidden (`--exclude-libs` on Linux, an exported-symbol list on
macOS), so a process that loads another libggml keeps its own copy. `-DKVBRIDGE_VULKAN=OFF`
builds a CPU-only library. ggml's Metal backend is off unless `-DKVBRIDGE_METAL=ON`, so on macOS
the GPU is Vulkan through MoltenVK. Without `KVBRIDGE_GGML_SOURCE`, CMake links an installed ggml
through `find_package(ggml)` instead. The CPU backend is built with `-march=native` (ggml's
`GGML_NATIVE`): pass `-DGGML_NATIVE=OFF` for a library you copy to other machines. A clean
configure and build took 70 s here (ccache warm, 16 jobs), with no warnings under `-Wall -Wextra`.

## Use

| Setting | Effect |
| --- | --- |
| device `vulkan` (`job["device"]`, `KVPARK_ENGINE_DEVICE`, `bench_engine.py --device`) | the ggml backend, or an error when the library or a Vulkan device is missing |
| device `auto` | torch when it imports and has a GPU (ROCm/CUDA, MPS), else ggml when the library loads and ggml sees a Vulkan GPU, else numpy |
| device `gpu` | the same order without the numpy fallback: no GPU backend is an error |
| `KVPARK_GGML_BRIDGE` | the library (default `runtime-ggml/build/libkvbridge.so`, or `runtime-ggml/build/Release/` with a multi-config generator; `.dylib` / `.dll` elsewhere) |
| `KVPARK_GGML_DEVICE` | `auto` (the first GPU; the default), `cpu`, a ggml device name such as `Vulkan1`, or a device index. `auto` and `gpu` only take ggml when this names a GPU; `cpu` (ggml's CPU backend) is reachable through device `vulkan` alone |

`backends.gpu_status("vulkan")` reports whether the library loads and what device it would use;
`ggml_backend.devices()` lists every device ggml sees. Errors map to the engine's fallback
rules:

- out of device memory, or an array larger than the device's largest buffer, raises
  `MemoryError`;
- an op the device does not support, a RoPE that disagrees with kvstate, or products that
  disagree with numpy (checked once per backend in `supports`) raise `NotImplementedError`;
- latents or rows that are not finite (a float16 overflow) raise `FloatingPointError`;
- anything else raises `RuntimeError`, which the daemon treats as a device fault.

In each case `engine.run` retries the job on numpy.

`engine_daemon.py` does not know the ggml backend yet: its warm-up, its GPU memory guard and its
eviction treat only torch as a GPU (see Integration below). Until that lands, a daemon on ggml
logs a `warm-up-error` per prewarm and counts only host memory.

## What runs where

Per hub side, the weights are uploaded once, as float16. The device transposes them into
`ggml_mul_mat`'s operand layout, so `x @ W` uses W stored as [cols, rows]. The format 2 bundles'
float16 files are uploaded as they are. Means, `z_mean` and the latents stay float32. A side's
per-layer matrices go up in one device graph per 256 MiB.

For a decoder, each cache section's layers are concatenated once on the device. When the sections
cover every layer, the per-layer copies are released, as in the torch backend.

| Call | Graph |
| --- | --- |
| `kvb_encode` | per source layer: upload the cell span of its stored rows, `get_rows` to the positions, de-rotate K, subtract the mean, `mul_mat` by the stage-one basis into a float16 stage-one scratch; then one `mul_mat` by the fused encoder, plus the bias −`z_mean` of the target's decoder (`encode(center=...)`); a finite check of the latents (`sum_rows`, then `sum`: a float16 overflow of the centred rows or the stage one is inf there), as torch's encode checks |
| `kvb_section` | `get_rows` of the mapped latents (or a view when they are 0..m−1), one `mul_mat` by the section's concatenated decoders, plus the mean, `set_rows` to their rows; the other rows get the mean (`repeat`) or the given fill rows |
| `kvb_cache_rows` | K rotation, cast to the cache dtype (f16/f32), a finite check per layer (`sum_rows`, then `sum`, of the stored values read back as f32: an f16 overflow is inf there), and the copy into each layer's host array |
| `kvb_rotate` | one RoPE on host rows (tests, and the backend's check of each model's RoPE against kvstate) |

Notes on these graphs:

- Every product is f16 × f16 with f32 accumulation: `ggml_prec_set_acc(GGML_PREC_F32)`, with the
  activations cast to f16 explicitly. ggml's default precision accumulates in f16 on Vulkan,
  which gave 1.2e-2 relative error on the hub's products (research notes, gpu.json).
- The encode, section and cache-row loops run `chunk_rows` rows at a time (4096 by default), fewer
  when a chunk's product or stored rows would exceed 256 MiB (`GRAPH_BYTES`) or the largest
  buffer. The graph allocator reuses intermediates within a graph. Device arrays (latents,
  section rows, scratch) come from a buffer pool, which `kvb_release` (the backend's `_released`)
  empties along with the graph allocator's buffer. Uploads convert and transpose through an
  allocator of their own, freed when the upload returns, so a side load leaves no graph buffer
  behind.
- Graphs run with `ggml_backend_graph_compute` on one backend, with no scheduler, so nodes run in
  the order they were built. That ordering is what makes the writes into views of a scratch tensor
  (the stage-one blocks, a partly rotated row) safe.

### RoPE: tables on the host, the multiply-adds on the device

The backend computes cos/sin per call from the positions. It uses kvstate's own angle formulas:

- the float32 theta recurrence of ggml-cpu;
- frequency factors, `freq_scale`, the YaRN mix and the magnitude;
- IMROPE sections, including [n, 4] M-RoPE positions.

cos/sin are evaluated in float64 and rounded to float32. kvstate uses libm `cosf`/`sinf`, which
are not correctly rounded for 1-3% of values. The two tables therefore agree to one unit in the
last place, and 97-99.9% of their entries are identical.

The device computes `a = x0·C − x1·S` and `b = x1·C + x0·S` on the rotated pairs only.

- Forward: (C, S) = (cos, sin).
- Inverse: (C, S) = (cos, −sin) / (cos² + sin²), kvstate.Rope.invert's division, done in float64
  on the host.
- NORM pairs (2i, 2i+1), NEOX and IMROPE pairs (i, i + n_rot/2), the MLA k_pe offset and NoPE
  layers (not rotated) all follow kvstate.

ggml's own `ggml_rope_ext` is not used because its Vulkan kernel evaluates sin/cos in float32 on
the GPU. Measured against `kvstate.cache_keys`, as max |err| / max |x|, with x of 1024 positions:

| Positions | `ggml_rope_ext` Vulkan, Muse NORM 128 | `ggml_rope_ext` Vulkan, NEOX 128 base 1e6 | `ggml_rope_ext` CPU | libkvbridge |
| --- | --- | --- | --- | --- |
| 0 – 1023 | 7.0e-5 | 5.5e-5 | 0 | ≤ 6.7e-8 |
| 32768 – 33791 | 2.1e-3 | 2.7e-3 | 0 | ≤ 1.3e-7 |
| 131072 – 132095 | 8.3e-3 | 1.1e-2 | 0 | ≤ 6.8e-8 |
| 261120 – 262143 | 1.5e-2 | 1.9e-2 | 0 | ≤ 6.4e-8 |

### Arrays larger than the device's largest buffer

ggml-vulkan's largest buffer is its suballocation block, 1 GiB (`GGML_VK_SUBALLOCATION_BLOCK_SIZE`),
whatever the driver allows. A 4096-row window of a section wider than 65,536 floats is larger:
a dense 70B-class target has 80 layers × 8 KV heads × 128 = 81,920. So a `DeviceArray` is held
in row blocks of at most the largest buffer (`blocks`, one block unless the array is larger),
`section` calls the library once per output block and latent block its rows fall in, and
`cache_rows` once per block. The library refuses a single array beyond the largest buffer as out
of memory (`MemoryError`), never as a device fault.

Measured on the 8060S, one 4096-row window of an 81,920-wide section (D = 256, NEOX keys at
positions 200000-204095, float16 cache): two blocks of 3276 and 820 rows; `section` 0.12 s,
`cache_rows` 0.16 s; peak device memory 1.80 GiB (the 1.25 GiB of rows plus one 256 MiB product
at a time); stored values within 5.8e-4 of numpy. A section weight larger than the largest
buffer (beyond 131,072 floats at D = 4096) is still refused, and that job runs on numpy.

### Reading results back on an integrated GPU

On a UMA device, ggml-vulkan allocates device buffers in host-visible, write-combined memory,
and its reads `memcpy` straight out of it. That ran at 0.2 GB/s here: 265 ms for 54 MB. The
library instead reads through a GPU copy into a pinned, host-cached staging buffer (ggml's
Vulkan host buffer type), then copies on up to 8 threads: 1.7-3.6 ms for the same 54 MB.

Tensors in that pinned memory cannot be graph nodes in this ggml version, because the Vulkan
backend's synchronisation reads every node's buffer as a device buffer, and the process crashed.
Results are therefore written to device memory first and copied out. Uploads are plain writes
to the mapped memory (20-28 GB/s).

## C API

All functions are `extern "C"` and every handle is opaque. A function returns 0 (or a non-null
handle) on success. On failure, `kvb_last_error()` and `kvb_last_error_code()` describe it
(−1 error, −2 out of memory). `kvb_encode` returns 1 when a latent is not finite, and
`kvb_cache_rows` when a value is not finite in the cache dtype.

| Function | Purpose |
| --- | --- |
| `kvb_version`, `kvb_ggml_version` | API version (1), ggml version and commit |
| `kvb_device_count`, `kvb_device_info(i, ...)` | ggml's devices: name, description, kind (0 cpu, 1 gpu, 2 igpu), free/total memory |
| `kvb_init(device, n_threads)` / `kvb_free` | a context on `"auto"`, `"cpu"`, a device name or a device index |
| `kvb_info`, `kvb_memory(out[8])`, `kvb_reset_peak`, `kvb_release`, `kvb_synchronize` | device, accounting (weights, live arrays, pooled, graph allocator, peak, device free/total, read staging), freeing caches |
| `kvb_new`, `kvb_upload`, `kvb_upload_many`, `kvb_concat_rows`, `kvb_read`, `kvb_write`, `kvb_shape`, `kvb_tensor_free` | device arrays [rows, cols] (f32/f16/i32): uninitialized, from host (converted, optionally transposed), row concatenation, row-range reads/writes |
| `kvb_encode(ctx, n, n_layers, kvb_encode_layer[], n_ropes, kvb_rope[], weight, bias, out, out_row0)` | latents of n positions into rows of `out` |
| `kvb_section(ctx, z, n_mapped, mapped, mapped_rows, z_mean, weight, mean, n_fill, fill_rows, fill, out, chunk_rows)` | one section's decoded rows |
| `kvb_cache_rows(ctx, values, row0, n, n_layers, kvb_cache_layer[], n_ropes, kvb_rope[], dtype, chunk_rows, &bad)` | rotated, cast and checked rows into host arrays |
| `kvb_rotate(ctx, x, n, heads, head_dim, rope, y)` | one rotation of host rows |

`kvb_rope` is {neox, n_rot, offset, pad, cos*, sin*}, with tables [n, n_rot/2]. The struct
layouts are mirrored in `ggml_backend.py` (`_Rope`, `_EncodeLayer`, `_CacheLayer`). The context
serialises calls with a mutex.

## Accuracy

On the synthetic fixtures, `test_ggml_backend.py` (42 tests, 1.6-6 s depending on the GPU's other
load) checks:

- RoPE against kvstate within 2e-6 of max |x| up to position 262143, for NORM, NEOX, IMROPE
  (including Qwen3.8's and Muse's geometry), YaRN, the MLA offset, odd NORM offsets and frequency
  factors. The measured error is 3e-8 to 1.3e-7 of max |x|.
- The float16 cast of identical float32 values is numpy's bit for bit: ties to even, subnormals,
  ±0 and 65504.
- `encode`, `section` and decoded values are within 1e-3 max relative error of numpy on hubs of
  real-sized products.
- Translations of all 11 synthetic model families (dense, iSWA, hybrid, MLA, DSA, DSA-iSWA,
  K-only hybrid, both ways), windowed and whole-section, are within 2e-3 relative Frobenius
  error of numpy, with recurrent rows byte-identical. A format 2 float16 bundle gives the same
  bytes as the format 1 float32 one.
- CPU fallback: the same translations on ggml's CPU backend.
- Row blocks: the products and translations again on a backend whose largest buffer is lowered
  to 16 KiB (latents in 4 blocks, section rows in up to 15) and to 64 bytes (1-2 rows a block).
- The product check refuses a device whose products disagree; latents that overflow float16 are
  refused; empty inputs give empty arrays (as numpy); an array beyond the largest buffer is a
  `MemoryError` that the engine does not count as a device fault; uploads leave no graph buffer.

On real states (`bench_engine.py accuracy --device vulkan`, run-v3b document 3b211f08, hub
hub4-d4096-v2), the ggml output is compared against numpy float32:

| Direction | T0 | Decoded max rel / rel Frobenius (worst section) | Stored max rel / rel Frobenius | f16 values identical |
| --- | --- | --- | --- | --- |
| Qwen3.8 → Muse | 12962 | 7.2e-4 / 4.2e-4 | 9.5e-4 / 4.9e-4 | 40-61% |
| Muse → Qwen3.8 | 13307 | 8.7e-4 / 4.7e-4 | 1.04e-3 / 5.5e-4 (recurrent rows byte-identical) | 41-43% |

The torch backend on the same document (bench-engine-fix-20261003) measured decoded 9.1e-4 /
4.6e-4 and stored 1.09e-3 / 5.2e-4. ggml is no worse: it keeps the latents in float32 and
subtracts `z_mean` before the float16 cast.

## Speed

`bench_engine.py` ran on the run-v3b long states with hub4-d4096-v2, on a Radeon 8060S (gfx1151,
RADV, Mesa 26.0.8) with OpenBLAS on 8 threads. No other job used the GPU; it was idle, or
settling from the previous run (6-23% busy in the samples before some runs), and the CPU load
average was 1.6-5.8. Raw reports: `evidence/universal-kv/bench-engine-vulkan-20261004/`, with
`runs.log` holding the GPU busy % and load next to each run.

Warm engine: the job runs in-process with the hub sides resident, as `engine_daemon.py` runs it.
The table gives the median of runs 1-4 (numpy: runs 1-2), with run 0 (the first job: side loads,
pipelines) in parentheses.

| Direction (T0) | ggml Vulkan | torch ROCm (hipBLASLt) | numpy |
| --- | --- | --- | --- |
| Qwen3.8 → Muse (12962) | 0.62 s (0.83) | 0.42 s (0.82) | 2.95 s (3.15) |
| Muse → Qwen3.8 (13307) | 0.82 s (1.07) | 0.56 s (0.95) | 4.02 s (4.27) |

The same warm runs, by stage:

| Direction | Backend | Encode | Decode | Assemble | Write |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 → Muse | ggml | 0.35 s | 0.11 s | 0.036 s | 0.043 s |
| Qwen3.8 → Muse | torch | 0.21 s | 0.04 s | 0.031 s | 0.046 s |
| Muse → Qwen3.8 | ggml | 0.08 s | 0.37 s | 0.09 s | 0.21 s |
| Muse → Qwen3.8 | torch | 0.05 s | 0.15 s | 0.07 s | 0.21 s |

The warm gap to torch is matrix-product throughput. ggml-vulkan runs f16 × f16 with f32
accumulation at about 9.8 TFLOPS on the fused encoder product ([4096 × 8192] by [8192 × 4096]
per 4096 rows), and at 13-15 TFLOPS on Muse's base section (4096 rows). It drops to 8-9 TFLOPS
on Muse's sliding section (2049 rows), and to 8.6-9.5 TFLOPS on Qwen3.8's attention section
(16384 columns) at every chunk size tried (2048-12288 rows). hipBLASLt reaches 25-35 TFLOPS on
these shapes.

Everything else is small:

- RoPE plus cast for 39 Muse layers of 4096 rows takes 15.8 ms of device time;
- reading back 54 MB takes 1.7-3.6 ms;
- loading a pair's sides takes 0.08-0.11 s.

One-shot engine: `python engine.py JOB` in a fresh process per job, with no daemon. This
includes interpreter start, library and Vulkan initialisation, side loads and pipeline
compilation. Each cell gives the two runs.

| Direction | ggml Vulkan | ggml, Mesa shader cache disabled | torch ROCm | numpy |
| --- | --- | --- | --- | --- |
| Qwen3.8 → Muse | 1.08 / 1.08 s | 1.15 / 1.14 s | 2.48 / 2.30 s | 3.26 / 3.29 s |
| Muse → Qwen3.8 | 1.33 / 1.31 s | 1.36 / 1.46 s | 2.59 / 2.42 s | 4.30 / 4.27 s |

ggml's first job is half torch's: torch's import and HIP initialisation alone take 1.4 s.
Compiling the Vulkan pipelines from scratch adds 0.06-0.1 s.

Memory, warm runs:

| Direction | Backend | Peak device memory | Resident pair | Host RSS peak |
| --- | --- | --- | --- | --- |
| Qwen3.8 → Muse | ggml | 0.88 GiB | 368 MiB | 1.55 GiB |
| Qwen3.8 → Muse | torch | 1.08 GiB | 368 MiB | 2.39 GiB |
| Qwen3.8 → Muse | numpy | — | 1.13 GiB (float32, host) | 3.45 GiB |
| Muse → Qwen3.8 | ggml | 1.17 GiB | 311 MiB | 1.50 GiB |
| Muse → Qwen3.8 | torch | 1.18 GiB | 312 MiB | 2.33 GiB |
| Muse → Qwen3.8 | numpy | — | 1.11 GiB (float32, host) | 4.30 GiB |

For ggml, peak device memory counts weights, live arrays, their pool and the graph allocator's
buffer; for torch it is the peak allocated. On ggml it stays bounded as T0 grows, because
translate's windows of 4096 target positions and the library's 4096-row chunks fix it. 8192-
and 16384-row windows were no faster: 0.63-0.67 s and 0.82-0.84 s, with peaks of 1.0-1.8 GiB.

When another process saturated the same GPU (`contended/`: llama-bench and a second engine
daemon), the warm switch took 1.16 s and 1.18-1.26 s on ggml, against 1.70-1.93 s and 1.70-1.83 s
on torch.

The review fixes (row blocks, the latent finite check, the product check, the uploads' own
allocator) were measured against the library and backend before them, alternating, under a CPU
load average of 11-14 from other work (`review-fixes/`; absolute times are therefore above the
tables'):

- Warm, median of runs 1-4 of two rounds: Qwen3.8 → Muse 0.769 s before, 0.776 s after; Muse →
  Qwen3.8 0.968 s before, 0.962 s after. Peak device memory is unchanged (0.88 / 1.17 GiB).
- Fresh process per job: 0.00-0.14 s more per job, of which the hub load (the product check in
  `supports`, and the uploads) is 0.02-0.04 s.

## Limits

- The products are about 1.5x slower than torch's on this GPU, as described above. Where a ROCm
  or CUDA torch is installed, `auto` keeps choosing torch.
- Other drivers. RADV reports a 4-byte `minStorageBufferOffsetAlignment`, so the views at
  4-byte offsets (NORM pairs, odd rope offsets) never took ggml-vulkan's misaligned-descriptor
  path here. Every op the graphs use other than `mul_mat` (binary, unary/copy, concat,
  `get_rows`/`set_rows`, `sum_rows`/`sum`) carries misalignment offsets in its push constants;
  `mul_mat` operands are always aligned (weights at a buffer's start, activations from the
  graph allocator); the per-layer checks sit in 256-byte slots. Mesa's llvmpipe (16-byte
  alignment, not UMA: ggml's own read path) ran the tests with `GGML_VK_VISIBLE_DEVICES=1`. RoPE
  (NORM pairs at 4-byte offsets included), cache rows, and the row-block products and
  translations (at most 42 rows per product) passed. But ggml-vulkan's products there are wrong
  past 64 rows that are not a multiple of 64: 0.6-1.2 relative error at 65, 90 and 150 rows,
  exact at 64, 128 and 256, and different from run to run. That is why `supports` checks the
  products once. On llvmpipe it refuses every other translation (`NotImplementedError`, after
  which `engine.run` translates on numpy), and the two tests that call the products directly
  fail. NVIDIA and Intel drivers, Windows, macOS (MoltenVK) and discrete GPUs remain untested.
- MLA latents go back on their RMS shell on the host (a read and a write of the decoded rows),
  and DSA indexer keys are computed on the host from the decoded rows (`backend.host`), as for
  torch. DeepSeek V4 targets translate on numpy, whatever the backend.
- The float16 cast matches numpy bit for bit where the device rounds float16 to nearest even and
  keeps float16 subnormals; ggml enables both through its float controls when the driver offers
  them (RADV does). Elsewhere, values within one float16 rounding are still checked by the
  translation tests.
- Tested on Linux with RADV (and llvmpipe, above) only. The CMake and ctypes code has Windows
  (`kvbridge.dll`, also under `Release/`) and macOS (`libkvbridge.dylib`, MoltenVK) paths, but
  neither was built or run.

## Integration

What the rest of the repository needs to use this backend fully (outside this directory):

- `engine_daemon.py`: run `backend.warm_products` for the warm-up and add `WARM_ROWS["ggml"] =
  (4096, 1536)`; treat a ggml backend on a GPU (`backend.on_gpu`) like torch in the GPU memory
  guard (`memory()["device_free"]`), the peak calibration (`reset_peak`, `peak_allocated`) and the
  eviction; report `backends.gpu_status("vulkan")` in `engine_info`.
- `src/kvpark/bridge`: `registry.ENGINE_DEVICES` gains `vulkan`, with the CLI help and its GPU
  warning; `docs/bridge.md` documents the device and this build.
- `scripts/build_runtime.py` (optional): build libkvbridge at the runtime's llama.cpp pin.
