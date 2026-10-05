# Runtime compatibility

This page describes the development tree after alpha `0.3.0a3`, whose runtime
was pinned to `85d5703a3`. See [backends](backends.md) for routing
capabilities. The rename from sloth-memory preserves snapshot format 2; the
[migration](updates.md#migrate-from-sloth-memory) keeps the archive path and runtime
unchanged so existing compatible snapshots remain usable.

## Runtime

The build script pins ggml-org/llama.cpp to
`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` (October 4, 2026) and applies the
patches in its ordered `PATCHES` list. Today that list holds one patch,
`patches/llama-slot-resume.patch`. This patch persists hybrid/recurrent context
checkpoints as well as target, draft, and speculative state. The `.bin.resume`
companion is mandatory: a stock `.bin` alone cannot be published as a usable
kvpark archive in the current implementation. Stock llama.cpp already has
slot save/restore APIs; the companion requirement is specific to this adapter.
This persistence patch is separate from the originating project's DFlash2 position
fix and is not restricted to DFlash2.

The previous pin was `85d5703a3b1b47243213a39059a6e3076c92733a`. The patch was
rebased without changing its behaviour: the `.bin.resume` layout (magic, version 1,
draft blob, speculative blob, checkpoints) and the slot file are byte-compatible.
A runtime update still changes the runtime identity, so conversations parked on
the old runtime need one fresh prefill and park on the new one. The new pin no
longer accepts `--mmap`, `--no-mmap`, `--mlock` or `--direct-io`; use `--load-mode`.
The state-format changes between the two pins that matter to the KV translator are
summarized in [validation](validation.md#llamacpp-a7b94df2c-runtime-upgrade--october-4-2026);
the full comparison is local evidence of the development checkout and is not part of
the repository. The translator (`kvpark bridge`) reads and writes both pins' states, so
servers on either pin can switch conversations with each other.

### DFlash drafts with images

The old pin failed an image request after about 2,000 text tokens when an M-RoPE
DFlash or DFlash2 draft ran with a multimodal projector: HTTP 500 `failed to
process mtmd chunk`. An image pins all its rows to one M-RoPE position, so the
draft's sliding-window cache could not free cells for it. The originating
deployment patched this with a draft-local position counter
(`llama-dflash-contiguous-positions.patch`). That counter restarts at 0 when a slot
is restored with an empty draft, so the draft then sits at positions far behind the
target's.

The new pin fixes the failure upstream, so kvpark ships no DFlash patch:

- An image's pinned rows are not injected into the draft (upstream `fa6769818`), so
  the draft's cache has a gap of one image's positions after each image.
- Drafts use the target's own positions, including after an image (upstream
  `b0dcb8192`).
- After a restore with an empty draft or with the bridge's aligned dummy, the draft
  stays at the target's positions.

On the old pin the failure was reproduced with a tiny M-RoPE fixture, a DFlash draft
and the real Qwen3.8 projector. The same check passes on the new pin, and so does
the real Qwen3.8 with the DFlash2 Q8 draft on ROCm (see
[validation](validation.md#llamacpp-a7b94df2c-runtime-upgrade--october-4-2026)).
Do not carry the contiguous-position patch forward.

### Production patch set

kvpark's default runtime applies only what kvpark needs. Patches that the
originating Qwen/ROCm deployment needed and that upstream does not have go in
`patches/production/`, which `build_runtime.py` never applies. At this pin every
serving patch of that deployment is either upstream or kvpark's own:

| Production patch (old pin) | At `a7b94df2c` |
| --- | --- |
| `llama-slot-resume.patch` | kvpark's `patches/llama-slot-resume.patch` (identical) |
| `llama-dflash-contiguous-positions.patch` | Superseded upstream (`fa6769818`, `b0dcb8192`); do not port |
| `llama-hip-gfx1151-host-access.patch` | Superseded: upstream `d4389a4dd` disables direct host-buffer access on every HIP device |
| `ggml-hip-topk-hipcub.patch` (never deployed) | Superseded: HIP `TOP_K` supports any row width (upstream `6a2743f02`); `test-backend-ops -o TOP_K -b ROCm0`: 525/525 pass |
| `llama-qwen38-flash-next-bringup.patch` | Superseded: qwen4exp graph budget, indexer state round trip and Hadamard rotations are upstream. Its Vulkan workaround (argsort instead of top-k, whose Vulkan pipeline stopped at k = 1023) is replaced by a pooled indexer top-k and a Vulkan radix-select path for large k |
| `llama-bench-depth-reps1-no-state-copy.patch` | Still needed for benchmarks; rebased as `patches/production/llama-bench-depth-reps1-no-state-copy.patch`. It changes only `llama-bench`, never `llama-server` |

To build a runtime with the production set, apply kvpark's patches first and then
the files in `patches/production/`, in name order, to a worktree of the pinned commit.
A worktree keeps Git metadata, so the build reports the pin in `build_info`:

```bash
git -C /path/to/llama.cpp fetch https://github.com/ggml-org/llama.cpp.git a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
git -C /path/to/llama.cpp worktree add --detach /path/to/llama.cpp-kvpark a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
cd /path/to/llama.cpp-kvpark
git apply /path/to/kvpark/patches/llama-slot-resume.patch
git apply /path/to/kvpark/patches/production/*.patch
```

This is a local format tied to the exact runtime identity. Do not exchange archive
files between installations or mix versions. The proxy fingerprints local file
identities, command arguments, and relevant runtime environment. These are
replacement checks, not cryptographic hashes of model contents.

| Component | Current scope |
| --- | --- |
| OS | Linux, macOS, Windows; backend hardware support varies |
| Python | 3.10+; psutil is installed as a runtime dependency |
| Inference server | Included pinned llama.cpp patch; local single-slot topology |
| Agent | Any server-side client implementing the session/replay contract |
| Wire API | Chat completions, including SSE; health and models discovery |
| Models | User-supplied GGUFs supported by the pinned runtime; validate each configuration |
| GPU | CMake opt-in; no universal hardware support claim |
| Drafts and images | Serialization is present; model/hardware-specific acceptance is required |
| Model switching / LoRA | Not supported as an archive reuse mechanism |

The initial public smoke test targets a small dense model on CPU. Historical Qwen
hybrid and DFlash results came from the originating project with additional local
runtime corrections; they are not blanket validation of this package's GPU path.
See [validation](validation.md) for the release's actual evidence.

A September 27, 2026 development acceptance run also verified native park/resume
for `Muse-Glimmer-30B-Q8_0.gguf` on the then-pinned `85d5703a3` CPU runtime, with F16 K/V caches and
a 4,096-token context. Its 2,467-token cold prompt exceeded Muse's 2,048-token
sliding window. A direct continuation after restarting both private processes
reused 2,462 tokens, processed 38, and exactly matched the resident response's
full message within an eight-token generation budget. See the
[Muse acceptance scope and reproduction command](validation.md#muse-cpu-snapshot-acceptance).
This verifies that particular CPU configuration; the existing Vulkan/DFlash
deployment was not validated or configured for kvpark by this test.

## Building

Default: `python scripts/build_runtime.py`. CMake builds only `llama-server` and
its dependencies with native CPU tuning disabled. No model downloads occur.
The CPU CI build adds `--cmake-arg=-DGGML_METAL=OFF`; use this for an initial
macOS smoke test. Building requires Git, CMake, and a C/C++ toolchain (Xcode Command
Line Tools on macOS, or the Visual Studio C++ build tools on Windows).
Use the binary path printed by the script and keep adjacent libraries with it.

To reuse an existing Git checkout containing the pin:

```bash
python scripts/build_runtime.py --source /path/to/llama.cpp --output runtime
```

The script exports the pinned commit, ignoring that checkout's modifications.
An existing output directory is refused; choose a fresh output for another build.
The export has no Git history, so the script passes the pin's upstream build number
and commit to CMake: `llama-server --version` and `/props` report `b11398-a7b94df2c`
for this pin. CMake also downloads llama.cpp's prebuilt web UI for that build number
from Hugging Face (with `HF_TOKEN` when it is set). kvpark starts its servers with
`--no-ui`, so `--cmake-arg=-DLLAMA_USE_PREBUILT_UI=OFF` builds without the UI and
without that download.
If a build is interrupted, resume its CMake build manually after resolving the
error, or use a new output directory.

Examples for installed GPU toolchains:

```bash
python scripts/build_runtime.py --output runtime-metal --cmake-arg=-DGGML_METAL=ON
python scripts/build_runtime.py --output runtime-cuda --cmake-arg=-DGGML_CUDA=ON
python scripts/build_runtime.py --output runtime-vulkan --cmake-arg=-DGGML_VULKAN=ON
python scripts/build_runtime.py --output runtime-rocm --cmake-arg=-DGGML_HIP=ON
```

These switches select upstream backends, not verified support for your GPU.

On October 4, 2026 the new pin was built three ways on an AMD Ryzen AI Max+ 395
(Radeon 8060S, gfx1151, ROCm 7.2.2), out of tree, with `-DCMAKE_BUILD_TYPE=Release
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON
-DLLAMA_BUILD_EXAMPLES=OFF` plus:

| Build | Additional CMake flags | Targets |
| --- | --- | --- |
| CPU | `-DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF` | `llama-server`, `test-llama-archs`, `llama-cli`, `llama-bench` |
| Vulkan | `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF` | the CPU targets and `test-backend-ops` |
| ROCm, production-equivalent | `-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.2/lib/llvm/bin/clang -DGGML_NATIVE=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=ON` | `llama-server`, `llama-bench`, `llama-cli`, `test-backend-ops` |

The GPU target and GGML options match the CMake cache of the existing production
build. That build named the same ROCm 7.2.2 LLVM's `clang++` as its HIP compiler,
enabled examples, used ccache and linked with absolute library paths. The ROCm
build's source also carried the production `llama-bench` patch; its `llama-server`
source is the same as the other builds'. The default `python scripts/build_runtime.py`
CPU build was also run end to end at this pin. See [validation](validation.md#llamacpp-a7b94df2c-runtime-upgrade--october-4-2026)
for what each build was tested with.

Use the resulting binary path with `kvpark backend` and pass any device
settings after `--`. Use explicit local file paths and long-form model/draft/
projector arguments. `LLAMA_ARG_*` environment defaults are refused to keep the
runtime identity reproducible. Other kernel environment settings are fingerprinted.

Keep the compiled binary with its libraries. GPU backend libraries loaded outside
the binary directory may require an explicit, stable `LD_LIBRARY_PATH`; rebuild
and repark after runtime updates.

### Switching a managed backend to a new runtime

1. Stop the services that use the old binary: the kvpark supervisor (for example
   `systemctl --user stop kvpark.service`) and any Hermes gateway that loaded the
   plugin. A running backend with another binary is reported as "another
   model/configuration" and is not replaced automatically.
2. Copy the new build's `bin` directory to a stable location. Builds made with
   `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON` find their libraries next to the binary.
3. Remove any `LD_LIBRARY_PATH` that points at an older runtime bundle, for example
   in the service's environment file. The library file names (`libllama.so.0`,
   `libggml-base.so.0`, …) are the same across versions, so an old path would load
   old libraries into the new binary.
4. Point the configuration at the new binary, with `/kvpark setup --server
   /absolute/path/bin/llama-server` in Hermes or `--server` for `kvpark backend`.
   Keep the model and backend arguments.
5. Start the services again and check `kvpark doctor --hermes`.

Snapshots parked under the old runtime remain on disk but are not restored by the
new one; the first return to such a conversation prefills it again, and parking it
then produces a compatible snapshot.

### Moving a checkout

The source does not depend on its parent directory. Run the documented commands
from the checkout root; explicit relative `--source` and `--output` paths resolve
from your current working directory. Runtime libraries in the Linux build tree
use paths relative to the binary, so the binary and its adjacent libraries can
move together.

Virtual environments and CMake build caches are specific to their creation path.
After moving the source, recreate the virtual environment, reinstall kvpark, and
build a new runtime instead of editing generated CMake files:

```bash
python scripts/build_runtime.py --output runtime-rebuilt --cmake-arg=-DGGML_METAL=OFF
```

Use the new server path in `kvpark backend` or Hermes setup, and update any model,
draft, projector, or library paths if those files moved too. Existing archives live
in the platform data directory unless you selected a custom location. Runtime
identity includes absolute paths, so a moved or rebuilt backend may require a
fresh prefill and park. Runtime output directories are local build artifacts and
must not be committed or included in Python distributions.

## Storage policy

New archives default to seven days after the last save, cleanup enabled, and a
32 GiB budget. The proxy checks at startup and hourly when inference is idle.
Restoring does not extend snapshot age. Expired unpublished generation files are
also cleaned; arbitrary files, symlinks, and live RAM state are not swept.

`settings --ttl-days 7 --max-gib 32 --cleanup` changes and persists preferences.
`--no-cleanup` disables scheduled deletion and age expiry on restore;
`--ttl-days 0` disables age expiry only. `kvpark cleanup` runs a manual sweep.
Explicit `serve --ttl-days … --max-gib … --[no-]cleanup` flags override saved
preferences; omitted flags preserve them.

The budget is enforced after saving and when reducing it, including when scheduled
cleanup is off. Least-recently-used snapshots can be evicted before seven days.
The limit applies to published snapshots; atomic replacement temporarily needs
extra space. An oversized snapshot is rejected, preserving the preceding generation.

Explicit parking always saves the current state. When switching away, an already
parked conversation is refreshed after 4,096 tokens of growth. A selected
foreground conversation is saved before detached background work even below
that threshold. Unparked one-off chats are disposable by default.
