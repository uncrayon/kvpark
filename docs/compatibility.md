# Runtime compatibility

This page describes alpha `0.3.0a2`. See [backends](backends.md) for routing
capabilities. The rename from sloth-memory preserves snapshot format 2; the
[migration](updates.md#migrate-from-sloth-memory) keeps the archive path and runtime
unchanged so existing compatible snapshots remain usable.

## Runtime

The build script pins ggml-org/llama.cpp to
`85d5703a3b1b47243213a39059a6e3076c92733a` and applies
`patches/llama-slot-resume.patch`. This patch persists hybrid/recurrent context
checkpoints as well as target, draft, and speculative state. The `.bin.resume`
companion is mandatory: a stock `.bin` alone cannot be published as a usable
kvpark archive in the current implementation. Stock llama.cpp already has
slot save/restore APIs; the companion requirement is specific to this adapter.
This persistence patch is separate from the originating project's DFlash2 position
fix and is not restricted to DFlash2.

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
Use the resulting binary path with `kvpark backend` and pass any device
settings after `--`. Use explicit local file paths and long-form model/draft/
projector arguments. `LLAMA_ARG_*` environment defaults are refused to keep the
runtime identity reproducible. Other kernel environment settings are fingerprinted.

Keep the compiled binary with its libraries. GPU backend libraries loaded outside
the binary directory may require an explicit, stable `LD_LIBRARY_PATH`; rebuild
and repark after runtime updates.

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
