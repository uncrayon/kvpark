# Alpha compatibility

## Runtime

The build script pins ggml-org/llama.cpp to
`85d5703a3b1b47243213a39059a6e3076c92733a` and applies
`patches/llama-slot-resume.patch`. This patch persists hybrid/recurrent context
checkpoints as well as target, draft, and speculative state. The `.bin.resume`
companion is mandatory: a stock `.bin` alone cannot be published as a usable
sloth-memory archive.

This is a local format tied to the exact runtime identity. Do not exchange archive
files between installations or mix versions. The proxy fingerprints local file
identities, command arguments, and relevant runtime environment. These are
replacement checks, not cryptographic hashes of model contents.

| Component | Alpha scope |
| --- | --- |
| OS | Linux with `/proc`, `flock`, and `ldd` |
| Python | 3.10+; standard library at runtime |
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
python scripts/build_runtime.py --output runtime-cuda --cmake-arg=-DGGML_CUDA=ON
python scripts/build_runtime.py --output runtime-vulkan --cmake-arg=-DGGML_VULKAN=ON
python scripts/build_runtime.py --output runtime-rocm --cmake-arg=-DGGML_HIP=ON
```

These switches select upstream backends, not verified support for your GPU.
Use the resulting binary path with `sloth-memory backend` and pass any device
settings after `--`. Use explicit local file paths and long-form model/draft/
projector arguments. `LLAMA_ARG_*` environment defaults are refused to keep the
runtime identity reproducible. Other kernel environment settings are fingerprinted.

Keep the compiled binary with its libraries. GPU backend libraries loaded outside
the binary directory may require an explicit, stable `LD_LIBRARY_PATH`; rebuild
and repark after runtime updates.

## Storage policy

`serve --ttl-days 7 --max-gib 32` sets the defaults. `--ttl-days 0` disables time
expiry. The budget applies to published snapshots; a save may temporarily need
extra space. A snapshot larger than the whole budget is rejected, preserving the
previous generation. Periodic GC evicts least-recently-used entries, so the budget
can be exceeded between GC runs and retention is not guaranteed.

Explicit parking always saves the current state. When switching away, an already
parked conversation is refreshed after 4,096 tokens of growth. A selected
foreground conversation is saved before detached background work even below
that threshold. Unparked one-off chats are disposable by default.
