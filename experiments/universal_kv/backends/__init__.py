"""Compute backends of the translation engine (translate.compute / translate.assemble).

  numpy  float32 on the CPU through numpy's BLAS; always available, the reference
  torch  float16 weights and activations with float32 accumulation on a GPU (ROCm or CUDA,
         Apple MPS); source and target RoPE computed on the device from positions
  ggml   the same arithmetic as ggml graphs on a Vulkan GPU (AMD, NVIDIA, Intel; no torch, no
         ROCm), through libkvbridge (ggml/README.md builds it; $KVPARK_GGML_BRIDGE locates it)

select(device) returns a process-wide backend instance. device is "cpu", "gpu", "vulkan" or
"auto" (job["device"], else the KVPARK_ENGINE_DEVICE environment variable, else "auto"):

  auto    torch when it imports and sees a GPU (ROCm/CUDA, MPS), else ggml when libkvbridge
          loads and ggml sees a Vulkan GPU, else numpy
  gpu     the same order without the numpy fallback: no GPU backend is an error
  vulkan  ggml only (an error without libkvbridge or a Vulkan GPU; ggml's CPU backend only when
          $KVPARK_GGML_DEVICE names it)
  cpu     numpy

Every backend keeps the hub sides it has used resident (an LRU the engine daemon trims); see
NumpyBackend for the interface.
"""

import os

DEVICES = ("auto", "cpu", "gpu", "vulkan")
ENV = "KVPARK_ENGINE_DEVICE"
_INSTANCES = {}


def requested(device=None):
    """The device asked for: the argument, else $KVPARK_ENGINE_DEVICE, else "auto"."""
    value = (device or os.environ.get(ENV) or "auto").strip().lower()
    if value not in DEVICES:
        raise ValueError(f"engine device must be one of {', '.join(DEVICES)}, not {value!r}")
    return value


def gpu_status(backend="torch", *, gpu=True):
    """(available, detail) of a GPU backend: "torch" (it imports and has a ROCm/CUDA or MPS device)
    or "vulkan" (libkvbridge loads and ggml sees its device, by default the first Vulkan GPU; with
    gpu=False also ggml's CPU backend when $KVPARK_GGML_DEVICE names it)."""
    if backend == "vulkan":
        from . import ggml_backend
        return ggml_backend.status(gpu=gpu)
    try:
        from . import torch_backend
    except ImportError as error:
        return False, f"torch unavailable ({error})"
    device = torch_backend.default_device()
    return (device is not None), (device or "torch has no GPU device")


def select(device=None):
    """The backend for a requested device (cached per process)."""
    want = requested(device)
    if want == "cpu":
        return _instance("numpy")
    if want == "vulkan":
        available, detail = gpu_status("vulkan", gpu=False)
        if not available:
            raise RuntimeError(f"no Vulkan backend: {detail}")
        return _instance("ggml")
    available, detail = gpu_status("torch")
    if available:
        return _instance("torch")
    vulkan, vulkan_detail = gpu_status("vulkan")
    if vulkan:
        return _instance("ggml")
    if want == "gpu":
        raise RuntimeError(f"no GPU backend: {detail}; {vulkan_detail}")
    return _instance("numpy")


def _instance(name):
    if name not in _INSTANCES:
        if name == "torch":
            from .torch_backend import TorchBackend
            _INSTANCES[name] = TorchBackend()
        elif name == "ggml":
            from .ggml_backend import GgmlBackend
            _INSTANCES[name] = GgmlBackend()
        else:
            from .numpy_backend import NumpyBackend
            _INSTANCES[name] = NumpyBackend()
    return _INSTANCES[name]


def loaded():
    """The backend instances created so far (name -> backend)."""
    return dict(_INSTANCES)
