"""Compute backends of the translation engine (translate.compute / translate.assemble).

  numpy  float32 on the CPU through numpy's BLAS; always available, the reference
  torch  float16 weights and activations with float32 accumulation on a GPU (ROCm or CUDA,
         Apple MPS); source and target RoPE computed on the device from positions

select(device) returns a process-wide backend instance. device is "cpu", "gpu" or "auto"
(job["device"], else the KVPARK_ENGINE_DEVICE environment variable, else "auto"); "auto"
takes torch when it imports and sees a device, numpy otherwise; "gpu" without one is an
error. Both backends keep the hub sides they have used resident (an LRU the engine daemon
trims); see NumpyBackend for the interface.
"""

import os

DEVICES = ("auto", "cpu", "gpu")
ENV = "KVPARK_ENGINE_DEVICE"
_INSTANCES = {}


def requested(device=None):
    """The device asked for: the argument, else $KVPARK_ENGINE_DEVICE, else "auto"."""
    value = (device or os.environ.get(ENV) or "auto").strip().lower()
    if value not in DEVICES:
        raise ValueError(f"engine device must be one of {', '.join(DEVICES)}, not {value!r}")
    return value


def gpu_status():
    """(available, detail): whether torch imports and has a ROCm/CUDA or MPS device."""
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
    available, detail = gpu_status()
    if available:
        return _instance("torch")
    if want == "gpu":
        raise RuntimeError(f"no GPU backend: {detail}")
    return _instance("numpy")


def _instance(name):
    if name not in _INSTANCES:
        if name == "torch":
            from .torch_backend import TorchBackend
            _INSTANCES[name] = TorchBackend()
        else:
            from .numpy_backend import NumpyBackend
            _INSTANCES[name] = NumpyBackend()
    return _INSTANCES[name]


def loaded():
    """The backend instances created so far (name -> backend)."""
    return dict(_INSTANCES)
