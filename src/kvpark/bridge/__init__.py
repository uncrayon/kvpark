"""kvpark bridge (experimental): move a live conversation's cache between different models.

kvpark parks and restores a model's own cache. The bridge translates a cache from one model
to another through a universal KV hub (per-model encoders into a shared per-token latent and
decoders out of it), so a conversation can continue on another model without a full prefill:
the new model restores the translated cache for the older part of the conversation and
prefills only a native tail.

Switches the translator cannot handle (architectures, media, draft models; capabilities.py)
are skipped with a reason, and the target prefills natively. On kvpark-patched servers the
bridge also writes the slot's .resume sidecar with the draft model's state (resume.py).

The orchestration here is standard library only. The translation itself runs in an engine
worker (numpy) given by the bridge configuration; see docs/bridge.md.
"""
