# Changelog

## 0.1.0a1 — first developer alpha

- Extracted the conversation archive proxy into an agent-independent Python package.
- Added backend launcher, runtime fingerprinting, directory ownership locks,
  and serve/status/park/forget/doctor commands.
- Included a pinned llama.cpp checkpoint persistence patch and a clean build script.
- Preserved foreground/background ownership, atomic snapshot publication,
  compatibility checks, retention, and actual completion cache metrics.
- Added a framework-free transcript client and a private CPU restart acceptance test.
- Added localhost API boundaries, optional bearer authentication, and request size limits.

Scope: Linux, one local slot, chat completions, user-supplied GGUF weights.
No bundled Hermes adapter, remote cache service, model weights, or GPU binaries.
