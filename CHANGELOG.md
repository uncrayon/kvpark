# Changelog

## 0.2.0.dev0 — unreleased

- Add the portable `/sloth` alias, preserve Telegram chat identity when parking, and document Telegram menu priority; cover dispatch and menu visibility in Hermes integration tests.

- Explain when a gateway conversation needs a normal model turn before its first park; cover Discord command dispatch and conversation isolation in integration tests.

- Clear stale startup errors after successful onboarding and show when Hermes is already connected.
- Document installation into an existing Hermes environment and migration from custom proxies, with live ROCm/DFlash2 acceptance evidence.

- Portable process identity, data directories, file locks, and managed process lifetime for Linux, macOS, and Windows.
- Select available ports, publish bound addresses, and discover them from CLI controls and Hermes.
- Configure Hermes through the local proxy, preserving served model IDs; refresh matching live OpenAI clients after a port change.
- Routing adapters for existing llama.cpp, Ollama, vLLM, and MLX-LM servers with explicit capability reporting.
- Native disk snapshots remain limited to the patched, managed llama.cpp runtime. Other adapters reject parking clearly.
- Cross-platform Python, Hermes integration, and native build CI.

## 0.1.0a2 — Hermes adapter and configurable cleanup

- Added native Hermes plugin discovery, automatic service startup/reuse, and
  `/sloth-memory setup` onboarding with future-session model routing.
- Added native park/delete/status commands plus retention, budget, cleanup,
  base URL, and autostart controls.
- Cleanup defaults to enabled, checks hourly, and expires disk snapshots seven
  days after saving; restores no longer extend their lifetime.
- Persisted cleanup settings across restarts; added manual sweeps and safe cleanup
  of expired unpublished snapshot generations.
- Enforced the snapshot budget after saves and budget reductions even when
  scheduled cleanup is disabled. MIT licensing is unchanged.

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
