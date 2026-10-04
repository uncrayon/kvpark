# Changelog

## Unreleased

- Move the patched runtime to llama.cpp a7b94df2c. The slot-resume patch carries over unchanged, so `.resume` archives keep their format. Upstream now keeps DFlash draft positions at the target's positions, which fixes images after long text with an M-RoPE DFlash draft; the production-only patches the user's builds carried are superseded upstream except a `llama-bench` one, kept in `patches/production/`.
- Add `kvpark bridge` (experimental): switch a live conversation between local models by translating its cache through a shared latent (`experiments/universal_kv`). The engine runs resident on the GPU (torch) or through Vulkan (ggml, no torch), writes `.resume` sidecars with DFlash and MTP draft state, renders Hermes' request options, starts hybrid models' recurrent state from a per-model mean, and replaces images, audio and video a target cannot read with text descriptions or transcripts. Hermes: `/kvpark switch`.
- Add `kvpark start --hermes`, which starts the connection configured in the active Hermes profile from a terminal, and `--supervise`, which keeps relaunching stopped services for a systemd or launchd boot service.

## 0.3.0a3 — safer routing and Hermes compatibility

- Point Hermes at the chosen model server instead of the kvpark proxy. The plugin inserts its proxy into each request while the proxy is healthy, so a stopped proxy or a plugin that fails to load no longer breaks every Hermes conversation. `/kvpark doctor` flags the previous proxy route as legacy; `/kvpark connect` switches it.
- Detect optional Hermes plugin APIs when loading. Hermes 0.20.1 and newer load the adapter; argument suggestions still need 0.21.0. A Hermes API change now surfaces as a `/kvpark` message naming the Hermes version instead of a silent load failure.
- Add `kvpark doctor --hermes` and extend `/kvpark doctor`: Hermes version, where kvpark loads from, whether `/kvpark` registered, the route mode, and proxy reachability. The installer runs it after enabling the plugin and stops with the reason when Hermes cannot load kvpark; its final instructions no longer contradict the chooser it opens.
- Install as a Hermes plugin directory with `hermes plugins install uncrayon/kvpark` and update with `hermes plugins update kvpark`; `/kvpark update` explains this for directory installs. Package installs keep the coordinated updater.
- Accept nonbreaking spaces from rich-text command input without changing quoted model names or paths, including setup choices and uninstall confirmations.
- Report unsupported disk saving before resolving Hermes conversation identity for routing connections. Keep configured users out of repeated onboarding and send unknown commands to help instead of restarting setup.

## 0.3.0a2 — easier installation and model discovery

- Add local model-server discovery and numbered setup choices in Hermes and the terminal. The installer opens the chooser automatically, and users can cancel without changing their model route.
- Add a one-command installer for macOS, Linux, and WSL with automatic Python setup, Hermes environment detection and plugin enablement, a standalone command, checksum verification, and optional patched-runtime builds.
- Accept Hermes desktop URL chips in setup and base-url commands, preserving the full host and port and applying the same URL validation.
- Offer kvpark action suggestions in Hermes' desktop composer and CLI while preserving free-text arguments; remove the completion entry when the plugin unloads.
- Make Hermes installation use its own Python interpreter explicitly and explain how to recover from installing into a separate standalone environment.

## 0.3.0a1 — kvpark rename and migration

- Rename the repository, Python distribution/module, CLI, and Hermes plugin to `kvpark`.
- Use `kvpark save / status / forget` and `/kvpark save / status / forget`; retain `park` as a compatibility alias and Hermes’ `delete` alias.
- Use `KVPARK_*` configuration names and `kvpark` default data directories for new installations; accept legacy `SLOTH_*` environment settings during migration.
- Add `python -m kvpark migrate --hermes` previews and `--confirm` application for existing sloth-memory profiles.
- Preserve the original archive directory, snapshot format, backend process, model configuration, and Hermes route backups; drain and stop only the verified legacy proxy.
- Disable the old plugin and enable kvpark. Migration requires stopped clients; restart Hermes afterward to launch and verify the new proxy before removing the old package.

Releases below were published under the former **sloth-memory** name. Their command names and validation scope are historical.

## 0.2.0a2 — uninstall and route recovery

- Add `/sloth uninstall` previews and `/sloth uninstall confirm`, with terminal equivalents for Hermes and standalone services.
- Save the original Hermes model route during setup and restore it on disconnect; preserve unrelated settings and refuse to guess missing backups.
- Disable plugin startup, drain and stop verified owned processes, and keep existing backend servers and saved data.
- Document package removal, optional archive/build cleanup, and manual uninstall for older alphas.

## 0.2.0a1 — coordinated release updates

- Add `/sloth update check` and `/sloth update`, plus equivalent terminal commands.
- Verify official GitHub release wheel hashes and compatibility before installation; retain a rollback wheel and configuration backups.
- Drain inference and save resident state before proxy replacement, keep the model backend running, and verify the replacement before reopening inference.
- Roll back package and proxy on handled installation/startup failures; report recovery paths when rollback fails.
- Restart supported Hermes gateways through their native lifecycle after a successful package update; show installed, loaded, and running versions.
- Add isolated real-package upgrade and failed-start rollback acceptance.

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
