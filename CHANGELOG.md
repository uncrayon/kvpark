# Changelog

## Unreleased

- Move the patched runtime to llama.cpp a7b94df2c. The slot-resume patch carries over unchanged, so `.resume` archives keep their format. Upstream now keeps DFlash draft positions at the target's positions, which fixes images after long text with an M-RoPE DFlash draft; the production-only patches the user's builds carried are superseded upstream except a `llama-bench` one, kept in `patches/production/`.
- Add `kvpark bridge` (experimental): switch a live conversation between local models by translating its cache through a shared latent (`experiments/universal_kv`). The engine runs resident on the GPU (torch) or through Vulkan (ggml, no torch), writes `.resume` sidecars with DFlash and MTP draft state, renders Hermes' request options, starts hybrid models' recurrent state from a per-model mean, and replaces images, audio and video a target cannot read with text descriptions or transcripts. Hermes: `/kvpark switch`.
- Experiments and bridge: the cache translator reads and writes the states of both llama.cpp versions (a7b94df2c, now the repository's `runtime/`, and the production 85d5703a3) and adds the GQA family (Llama, Mistral 3, Qwen3-MoE, Hy3, Phi-3/4, Gemma 3/4 with Gemma 4's MTP head, gpt-oss, Laguna, MiMo) and GLM5-Next, Gemma 4 and SentencePiece vocabularies, and chat-template rendering that matches llama-server byte for byte on every local model (on tokenizer twins at both pins); the bridge skips embedding models, LongRoPE models and vocabularies an older engine cannot read.
- Bridge: `kvpark bridge add --no-source` registers a model as a target only. Switches out of it are skipped (`source-refused`) before either server is touched, and the next model prefills natively; switches into it still translate. Use it for a model the hub translates out of worse than no translation, such as Nemotron-3.5 with the hub6 bundle.
- Add `kvpark start --hermes`, which starts the connection configured in the active Hermes profile from a terminal, and `--supervise`, which keeps relaunching stopped services for a systemd or launchd boot service.
- Add OpenCode support (tested with 1.18.34): `kvpark setup --opencode` adds a `kvpark` provider and a plugin to OpenCode's config (merged, backed up, idempotent) and, with `--server` and `--model`, starts kvpark's patched llama-server for disk park/resume; `--url` shares a running proxy such as Hermes'. The plugin tags each request with its OpenCode session and tells the session's own requests from title, summary, compaction and subagent requests. Setup refuses a context window under 16,384 tokens, which leaves OpenCode almost no room after its system prompt and tools. `kvpark doctor --opencode` checks the setup and `kvpark uninstall --opencode` removes it. See docs/opencode.md.
- Proxy: a request may set `slot_archive_autosave`. Its conversation is then saved before background work takes the slot, and saved like `KVPARK_AUTO_SAVE` when another conversation takes it. A new autosave conversation that finds the slot without its system prompt and tools starts from the smallest saved snapshot with the same system prompt, tools and template settings, so that shared prefix is not prefilled again (event `seeded`), unless the proxy has measured that prefilling it is faster than that restore on this machine (event `seed-skipped`).
- Proxy: a saved conversation whose snapshot no longer matches it (a changed system prompt or tool list, or compaction) is saved again on its next save whatever its growth; before, it was prefilled on every return until it grew by 4,096 tokens. A snapshot that already holds what is in the slot is not written again when background work takes the slot.
- Add `kvpark stop` (`--hermes`, `--archive-dir`): waits up to a minute for the reply in progress, saves the conversation in the slot, and stops the proxy and the managed model server, keeping the setup for `kvpark start`. The documented systemd unit runs it as `ExecStop`, so stopping or restarting the service and shutting down save the conversation first.
- `kvpark start` exits with status 78 when the setup needs a change: Hermes cannot load, or no model is configured or connected. The documented systemd unit doesn't restart on 78 and stops after five failed starts in five minutes. Before, a Hermes update that its environment couldn't import restarted the service every 10 seconds. The message now names the module Hermes failed to import, not a generic hint.

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
