# Alpha validation

## Coordinated updates — 0.2.0a1

Local release checks on September 6, 2026:

- 75 Python tests: 74 passed, one optional model test skipped.
- 18 Hermes integration tests: 17 passed, one optional model test skipped,
  against both the installed Hermes and pinned upstream source.
- Real wheel installation in a disposable Python 3.14 virtual environment:
  upgrade succeeded; a subsequent package with a deliberately broken proxy
  startup rolled back to the previous wheel and a healthy proxy. The proxy
  address and upstream backend were retained. No production model was used.
- Tests cover maintenance gating, drain timeout, failed snapshot saves, invalid
  release hashes/formats, wrong Python environments, interrupted installation,
  failed replacement shutdown, and a lost acknowledgement when reopening traffic.
- Release wheel and source archive passed package metadata validation.

The CI workflow repeats package upgrade and failed-start rollback acceptance on
Linux, macOS, and Windows. Native gateway restart scheduling is covered through
Hermes' real dispatch with the final restart operation mocked. This is separate
from the earlier live model park/resume evidence below.

## Live Hermes migration — main development version

On 2026-09-05, sloth-memory was installed into the existing Hermes Python 3.11
environment using this repository's installation guide. The original qwen-slot
plugin, provider override, and system services were retired after a private
rollback backup and parking the resident conversation. Setup ran through the real
Hermes CLI slash command, rather than writing the plugin's settings by hand.

This run used Linux, Hermes 0.21.0 (installed upstream revision `79445a49` with
pre-existing local fixes), and Qwen3.8-27B Q8 with DFlash2 on ROCm. It reused the
previously validated runtime bundle, including its slot-resume extension, DFlash
position fix, and AMD gfx1151 correction. It does **not** establish that the
standard package build includes those hardware/model-specific fixes.

| Check | Observed result |
| --- | --- |
| Installation and discovery | Installed `0.2.0.dev0` in Hermes' environment; `/sloth-memory setup` available without granting tool overrides |
| Native setup | Plugin launched backend and proxy; preserved context size 262,144, GPU/draft/projector and reasoning settings |
| Actual request ownership | A real Hermes conversation reached the proxy with a verified foreground archive key |
| Park | Slash command saved 666 tokens, approximately 0.665 GiB |
| Process-loss recovery | Both backend and proxy were stopped, then restarted by plugin discovery; RAM prompt caching was disabled |
| Occupied proxy port | A test listener held port 8080; the proxy selected 41283 and Hermes persisted and followed that route |
| Restored continuation | 666 cached / 35 processed tokens; 0.135 s disk restore and 0.815 s time to first token; correctly recalled the project code and mascot |
| Vision and reasoning | Through Hermes with `xhigh` reasoning, read the original document heading, both page numbers, and the nonlinear terms correctly |
| Cleanup controls | With cleanup enabled, an eight-day-old disposable orphan was removed while a six-day-old orphan and unrelated file remained; restored seven-day/on defaults |
| Delete | Slash command deleted only the acceptance conversation's saved snapshot; transcript retained |
| Gateway | Existing Hermes gateway restarted on the selected proxy route; runtime identity and single-slot health checks passed |

The test exposed a stale startup-error message after successful onboarding. The
retry path now clears that error, and onboarding recognizes an already connected
configuration. A regression test covers recovery after an initial startup failure.
The installation guide now covers the tool-override prompt, environment selection,
legacy-service migration, and preserving runtime arguments and library paths.

Validation after the fix: 55 package tests (one Windows-only skip on Linux), plus
nine passing integration checks against the installed Hermes environment (the
optional isolated real-model test was skipped; the live workflow above supplied
real-model evidence). This is a functional installation test with a short text
conversation, not a matched performance benchmark, a fresh 100K-context test,
or a macOS/Windows hardware acceptance run. No full OS reboot was performed.

## Discord first-park follow-up

The first live Discord attempt after migration exposed a gap in the installation
test: restarting the gateway and issuing `park` had not sent that existing
conversation through the new proxy. The plugin reported an identity error even
though the missing item was saved or resident state for that chat.

The message and setup guide now explain the required first normal message.
Integration tests exercise Hermes' actual idle gateway command dispatcher with a
Discord event: first park without state, a matching resident conversation, and a
different chat's resident state. The matching case resolves the correct key through
the hook without a bound session ID; unrelated process environment identity is
ignored. These tests use an isolated archive and stub the final save operation;
they do not send Discord messages or constitute another real-model cache test.
Twelve integration tests pass on both the installed and pinned Hermes versions;
the optional isolated real-model test is skipped in these runs.

## Hermes and cleanup — alpha.2

The standalone suite now covers persistent cleanup controls, expiry by save date,
crash-orphan cleanup, directory ownership, and safe startup/attachment. Separate
tests load the actual pip entry point through Hermes in temporary profiles,
dispatch native slash commands, persist settings through Hermes' config writer,
and exercise real request middleware with foreground/background turn bindings.

An optional real-model test also verifies that Hermes discovery itself starts
the CPU backend and proxy, native park/delete operate on a real conversation,
unloading Hermes leaves cleanup services running, and reloading attaches without
duplicate processes. It passed locally with Qwen3-4B and the pinned CPU runtime.

Run the integration tests in Hermes' Python environment with sloth-memory installed:

```bash
python tests/hermes/test_integration.py -v
```

For the optional real-model test, set `SLOTH_TEST_SERVER` and `SLOTH_TEST_MODEL`
to absolute local paths. It uses temporary profiles, ports, and archive files,
and terminates only the processes it launches. The Hermes CI job pins upstream
`9dd6634c5635321cf38840cc30e9b51226689128` and runs without model downloads.

## Standalone release test — September 5, 2026

The standalone package was exercised without Hermes against a clean CPU build of
the pinned llama.cpp commit plus the included persistence patch. The model was
`Qwen3-4B-Instruct-2507-Q8_0.gguf`, with a 4,096-token context, eight checkpoints,
128-token checkpoint spacing, four CPU threads, and RAM prompt cache disabled.

The acceptance harness used a synthetic workshop conversation with an exact
project-code answer. It parked the first response, measured a resident
continuation, stopped both private processes, and launched fresh processes before
replaying the original prompt and its continuation.

| Stage | Cached tokens | Processed tokens | Whole response |
| --- | ---: | ---: | ---: |
| Cold prompt | 0 | 915 | 30.139 s |
| Resident continuation | 922 | 22 | 2.253 s |
| Replay after both restarts | 914 | 1 | 1.327 s |
| Following continuation | 922 | 22 | 2.252 s |

Both restored outputs exactly matched their corresponding baseline outputs.
These are one-run functional observations on an AMD Ryzen AI Max+ 395 host,
not broad model quality or hardware performance claims. The OS page cache was
not flushed. The timings are whole responses, not time to first token or cold
physical SSD throughput. Run `scripts/acceptance.py` on your own configuration.

The machine's existing production inference services were not stopped or changed.
The script used CPU-only inference, temporary ports, and a private archive directory.
The release includes the synthetic acceptance JSON as a separate asset.

## Automated coverage

The Python suite covers explicit ownership, background forks, streaming and
concurrent generation serialization, restart identity, incompatible weights and
prompt settings, expiry, atomic overwrite failures, oversized snapshots, failed
restores, control endpoints, optional authentication, and launcher ownership.

GitHub CI installs and tests Python 3.10, 3.12, and 3.14, checks source/wheel
packaging, and compiles the pinned CPU runtime. CI does not download model weights
or run GPU/model acceptance tests.

## Historical context

The originating local Qwen/Hermes implementation recorded a separate roughly
102K-token image conversation restored after process loss, with 102,010 cached
tokens and 35 newly processed tokens. That setup used ROCm, DFlash2, and additional
hardware/speculative fixes that are not shipped in this alpha.

That result motivated this project. It is not a benchmark for the standalone
release and is not evidence of universal hybrid-model, draft, image, or GPU support.
The public release's tested scope is the standalone CPU flow described above.
