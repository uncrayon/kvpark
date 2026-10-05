# Alpha validation

Each result below applies to its stated revision, model, and environment.
The September 2026 historical release and migration sections were recorded under
the former **sloth-memory** name; their commands now use kvpark's package and
environment-variable names. New development results are identified separately.

## Muse CPU snapshot acceptance

On September 27, 2026, the development checkout passed `scripts/acceptance.py`
with the real `Muse-Glimmer-30B-Q8_0.gguf` weights. It used the pinned llama.cpp
commit `85d5703a3b1b47243213a39059a6e3076c92733a` with the included slot-resume
patch, CPU inference, F16 K/V caches, a 4,096-token context, 12 threads, and
256-token batch/microbatch sizes. Flash attention and weight repacking were
disabled; weights were memory mapped. Eight context checkpoints used 128-token
minimum spacing.

The harness launched its own backend and proxy on temporary ports with a private
archive directory and `--cache-ram 0`. It processed a 2,467-token prompt, exceeding
Muse's 2,048-token sliding window, then parked 2,474 tokens. After measuring a
resident continuation, it stopped both private processes and launched fresh ones.
The first request after restart was the continued transcript, without replaying
the original request first.

| Stage | Cached tokens | Processed tokens | Whole response |
| --- | ---: | ---: | ---: |
| Cold prompt | 0 | 2,467 | 104.900 s |
| Resident continuation | 2,474 | 26 | 3.487 s |
| Direct continuation after both restarts | 2,462 | 38 | 4.056 s |

The archive contained a 114,788,284-byte `.bin` and a 164,988,268-byte `.bin.resume`
companion. Both persisted across the restart, matched the published metadata,
and the restore event reported the parked token count. The resident and restored
continuations returned exactly equal complete message objects, including
`reasoning_content`.

The eight-token output budget ended during reasoning; neither continuation
produced a final answer. The request used `reasoning_effort="none"`, but this
model still returned reasoning within that budget. This is evidence of snapshot
persistence, substantial cache reuse, and deterministic continuation, not task
answer accuracy. Durations measure complete HTTP responses, excluding process
startup and model loading; they are not TTFT. The OS page cache was not flushed.

The local evidence file is `evidence/linear-transfer/muse-real-acceptance-v2.json`;
generated evidence is not included in the package. Reproduce with an installed
kvpark checkout and the matching patched runtime:

```bash
python scripts/acceptance.py \
  --server runtime/build/bin/llama-server \
  --model /path/to/Muse-Glimmer-30B-Q8_0.gguf \
  --output evidence/muse-acceptance.json \
  --ctx-size 4096 --prompt-repetitions 220 --max-tokens 8 --threads 12 \
  --batch-size 256 --ubatch-size 256 \
  --cache-type-k f16 --cache-type-v f16 --flash-attn off \
  --load-mode mmap --no-repack \
  --min-prompt-tokens 2049 --direct-continuation
```

This run validates the stated CPU configuration only. It neither validates the
existing Vulkan/DFlash serving configuration nor configures that deployment for
kvpark. GPU/speculative deployment and cross-model transfer require separate
acceptance tests.

## Coordinated updates — 0.2.0a1

Local release checks on September 6, 2026:

- 77 Python tests: 76 passed, one optional model test skipped.
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

Package upgrade and failed-start rollback acceptance passed in CI on Linux,
macOS, and Windows. Tests also cover Python environment aliases and native
launchers whose process differs from the running proxy. Native gateway restart scheduling is covered through
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

Run the integration tests in Hermes' Python environment with kvpark installed:

```bash
python tests/hermes/test_integration.py -v
```

For the optional real-model test, set `KVPARK_TEST_SERVER` and `KVPARK_TEST_MODEL`
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
The original public release's tested scope was the standalone Qwen CPU flow;
newer dated results above retain their own stated scope.

## llama.cpp a7b94df2c runtime upgrade — October 4, 2026

The runtime pin moved from `85d5703a3b1b47243213a39059a6e3076c92733a` to
`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e`, 583 upstream commits later.
`patches/llama-slot-resume.patch` was rebased: only its hunk positions changed, and
the patched `server-context.cpp` is identical to applying the old patch with offsets.
No other patch is needed: upstream now fixes the DFlash image failure that the
originating deployment patched locally. [Compatibility](compatibility.md#production-patch-set)
explains the patch split. Everything ran on an AMD Ryzen AI Max+ 395 (Radeon 8060S,
gfx1151, ROCm 7.2.2) with private servers on ports 18600-18619. The production
services stayed stopped and were not changed. Other development work shared the
machine during some runs, so the timings are single observations, not benchmarks.
The evidence, including the check scripts named below, is local to the development
checkout (`evidence/llamacpp-upgrade/`) and is not part of the repository.

**Builds.** `python scripts/build_runtime.py` built the default CPU runtime end to end
(51 s at `--jobs 24`). Three more out-of-tree builds of the same `llama-server` source
used the flags in [compatibility](compatibility.md#building): a CPU build with tests, a
Vulkan build, and a ROCm gfx1151 build with the production CMake options, whose source
also carried the production `llama-bench` patch. Those three report
`b11398-a7b94df2c`. The first script build reported the enclosing kvpark commit
(`b34-b14e742`) and embedded the newest web UI instead of the pin's, because its export
has no Git metadata. The script now passes the pin's build number and commit to CMake;
its rebuild (74 s) reports `b11398-a7b94df2c`, embeds the `b11398` UI, has the same
patched source, and passed the Qwen3-4B acceptance below again (cold 0/915 in 8.578 s,
resident 922/22, replay after both restarts 914/1, outputs identical). On ROCm,
`test-backend-ops -o TOP_K -b ROCm0` passed 525 of 525 cases, with no unsupported rows.

**kvpark park/resume on the new CPU runtime** (`scripts/acceptance.py`, script build,
ports from the session's range):

| Model and settings | Stage | Cached | Processed | Whole response |
| --- | --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 Q8_0, 4,096 context, 8 threads | Cold prompt | 0 | 915 | 8.785 s |
| | Resident continuation | 922 | 22 | 0.677 s |
| | Replay after both restarts | 914 | 1 | 0.455 s |
| | Following continuation | 922 | 22 | 0.687 s |
| Muse-Glimmer-30B Q8_0, the September 27 settings | Cold prompt | 0 | 2,467 | 111.462 s |
| | Resident continuation | 2,474 | 26 | 3.645 s |
| | Direct continuation after both restarts | 2,462 | 38 | 4.238 s |

Both runs passed, and the restored messages equalled the resident ones exactly. The
Muse snapshot (`.bin` 114,788,284 bytes, `.bin.resume` 164,988,268 bytes) has exactly
the sizes recorded on the old pin.

**Bridge sidecars on the new CPU server.**
`experiments/universal_kv/fixtures_bridge/live_resume_check.py` passed all seven cases:
no draft, an M-RoPE DFlash draft, a NEOX DFlash draft, MTP, media, and bridge switches
into a DFlash target and an MTP target. Every translated restore was reused exactly
(64, 321 and 288 tokens), and the negative checks (no `.resume`, an empty draft blob)
were refused.

**DFlash position fix.** `evidence/llamacpp-upgrade/tools/dflash_image_check.py` ran a
tiny 5120-wide Qwen3.5 hybrid (IMROPE), a two-layer M-RoPE DFlash draft (8-token
window) and the real Qwen3.8 F16 projector on the CPU. Each run sent 2,600 text
tokens, then the same text followed by a 1024×1024 image (1,024 image tokens, two
512-row microbatches). It then restored the text save twice, once with an empty draft
blob and once with the bridge's aligned dummy, and continued:

| Binary | Image after 2,600 text tokens | Draft cells after empty-draft restore (target at 2,689) |
| --- | --- | --- |
| Old pinned runtime (`85d5703a3` + slot resume) | HTTP 500 `failed to process mtmd chunk` (`failed to find a memory slot`) | 2,607-2,689 (aligned) |
| Old production binary (+ contiguous-position patch), CPU device | Works, but afterwards the draft's window sits at positions 34-41 while the target is at 2,642 | 0-169 (misaligned; `process() did not run` warning) |
| New pin (`a7b94df2c` + slot resume) | Works; the draft skips the image's positions | 2,607-2,689 (aligned) |

With the aligned dummy, all three binaries kept the draft at the target's positions.

A 2-second 512×512 video of the same image after the 2,600 text tokens worked on both
pins: its eight frames arrived as five image chunks of about 260 rows, and each chunk
fit the draft's free cells. The new pin's draft skipped the video's rows (2,634 draft
cells for 3,914 target tokens, its window at the target's last positions); the old
pin's draft held all 3,914.

The failure also appears with the real model. With the production arguments moved to
the CPU (devices `none`, a 16,384 context), Qwen3.8-27B Q8_0 with the DFlash2 Q8 M-RoPE
draft and the F16 projector returned HTTP 500 `failed to process mtmd chunk` (`failed
to find a memory slot for batch of size 438`) for the 950-token image after about
2,650 text tokens on the old pinned runtime. The new pin's `build_runtime.py` build
answered the same request (3,607 prompt tokens at 20 t/s on the CPU, draft acceptance
0.49) and read all 13 checked strings of the image described below.

**Production-equivalent ROCm build, Qwen3.8-27B Q8_0.** The production arguments were
used as they are, except for the port, the slot directory, the alias and a 65,536
context: DFlash2 Q8 M-RoPE draft (depth 7), F16 projector on ROCm0, F16 caches, flash
attention, `--load-mode none`. Each prompt ran cold with the slot erased, at
temperature 0 and with 256 generated tokens under the production reasoning settings.
The old production binary ran the same script.

| Request | New build, DFlash2 | Old production, DFlash2 | New build, MTP |
| --- | --- | --- | --- |
| 2,644-token prompt: prefill, decode, acceptance | 287.3 t/s, 19.43 t/s, 0.367 | 291.4 t/s, 20.26 t/s, 0.374 | 263.2 t/s, 14.57 t/s, 0.520 |
| 14,295-token prompt | 294.7 t/s, 18.36 t/s, 0.385 | 284.7 t/s, 20.18 t/s, 0.396 | 309.0 t/s, 17.43 t/s, 0.589 |
| Image (950 tokens) after about 2,650 text tokens | 263.4 t/s, 24.13 t/s, 0.491 | 243.8 t/s, 26.70 t/s, 0.553 | 289.4 t/s, 19.55 t/s, 0.645 |
| Transcription of the image's 13 items | 13/13 | 13/13 | 13/13 |
| Slot save, 3,718 tokens with the image (`.bin` / `.resume`) | 400.7 MB / 595.5 MB | 400.7 MB / 583.0 MB | 400.7 MB / 503.9 MB |
| Restore, then continuation: reused / processed | 3,718 / 39, 163 ms restore | 3,719 / 38, 345 ms restore | 3,718 / 39, 143 ms restore |
| Restored continuation equals resident continuation | Yes | Yes | Yes |

The image was a synthetic 1200×800 document page with a Spanish heading, a four-row
table and a footer. Every run read all 13 checked strings, including `7.3. Inventario
de la Sala Norte`, `1,284`, `3,907` and `5,989`. The servers logged no `failed to
process mtmd chunk`, memory-slot, position or restore errors. MTP used
`--spec-type draft-mtp --spec-draft-n-max 4` with the Qwen3.8 MTP-only Q8_0 head, and
the new build runs it unchanged.

**Parking across processes, and video.** A second run of each speculative mode on the
new build killed the server after the resident continuation and restored the slot in a
fresh process with the same arguments, as parking does. Both reused all 3,718 tokens
(restore 162 ms with DFlash2, 126 ms with MTP), processed 39, and repeated the resident
answer exactly. The fresh process then took a 2-second 800×534 video of the same page
after the 2,623 cached text tokens: eight frames, sent as five image chunks, 2,182 new
tokens. With DFlash2 (acceptance 0.50) and with MTP (0.74) the answer contained all 13
checked strings, and neither process logged an error.

Speculative decode speed depends on what the model generates, so the decode
differences above are not raw kernel speed. The raw comparison below used
`llama-bench` without a draft (`-p 512 -n 128 -d 0,8192 -r 3`), alternating the new
build and the old production libraries for four rounds, three starting with the new
build and one with the old:

| `llama-bench`, tokens/s (4 rounds × 3 samples) | New build | Old production |
| --- | --- | --- |
| 512-token prefill, empty context | median 332 (271-343) | median 335 (313-351) |
| 128-token decode, empty context | median 7.68 (4.95-7.76) | median 7.75 (6.90-7.76) |
| 512-token prefill after 8,192 tokens | median 300 (161-311) | median 293 (282-299) |
| 128-token decode after 8,192 tokens | median 7.53 (4.91-7.60) | median 7.61 (7.57-7.62) |

The low new-build values come from one run, the first of the sequence, in which every
decode sample stayed between 4.9 and 6.0 t/s. Its cause was not found: other
development work shared the machine, and other processes were logged only at the
start of each run. Without that run the new build's medians are 331, 7.73, 305 and
7.54 t/s. Prefill is then within 1% on an empty context and 4% faster after 8,192
tokens. Decode is within 0.3% on an empty context and about 1% slower after 8,192
tokens (7.48-7.60 against 7.57-7.62 t/s). An earlier two-round comparison also had
low new-build values in its first run, and in its second round a 13% lower
empty-context prefill for the new build (300 against 344 t/s) that the four rounds
above did not reproduce. These are observations on a shared machine, not a
quiet-machine benchmark. A first attempt, made while another 30B server shared the
GPU, was discarded.

**Vulkan build.** The slot save and restore cycle passed on `Vulkan0`
(`tools/slot_cycle.py`, full offload, flash attention, 8,192 context). The first two
rows restored in the same process after an erase; the last one restored in a fresh
process after the server was killed, with a prompt longer than Muse's 2,048-token
sliding window:

| Model | Saved | Reused after restore | Restored answer equals resident |
| --- | ---: | ---: | --- |
| Qwen3-4B-Instruct-2507 Q8_0 | 1,804 | 1,804 | Yes |
| Muse-Glimmer-30B Q8_0 | 1,506 | 1,478 (from a checkpoint) | Yes, including the reasoning text |
| Muse-Glimmer-30B Q8_0, 2,724-token prompt, fresh process | 2,747 | 2,719 (from a checkpoint) | Yes, including the reasoning text |

For Qwen3-4B, prefill ran at about 2,090 t/s and decode at 45 t/s. For Muse, prefill
ran at about 350 t/s and decode at 7.8 t/s.

**State formats.** Slot files, sequence blobs, KV and recurrent sections, server token
lists, mtmd chunks and the `.resume` sidecar did not change. The KV translator's
readers are affected in three places: nextn layers are no longer in the trunk cache
of deepseek2 and similar models, qwen4exp's indexer cache is K-only with k-pool rows,
and bailingmoe3 uses M-RoPE when its GGUF has rope sections.
[experiments/linear_transfer/STATE_FORMAT_a7b94df.md](../experiments/linear_transfer/STATE_FORMAT_a7b94df.md) lists the changes.
`kvstate.py` and the translator now read and write both pins' states (a state is read
with either pin's layout, whichever fits), and every experiment suite passes with
either pin's binaries; the commands are in
[experiments/linear_transfer/README.md](../experiments/linear_transfer/README.md#tests-on-both-pinned-runtimes).
Since October 5, 2026 the repository's `runtime/` is a `scripts/build_runtime.py` build
of a7b94df2c; the 85d5703a3 build that was there is kept at `runtime-85d5703a3/`.

### Switching the local Qwen3.8 service to the new ROCm build

These are the commands for this machine; run them yourself. The Qwen3.8 backend
runs under the user unit `kvpark.service` (`kvpark start --hermes --supervise`), and
its binary path is stored in the Hermes profile's kvpark plugin settings.

```bash
OLD=/home/uncrayon/Documents/benchmark_models_linux/production/qwen38-dflash2-rocm-resume-20260905
NEW=/home/uncrayon/Documents/benchmark_models_linux/production/qwen38-dflash2-rocm-a7b94df2c-20261004

# 1. Stop the supervisor, proxy and backend, and the gateway that loads the plugin.
systemctl --user stop hermes-gateway.service kvpark.service

# 2. Keep a stable copy of the new build (its libraries resolve next to the binary).
mkdir -p "$NEW"
cp -a /home/uncrayon/Documents/kvpark/runtime-next/build-rocm/bin "$NEW/"
"$NEW/bin/llama-server" --version   # version: 0.5.0-dev (build 11398, commit a7b94df2c)

# 3. The old bundle was selected with LD_LIBRARY_PATH. Remove that line, or the new
#    binary would load the old libraries, which have the same file names.
grep -n LD_LIBRARY_PATH ~/.hermes/.env
systemctl --user show-environment | grep LD_LIBRARY_PATH   # if it is not in .env

# 4. Point the kvpark plugin at the new binary; the model and arguments stay the same.
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.before-a7b94df2c
sed -i "s#$OLD/bin/llama-server#$NEW/bin/llama-server#g" ~/.hermes/config.yaml
grep -n "$NEW" ~/.hermes/config.yaml

# 5. Start again and check.
systemctl --user start kvpark.service hermes-gateway.service
~/.hermes/hermes-agent/venv/bin/python -m kvpark doctor --hermes
curl -s http://127.0.0.1:8090/props | python3 -c 'import json, sys; print(json.load(sys.stdin)["build_info"])'
```

The last command should print `b11398-a7b94df2c`. Conversations parked under the old
binary are not restored by the new one: the first return to each prefills it again,
and parking it then produces a compatible snapshot. To roll back, stop both units,
restore `~/.hermes/config.yaml.before-a7b94df2c` and the `LD_LIBRARY_PATH` line, and
start them again.
