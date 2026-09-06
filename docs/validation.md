# Alpha validation

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
