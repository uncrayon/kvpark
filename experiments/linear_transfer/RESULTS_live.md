# Live handoff on real llama-servers

On October 1, 2026, the handoff ran end to end on real servers for the first time, using
`live_handoff.py` with the deployed services stopped. The script started two throwaway
instances, on their own ports and with private slot directories:

- **Qwen:** the production binary with the production flags. It ran on ROCm, without the
  vision projector, with ctx 36,864.
- **Muse:** upstream llama.cpp 85d5703a3, built with Vulkan into
  `evidence/linear-transfer/muse-server-85d5703a3`. It used the production flags from
  `start-muse-server.sh`'s dry run, including DFlash, with ctx 36,864 and
  `--slot-save-path`.

Both servers stayed resident the whole time, as in the intended use. The test covered 9
held-out conversations of 4.9K–27.2K Muse tokens: 3 chats of 4K–8K from v3, plus 6
concatenated sessions, agent traces and real chats of 8K+ from v3b. Each prompt was the
Muse token ids before the final assistant message, followed by 256 greedy tokens.

## Mechanics: it works on a real server

| Check | Result |
| --- | --- |
| Built file restored, reused prefix (`cache_n`) equals T0 | **9 of 9** (handoff and oracle rebuild) |
| Muse's own slot save restored and reused | 0 of 9, as predicted: 2,048 sliding cells fail the server's reuse check, and 2,049 pass |
| Restore time | 28–45 ms |
| Qwen slot save | 0.16–0.39 s (0.5–2.0 GB) |
| Server errors or warnings, crashes | none |
| DFlash draft acceptance (native / handoff) | 0.16–0.43 / 0.18–0.48, no systematic change |
| Lowest MemAvailable (both servers, the mem0 servers, the build) | 38.7 GB |

## Latency: about 22–28 s to hand over, whatever the length

Native means Muse prefills the whole prompt. Handoff is the sum of these steps:

1. Qwen slot save.
2. Read and parse the Qwen file.
3. Build the Muse file (v3 mapper, table fill, W = 2048, 2,049 sliding cells).
4. Write the file.
5. Erase and restore Muse's slot.
6. Prefill the native tail and produce the first token.

| Prompt (Muse tokens) | Content | Native TTFT | Build | Tail TTFT | Handoff total | Saving |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,909 | chat | 33.6 s | 6.3 s | 16.8 s | 23.5 s | 10 s (30%) |
| 6,050 | chat | 41.2 s | 4.4 s | 17.7 s | 22.5 s | 19 s (45%) |
| 7,933 | chat | 53.5 s | 4.4 s | 16.4 s | 21.3 s | 32 s (60%) |
| 8,309 | concatenated | 56.0 s | 4.9 s | 15.5 s | 20.9 s | 35 s (63%) |
| 12,397 | concatenated | 83.9 s | 5.5 s | 15.7 s | 21.9 s | 62 s (74%) |
| 15,615 | agent | 106.0 s | 5.9 s | 16.8 s | 23.5 s | 83 s (78%) |
| 17,711 | agent | 120.7 s | 6.4 s | 17.5 s | 24.7 s | 96 s (80%) |
| 23,376 | concatenated | 160.4 s | 9.0 s | 17.9 s | 28.1 s | 132 s (83%) |
| 27,185 | concatenated | 187.5 s | 10.1 s | 16.2 s | 27.4 s | 160 s (85%) |

- **The server prefills more slowly than assumed.** Muse prefills at 147 tok/s with
  DFlash on, not the 235 tok/s the probe measured without a draft model. Native TTFT is
  therefore about P / 147 s, and the handoff pays off from roughly 3.5K tokens on.
- **Most handoff time is the native tail** (2,048–2,500 tokens, about 16–18 s). The
  build grows with T0, from 4.4 s to 10.1 s. In this test it ran on the CPU inside the
  test process (8 BLAS threads), and its mapper files were already in the page cache.
- **The design's savings table was too pessimistic.** At 8K the measured saving is
  35 s, not 16–21 s; at 27K it is 160 s.

## What this test does not measure

- **Quality.** Greedy continuations diverge early even between two native runs of the
  same prompt (as early as token 18, from batch-shape differences). First-divergence
  positions are therefore not a quality measure. Quality comes from the teacher-forced
  evaluations in [RESULTS_v3.md](RESULTS_v3.md) and [RESULTS_v3b.md](RESULTS_v3b.md),
  and those say the handoff is fit for chat and not for agent traces.
- **Hermes requests.** Prompts were sent to `/completion` as token ids, not as Hermes
  chat bodies, so `/apply-template` + `/tokenize`, tools rendering and the Hermes
  system-prompt rebuild were not exercised.
- **A busy Qwen.** Qwen was idle during the Muse steps, and the build did not run as a
  separate memory-capped worker.

## Reproduce

`evidence/linear-transfer/live-handoff-20261001/` holds `results.jsonl` (per document:
timings, cache counts, draft statistics, generated token ids, no text), `meta.json`
(both argvs, build info `b10815-85d5703a3`, memory) and both server logs.

```
live_handoff.py --select run-v3:4096-8192 --select run-v3b:8192-16384 \
  --select run-v3b:16384-29500 --per-bucket 3 --mapper run-v3/refit/mapper \
  --muse-server muse-server-85d5703a3/bin/llama-server --qwen-server \
  --output live-handoff-20261001
```
