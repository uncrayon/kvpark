# Qwen → Muse v3: long-context handoff on GPU

The v3 run finished on September 30, 2026. It tested the shippable handoff design
("Design A") on 30 held-out conversations with prefixes of 4.3K–28.4K Muse tokens.

- **Design A:** Muse receives a cache for the older part of the conversation, [0, T0),
  and natively prefills only the last W tokens before continuing.
- **Where the cache comes from:** a real Qwen slot state saved on GPU, passed through
  the linear mapper.
- **Chat-template, special and unmatched tokens** in the old region are filled from a
  per-token table.

The target was a loss increase under 0.05 nats/token. The pre-registered headline
condition (W = 2048, v3 mapper, table fill) misses it overall, **+0.242 [0.163,
0.331]**, but the average hides a clean split:

| Prefix bucket and content | Docs | Share of prefix handed over | ΔNLL, headline condition | Information-free floor | Share of the gap recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K–8K, multi-turn chat | 10 | 60% | **+0.021 [−0.005, 0.044], meets target** | +0.232 | **91%** |
| 8K–16K, long-document Q&A (LongAlign) | 10 | 80% | +0.342 [0.220, 0.462] | +0.398 | 14% |
| 16K–29K, long-document Q&A (LongAlign) | 10 | 90% | +0.364 [0.236, 0.519] | +0.384 | 5% |

Handing over a chat conversation works: 91% of what the old context contributes
survives, and the loss is within the target. Handing over a long document the model
must answer questions about does not work: the mapped cache is barely better than an
information-free one. Every 4K–8K document is chat, and every 8K+ document is
long-document Q&A, so this run cannot tell whether the failure comes from length or
from the kind of content. The two long buckets behave alike (+0.34 vs +0.36), which
points to content.

## Checks

- The handoff builder is exact on real data. Rebuilding Muse's cache from its own
  native values produced a file byte-identical to Muse's native save for every document
  (59 of 59).
- Restoring Muse's native cache on GPU gave KL exactly 0 on all 118 checks. With flash
  attention on and F16 caches, this is the deployment layout.
- Both information-free floors change every document's predictions (positive control).
- Scoring used 2.54M decoded tokens and ~590 built states, all on the GPU.

## What else the run showed

| Condition (all documents) | ΔNLL | Top-1 agreement |
| --- | ---: | ---: |
| Native tail W = 4096, table fill, v3 mapper | +0.217 | 83.5% |
| **Native tail W = 2048, table fill, v3 mapper (headline)** | **+0.242** | 81.0% |
| Same, template cells from Muse's native values (upper bound) | +0.245 | 82.1% |
| Same, v2 mapper | +0.259 | 80.5% |
| No native tail (W = 0), everything mapped | +0.292 | 74.4% |
| Information-free old region, native tail W = 2048 | +0.338 | 77.9% |
| Information-free, no tail | +0.770 | 64.7% |

- **The native tail matters for chat.** In the 4K–8K bucket, mapping everything
  (W = 0) costs +0.152. A native tail of 2,048 tokens brings that to +0.021, and 4,096
  tokens to +0.002.
- **Filling template tokens from the table is fine.** It is within noise of filling them
  with Muse's own native values: paired difference −0.003 [−0.019, 0.015].
- **More training data helps a little.** The v3 mapper (2.5× the training pairs) beats
  v2 by −0.016 [−0.025, −0.009] nats/token. On the v2 test split its R² is K 0.766 →
  0.780 and V 0.615 → 0.642.
- **Mapped values are worse on long documents.** The directly measured R² of mapped vs
  native full-layer K/V inside the handed-over region is K ≈ 0.70 / V ≈ 0.62 on a chat
  document, and K 0.42–0.61 / V 0.33–0.53 on two LongAlign documents. It does not fall
  with position (0 to 8K), so this is a content/domain effect, not positional
  extrapolation. Answering questions about a long document needs precise retrieval of
  specific tokens, which R² ≈ 0.5 keys and values cannot support. Continuing a chat
  mostly needs the gist, which survives.

## Speed

Measured on the Radeon 8060S (Vulkan) with the live servers stopped:

- Qwen prefill with state save: 198 tok/s.
- Muse prefill with state save: 235 tok/s.
- Muse seed-score: 211–230 tok/s.
- Building a handoff state on CPU with 8 threads: 30 documents in 31 minutes, including
  every condition. That is about 1 s per state for short documents and minutes for
  29K-token documents.

Handoff latency through a live server has not been measured.

## Limits

- 10 documents per bucket. Content type and length are confounded between the 4K–8K
  bucket and the 8K+ buckets.
- Teacher-forced scoring of the first 256 tokens of the final message; no free
  generation.
- The mapper was trained on chat-style conversations of 4.6K tokens or fewer, with CPU
  (v2) and GPU (v3 extra) numerics.
- States were restored by `kvpark-kv-probe`, not by a live llama-server, and DFlash
  drafting was not exercised.

## Reproduce

Run directory `evidence/linear-transfer/run-v3/` (not shipped). `long/report.md`
holds every table, including per-document results and ΔNLL against prefix length.
The command is in `run_long.py --help`. The corpus is described in
[corpus_long.md](corpus_long.md).
