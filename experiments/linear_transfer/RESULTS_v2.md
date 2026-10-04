# Qwen → Muse v2 results: multi-layer K/V mapping and prefix injection

The v2 run finished on September 28, 2026. It used the real Q8 checkpoints on CPU
and [corpus v2](corpus_v2.md), which is chat-formatted and split into train, dev and
test. The mapper fills Muse's cache for the conversation prefix from Qwen's cached
keys and values across all 52 Muse layers.

The main test covers documents whose transferred prefix is at least 256 tokens.
There, Muse's next-token loss on the continuation rises by **0.11 nats/token (about
+11% perplexity)**. The mapped prefix recovers about **83%** of the loss gap between
an information-free cache and Muse's own prefill. The sample is small (5 long-prefix
documents), so read these numbers as a first measurement, not a validated
capability. The tool that writes a mapped cache into a running server and the
latency comparison against native prefill are still to be built.

## Step 1: v1 re-analysis

These results reuse the v1 captures, with λ chosen by document-grouped CV on the
training documents only (`probe.py --ridge auto`, `reanalyze_v1.py`). The source is
Qwen layer 31 V and the target is Muse layer 27 V.

| Held-out (8 v1 eval documents) | R² |
| --- | ---: |
| Naive ridge, λ = 0.01 (effectively least squares) | 0.398 |
| Ridge with CV-selected λ = 1,585 | **0.451** |
| Same, in Muse's output space (through W_o) | 0.444 |
| Muse's own token embeddings (current + 3 previous tokens + position) | 0.142 |
| Qwen values beyond that receiver-side baseline | +0.316 (CV +0.324, every fold ≥ +0.313) |

The cross-model signal is contextual, not lexical.

## Collection

`kvpark-kv-probe` captured K (before RoPE) and V for every attention layer in one
pass per model. It covered the 16 full-attention layers of Qwen and all 52 layers of
Muse, and additionally captured Muse's Q, output gate and attention output on the
dev and test evaluation layers.

| Model | Tokens | Time | Throughput |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 157,168 | 1 h 47 m | 24.6 tok/s |
| Muse-Glimmer-30B | 154,036 | 1 h 32 m | 27.8 tok/s |

Alignment pairs tokens that end at the same byte of the same message segment in
both chat renderings. It covers 98% of Muse's content tokens in every split, which
gives 83,893 train, 18,064 dev and 20,971 test pairs. Template and special tokens
(13–21% of Muse tokens) always stay native.

## Step 4: multi-layer fit

Each Muse layer gets its own map for K and for V. The inputs are concatenated from
the top-k Qwen full-attention layers, ranked by single-layer dev R², with k in {1, 2,
4, 8, 16}. The ridge strength is chosen on dev, and test was evaluated once.

| Test R² (mean over layers) | K | V |
| --- | ---: | ---: |
| Muse full-attention layers (13) | 0.756 | 0.641 |
| Muse sliding-window layers (39) | 0.769 | 0.607 |
| All 52 layers | **0.766** | **0.615** |

Dev and test agree to within 0.01, and 83 of the 104 maps chose k = 16. For Muse
layer 27, V is 0.572, against 0.451 for v1's single tuned source layer (on a different
corpus, so only roughly comparable). Keys are about
0.15 R² easier than values. The weakest maps are late sliding-layer values: layer 49
V 0.26, layers 46 and 48 V about 0.4. Per-head renormalization of K never won on
dev. Every native Muse K has per-head RMS exactly 1.0 (its k_norm weight is ones).

## Step 5a: attention output, offline

On real data, the numpy attention reproduces Muse's captured attention output. The
relative error is at most 7.9e-7 on a full layer (27) and 6.2e-7 on a sliding layer
(26).

At the positions injection scores, mapped K+V keeps the cosine of each layer's
residual contribution at 0.986–1.000. Because queries attend mostly to native tokens,
the information-free floor (the training-mean K/V) is already 0.95–0.998. The skill
is (cos − floor) / (1 − floor), and for K+V it is 0.45–0.80 per layer; it is lowest on
late sliding layers (layer 50: 0.45).

## Step 5b: prefix injection into Muse

Muse processes each test conversation through the start of its final assistant
message, and the first 256 tokens of that message are scored. Patched K/V replace
Muse's own at aligned content positions of the prefix, and all other positions are
recomputed natively. 12 test documents contributed 2,377 scored tokens.

The sanity condition, which patches Muse's own values back, gives KL exactly 0 on
every token. The positive control changes predictions on every document.

Long transferred prefix (≥ 256 patched positions, 5 documents, 813 tokens; mean [95%
bootstrap CI over documents]):

| Condition | ΔNLL vs native (nats/token) | KL(native ‖ condition) | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Layer 27 V only mapped | 0.000 [−0.020, 0.011] | 0.004 | 97.9% |
| Layer 27 K+V mapped | 0.011 [−0.005, 0.021] | 0.003 | 98.4% |
| 13 full-attention layers mapped | 0.050 [−0.002, 0.091] | 0.042 | 94.3% |
| **All 52 layers mapped (complete transfer)** | **0.108 [0.041, 0.185]** | **0.119** | **89.1%** |
| All 52, last 64 prefix tokens recomputed natively | 0.118 [0.042, 0.193] | 0.093 | 90.9% |
| All 52, system message recomputed natively | 0.107 [0.037, 0.187] | 0.118 | 88.9% |
| Information-free floor (training-mean K/V, all layers) | 0.648 [0.403, 0.958] | 0.678 | 75.5% |

Paired against the floor, complete transfer lowers NLL by 0.54 [0.35, 0.77]
nats/token. When the transferred prefix is short (7 documents with 24–138 mapped tokens, mostly
system and user turns), complete transfer changes NLL by −0.005 [−0.018, 0.008], with
KL 0.053 and 92% top-1 agreement.

Most of the loss comes from the sliding-window layers. Mapping the 13 full layers
costs 0.05 nats/token, and adding the 39 sliding layers doubles that. Recomputing the
last 64 prefix tokens lowers KL but does not improve NLL. The largest single loss is a
1,402-token multi-turn prefix (ΔNLL 0.22). The one prefix longer than Muse's
2,048-token window (a 2,770-token agent trace) has ΔNLL 0.07 over 34 scored tokens.

## Interpretation

- A closed-form linear map carries most of what Muse needs from a Qwen cache
  across families, tokenizers and cache geometries (4×256 → 2×128). This is the
  setting the NVIDIA paper lists as open. The +11% perplexity on long prefixes is
  near NVIDIA's best within-family pair (Qwen3 14B → 32B, WikiText-2 perplexity 7.33
  vs 6.79, +8%). The protocols differ, so this is a loose comparison.
- R² alone would have been misleading. V R² of 0.6 looks weak, yet the continuation
  loses only 0.11 nats/token, because attention averages errors and most queries lean
  on native template tokens.
- The next accuracy lever is the sliding-window layers. They are the weakest values
  by R² and cause most of the injection loss. With 84K training rows, a nonlinear or
  attention-weighted fit for those layers is now worth testing.

## Limits

- Few documents: 12 injected, 5 with long prefixes; prefixes up to 2.8K tokens. One
  corpus, one split, one seed.
- Scoring is teacher-forced on the dataset's reference answers, over the first 256
  tokens of the final message. It is not free generation, and final-answer accuracy
  was not tested.
- Captures and injection used the F32 cache on CPU. The live servers use F16 caches
  on Vulkan/ROCm.
- Template and special tokens are recomputed by Muse in every condition. A real
  handoff must also run those tokens on top of the mapped cache. Injection emulates
  exactly that, but no kvpark code path does it yet.
- Latency is not measured. Applying the mapper costs roughly 1–2% of Muse's
  per-token prefill compute (52 layers × 2 maps of up to 16,384 × 256). Writing a
  mapped cache into llama.cpp's state format and timing a GPU handoff are future work.

## Reproduce

Artifacts live under `evidence/linear-transfer/run-v2/` (about 62 GB, not shipped).
`summary.md` there holds every table in full: per-layer R² for each k, per-layer
attention skill and per-document injection results. The commands are in the
[README](README.md#v2-pipeline).
