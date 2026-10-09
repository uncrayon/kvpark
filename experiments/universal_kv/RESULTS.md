# Universal KV translator: results (October 2-9, 2026)

One shared latent ("hub") now translates attention caches between four local models of
three architectures and three tokenizers, and a `kvpark bridge switch` moves real
conversations between running llama-servers on an EVO-X2-class machine (Ryzen AI Max+ 395,
128 GB, Radeon 8060S). Design: [ARCHITECTURE.md](ARCHITECTURE.md). Survey of the top-20 open
models: [survey/SURVEY.md](survey/SURVEY.md).

| Model | Architecture | Attention cache | Tokenizer |
| --- | --- | --- | --- |
| Qwen3.8-27B (Q8) | hybrid: 48 gated-DeltaNet + 16 full-attention layers | 4 × 256, partial M-RoPE | Qwen3.5 (248K) |
| Ornith-1.5-9B (Q8) | same family, 24 + 8 layers | 4 × 256 | Qwen3.5 (same) |
| Qwen3-4B-Instruct-2507 (Q8) | dense GQA, 36 layers | 8 × 128, NeoX RoPE | Qwen2/3 (152K) |
| Muse-Glimmer-30B (Q8) | 39 sliding (2,048) + 13 global NoPE layers | 2 × 128 | Llama 4 (202K) |

## Summary

The headline numbers, each with the section that measures it. Every live number comes from one
machine shared with other work; the sections give the load and memory at each run.

| What | Result | Section |
| --- | --- | --- |
| Hub quality (4 models, 3 tokenizers, 4,096 dims; held-out spans) | K R² 0.67–0.77, V 0.49–0.67 in all 12 directions; Qwen3.8 → Muse reaches 97% / 93% of a dedicated pairwise map | [1](#1-one-hub-for-four-models-test-tokens) |
| Loss on held-out long chats (4.5–15K tokens, ≈2,048-token native tail) | Qwen3.8 → Ornith +0.016, → Muse +0.029, → Qwen3-4B +0.008 nats/token (the last with the native prompt start); Qwen3-4B as the source recovers 0–15% of the gap | [2](#2-end-to-end-loss-on-the-target-model) |
| Agent traces with tool output | +0.295 nats/token Qwen3.8 → Muse (2-model hub); KL 0.40–0.50 into Qwen3.8: not usable | [2](#2-end-to-end-loss-on-the-target-model), [7](#7-hybrid-targets-what-the-recurrent-cell-starts-from-october-4-2026) |
| Hybrid target, recurrent cell from the model's mean instead of zeros (Qwen3.8, native attention, chats, tails ≈1.4K / 2.4K / 4.4K) | +0.053 / +0.039 / +0.029 → +0.008 / +0.002 / +0.003 nats/token; KL lower on 12 of 12 documents at every tail on Qwen3.8 and on Ornith | [7](#7-hybrid-targets-what-the-recurrent-cell-starts-from-october-4-2026) |
| The same with translated attention (Muse → Qwen3.8) | +0.129 / +0.102 / +0.061 → +0.097 / +0.064 / +0.027 nats/token | [7](#7-hybrid-targets-what-the-recurrent-cell-starts-from-october-4-2026) |
| Live switches among Qwen3.8 (production DFlash2 build), Muse and Qwen3-4B, 11–14K tokens | 24 of 24 hops reused exactly T0; with the GPU engine a switch took 7.7–18.0 s against 14.8–86.0 s of native prefill | [6](#6-stage-2-live-october-3-2026-engine-daemon-the-production-qwen38-build-mtp-images) |
| Engine, warm, 13K tokens | iGPU 0.45–0.47 s (Qwen3.8 → Muse) and 0.72–0.78 s (Muse → Qwen3.8); CPU with 16 BLAS threads 2.1–2.8 s; the one-shot engine before: 13.3 s | [6](#engine-daemon-first-job-after-a-prewarm-blas-threads) |
| Into the dense Qwen3-4B after the assemble work | assemble 0.23–0.36 → 0.13–0.15 s on the same jobs; live engine 0.77–1.10 s | [8](#8-stage-4-october-4-2026-the-recurrent-mean-in-the-bridge-a-faster-assemble-deepseek-v4-switches) |
| Hermes' request template options (`reasoning_effort`, `preserve_reasoning`, `enable_thinking: false`) | 8 of 8 switches into the production Qwen3.8 reused exactly T0 (stage 2 skipped them) | [6](#request-template-options-stage-3-october-3-2026) |
| Muse → production Qwen3.8 with the recurrent mean | 6 of 6 reused exactly T0 with the right group; switch 0.9–3.2 s plus an 8.0–8.4 s tail against 41–59 s of native prefill | [8](#8-stage-4-october-4-2026-the-recurrent-mean-in-the-bridge-a-faster-assemble-deepseek-v4-switches) |
| DFlash2 after a translated restore | 37% of drafted tokens accepted against 41% natively; the aligned dummy alone: 43.4% against 43.7% with the real draft state | [6](#6-stage-2-live-october-3-2026-engine-daemon-the-production-qwen38-build-mtp-images) |
| MTP after a translated restore | 48% against 61%; the empty head cache alone costs about 5 points (55% against 60%) | [6](#6-stage-2-live-october-3-2026-engine-daemon-the-production-qwen38-build-mtp-images) |
| Images, Muse → Qwen3.8 (vision) | every switch reused exactly T0; an image in the newest message got the native answer; with an image in an earlier message the translated prefix changed the outcome in 3 of 3 conversations (the target's own cache restored the same way did not) | [6](#6-stage-2-live-october-3-2026-engine-daemon-the-production-qwen38-build-mtp-images) |
| Onboarded models as sources (DeepSeek-V2-Lite, LFM2-2.6B, Qwen3.6-35B-A3B → Qwen3.8; 8 held-out chats) | the onboarded encoder shrank its latents to half a jointly fitted model's spread; the new default (`reverse`) removes 17% / 24% / 41% of the floor's KL instead of 3% / 4% / 11% (Qwen3-4B, fitted jointly: 30%), and 20% / 25% / 40% against 5% / 6% / 12% on 11 chats (lower than ridge on 10 of 11 each); GLM-4.7-Flash, onboarded with it, 38%; translations into them unchanged. A seven-model joint refit on the models' common spans did worse out of all four sources (individually significant for LFM2 and Qwen3.6, exact sign-flip p 0.023). Selection-set results: the method was chosen on the same chats | [9](#9-onboarded-models-as-translation-sources-october-6-2026) |
| hub6: fifteen models, eleven onboarded (Gemma 4 iSWA, MLA, MLA + linear-attention and Mamba hybrids among them; 8 held-out chats each way with Qwen3.8, selection-set results) | into the new models, on the six chats every run shares, a switch removes 16-83% of the floor's KL for nine of the twelve and nothing measurable into Granite, Falcon-H1 and Jamba (6-10%; their pooled 42-51% are mostly one reply that rewrites the earlier answer); out of them into Qwen3.8 25-42% where the onboarding's encoder warnings stay quiet (Falcon-H1, Kimi-Linear, Ling, Gemma 4 26B/31B; Qwen3-4B 30%, none distinguishable from it at p < 0.05), and where they fire `reverse` did no good or harm (Granite 5%, Nemotron −115% as the servers render, Jamba −1031%): hub6-20261009 carries `ridge` encoders for Granite (1.3%, harmless) and Nemotron (still −44% as the servers render: a target only, new `kvpark bridge add --no-source`) and leaves Jamba out; a fifteen-model joint refit does not fit the machine | [10](#10-hub6-eight-more-models-onboarded-october-8-9-2026) |
| Live switches with hub6 (Qwen3.8 → Granite-4.0-H-Tiny and → Gemma-4-26B, Gemma-4-26B and Kimi-Linear → Qwen3.8, Kimi-Linear → Gemma-4-26B; 8-15K tokens; the second series) | 10 of 10 reused exactly T0; into Qwen3.8 a switch took 9.7-12.5 s of wall time against 48-52 s of native prefill, 3.3-6.8 times faster where the native prefill met few guard pauses (the guard's pauses make the exact ratio uncertain); into Gemma-4-26B 2.0-3.1 times, into Granite 1.8-3.0 times; the first 32 tokens equal native's in 5 of 6 switches into Gemma and Granite (mostly what the native tail holds) | [10](#live-switches-with-hub6-october-9) |
| Other cache families: DSA, the remaining hybrids and GQA architectures, DeepSeek V4 as a target | translated on random-weight fixtures and restored in llama.cpp; no local model, so no quality number | [8](#8-stage-4-october-4-2026-the-recurrent-mean-in-the-bridge-a-faster-assemble-deepseek-v4-switches), [ARCHITECTURE.md](ARCHITECTURE.md#what-the-translator-supports-today) |

## 1. One hub for four models (test tokens)

Fitted in closed form on the CPU (2 h) from 185,672 training spans: tokens that end at the
same byte of the same chat segment in all four renderings (89% of the pairwise-aligned
tokens). Hub of 4,096 dimensions per token, one for keys and one for values. Variance-
weighted R² of the translated K / V against each target model's own cache, on 18,415
held-out spans:

| from \ to | Qwen3.8 | Muse | Qwen3-4B | Ornith |
| --- | --- | --- | --- | --- |
| Qwen3.8 | (0.87 / 0.81) | **0.747 / 0.616** | 0.769 / 0.539 | 0.749 / 0.668 |
| Muse (global layers only) | 0.669 / 0.526 | (0.86 / 0.67) | 0.738 / 0.494 | 0.683 / 0.545 |
| Qwen3-4B | 0.689 / 0.546 | 0.746 / 0.605 | (0.90 / 0.79) | 0.707 / 0.578 |
| Ornith | 0.715 / 0.632 | 0.731 / 0.601 | 0.755 / 0.538 | (0.90 / 0.85) |

- **Against a dedicated pairwise map** (Qwen3.8 → Muse, the v3 mapper, same spans): K 0.768,
  V 0.665. The 4-model hub reaches 97% of the pairwise K R² and 93% of V while serving all
  12 directions with 8 adapters. A 2-model hub at 2,048 dimensions reached 96% / 89%; Muse's
  value spectrum needed the wider hub.
- **The models share most of their cache geometry.** The top canonical eigenvalues of the
  4-view CCA are 3.99 (K) and 3.98 (V) out of a maximum of 4: the leading directions are
  near-perfectly shared by all four models. Even the 4,096th direction carries 1.35.
- **Sliding-window sources work from their global layers.** Old tokens of Muse keep only its
  13 global layers; the hub's dedicated encoder from those layers still reaches K 0.67–0.74.
- **Same family is easiest.** Qwen3.8 ↔ Ornith (same tokenizer and layer types) has the best
  values (0.67 / 0.63); values into Qwen3-4B are the hardest (0.49–0.55).

### Adding a model without refitting the others

Qwen3-4B was added to the existing Qwen3.8 + Muse hub (2,048 dims) with `fit_hub.py --base`,
using Qwen3.8 and Muse as pivots and only Qwen3-4B's own captures; the other models' adapters
stayed bit-identical. Against a 3-model hub fitted jointly at the same size:

| Pair | K onboarded | K joint | V onboarded | V joint |
| --- | --- | --- | --- | --- |
| Qwen3.8 → Qwen3-4B | 0.754 | 0.756 | 0.514 | 0.516 |
| Muse → Qwen3-4B | 0.727 | 0.729 | 0.468 | 0.471 |
| Qwen3-4B → Qwen3.8 | 0.655 | 0.675 | 0.482 | 0.511 |
| Qwen3-4B → Muse | 0.715 | 0.735 | 0.533 | 0.555 |
| Qwen3.8 → Muse (existing) | 0.740 | 0.737 | 0.591 | 0.587 |

Decoding *into* the new model is as good as a joint fit; encoding *from* it loses 0.02–0.03.
A new model therefore costs one capture run on the hub corpus and a few ridge fits (15 minutes
on the CPU for a 36-layer model), not a refit of every adapter.

## 2. End to end: loss on the target model

Seed-score on the GPU: the target restores the translated cache for [0, T0), natively
prefills a tail of ≈2,048 tokens, and is scored (teacher-forced) on the first 256 tokens of the
reply against its own full prefill. Restoring the target's own native state gives exactly 0
in every run (sanity).

**Qwen3.8 → Muse, 22 held-out conversations (2-model hub, 2,048 dims; table fill):**

| Content | Hub | Dedicated pairwise map |
| --- | --- | --- |
| Chat, 4–8K tokens (10) | **+0.038** | +0.021 |
| Real multi-turn chat, 8–16K (3) | **+0.034** | +0.029 |
| Concatenated sessions, 8–16K (5) | +0.060 | +0.056 |
| Agent traces with tool output, 8–16K (4) | +0.295 | +0.282 |
| All, token-weighted | +0.059 [0.036, 0.089] | +0.048 [0.026, 0.075] |

The hub costs +0.012 [0.004, 0.023] nats/token over a map trained for exactly this pair; chat
stays within the 0.05 target. Agent traces fail with either (see experiments/linear_transfer
RESULTS_v3b.md: two-thirds of that loss is in table-filled template cells).

**All four models, 4-model hub (4,096 dims), 8 held-out long chats per pair (4.5K–15K tokens;
SmolTalk2 real and concatenated multi-turn), loss increase vs the target's own prefill
(token-weighted, document-bootstrap 95% CI) and mean KL.** Conditions: `hub` = translated
cache, uncovered cells filled with the decoder mean; `floor` = no information (every cell the
mean); `_prefix` = the same with the target's own values for its prompt start (system message,
14–30 tokens), which a live switch gets from a short native prefill; `hub_table_prefix` = prefix
plus per-token-id table fill.

| Source → target | Docs | hub | floor | hub_prefix | floor_prefix | hub_table_prefix |
| --- | ---: | --- | --- | --- | --- | --- |
| Qwen3.8 → Ornith | 8 | +0.016 [-0.011, +0.044] · KL 0.05 | +0.737 [+0.120, +1.559] · KL 0.80 | +0.027 [+0.004, +0.052] · KL 0.06 | +0.054 [+0.026, +0.095] · KL 0.10 | — |
| Qwen3.8 → Muse | 8 | +0.029 [-0.011, +0.061] · KL 0.07 | +0.066 [+0.029, +0.106] · KL 0.13 | +0.083 [+0.038, +0.123] · KL 0.06 | +0.156 [+0.099, +0.221] · KL 0.14 | — |
| Qwen3.8 → Qwen3-4B | 8 | +0.201 [+0.045, +0.362] · KL 0.77 | +0.188 [+0.053, +0.341] · KL 0.68 | +0.008 [-0.036, +0.059] · KL 0.06 | +0.033 [-0.016, +0.085] · KL 0.10 | +0.011 [-0.033, +0.066] · KL 0.06 |
| Qwen3-4B → Muse | 8 | +0.066 [+0.011, +0.115] · KL 0.09 | +0.066 [+0.029, +0.106] · KL 0.13 | +0.126 [+0.068, +0.178] · KL 0.09 | +0.156 [+0.099, +0.221] · KL 0.14 | +0.098 [+0.079, +0.119] · KL 0.09 |
| Qwen3-4B → Qwen3.8 | 8 | +0.068 [+0.036, +0.112] · KL 0.08 | +0.080 [+0.042, +0.128] · KL 0.10 | +0.067 [+0.040, +0.104] · KL 0.07 | +0.077 [+0.042, +0.119] · KL 0.09 | +0.070 [+0.044, +0.104] · KL 0.07 |
| Qwen3-4B → Qwen3-4B (control: its own cache through the hub) | 8 | +0.193 [+0.027, +0.370] · KL 0.77 | +0.188 [+0.053, +0.341] · KL 0.68 | +0.014 [-0.020, +0.054] · KL 0.05 | +0.033 [-0.016, +0.085] · KL 0.10 | +0.014 [-0.021, +0.052] · KL 0.05 |

What the pairs show:

- **The attention sink decides dense and hybrid Qwen targets.** Qwen3-4B (dense) and the
  Qwen3.5-family hybrids collapse when the first tokens of the prompt hold mean values: even
  Qwen3-4B's *own* cache passed through the hub scores +0.193 (KL 0.77) that way, and +0.014
  (KL 0.05, top-1 93%) once its first 14 tokens are native. With the native prompt start,
  Qwen3.8 → Qwen3-4B meets the target (+0.008).
- **Muse is the opposite:** it does better with the hub's values for its prompt start (+0.029)
  than with its own (+0.083). The bridge therefore enables the native prefix per target model.
- **Same family is easiest:** Qwen3.8 → Ornith +0.016 even with Ornith's recurrent state reset.
- **The source matters more than R² suggests.** Qwen3.8 → Muse recovers 56% of the gap
  (+0.029 vs floor +0.066) while Qwen3-4B → Muse recovers nothing (+0.066 vs +0.066) on the same
  eight conversations, although both have K/V R² ≈ 0.75 / 0.61 into Muse; Qwen3-4B → Qwen3.8
  recovers 15%. A small model's cache carries less of what a larger model needs from the old
  context. Translating *into* the small model works (+0.008).
- **Token tables** add nothing measurable on chat (+0.008 vs +0.011 into Qwen3-4B) but soften the
  native-prefix penalty on Muse (+0.126 → +0.098); their value should show on template-heavy
  agent conversations, which these pairs did not test.

Recommended per-target setting (`kvpark bridge add`): native prefix **on** for the Qwen family
(dense Qwen3, hybrid Qwen3.5/3.8/Ornith), **off** for Muse, and for the hybrids the recurrent
mean of section 7 (`--recurrent-mean`); prefer a larger model as the source. Pairs that recover less than the floor gap are no better than a native prefill of the
tail alone, so a switch from Qwen3-4B into a larger model is fast but adds no memory of the
old context beyond what the native tail gives.


## 3. Hybrid targets: resetting the recurrent state

A hybrid target's recurrent state (gated DeltaNet) has no per-token form. Strategy R0 zeroes it
and lets the native tail rebuild it. Measured alone (Qwen3.8 with its *native* attention cache,
recurrent state zeroed at T0), six conversations:

| Native tail | Chat (4 docs) | Agent trace (16.8K) |
| --- | --- | --- |
| ≈2.1–2.4K tokens | +0.028 to +0.064 | +0.112 |
| ≈3.9–4.6K tokens | +0.016 to +0.033 | +0.123 |

R0 is a usable baseline for chat with a 4K tail. Section 7 starts the cell from the model's mean
of native cells instead, which removes most of this cost on chats; replaying the delta rule from
decoded per-token inputs is future work.

## 4. Live: switching freely between three running models

Throwaway upstream llama-servers (85d5703a3, Vulkan, F16 cache, flash attention, one slot,
`--slot-save-path`) for Qwen3.8-27B, Muse-Glimmer-30B and Qwen3-4B, all resident at once
(MemAvailable 37 GB left). Two held-out long chats, each primed on Qwen3.8 and then moved with
`kvpark bridge switch` along Qwen3.8 → Muse → Qwen3-4B → Qwen3.8 (4-model hub; native prefix
on the Qwen targets). Before each hop the target also prefilled the same prompt natively, for
the reference time.

| Hop | Tokens reused by the target (= T0) | Time to first token: bridge vs native | Engine |
| --- | --- | --- | --- |
| Qwen3.8 → Muse | 12,962 | 9.7 s vs 63.7 s | 41.6 s (first use: model profiles built) |
| Muse → Qwen3-4B | 6,136 | 2.8 s vs 7.8 s | 24.4 s (first use) |
| Qwen3-4B → Qwen3.8 | 13,307 | 11.9 s vs 72.2 s | 15.8 s |
| Qwen3.8 → Muse | 6,258 | 9.4 s vs 34.8 s | 9.2 s |
| Muse → Qwen3-4B | 1,167 | 1.5 s vs 2.1 s | 6.5 s |
| Qwen3-4B → Qwen3.8 | 6,464 | 10.8 s vs 39.0 s | 8.8 s |

- **Every hop restored exactly T0 tokens and the target reused all of them**, across three
  architectures and three tokenizers, in both directions and in a round trip.
- **Cost of a hop with warm caches: engine 6–16 s (the one-shot CPU engine of the time; the
  engine daemon of section 6 takes 0.5–3.7 s) + the native tail (~10 s on the big models)**,
  i.e. 15–28 s against 35–72 s of native prefill for 6–13K-token conversations on the 27–30B
  models; the gap grows with length (native ≈ P / 150 tok/s). Into the small Qwen3-4B a native
  prefill is already quick (2–8 s), so a switch into it is not worth translating; the bridge's
  per-target `min_handover` lets small models opt out.
- **A bug the live runs found:** the bridge stored the engine interpreter with symlinks
  resolved, so a venv's `bin/python` became the base interpreter and the engine ran on the
  system numpy with reference (single-threaded) BLAS: 350 s instead of 6–10 s per translation.
  Fixed (`registry.set_engine` keeps the venv path; regression test) together with a
  per-model profile cache (GGUF metadata parsed once) and batched RoPE.
- Earlier runs with two models (Qwen3.8 ↔ Muse, three chats) gave the same picture: 5 of 6 hops
  restored and reused T0 (the sixth found a recurrent-state format detail, since fixed).

## 5. Reproduce

All under evidence/universal-kv/ (not shipped): captures (Qwen3-4B, Ornith; Qwen3.8 and Muse
reuse linear-transfer run-v2/run-v3), hub2-d2048, hub4-d4096 (+ .log.jsonl), e2e-hub2-qwen38-muse,
recurrent-reset-qwen38, live-bridge-qwen38-muse, live-bridge-3models-v2, pair-*, pair2-*, pair-self-*, token-tables. Commands are in each script's `--help`:
fit_hub.py, score_translate.py, pair_eval.py, recurrent_reset.py, live_bridge.py. Sections 6-8
list their own evidence directories.

## 6. Stage 2 live (October 3, 2026): engine daemon, the production Qwen3.8 build, MTP, images

The bridge now runs its engine as a resident daemon (`engine_daemon.py`, started and fed by
`kvpark.bridge.engine_client`; the switch sends the job over a socket) on the iGPU (torch fp16,
`~/venvs/rocm`) or the CPU (numpy, OpenBLAS). Same machine as section 4; every server was a
throwaway instance on ports 18530-18536 with slot directories in scratch space, started and
stopped by `live_bridge.py`. The load average and MemAvailable are recorded next to every timing
in the evidence; other agents compiled and tested on the same machine during the runs (load
up to 26), and the table gives the 1-minute load at each switch.

- **Qwen3.8-27B (Q8)**: the production binary itself (kvpark-patched ROCm build with the
  `.resume` sidecar, the DFlash2 Q8 M-RoPE draft at depth 7, the F16 mmproj, the contiguous
  draft-position patch, reasoning effort xhigh, 8 context checkpoints) with the production
  flags. Context 65,536 for the three-server hop table (memory: with all three servers and two
  engine daemons MemAvailable was 20-27 GB and dipped to 4.9 GB while other agents ran builds;
  262,144 would have taken 12 GB more), the production 262,144 for the draft, MTP and image runs.
- **Muse-Glimmer-30B (Q8)** and **Qwen3-4B-Instruct-2507 (Q8)**: the upstream Vulkan build
  85d5703a3 with the flags of section 4 (context 36,864).
- **Engine**: hub4-d4096 converted to format 2. GPU: the daemon under the ROCm venv (torch
  2.12, ROCm 7.14, `--device gpu`). CPU: the daemon under the experiment venv (numpy 2.5.3,
  OpenBLAS, `--device cpu`, 16 BLAS threads, the new default). Before every switch the pair was
  prewarmed as staging a switch in Hermes does. Native prompt start on for the Qwen targets.

### Hop table: every ordered pair, GPU and CPU engine

Two held-out chats that every template renders long: `multi_turn_reasoning_if_think:23743`
(13,627 / 12,916 / 12,662 tokens on Qwen3.8 / Muse / Qwen3-4B) and `:18676` (12,076 / 11,862 /
11,201). Each was primed on Qwen3.8 and moved along Qwen3.8 → Muse → Qwen3-4B → Qwen3.8 →
Qwen3-4B → Muse → Qwen3.8, so every ordered pair ran once per conversation and engine. At each
hop the target first prefilled the prompt natively (reference), then the bridge switched with
the GPU engine and again with the CPU engine; a warm request prefilled the native tail
(2,048 tokens). Engine s: the switch's engine step (client to daemon and back, translated slot
written); switch total: everything from the source's save to the end of the tail prefill.

| Conv. | Hop | T0 / P | Engine | Engine s | Restore s | Tail prefill s | Switch total s | Native prefill s | Reuse (cache_n = T0) | Load |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 23743 | Qwen3.8 → Muse | 10,868 / 12,916 | GPU | 0.50 | 0.02 | 12.4 | 13.5 | 73.2 | yes | 2 |
| 23743 | Qwen3.8 → Muse | 10,868 / 12,916 | CPU | 2.57 | 0.02 | 12.8 | 15.9 | 73.2 | yes | 2 |
| 23743 | Muse → Qwen3-4B | 10,614 / 12,662 | GPU | 2.88 | 0.19 | 4.1 | 7.7 | 15.6 | yes | 3 |
| 23743 | Muse → Qwen3-4B | 10,614 / 12,662 | CPU | 9.39 | 0.19 | 4.1 | 14.3 | 15.6 | yes | 3 |
| 23743 | Qwen3-4B → Qwen3.8 | 11,579 / 13,627 | GPU | 3.65 | 0.14 | 8.5 | 14.8 | 47.2 | yes | 8 |
| 23743 | Qwen3-4B → Qwen3.8 | 11,579 / 13,627 | CPU | 11.96 | 0.10 | 8.1 | 21.7 | 47.2 | yes | 10 |
| 23743 | Qwen3.8 → Qwen3-4B | 10,614 / 12,662 | GPU | 3.02 | 0.21 | 4.1 | 7.7 | 15.6 | yes | 7 |
| 23743 | Qwen3.8 → Qwen3-4B | 10,614 / 12,662 | CPU | 11.76 | 0.18 | 4.1 | 16.4 | 15.6 | yes | 9 |
| 23743 | Qwen3-4B → Muse | 10,868 / 12,916 | GPU | 1.27 | 0.06 | 14.8 | 17.7 | 74.4 | yes | 14 |
| 23743 | Qwen3-4B → Muse | 10,868 / 12,916 | CPU | 6.13 | 0.04 | 15.0 | 22.0 | 74.4 | yes | 19 |
| 23743 | Muse → Qwen3.8 | 11,579 / 13,627 | GPU | 2.35 | 0.22 | 10.0 | 13.2 | 56.8 | yes | 21 |
| 23743 | Muse → Qwen3.8 | 11,579 / 13,627 | CPU | 6.38 | 0.17 | 11.0 | 18.3 | 56.8 | yes | 23 |
| 18676 | Qwen3.8 → Muse | 9,814 / 11,862 | GPU | 0.89 | 0.05 | 15.1 | 16.6 | 83.5 | yes | 21 |
| 18676 | Qwen3.8 → Muse | 9,814 / 11,862 | CPU | 4.96 | 0.04 | 15.4 | 21.0 | 83.5 | yes | 21 |
| 18676 | Muse → Qwen3-4B | 9,153 / 11,201 | GPU | 2.95 | 0.31 | 4.0 | 8.0 | 15.4 | yes | 20 |
| 18676 | Muse → Qwen3-4B | 9,153 / 11,201 | CPU | 13.56 | 0.25 | 4.0 | 18.5 | 15.4 | yes | 20 |
| 18676 | Qwen3-4B → Qwen3.8 | 10,028 / 12,076 | GPU | 2.59 | 0.17 | 9.2 | 12.6 | 51.4 | yes | 26 |
| 18676 | Qwen3-4B → Qwen3.8 | 10,028 / 12,076 | CPU | 12.71 | 0.18 | 9.5 | 23.0 | 51.4 | yes | 24 |
| 18676 | Qwen3.8 → Qwen3-4B | 9,153 / 11,201 | GPU | 3.30 | 0.28 | 4.1 | 8.3 | 14.8 | yes | 22 |
| 18676 | Qwen3.8 → Qwen3-4B | 9,153 / 11,201 | CPU | 7.56 | 0.25 | 4.1 | 12.4 | 14.8 | yes | 23 |
| 18676 | Qwen3-4B → Muse | 9,814 / 11,862 | GPU | 1.06 | 0.06 | 16.2 | 18.0 | 86.0 | yes | 19 |
| 18676 | Qwen3-4B → Muse | 9,814 / 11,862 | CPU | 4.86 | 0.04 | 16.0 | 21.8 | 86.0 | yes | 19 |
| 18676 | Muse → Qwen3.8 | 10,028 / 12,076 | GPU | 1.71 | 0.12 | 7.8 | 10.2 | 47.7 | yes | 15 |
| 18676 | Muse → Qwen3.8 | 10,028 / 12,076 | CPU | 4.18 | 0.10 | 7.9 | 12.8 | 47.7 | yes | 13 |

- **All 24 switches restored T0 cells and the next request reused exactly T0 tokens**, into
  and out of the production DFlash2 server, across three architectures and three tokenizers.
- **The engine is no longer the cost of a switch on the GPU.** Qwen3.8 → Muse took 0.50-0.89 s
  (the one-shot engine: 13.3 s on a quiet machine, 9.2-41.6 s in section 4), Muse → Qwen3.8
  1.7-2.4 s, Qwen3-4B → Muse 1.1-1.3 s. Switches into Qwen3-4B took 2.9-3.3 s and Qwen3-4B →
  Qwen3.8 2.6-3.7 s. The dense model keeps 36 layers × 8 KV heads (36,864 values per token and
  kind, against 16,384 for Qwen3.8's attention layers and 3,328 for Muse's global ones), so its
  states are 1.3-1.6 GB at these lengths. Most of the extra time follows the target: into
  Qwen3-4B, 1.4-2.0 s of each GPU switch went to the assemble stage (device-to-host copies into
  the output arrays) against 0.05-0.2 s for the other targets (Qwen3-4B → Qwen3.8 paid it once
  too, 2.1 s, and once 1.2 s of reading inputs at load 26 instead). The products themselves took
  0.4-1.3 s with Qwen3-4B on either side against 0.2-0.4 s between Qwen3.8 and Muse. Section 8
  takes the assemble stage of these targets to 0.13-0.16 s.
- **CPU engine**: Qwen3.8 → Muse 2.6 s at load 2 and 5.0 s at load 21, Muse → Qwen3.8
  4.2-6.4 s at load 13-23, pairs with Qwen3-4B 4.9-13.6 s (Qwen3-4B → Muse 4.9-6.1 s, the
  others 7.6-13.6 s). A switch with it still beats the native prefill everywhere except into
  Qwen3-4B, where it takes about as long as the 15-s native prefill.
- **The native tail now dominates**: 12.4-16.2 s on Muse (Vulkan) for 2,048 tokens at
  10-11K depth, 7.8-11.0 s on Qwen3.8 (ROCm), 4.0-4.1 s on Qwen3-4B. A switch with the GPU
  engine took 13.5-18.0 s into Muse against 73-86 s of native prefill, 10.2-14.8 s into
  Qwen3.8 against 47-57 s, and 7.7-8.3 s into Qwen3-4B against 15 s.
- **Under load** the CPU-side stages slow down: for Qwen3.8 targets, reading the inputs
  (renders, the hybrid template and native-prefix states) took 1.1-1.8 s in the 6 switches at
  load 13-26 and 0.14-0.19 s at load 8-10.
- **Prewarm (stage time, not part of a switch)**: 20-22 s for the first pair of a new bridge
  directory (GGUF profiles parsed, hub files hashed once), then 0.2-7.7 s; the warm-up
  products took 0.2-0.9 s on the GPU and 0.1-4.3 s on the CPU.
- **Into the production DFlash2 server** the bridge read the draft from the template's own
  `.resume` (base without layers, 5 sliding-window layers, M-RoPE cells), wrote the 41,216-byte
  aligned dummy, and the server accepted the restore. The 32 tokens generated after each of the
  8 translated restores drafted 25/40 to 25/42, 26/31 and 25/32 accepted (GPU and CPU runs
  alike) against 23/48 and 24/39 after a native prefill. The server log has no draft-position
  warning (`ctx_dft pos_max`), no position error and no failed decode.
- **Agreement of the first 32 generated tokens with the native run**: into Muse 19 (`:23743`)
  and 1 (`:18676`), into Qwen3-4B 0 and 32, into Qwen3.8 0 and 2 (its output starts with its
  reasoning). Quality is
  measured in sections 2-3, not here.
- The first conversation picked (`:18905`, 12,768 tokens on Qwen3.8) renders to 121 tokens with
  Qwen3-4B's template, which drops earlier reasoning: Muse → Qwen3-4B was correctly skipped as
  `short`, and the run was restarted with conversations that every template renders long.

### Speculative decoding after a translated restore (production Qwen3.8 target)

Muse → Qwen3.8, three held-out chats, GPU engine, no warm request (as Hermes switches). The
target generated 256 tokens at temperature 0 after a native prefill and after the switch
(the same prompt tokens). DFlash2: the production configuration; MTP: the same binary with
the rollback settings of the production configuration (`--spec-type draft-mtp`, the MTP-only
Q8 head, depth 4). The bridge recognized the MTP state (one full-attention layer, `/slots`
`draft-mtp`) and restored it with the 16-byte empty marker.

| Draft | Conv. | P / T0 | Native: accepted / drafted | Translated: accepted / drafted | Tokens/s native / translated | Native prefill s | Switch + prefill s | Reuse |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| DFlash2 | 18905 | 12,768 / 10,720 | 200/381 = 0.52 | 182/499 = 0.36 | 25.4 / 19.4 | 44.1 | 2.0 + 8.1 | yes |
| DFlash2 | 23743 | 13,627 / 11,579 | 194/420 = 0.46 | 175/547 = 0.32 | 22.8 / 17.4 | 47.2 | 1.5 + 8.1 | yes |
| DFlash2 | 595 | 15,355 / 13,307 | 172/576 = 0.30 | 193/427 = 0.45 | 16.7 / 22.3 | 54.3 | 1.8 + 8.4 | yes |
| MTP | 18905 | 12,768 / 10,720 | 189/264 = 0.72 | 166/349 = 0.48 | 20.5 / 15.4 | 43.6 | 2.0 + 7.8 | yes |
| MTP | 23743 | 13,627 / 11,579 | 184/278 = 0.66 | 162/368 = 0.44 | 19.1 / 14.7 | 48.4 | 1.5 + 8.0 | yes |
| MTP | 595 | 15,355 / 13,307 | 168/348 = 0.48 | 172/329 = 0.52 | 15.4 / 16.3 | 53.7 | 1.6 + 8.3 | yes |

- **DFlash2 keeps working after a translated restore**: 550/1473 = 0.37 accepted against
  566/1377 = 0.41 natively, overall. Per conversation the difference goes both ways (0.36 vs
  0.52, 0.32 vs 0.46, 0.45 vs 0.30), because the two runs generate different text (they agree
  on no generated token: the translated cache changes the answer). The aligned dummy worked
  on the production build's draft-local position counter: no `ctx_dft pos_max` warning and no
  position error in any of the runs above.
- **MTP loses more**: 500/1046 = 0.48 against 541/890 = 0.61 natively (0.48 vs 0.72, 0.44 vs
  0.66, 0.52 vs 0.48). The MTP head starts empty after a restore and sees only the native tail
  and what follows, as expected from the empty marker. Generation ran at 14.7-16.3 tokens/s
  after the restore against 15.4-20.5 natively. The text differs between the runs as well; the
  controls below separate the two: the empty marker alone costs 0.05 of the 0.13.
- Either way the switch saves most of the prefill: 1.5-2.0 s for the switch plus 7.8-8.4 s for
  the 2,048-token tail against 44-54 s of native prefill.

### Images: Muse (no vision) → Qwen3.8 (vision)

The 7 messages of `:18905` with an image, a generated bar chart (five bars,
blue, orange, green, red, purple; the red one tallest; `media/chart.png`). The source Muse has
no vision, so it was primed on the conversation without the image; the real chat request
then went to Qwen3.8 with the image, after a native prefill and after the switch. Question:
"How many bars does this chart have, and which color is the tallest bar?"

| Case | T0 (capped at) | Reused (cache_n) | Prefilled by the target | Native prefill | Switch | Answer after the switch |
| --- | --- | --- | --- | --- | --- | --- |
| image in the newest message | 11,403 (P − tail) | 11,403 | 2,201 tokens in 9.4 s | 13,604 tokens in 47.2 s | 2.6 s | 'The chart contains **5 bars** (blue, orange, green, red, and purple, from left to right). The tallest bar is **red**.' |
| image in message 3 of 8 | 7,311 (start of message 3) | 7,311 | 6,300 tokens in 23.9 s | 13,611 tokens in 47.2 s | 1.3 s | 'the chart has 5 bars, and the tallest bar is red.' |

- **Both cases reused exactly T0**; the image and everything after it were prefilled natively.
  With the image in the newest message the request took 22.2 s end to end (prefill and 283
  generated tokens) after a 2.6-s switch, against 62.3 s natively (326 tokens), and the answer
  was identical to the native one ("5 bars (blue, orange, green, red, and purple, from left to
  right). The tallest bar is red."). DFlash2 accepted 212/497 after the switch against 242/588
  natively.
- **Image in an earlier message**: T0 stopped at the start of that message (7,311 of 13,487
  tokens), and the target prefilled 6,300 tokens with the image in 23.9 s against 47.2 s. The
  final answer was right ("the chart has 5 bars, and the tallest bar is red"), but the reasoning
  before it began by stating that no chart was attached, and ran 751 tokens against 522
  natively. Positions and cache bookkeeping were exact (the image cells sit at native M-RoPE
  positions after T0; no position or mtmd error in the log). The controls below restore the
  target's own cache of the same [0, T0) the same way, and that run reasons and answers like
  the native one: the doubt comes from the translated values in front of the image, the
  caveat of plan section 1, not from the restore.
- **Qwen3.8 → Muse with the image is skipped** (`target-no-vision`) before either server is
  touched.
- **A first attempt reused nothing** (cache_n 0 in both cases; evidence in
  `media-thinking-off-requests`): the test's chat requests carried
  `chat_template_kwargs: {enable_thinking: false}`. Qwen3.8's template writes the reasoning
  effort into the system prompt and drops it without thinking, so the real request differed
  from the bridge's rendering right after `<|im_start|>system`, and the hybrid model, which
  cannot roll back, reprocessed everything. The bridge now also renders with the request's
  `chat_template_kwargs` and `reasoning_effort` and stops T0 where that rendering departs from
  the default one; for Qwen3.8 that is the system prompt, so such a switch is skipped
  (`template-options`, verified on the same server) instead of translating cells nothing will
  reuse (a template whose options change only the generation prompt, like classic Qwen3's
  `enable_thinking`, loses nothing). Hermes passes these fields from the real request. Stage 3
  made the engine render with them, and such switches now translate: see
  [Request template options](#request-template-options-stage-3-october-3-2026) below.

### Controls: the target's own cache restored like a translated one

The comparisons above set a translated restore against a native prefill, and the two runs
generate different text, so they mix three things: the draft state the bridge synthesizes,
the translated cells, and the text those lead to. `live_bridge.py drafts-control` and
`media-control` add the missing control with the same server setup (the production Qwen3.8
binary at context 262,144 with the flags above; Muse on Vulkan as the source; GPU engine;
load 2.4-4.7, MemAvailable 21.8-27.8 GiB): after the native run and the translated switch,
the target prefills [0, T0) of the same prompt itself, saves its slot, erases it and restores
it as a switch restores a translated file, either with the draft state the bridge synthesizes
(the 41,216-byte DFlash aligned dummy, the 16-byte empty MTP marker) or with the server's own
`.resume`. Its target cells are then native, so only the draft state differs between the
two. The native and translated runs reproduced the counts above exactly (temperature 0).

| Draft | Conv. | T0 / P | Native prefill | Translated switch | Own cache, synthesized draft state | Own cache, its real draft state | Own runs agree with native for |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| DFlash2 | 18905 | 10,720 / 12,768 | 200/381 = 0.52 | 182/499 = 0.36 | 195/412 = 0.47 | 196/405 = 0.48 | 30 tokens |
| DFlash2 | 23743 | 11,579 / 13,627 | 194/420 = 0.46 | 175/547 = 0.32 | 195/418 = 0.47 | 195/418 = 0.47 | 231 tokens |
| DFlash2 | 595 | 13,307 / 15,355 | 172/576 = 0.30 | 193/427 = 0.45 | 184/493 = 0.37 | 184/493 = 0.37 | 20 tokens |
| DFlash2 | all | | 566/1377 = 0.411 | 550/1473 = 0.373 | 574/1323 = 0.434 | 575/1316 = 0.437 | |
| MTP | 18905 | 10,720 / 12,768 | 189/264 = 0.72 | 166/349 = 0.48 | 177/308 = 0.57 | 181/292 = 0.62 | 30 tokens |
| MTP | 23743 | 11,579 / 13,627 | 184/278 = 0.66 | 162/368 = 0.44 | 180/298 = 0.60 | 185/278 = 0.67 | 231 tokens |
| MTP | 595 | 13,307 / 15,355 | 168/348 = 0.48 | 172/329 = 0.52 | 168/344 = 0.49 | 173/327 = 0.53 | 20 tokens |
| MTP | all | | 541/890 = 0.608 | 500/1046 = 0.478 | 525/950 = 0.553 | 539/897 = 0.601 | |

- **The DFlash2 aligned dummy costs nothing measurable**: 0.434 against 0.437 with the
  server's real draft state, the same counts in two of the three conversations (the window
  of 2,047 native tail tokens covers everything the draft attends to at its first step).
- **The empty MTP marker costs about 0.05**: 0.553 against 0.601, and here both own-cache runs
  generated the same 256 tokens in every conversation, so this is the MTP head missing
  [0, T0) and nothing else. Generation ran at 15.6-18.1 tokens/s with the marker against
  16.3-19.2 with the real state.
- **The rest is the translated cells and the text they lead to.** Even the target's own cache,
  prefilled in two steps ([0, T0), then the tail) instead of one, generates text that departs
  from the single native prefill after 20-231 tokens, and acceptance over 256 tokens moved
  with it by -0.05, 0 and +0.07 (DFlash2). The translated runs differ from the own-cache runs
  by -0.11, -0.15 and +0.08 (DFlash2) and -0.10, -0.16 and +0.03 (MTP against the
  synthesized marker): with three conversations, a cost of the translated cells for drafting
  is likely but not separated from text variance. No draft-position warning, position error
  or failed decode in any run.

The same for the image in an earlier message, on three conversations (case (ii) of the
previous section, 1,200 answer tokens with reasoning): the image's message and everything
after it are prefilled natively in all three runs, and only the cells in front of the image
differ.

| Conv. | Image in message | T0 / P | Native prefill | Translated switch | Own cache, restored the same way |
| --- | --- | --- | --- | --- | --- |
| 18905 | 3 of 8 | 7,311 / 13,487 | 5 bars, red tallest; 522 tokens | right answer, but the reasoning first says no chart is attached; 751 tokens | the native reasoning and answer, word for word; 426 tokens |
| 23743 | 3 of 8 | 6,242 / 14,228 | 5 bars, red (the fourth) tallest; 425 tokens | the reasoning, in a terse style, finds 5 bars and red, then runs to the 1,200-token limit without an answer | the native reasoning opening, the same answer in other words; 433 tokens |
| 595 | 15 of 26 | 9,785 / 16,052 | 5 bars, red tallest, written in capitals; 284 tokens | right answer, in normal case; 394 tokens | identical to native (the same 284 tokens and draft counts) |

- **Every run reused exactly T0** (cache_n = T0) and the server logged no position, decode or
  mtmd error. The switch took 1.4-1.8 s, and the request after it prefilled 6,300-8,110
  tokens with the image in 24-31 s against 48-59 s natively.
- **The restore path is not what changes the answers.** The target's own cache restored
  through the same path (with the bridge's aligned dummy) answered like the native prefill in
  all three conversations, so the image cells after a restored prefix sit where they should.
  The translated prefix did change the outcomes: a doubted image, a reasoning that never
  reached its answer within 1,200 tokens, and a formatting request from earlier in the
  conversation that was no longer followed. This is the translation quality of sections 2-3
  (plan section 1), seen here in a vision request.
- **Request template options** (the stage 2 rule, on the same server; stage 3 translates such
  switches, below): a switch whose request
  sets `chat_template_kwargs: {enable_thinking: false}` rendered differently from token 3 on
  (inside Qwen3.8's system prompt) and was skipped (`template-options`); one with
  `reasoning_effort: high`, which Qwen3.8's template maps to its default, rendered the same
  and switched as usual (T0 10,720, reused exactly).

### Engine daemon: first job after a prewarm, BLAS threads

`bench_engine.py prewarm` (a fresh daemon per run, started by `engine_client` as a switch
starts it, prewarmed for the pair, then a first and a second job; run-v3b document 3b211f08,
T0 12,962 / 13,307, load 2.4-4.2) and `bench_engine.py warm --device cpu` (load 3.7-8.8,
partly from the runs themselves):

| | Qwen3.8 → Muse | Muse → Qwen3.8 |
| --- | --- | --- |
| GPU, first job after a plain load | 0.69-0.71 s | 0.95-0.96 s |
| GPU, first job after load + warm-up products (0.35 s) | 0.51-0.52 s | 0.76 s |
| GPU, second job | 0.45-0.47 s | 0.72-0.78 s |
| CPU, 8 BLAS threads, warm | 2.75-2.85 s | 3.94-3.96 s |
| CPU, 16 BLAS threads, warm | 2.10-2.16 s | 2.75-2.78 s |

The warm-up (the switch's products once on zero rows, the RoPE checks, both vocabularies and
the chat template) removes most of the first-job overhead; what remains (about 0.05 s) is
reading a new source file, allocating the output and the job's last window, whose row count
depends on T0 and so is not one of the warmed shapes. The warm-up holds only while the sides
stay loaded: the daemon drops them after `--idle-unload` (120 s) and exits after
`--idle-exit` (600 s), while a staged switch lasts 15 minutes. The CPU default is now one BLAS thread
per physical core, at most 16: the servers sit idle while a switch translates.

Evidence: `evidence/universal-kv/live-stage2-20261003/` (`hops/`, `drafts-dflash2/`,
`drafts-mtp/`, `media/`, `media-thinking-off-requests/`, `drafts-control-dflash2/`,
`drafts-control-mtp/`, `media-control/`: server logs and JSON per switch; `engine-logs/`: the
daemons' logs; `bench/`: the prewarm and thread benchmarks). Commands: `live_bridge.py
hops|drafts|media|drafts-control|media-control --help`, `bench_engine.py prewarm|warm --help`.

### Request template options (stage 3, October 3, 2026)

The engine now renders the target with the request's `chat_template_kwargs` and
`reasoning_effort`, building the template context llama-server builds for a request
(`chat.server_context`: the kwargs, the effort, `preserve_reasoning`, the server's default
variables), and checks that the target server's tokens spell that rendering byte for byte.
`live_bridge.py options`: Muse (Vulkan 85d5703a3) primed two held-out chats and was stopped; a
stub then served its saved slots while the production Qwen3.8 binary (DFlash2 draft, F16 mmproj,
production flags, context 65,536) and Qwen3-4B ran with the GPU engine daemon. Each switch had
no warm request, as Hermes switches; the real chat request followed with the same options. The
1-minute load was 5-9 with other agents' GPU work, MemAvailable 41-47 GiB.

| Conv. | Options | P | T0 = cache_n | Switch s | Engine s | Tail prefill s | Native prefill s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23743 | none | 13,627 | 11,579 | 4.07 | 1.45 | 20.1 | 79.7 |
| 23743 | Hermes' `reasoning_effort: medium` | 13,589 | 11,541 | 2.86 | 1.41 | 19.2 | |
| 23743 | Hermes' pair (`medium` and `preserve_reasoning: true`) | 13,589 | 11,541 | 1.80 | 1.30 | 19.6 | |
| 23743 | `enable_thinking: false` | 13,591 | 11,543 | 1.43 | 1.06 | 18.7 | |
| 18676 | none | 12,076 | 10,028 | 2.36 | 1.84 | 18.2 | 102.3 |
| 18676 | Hermes' `reasoning_effort: medium` | 12,038 | 9,990 | 2.01 | 1.47 | 21.0 | |
| 18676 | Hermes' pair | 12,038 | 9,990 | 1.70 | 1.17 | 18.1 | |
| 18676 | `enable_thinking: false` | 12,040 | 9,992 | 2.12 | 1.76 | 17.0 | |

- **All 8 switches reused exactly T0.** The requests with options share 3 tokens with the
  default rendering (Qwen3.8 writes the effort into its system prompt), so stage 2 skipped them
  (`template-options`). The template's own defaults matched the server's tokens every time
  (`rendering: template`).
- The switch times include the native prompt start's prefill where it was not cached yet (1.7 s
  for the first switch, 0.9 s for the first one whose system prompt the options change). Native
  prefill took 80-102 s and the 2,048-token tail 17-21 s here, against 47-57 s and 8-11 s in the
  hop table above: the GPU was shared, so compare ratios, not seconds. Rendering with options
  cost 3-12 ms per job.
- DFlash2 got the 41,216-byte aligned dummy each time; the Qwen3.8 log has no draft-position
  warning, position error or failed decode.
- Into Qwen3-4B on the same run (default rendering, a warm request prefilling the tail): 4 of 4
  reused T0 (10,614 and 9,153); engine 1.93-2.21 s, of which assemble 0.49-0.90 s; switch total
  8.9-9.2 s with a 6.1-6.5 s tail. Section 8 has the same four hops after the assemble work.

Evidence: `evidence/universal-kv/stage3-engine-20261003/` (`options/`: `options.json`, server
and engine logs; `kept/`: the Qwen3-4B jobs; `bench-loaded/`, `bench-unloaded/`: assemble
replays). Commands: `live_bridge.py options --help`, `bench_engine.py job --help`.

## 7. Hybrid targets: what the recurrent cell starts from (October 4, 2026)

Section 3 reset a hybrid target's recurrent cell to zero at T0 - 1 (R0) and let the native tail
rebuild it. `recurrent_init.py` now builds that cell with one of several strategies, and
`r0_experiments.py` measures each one on Qwen3.8-27B and Ornith-1.5-9B the way
`recurrent_reset.py` did: the target keeps its own native attention cache at T0, only the
recurrent cell (the conv rows R and delta-rule rows S of every recurrent layer) is replaced, and
the first 256 tokens of the final message are seed-scored against the target's native full
prefill (Vulkan probe, F16 cache, flash attention). A second experiment does the same with
translated attention (Muse → Qwen3.8 through the hub).

| Strategy | The cell at T0 - 1 |
| --- | --- |
| zero | every row zero (R0; what translate.py wrote before section 8, and still its default without a mean) |
| mean | per-layer mean of the target's native cells over training conversations (all of them) |
| mean-match | the same mean over the conversation's group: `tools` when it declares tools, else `chat` |
| mean_conv | R from the mean, S zero |
| mean_s0.5 | R from the mean, S half the mean |
| prefix | the cell after the native prompt start that the bridge already prefills for Qwen targets (the bridge's rule: the system message less 2 tokens, H = 48 tokens on Qwen3.8 and 10 on Ornith; 497 / 459 for the agent traces' tool prompt) |

**Calibration** (`r0_experiments.py calibrate`, once per model). The source conversations are the
training split of corpus v2 and v3: 204 conversations, none of them an evaluation conversation.
They are joined per source into long sessions: 16 chat sessions of about 48K characters and 3
smolagents sessions that share one tool list. The 33 Hermes function-calling conversations each
have their own tools and stay alone. A native save is taken every 1,024 tokens from depth 1,024
to 16,384: on Qwen3.8 209 saves of 25 documents (227K tokens), 166 in the `chat` group and 43 in
`tools`; on Ornith, whose tokenizer makes one chat session shorter, 208 saves (165 `chat`). The
`tools` group is mostly smolagents: 36 of its 43 saves come from the 3 smolagents sessions (39K
tokens), 7 from Hermes function-calling conversations of 1-2K tokens. The smolagents sessions
share the evaluation traces' system prompt and tool list.

- **The GDN state has a large shared part.** On the calibration saves the mean explains 82% of
  the S rows' energy on Qwen3.8 and 83% on Ornith (48-85% per layer), and 63% / 65% of the R
  rows'. On the held-out documents below, the mean's S error is 0.12-0.36 of the true state's
  energy on chats and 0.18-0.43 on the agent traces (0.12-0.31 for the matched `tools` mean).
  Zero's error is 1 and prefix's 0.45-0.76.
- **One mean serves every depth.** The means of the depth bins 1K, 2-3K, 4-7K and 8-16K lie
  within 0.2-4% of the energy from the overall mean (4% only for the 1K bin's S).
- **The groups differ.** The `chat` and `tools` means are 9% of the S energy apart on Ornith.
  That is more than half of what the mean leaves unexplained.
- **Cost.** The calibration is 227K tokens of native prefill per model: 24 min for Qwen3.8 and
  13 min for Ornith, with the two runs concurrent on the 8060S. Alone on the GPU, the save passes
  of the exact-tail runs below prefilled about 210 tokens/s on Qwen3.8 and 720 on Ornith (model
  load included), so the calibration alone takes about 18 min and 5 min. Reading the saves
  adds 86 s / 45 s. Stored as float16 npz, a group takes 78 MB for Qwen3.8 and 26 MB for Ornith
  (all three groups: 235 MB / 79 MB). A switch needs one group. Per switch, loading it takes
  0.16 s / 0.07 s (cacheable in the engine daemon) and building the cell 2-120 ms. There is no
  extra prefill.

**Documents.** These are the 12 held-out conversations of run-v3b's 8-16K bucket, each rendered
with the target's own template. 8 are chats (5 concatenated sessions, 3 multi-turn reasoning)
and 4 are smolagents tool-calling traces, P = 8.5K-16.8K tokens. T0 = P - W rounded down to the
512 grid for W = 1,024 / 2,048 / 4,096. The native tails are then 1.0-1.5K / 2.1-2.6K / 4.1-4.6K
(means in the table headers), and T0 runs from 4,096 to 15,360. In these grid runs the sanity
condition (the native save at T0 itself) scores exactly 0 on every document. A first attempt saved the prefix state in
the same pass as the T0 states. That split the first ubatch at H, so the saves at T0 came from
another ubatch partition than the reference, and the sanity condition read up to 0.005 (Ornith).
The prefix is now saved by a separate probe record.

**How to read the agent rows.** On the agent traces, the loss on the reference final message is
not a fidelity measure. Resetting the state *lowers* it below the native prefill on all four
traces on Ornith (-0.06 to -0.29). On Qwen3.8 it is -0.04 to -0.08 on two traces (+0.008 at one
tail) and +0.08 to +0.14 on the other two (the 16.8K one is section 3's trace). What a switch
should preserve is the target's own next-token distribution, so the agent tables give KL to the
native prefill first.

**The agent rows are indicative.** There are 4 traces, all smolagents, and they score 46, 48, 168
and 44 tokens, so the 168-token trace carries 55% of every token-weighted agent mean. A bootstrap
over 4 documents has at most 35 distinct resamples, so the agent CIs are much rougher than the
chat ones. The per-trace KL values are in `summary.md` (column "KL doc mean": the plain mean over
traces) and below where they matter.

### Native attention + each strategy

Cells: chats ΔNLL [95% document-bootstrap CI] · mean KL; agent traces KL [95% CI] · ΔNLL. In
parentheses: on how many documents the strategy's KL is below zero's. Tails are P - W rounded
down to the 512 grid, so the columns hold the measured mean tails; the exact tails 512, 1,024 and
2,048 follow in their own table below.

**Qwen3.8-27B, 8 chats:**

| Strategy | tail ≈1.4K | tail ≈2.4K | tail ≈4.4K |
| --- | --- | --- | --- |
| zero | +0.053 [+0.033, +0.080] · 0.045 | +0.039 [+0.021, +0.064] · 0.037 | +0.029 [+0.013, +0.050] · 0.024 |
| mean | +0.010 [+0.001, +0.021] · 0.015 (8/8) | +0.007 [-0.003, +0.020] · 0.012 (8/8) | +0.006 [-0.002, +0.017] · 0.010 (8/8) |
| **mean-match** | **+0.008 [+0.002, +0.014] · 0.015 (8/8)** | **+0.002 [-0.002, +0.007] · 0.012 (8/8)** | **+0.003 [+0.001, +0.006] · 0.010 (8/8)** |
| mean_conv | +0.035 [+0.011, +0.071] · 0.028 (8/8) | +0.029 [+0.004, +0.065] · 0.023 (8/8) | +0.016 [-0.003, +0.048] · 0.018 (8/8) |
| mean_s0.5 | +0.020 [+0.008, +0.037] · 0.016 (8/8) | +0.011 [+0.001, +0.028] · 0.013 (8/8) | +0.006 [-0.005, +0.024] · 0.011 (8/8) |
| prefix | +0.049 [+0.032, +0.069] · 0.045 (3/8) | +0.037 [+0.021, +0.055] · 0.033 (4/8) | +0.017 [+0.003, +0.033] · 0.021 (5/8) |

**Qwen3.8-27B, 4 agent traces:**

| Strategy | tail ≈1.3K | tail ≈2.3K | tail ≈4.3K |
| --- | --- | --- | --- |
| zero | 0.063 [0.046, 0.072] · -0.008 | 0.064 [0.046, 0.075] · -0.026 | 0.058 [0.046, 0.081] · +0.001 |
| mean | 0.063 [0.051, 0.095] · -0.076 (1/4) | 0.061 [0.046, 0.099] · -0.075 (1/4) | 0.057 [0.045, 0.086] · -0.077 (3/4) |
| **mean-match** | **0.026 [0.018, 0.045] · -0.050 (4/4)** | **0.019 [0.014, 0.031] · -0.036 (4/4)** | **0.015 [0.013, 0.019] · -0.034 (4/4)** |
| mean_conv | 0.032 [0.025, 0.042] · +0.029 (4/4) | 0.032 [0.025, 0.041] · +0.023 (3/4) | 0.032 [0.025, 0.040] · +0.018 (4/4) |
| mean_s0.5 | 0.040 [0.033, 0.056] · -0.031 (4/4) | 0.039 [0.033, 0.057] · -0.021 (4/4) | 0.038 [0.033, 0.054] · -0.029 (4/4) |
| prefix | 0.042 [0.037, 0.050] · +0.001 (3/4) | 0.030 [0.026, 0.032] · +0.006 (4/4) | 0.020 [0.016, 0.031] · +0.022 (4/4) |

**Ornith-1.5-9B, 8 chats:**

| Strategy | tail ≈0.9K | tail ≈1.4K | tail ≈2.4K | tail ≈4.5K |
| --- | --- | --- | --- | --- |
| zero | +0.071 [+0.042, +0.106] · 0.100 | +0.065 [+0.040, +0.093] · 0.082 | +0.035 [+0.022, +0.047] · 0.046 | +0.021 [+0.009, +0.034] · 0.030 |
| mean | -0.001 [-0.011, +0.007] · 0.018 (8/8) | -0.003 [-0.011, +0.004] · 0.016 (8/8) | -0.004 [-0.009, +0.000] · 0.012 (8/8) | -0.004 [-0.009, +0.001] · 0.009 (8/8) |
| **mean-match** | **-0.004 [-0.016, +0.007] · 0.018 (8/8)** | **-0.006 [-0.019, +0.003] · 0.016 (8/8)** | **-0.005 [-0.014, +0.001] · 0.013 (8/8)** | **-0.005 [-0.012, +0.001] · 0.011 (8/8)** |
| mean_conv | +0.025 [+0.013, +0.036] · 0.027 (8/8) | +0.021 [+0.011, +0.033] · 0.024 (8/8) | +0.014 [+0.004, +0.025] · 0.018 (8/8) | +0.014 [+0.003, +0.026] · 0.014 (8/8) |
| mean_s0.5 | — | +0.006 [-0.001, +0.014] · 0.015 (8/8) | +0.002 [-0.003, +0.009] · 0.011 (8/8) | -0.001 [-0.004, +0.004] · 0.008 (8/8) |
| prefix | — | +0.048 [+0.022, +0.079] · 0.074 (7/8) | +0.025 [+0.014, +0.036] · 0.040 (8/8) | +0.010 [+0.000, +0.020] · 0.024 (8/8) |

**Ornith-1.5-9B, 4 agent traces:**

| Strategy | tail ≈0.7K | tail ≈1.2K | tail ≈2.3K | tail ≈4.3K |
| --- | --- | --- | --- | --- |
| zero | 0.154 [0.130, 0.192] · -0.045 | 0.125 [0.080, 0.144] · -0.084 | 0.108 [0.074, 0.125] · -0.123 | 0.084 [0.069, 0.091] · -0.120 |
| mean | 0.077 [0.063, 0.110] · -0.084 (4/4) | 0.072 [0.060, 0.098] · -0.063 (3/4) | 0.067 [0.057, 0.094] · -0.051 (3/4) | 0.070 [0.057, 0.085] · -0.051 (2/4) |
| **mean-match** | **0.054 [0.042, 0.070] · -0.096 (4/4)** | **0.062 [0.050, 0.081] · -0.088 (4/4)** | **0.047 [0.036, 0.068] · -0.087 (4/4)** | **0.048 [0.038, 0.063] · -0.073 (4/4)** |
| mean_conv | 0.070 [0.054, 0.116] · +0.070 (4/4) | 0.053 [0.044, 0.057] · +0.010 (4/4) | 0.050 [0.040, 0.054] · -0.018 (4/4) | 0.058 [0.040, 0.065] · -0.017 (4/4) |
| mean_s0.5 | — | 0.054 [0.045, 0.062] · -0.017 (4/4) | 0.054 [0.042, 0.072] · -0.025 (3/4) | 0.056 [0.046, 0.065] · -0.031 (4/4) |
| prefix | — | 0.101 [0.079, 0.163] · +0.160 (1/4) | 0.064 [0.045, 0.085] · +0.112 (4/4) | 0.059 [0.052, 0.075] · +0.116 (3/4) |

The ≈0.9K / ≈0.7K columns are a separate Ornith run with W = 512 (zero, mean, mean-match and
mean_conv only).

What the tables show:

- **The mean beats zero on every document, at every tail, on both models.** mean-match's KL is
  below zero's on 12 of 12 documents at every tail, on Qwen3.8 and on Ornith. On chats it removes
  most of the reset's cost. Qwen3.8 goes from +0.053 / +0.039 / +0.029 to +0.008 / +0.002 /
  +0.003, and Ornith from +0.065 / +0.035 / +0.021 to -0.006 / -0.005 / -0.005, with KL 2.4-5
  times lower.
- **With the mean, a short tail does better than zero with a long one.** Qwen3.8 chats at ≈1.4K
  with mean-match reach KL 0.015, against 0.024 for zero at ≈4.4K. Ornith chats at ≈0.9K reach KL
  0.018 (ΔNLL -0.004), against 0.030 (+0.021) for zero at ≈4.5K.
- **Agent traces need a matching group, and here the match is a best case.** The pooled mean
  does nothing for Qwen3.8's agent traces: KL 0.063 against zero's 0.063, lower on only 1, 1 and
  3 of 4 traces. The `tools` mean cuts KL to 0.026 / 0.019 / 0.015, lower on every trace at every
  tail (per-trace mean 0.030 / 0.021 / 0.016 against zero's 0.061, so the gain does not come from
  the one long trace). On Ornith the pooled mean already helps (0.072 / 0.067 / 0.070 against
  0.125 / 0.108 / 0.084), and the `tools` mean helps more (0.062 / 0.047 / 0.048). But the
  `tools` mean is mostly smolagents sessions with these traces' own system prompt and tool list.
  What it shows is a group calibrated on the deployment's own agent prompts; a `tools` group from
  other agents may do no better than the pooled mean did on Qwen3.8. Chats lose nothing from
  matching, since the `chat` mean holds 80% of the pooled samples.
- **The conv rows carry a large part of it.** mean_conv (mean R, zero S) closes 27-62% of the
  chat ΔNLL gap between zero and mean-match, and 43-88% of the KL gap. On Ornith's agent traces
  at ≈1.2K-4.3K it does about as well as mean-match (KL 0.050-0.058 against 0.047-0.062), but not
  at ≈0.7K (0.070 against 0.054). R holds only the last three projections, so it acts only on the
  first tail tokens. With a zero window, those tokens write wrong values into the long memory,
  and the memory keeps them.
- **No shrinkage.** Halving the mean's S (mean_s0.5) is worse than the matched mean on Qwen3.8
  (chat ΔNLL +0.020 / +0.011 / +0.006, agent KL 0.040 / 0.039 / 0.038) and about level with it
  on Ornith.
- **The native prompt start helps only with a long prompt.** On chats H is a short generic
  system prompt (48 tokens on Qwen3.8, 10 on Ornith). There prefix is within noise of zero on
  Qwen3.8 at ≈1.4K and ≈2.4K (lower KL on 3 and 4 of 8 documents) and 0.013 better at ≈4.4K, and
  on Ornith it gains 0.010-0.018. Its S error stays at 0.45-0.76 of the state's energy. On the
  agent traces H is the 497-token tool prompt, and on Qwen3.8 prefix does cut KL, from 0.063 /
  0.064 / 0.058 to 0.042 / 0.030 / 0.020 (3-4 of 4 traces): better than the pooled mean, though
  not the `tools` mean. On Ornith's traces it lowers KL less and raises ΔNLL to +0.11 to +0.16.
  Longer deployment prompts (an agent's instructions with its tool definitions) are untested;
  there the prompt's own cell may be worth more.

### Exact tails (T0 = P - W, as the engine sets it)

The tables above round T0 down to the 512 grid, so their tails run W to W + 511 tokens. A
second run (`--grid 1`) put T0 at exactly P - W for the tails a deployment would use: 1,024 and
2,048 on Qwen3.8, 512 and 1,024 on Ornith, with zero, the pooled mean and mean-match on the same
12 conversations. The save at T0 now ends inside one ubatch of the native prefill, and the
restored tail starts there, so that ubatch is computed in two parts. The sanity condition (the
native save at T0, at every tail) measures what that alone costs: KL 0.0018-0.0021 on chats and
0.0008-0.0024 on the agent traces, ΔNLL within ±0.003 on chats (max |ΔNLL| on one document
0.005-0.015). On Ornith's agent traces the sanity ΔNLL is +0.008 / +0.012, from scored windows of
44-48 tokens. This floor is part of every cell below, as it is of a real switch. Before scoring,
every native save at T0 was also rewritten through the strategy write path (`section`,
`replace_recurrent`, `state_writer`) with its own cell, and each gave its exact bytes back, so the
conditions differ from the sanity condition by the recurrent cell alone.

| Model, tail W | Chats: zero | Chats: mean | Chats: mean-match | Agents KL: zero | mean | mean-match |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8, 1,024 | +0.057 [+0.032, +0.092] · 0.053 | +0.013 · 0.015 (8/8) | **+0.008 [-0.000, +0.017] · 0.016 (8/8)** | 0.084 [0.051, 0.099] | 0.069 (2/4) | **0.029 [0.019, 0.051] (4/4)** |
| Qwen3.8, 2,048 | +0.041 [+0.018, +0.075] · 0.036 | +0.013 · 0.013 (8/8) | **+0.007 [-0.001, +0.015] · 0.013 (8/8)** | 0.057 [0.043, 0.071] | 0.058 (1/4) | **0.020 [0.014, 0.032] (4/4)** |
| Ornith, 512 | +0.098 [+0.067, +0.135] · 0.132 | +0.002 · 0.024 (8/8) | **-0.002 [-0.016, +0.008] · 0.023 (8/8)** | 0.148 [0.135, 0.168] | 0.086 (4/4) | **0.065 [0.053, 0.084] (4/4)** |
| Ornith, 1,024 | +0.076 [+0.046, +0.115] · 0.097 | -0.001 · 0.017 (8/8) | **-0.006 [-0.018, +0.002] · 0.017 (8/8)** | 0.126 [0.092, 0.168] | 0.072 (4/4) | **0.059 [0.049, 0.084] (4/4)** |

Chat cells: ΔNLL [95% CI] · KL; agent cells: KL. Per trace, mean-match's agent KL on Qwen3.8 is
0.058 / 0.016 / 0.020 / 0.043 at 1,024 and 0.039 / 0.012 / 0.018 / 0.018 at 2,048 (zero's 0.087 /
0.033 / 0.090 / 0.109 and 0.084 / 0.052 / 0.057 / 0.033). The exact tails confirm the grid
results at the lengths a deployment uses: with mean-match, Qwen3.8 at 1,024 (chat KL 0.016,
agents 0.029) is better than zero at ≈4.4K (0.024 / 0.058), and Ornith at 512 (0.023 / 0.065) is
better than zero at ≈4.5K (0.030 / 0.084). Halving Ornith's tail from 1,024 to 512 costs 0.006
in chat KL with the mean, against 0.035 with zero.

### Translated attention + each strategy (Muse → Qwen3.8)

The same 12 conversations, switched the way the bridge switches into Qwen targets. Muse saves at
its own P_s, and hub4-d4096-v2 (format 2, numpy engine, 1.4-4.3 s per translation) translates
into Qwen3.8 at T0 with the native prompt start (H tokens copied from Qwen3.8's own save of
them). The strategy then replaces the recurrent cell. Same metrics as above.

| Strategy | Chats, tail ≈1.4K | ≈2.4K | ≈4.4K | Agent traces, tail ≈1.3K | ≈2.3K | ≈4.3K |
| --- | --- | --- | --- | --- | --- | --- |
| zero | +0.129 [+0.084, +0.177] · 0.130 | +0.102 [+0.060, +0.145] · 0.102 | +0.061 [+0.031, +0.095] · 0.062 | 0.496 [0.155, 0.642] · +0.339 | 0.462 [0.120, 0.610] · +0.298 | 0.454 [0.110, 0.602] · +0.293 |
| mean | +0.100 [+0.065, +0.138] · 0.108 (6/8) | +0.069 [+0.032, +0.106] · 0.080 (8/8) | +0.031 [+0.013, +0.052] · 0.046 (8/8) | 0.446 [0.150, 0.573] · +0.242 (3/4) | 0.421 [0.109, 0.555] · +0.211 (3/4) | 0.399 [0.095, 0.529] · +0.183 (4/4) |
| **mean-match** | **+0.097 [+0.060, +0.139] · 0.108 (8/8)** | **+0.064 [+0.031, +0.099] · 0.080 (8/8)** | **+0.027 [+0.011, +0.045] · 0.047 (8/8)** | **0.463 [0.163, 0.592] · +0.287 (3/4)** | **0.445 [0.107, 0.591] · +0.267 (2/4)** | **0.432 [0.113, 0.569] · +0.238 (2/4)** |
| prefix | +0.128 [+0.081, +0.182] · 0.137 (3/8) | +0.101 [+0.055, +0.149] · 0.102 (3/8) | +0.045 [+0.021, +0.072] · 0.056 (7/8) | 0.479 [0.143, 0.623] · +0.365 (3/4) | 0.453 [0.123, 0.595] · +0.319 (3/4) | 0.450 [0.098, 0.602] · +0.323 (3/4) |

- **The mean takes off about as much as with native attention.** On chats it saves 0.032-0.038
  nats/token (mean-match's KL is lower on 8 of 8 documents at every tail). The rest is the
  translated cells. At ≈4.4K the mean halves the total (+0.061 → +0.027). The mean is worth
  about the step from a 2,048 to a 4,096 tail in ΔNLL: mean-match at ≈2.4K scores +0.064,
  against +0.061 for zero at ≈4.4K. In KL it is worth less (0.080 against 0.062).
- **Agent traces stay broken by the translated cells** (KL 0.40-0.50, ΔNLL +0.18 to +0.34; see
  section 2). The mean lowers KL by 0.02-0.06. Here the pooled mean does better than the `tools`
  mean: its paired KL difference to zero is -0.050 [-0.150, -0.003] / -0.042 [-0.112, -0.011] /
  -0.056 [-0.130, -0.015] (lower on 3-4 of 4 traces), while the `tools` mean's CIs include 0
  (-0.033 [-0.113, +0.011] / -0.017 [-0.086, +0.010] / -0.022 [-0.112, +0.003], lower on 2-3 of
  4). Neither makes such a switch usable.
- prefix is again level with zero except at ≈4.4K (+0.045).

**Top-1 agreement with the native prefill** (token-weighted share of scored tokens whose argmax
matches the native one; zero → mean-match):

| Tail W (grid) | Qwen3.8 chats | Qwen3.8 agents | Ornith chats | Ornith agents | Muse → Qwen3.8 chats | Muse → Qwen3.8 agents |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | — | — | 0.877 → 0.945 | 0.853 → 0.931 | — | — |
| 1,024 | 0.909 → 0.941 | 0.931 → 0.964 | 0.883 → 0.944 | 0.876 → 0.922 | 0.870 → 0.888 | 0.804 → 0.824 |
| 2,048 | 0.925 → 0.953 | 0.931 → 0.961 | 0.909 → 0.948 | 0.899 → 0.931 | 0.887 → 0.893 | 0.824 → 0.810 |
| 4,096 | 0.937 → 0.956 | 0.938 → 0.977 | 0.917 → 0.954 | 0.912 → 0.935 | 0.912 → 0.917 | 0.830 → 0.807 |

With native attention the mean takes back 2-8 points of top-1 agreement. With translated
attention it adds 0.5-2 points on chats, and on agent traces the change (-2 to +2) is noise.

### Recommendation

- **Default for hybrid targets: a per-model mean recurrent cell instead of zero.** Use the `chat`
  mean for conversations without tools. For conversations with tools, use a `tools` mean only
  when that group was calibrated on the deployment's own agent prompts; otherwise use the pooled
  mean (`all`). The pooled mean was level with zero or better everywhere: level on Qwen3.8's
  agent traces with native attention (KL difference -0.015 to +0.001), better on Ornith's and in
  every translated case. The matched `tools` mean measured here is a best case (its sessions
  share the traces' prompt and tools). The mean comes from a per-model npz built once by `r0_experiments.py calibrate` from native
  saves of training conversations (any `recurrent_init.MeanAccumulator` over native saves will
  do). With native attention the chat mean beat R0 on every chat, model and tail, grid or exact;
  with translated attention it did so on every chat. It needs no extra prefill. Per switch it
  costs under 0.3 s: 0.16 s to load the group, which the daemon can cache, and up to 0.12 s to
  build the cell. On disk two groups take 157 MB for Qwen3.8 and 53 MB for Ornith (float16).
- **Tail per hybrid target with the mean** (measured at exactly these tails, above). For Qwen3.8
  behind the production DFlash2 draft, keep 2,048, because the draft needs 2,047 native tail
  tokens (section 6): mean-match's native-attention KL is 0.013 on chats and 0.020 on agent
  traces there. Without a draft, use 1,024: KL 0.016 / 0.029, better than R0 with ≈4.4K (0.024 /
  0.058). For Ornith, use 1,024: KL 0.017 / 0.059. 512 costs a little (0.023 / 0.065) and is
  still better than R0 with ≈4.5K (0.030 / 0.084). The agent figures assume a matching `tools`
  group. With the pooled mean they are 0.069 at 1,024 and 0.058 at 2,048 on Qwen3.8 (zero: 0.084
  / 0.057, lower on 2 and 1 of 4 traces), and 0.072 at 1,024 and 0.086 at 512 on Ornith (zero:
  0.126 / 0.148). With translated attention the translation's own cost still falls with
  the tail, so the tail remains a quality lever. There the mean buys about one doubling of the
  tail.
- **Not recommended:** prefix with short system prompts (no gain on Qwen3.8 chats; untested with
  long ones), and shrinking the mean.
- **Caveats.** These are 12 conversations from one bucket (8-16K tokens). The 4 agent traces are
  all smolagents, one of them holds 55% of the scored agent tokens, and 36 of the calibration's
  43 `tools` saves come from smolagents sessions with the same tool list and system prompt.
  Hermes-agent traffic (another system prompt and other tools) is untested. On Qwen3.8 the
  pooled mean did nothing for agent traces, so a group that does not match the deployment's
  prompts may help no more than that. Calibrating on representative conversations is the safe
  default. A bridge can do that online at no prefill cost: each switch away from a hybrid model
  saves that model's native slot, and the slot's recurrent cell can be added to a
  MeanAccumulator per model and group. The mean leaves 12-43% of the S energy unexplained; that
  remainder is the R2 replay's target.

**Integration (translate.py, engine, bridge).** Stage 4 did this as listed (section 8), with two
differences: with tools the engine takes the pooled `all` group unless the model was registered
with `--recurrent-tools`, and the default tail is unchanged (a shorter one is a per-model `--tail`).

- In `translate.build_state`, the recurrent branch should call
  `recurrent_init.initial_state(part, t0 - 1, plan.recurrent, mean=plan.recurrent_mean,
  prefix=plan.native_state, group=plan.recurrent_group)` instead of
  `kvstate.recurrent_reset(part, t0 - 1)`.
- `translate()` and `prepare()` take `recurrent="zero"`, `recurrent_mean=None` and
  `recurrent_group="all"` into the Plan. Zero stays the default, so existing callers and tests do
  not change. The report records the strategy and the group.
- The prefix strategy must refuse a `native_state` that holds more than `native_prefix` tokens.
  pair_eval passes the native save at T0 as `native_state`, and its recurrent cell is the answer
  itself.
- `engine.run` takes `target.recurrent_mean` (the npz path) and `target.recurrent` (default
  `mean` when a mean is given). It picks the group with
  `recurrent_init.match_group(RecurrentMean.stored_groups(path), conversation.get("tools"))`:
  `tools` or `chat`, else `all` when the file lacks that group, else None (then zero).
- The daemon caches `RecurrentMean.load(path, groups=[group])` by path, size and mtime.
- What the file holds is the deployment's policy. Ship `chat` and `all` by default (157 MB for
  Qwen3.8, 53 MB for Ornith), so tool conversations use the pooled mean. Add a `tools` group
  only when it was calibrated on the deployment's own agent prompts.
  `RecurrentMean.load(path, groups=[...]).save(new_path)` writes such a subset.
- The bridge stores a per-model `recurrent_mean` (`kvpark bridge add --recurrent-mean`) and passes
  it in the target job for hybrid targets. With the mean, a 1,024-token tail is enough for chat
  targets without a draft window; tool conversations on the pooled mean keep the 2,048 default
  (section 8).

**Reproduce.** The evidence is in `evidence/universal-kv/recurrent-init-20261003/`:

- `calibrate-{qwen38,ornith}/`: `recurrent-mean.npz` and `report.json` with energy shares, depth
  bins and timings.
- `run-qwen38/`, `run-ornith/` and `run-ornith-tail512/`: `report.json` with per-document scores
  and state errors, `summary.json` and `summary.md`.
- `combined-muse-qwen38/`: the translated-attention run.
- `run-ornith-split-h/`: the first, superseded chunk with the split ubatch.
- `run-qwen38-aborted/`: a start that waited 9 min on the memory guard while another stream's
  Qwen3.8 server ran.
- `run-qwen38-exact/` and `run-ornith-exact/`: the exact tails (`--grid 1`).
- `summary.json` and `summary.md` of every run were regenerated after the summary gained the
  per-document KL mean, the tail range, the sanity KL and a token-weighted top-1. The report.json
  files are unchanged.

Commands are in `r0_experiments.py calibrate|run|combined|summarize --help`. For example, `run
--model GGUF --mean NPZ --select evidence/linear-transfer/run-v3b:8192-16384 --strategies
zero,mean,mean:match,mean_conv,mean_s0.5,prefix --probe evidence/linear-transfer/
kv-probe-build-v1gpu/kvpark-kv-probe`; the exact tails add `--tails 1024,2048 --grid 1
--strategies zero,mean,mean:match`. The exact runs took 43 min (Qwen3.8) and 11 min (Ornith).

## 8. Stage 4 (October 4, 2026): the recurrent mean in the bridge, a faster assemble, DeepSeek V4 switches

Three things changed in the engine and the bridge, and one live run checks the first two on the
production Qwen3.8 build.

- **Hybrid targets start from the mean (section 7).** `kvpark bridge add NAME --recurrent-mean
  NPZ` registers a hybrid model's mean of native recurrent cells (checked against its GGUF header:
  architecture, layer count, and for `qwen35`, `qwen35moe` and `qwen3next` the recurrent layers
  and row sizes). The
  target job carries it, `translate` writes the cell at T0 - 1 with `recurrent_init.initial_state`,
  and the engine picks the group as section 7 recommends: `chat` without tools, the pooled `all`
  with tools, `tools` only after `--recurrent-tools` (that group was calibrated on the deployment's
  own agent prompts). A file without the group falls back to `all`, then to zero, and so does a
  file moved or damaged since it was registered; the report says which (`report["recurrent"]`). The engine daemon keeps the loaded group per file identity until
  its idle unload. Zero stays the default without a mean, and `prefix` is refused when the native
  state holds more than the native prefix. Tails are not changed for anyone: a DFlash draft keeps
  its window rule, and the shorter tail the mean allows (section 7) is left to `--tail`. That
  shorter tail holds for chats, and for tool conversations only with a matched `tools` group:
  with the pooled mean, Qwen3.8's agent traces score KL 0.069 at 1,024 against zero's 0.057 at
  the default 2,048, so a deployment whose tool traffic gets the pooled mean (all Hermes traffic
  today) keeps the default tail.
- **Assemble for dense targets.** The torch backend's forward K RoPE runs in float32 with
  `torch.addcmul` (one fused multiply-add per element), which on the 8060S gives exactly the
  bits of the float64 emulation it replaces (four RoPE variants, every float equal) at 0.034
  instead of 0.081 s per 4,096-row Qwen3-4B window. The fresh output arrays are touched on four
  threads while the GPU encodes and decodes (`translate.Prefault`; 1.5 GB took 0.12 s to copy into
  fresh pages against 0.05 s into touched ones). `kvstate.truncate_state` slices a run of kept
  cells instead of copying it (a media source read memory-mapped stays a view).
- **DeepSeek V4 targets through the bridge** (tiny fixture only, no real model): the engine keeps
  the T0 that `dsv4_target` rounds down to whole 128-token blocks, so the restore check, the
  `.resume` and the reused cells follow it; DeepSeek V4 jobs run on numpy in the daemon too, with
  the head decoders resident and no head encoders loaded; format 2 bundles keep the head files;
  the lightning-indexer gate no longer skips DeepSeek V4 (its indexer keys are pooled LID rows the
  head decoders fill); an upstream server with a draft is skipped like a patched one;
  `kvpark bridge add --native-prefix` is refused for it and an older entry's flag is ignored
  (the engine synthesizes the cache whole). A switch
  from a qwen3 fixture at T0 = P - 40 = 332 restored 256 cells and reused 256
  (`test_dsv4.EngineBridgeTests`). The synthesis itself is exact on the fixture (stage 3,
  `test_dsv4.OracleTests`): a state built from the native run's own per-token compressor inputs
  restores within 9.3e-10 of a restored native save (logit scale 9e-3; q8_0 caches identical),
  while zeroing the raw window, any pooled rows or the CSA/LID rings moves the logits by 2.1e-7
  to 1.5e-5. Pooling in numpy costs about 21 s per 32K tokens at V4-Flash geometry, before the
  hub decode. No real DeepSeek V4 model has been run.

### Live: Muse → production Qwen3.8 with the recurrent mean

`live_bridge.py options` as in stage 3: Muse (Vulkan 85d5703a3, port 18590) primed three held-out
conversations alone and was stopped; then the production Qwen3.8 binary (DFlash2 draft, F16 mmproj,
production flags, context 65,536, port 18591, registered with
`recurrent-init-20261003/calibrate-qwen38/recurrent-mean.npz` and a native prompt start) and
Qwen3-4B (port 18592) ran next to a stub serving Muse's saved slots, with the GPU engine daemon.
Each switch had no warm request (as Hermes switches); the real chat request followed with the same
options. MemAvailable stayed at 52-55 GiB, the 1-minute load at 2.4-4.9.

| Conv. | Options | P | T0 = cache_n | Group | Switch s | Engine s | Mean load s | Tail prefill s | Native prefill s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 23743 | none | 13,627 | 11,579 | chat | 1.96 | 0.90 | 0.20 | 8.27 | 46.96 |
| 23743 | Hermes' `medium` | 13,589 | 11,541 | chat | 1.41 | 0.63 | kept | 8.15 | 46.96 |
| smolagents 7414 (tools) | none | 16,844 | 14,796 | all | 3.17 | 0.94 | 0.21 | 8.33 | 59.11 |
| smolagents 7414 (tools) | Hermes' `medium` | 16,806 | 14,758 | all | 2.91 | 0.73 | kept | 8.35 | 59.11 |
| 18676 | none | 12,076 | 10,028 | chat | 0.86 | 0.55 | kept | 8.03 | 41.48 |
| 18676 | Hermes' `medium` | 12,038 | 9,990 | chat | 0.89 | 0.55 | kept | 8.00 | 41.48 |

- **Every switch reused exactly T0** in the real chat request, with the mean of the right group:
  `chat` for the two chats, the pooled `all` for the tool-calling trace (the file's `tools` group
  is smolagents-calibrated, and it was registered without `--recurrent-tools`).
- The group loads once (0.20-0.21 s, then kept by the daemon). The switch's other costs did not
  change: the native prompt start (0.5 s when not cached; 1.8-1.9 s for the trace, whose system
  prompt holds its tool list) and the engine (0.55-0.94 s).
- DFlash2 got the 41,216-byte aligned dummy each time; the Qwen3.8 log has no draft-position
  warning, position error or failed decode.

Into Qwen3-4B on the same run (default rendering, a warm request prefilling the 2,048-token tail):

| Conv. | Hop | T0 / P | Engine s | Assemble s | Switch total s | Reuse |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 23743 | Muse → Qwen3-4B | 10,614 / 12,662 | 1.00 | 0.161 | 5.53 | yes |
| 23743 | Qwen3.8 → Qwen3-4B | 10,614 / 12,662 | 1.10 | 0.159 | 5.82 | yes |
| 18676 | Muse → Qwen3-4B | 9,153 / 11,201 | 0.77 | 0.129 | 4.65 | yes |
| 18676 | Qwen3.8 → Qwen3-4B | 9,153 / 11,201 | 0.95 | 0.141 | 5.19 | yes |

Stage 3's run of the same four hops took 1.93-2.21 s of engine, 0.49-0.90 s of assemble and
8.9-9.2 s per switch, with another agent's work on the GPU (load 5-9; here 3.8-4.5). The
tool-calling trace into Qwen3-4B was skipped as `template-mismatch`: chat.py's check that a template
renders message content unchanged fails on Qwen3-4B's tool-call rendering, so the target prefilled
natively.

**The same jobs, old and new engine, same conditions.** `bench_engine.py job` on the four kept
Qwen3-4B jobs, the stage 3 engine and this one alternating (median of the warm runs):

| Job | Assemble s, idle | Assemble s, servers loaded | Engine s, servers loaded |
| --- | --- | --- | --- |
| 18676 Muse → Qwen3-4B | 0.234 → 0.128 | 0.234 → 0.127 | 0.84 → 0.72 |
| 18676 Qwen3.8 → Qwen3-4B | 0.308 → 0.136 | 0.307 → 0.136 | 1.06 → 0.90 |
| 23743 Muse → Qwen3-4B | 0.277 → 0.143 | 0.276 → 0.143 | 0.95 → 0.82 |
| 23743 Qwen3.8 → Qwen3-4B | 0.358 → 0.150 | 0.350 → 0.154 | 1.23 → 1.04 |

The RoPE change alone took assemble to 0.128 / 0.199 / 0.143 / 0.225 s. Touching the output pages
ahead took off the rest of the Qwen3.8-sourced jobs' (0.199 → 0.135, 0.225 → 0.151 s): their
output lands in fresh pages every run, while the Muse-sourced jobs reuse freed heap. Assemble is
now 0.13-0.15 s; compute (0.30-0.51 s) and writing the 1.3-1.6 GB file (0.25-0.30 s) are the
larger parts of a dense target's engine time.

Evidence: `evidence/universal-kv/live-stage4-20261004/` (`options/`: run records and server logs;
`kept/`: the Qwen3-4B jobs; `bench-loaded/` and `bench-idle/`: the old/new replays with
`summary.txt`; `rope/`: the RoPE and first-touch micro-benchmarks; `servers.json`,
`run-options.sh`, `after.sh`).

## 9. Onboarded models as translation sources (October 6, 2026)

The first three onboarding campaigns (README "Onboarding campaign") translated well *into* the new
models (Qwen3.8 → DeepSeek-V2-Lite, → Qwen3.6-35B-A3B, → LFM2-2.6B removed 37-47% of the floor's KL on the
12-13 chats each scored, 31-51% on the six chats all three render long enough; 93% of DeepSeek-V2-Lite's floor
and 92% of LFM2's sit in one chat, a5add943, section 10)
but poorly *out of* them into Qwen3.8: 3% (DeepSeek-V2-Lite), 4% (LFM2) and 11% (Qwen3.6) of the floor's
KL, against 30% for Qwen3-4B, a weaker model that was fitted jointly with the hub. This section finds
why, compares onboarding encoders, and changes the default.

Evidence: `evidence/universal-kv/onboard-encoders/` (`onboard_encoders.py`: the diagnosis and every
method's encoder on the onboarding's own statistics, the bundles, held-out R² of assembled hubs;
`e2e.py`, `e2e_table.py`: the end-to-end runs; `diag/`, `bundles/`, `assembled/`, `e2e/`).

### Diagnosis

Measured on each pilot's own spans (train for the statistics, test for the rest), with its onboarded
bundle (the pilot's transform, decoders and ridge encoder; GPU products, guard on):

- **The ridge target is shrunk by construction.** An onboarded encoder regressed the new view onto the
  mean of the four pivots' estimates. In a MAXVAR hub the M views' estimates sum to λ·Z per dimension,
  so that mean is (λ/M)·Z: λ runs from 1.35 to 3.99 (mean 2.33) of M = 4, and the mean's variance is
  0.33 per dimension against 0.54 for a single pivot's estimate (what every decoder was fitted on) and
  1.0 for Z. A unit test checks the identity exactly: with full-view pivots the ridge encoder is the
  consensus encoder times λ/M.
- **The new view then explains only part of that target.** The ridge encodings' per-dimension variance
  is 4-6% of the pooled pivot estimates' in the weakest quarter of the hub dimensions (λ 1.35-1.67) and
  51-58% in the strongest; Qwen3-4B's own estimates reach 101-140%. Their spread (square root of the
  summed variances) is 0.51-0.54 of the pooled estimates' and 0.48-0.51 of Qwen3.8's own encoding;
  Qwen3-4B's is 1.07. The per-dimension correlation with Z is 0.19-0.22 in the weakest quarter and
  0.71-0.77 in the strongest (Qwen3-4B: 0.55-0.58 and 0.88-0.91): a jointly fitted view helped define
  Z, an onboarded one did not.
- **R² cannot see it.** The R²-optimal uniform gain on the ridge encodings decoded into Qwen3.8 is
  1.03-1.06 (K), 1.11-1.15 (V): for squared error they are nearly calibrated. But the rows they decode
  carry 0.55-0.61 of the native variance (K), against 0.70-0.72 from Qwen3-4B. Under-dispersed K rows
  flatten attention, and that costs KL that a 0.1 gap in R² does not show.

### Methods compared (hub.solve_onboard_encoder, `fit_hub.py --onboard-encoder`)

All of them change only the new model's encoder: its transform and decoders, and every other model's
adapters, stay bit-identical, so translations into the new model and between the base models do not
change (held-out R² into each pilot: identical to the pilot's).

| Encoder | What it is |
| --- | --- |
| ridge | the pilots': the ridge regression onto the mean of the pivots' estimates |
| consensus | the ridge regression onto Z itself (Z = the summed full-view estimates of the four models the GCCA was fitted on, over λ), the estimand of a jointly fitted encoder |
| tls | ridge, rescaled per dimension to the target's variance: the geometric mean of the forward and the inverted reverse slope, i.e. reduced-major-axis scaling (not orthogonal total least squares; the name stayed) |
| match | ridge, rescaled per dimension to the pooled pivot estimates' variance (what the decoders were fitted on) |
| reverse | ridge, rescaled per dimension by var(target) / cov(target, encoding): the inverted reverse regression, unbiased given the target instead of shrunk toward its mean |
| cca | the CCA of the new view with Z, every canonical pair at unit gain (the encodings carry Z's covariance, not scaled like the base views' estimates: that variant, `ccam`, came with the review, below) |
| match@z, tls@z, reverse@z | match, tls and reverse onto Z instead of the mean (reverse@z is `reverse_z`) |

### Held-out (test spans of each pilot's onboarding)

Decoded into Qwen3.8 (variance-weighted R² of K / V), spread of the K encodings against the pooled pivot
estimates (the V spreads are within 0.04 of them: consensus 0.79-0.80, cca 1.44-1.49, reverse 1.28-1.45), and
the decoded K rows' variance against the native rows':

| Encoder | DS-V2-Lite R² K / V | LFM2 R² K / V | Qwen3.6 R² K / V | K spread DS / LFM2 / Q3.6 | decoded K variance DS / LFM2 / Q3.6 |
| --- | --- | --- | --- | --- | --- |
| ridge (pilots) | 0.604 / 0.454 | 0.577 / 0.455 | 0.694 / 0.537 | 0.53 / 0.51 / 0.54 | 0.57 / 0.55 / 0.61 |
| consensus | 0.606 / 0.460 | 0.578 / 0.463 | 0.705 / 0.554 | 0.78 / 0.76 / 0.77 | 0.64 / 0.61 / 0.69 |
| tls | 0.605 / 0.459 | 0.575 / 0.462 | 0.706 / 0.553 | 0.77 / 0.78 / 0.80 | 0.66 / 0.64 / 0.70 |
| match | 0.596 / 0.452 | 0.563 / 0.453 | 0.702 / 0.552 | 0.97 / 1.01 / 1.02 | 0.71 / 0.69 / 0.76 |
| **reverse** | 0.570 / 0.425 | 0.526 / 0.422 | 0.673 / 0.526 | 1.29 / 1.41 / 1.46 | 0.81 / 0.80 / 0.87 |
| cca | 0.542 / 0.375 | 0.507 / 0.385 | 0.676 / 0.499 | 1.40 / 1.42 / 1.43 | 0.79 / 0.78 / 0.80 |
| match@z | 0.599 / 0.455 | 0.566 / 0.456 | 0.703 / 0.553 | 0.97 / 1.01 / 1.02 | 0.70 / 0.68 / 0.75 |
| tls@z | 0.576 / 0.429 | 0.540 / 0.429 | 0.682 / 0.533 | 1.35 / 1.40 / 1.41 | 0.77 / 0.75 / 0.84 |
| reverse@z | 0.418 / 0.251 | 0.325 / 0.204 | 0.439 / 0.281 | 2.68 / 3.01 / 3.24 | 1.07 / 1.11 / 1.28 |
| Qwen3-4B (jointly fitted, same spans) | 0.697 / 0.559 | 0.655 / 0.506 | 0.689 / 0.546 | 1.07 / 1.06 / 1.07 | 0.72 / 0.70 / 0.72 |

Into the pilots (unchanged by any encoder): K 0.46-0.53 / V 0.44-0.49 into DeepSeek-V2-Lite, K
0.70-0.76 / V 0.50-0.56 into LFM2, K 0.70-0.75 / V 0.58-0.65 into Qwen3.6. On the seven models' common
spans (hub4 + the three pilots assembled, 8,884 test spans) the onboarded models into each other move the
same way: LFM2 → DeepSeek-V2-Lite K 0.472 (ridge), 0.479 (consensus), 0.468 (match), 0.427 (reverse),
0.103 (reverse@z). R² is flat from ridge to match (within 0.015) and falls beyond; spread and decoded
variance rise steadily.

### End to end (pair_eval, the pilots' own commands and chats)

Each pilot's NAME → Qwen3.8 run again with the very command campaign.py ran (the eight fixed chats,
templates' rendering, Qwen3.8's token tables and recurrent mean, native prompt start), every method's
bundle as an `--extra-hub` condition in the same run: the same documents, states, native scores and
floors, so the encoders compare paired. The rerun's own conditions equal the pilot's to the last digit
(also with 8 instead of 16 probe threads, and each document scored by its own probe run), and its floors
equal the Qwen3-4B → Qwen3.8 baseline's. Share of the floor's KL removed (document-bootstrap 95% CI), and
the paired KL difference against Qwen3-4B → Qwen3.8 (positive: worse than Qwen3-4B; lower on n of 8):

| Encoder | DS-V2-Lite → Qwen3.8 | vs Qwen3-4B | LFM2 → Qwen3.8 | vs Qwen3-4B |
| --- | --- | --- | --- | --- |
| ridge (pilots) | 2.9% [1.1%, 3.6%] | +0.0182 [+0.0058, +0.0334], 1/8 | 4.0% [2.6%, 5.3%] | +0.0175 [+0.0052, +0.0332], 1/8 |
| consensus | 6.0% [3.7%, 7.3%] | +0.0161 [+0.0049, +0.0299], 1/8 | 8.2% [3.0%, 12.8%] | +0.0146 [+0.0044, +0.0277], 1/8 |
| match | 11.3% [5.9%, 16.1%] | +0.0125 [+0.0036, +0.0235], 1/8 | 14.4% [6.7%, 22.5%] | +0.0105 [+0.0017, +0.0208], 1/8 |
| **reverse** | **16.9% [9.0%, 22.7%]** | +0.0087 [+0.0020, +0.0166], 1/8 | **23.5% [13.3%, 31.1%]** | +0.0043 [+0.0001, +0.0097], 3/8 |
| reverse@z | 25.6% [15.4%, 31.6%] | +0.0029 [−0.0008, +0.0072], 2/8 | 16.3% [−0.5%, 28.9%] | +0.0091 [+0.0016, +0.0193], 2/8 |
| Qwen3-4B → Qwen3.8 (baseline) | 29.8% [17.0%, 36.2%] | | 29.8% [17.0%, 36.2%] | |

- Against the pilots' ridge, reverse lowers KL by −0.0095 [−0.0179, −0.0025] (DS-V2-Lite, lower on
  7/8) and −0.0132 [−0.0251, −0.0036] (LFM2, 6/8), and ΔNLL against the floor by −0.0147 [−0.0321,
  −0.0023] and −0.0190 [−0.0376, −0.0032] nats/token.
- KL falls steadily with the spread from ridge to reverse in both runs, on the chats where the floor
  is large (the replies that draw on the earlier context: 2ddc2137, b19deab5, f2eef633), from 2-5% to
  13-38% of their floor removed. On the two chats whose floor is already small (f3fcee86, 1749e42d), the
  encoders from ridge to reverse all stay within 0.0025 KL of the floor.
- reverse@z, a variance up to 40 times the pooled estimates' in the weak dimensions, helps
  DeepSeek-V2-Lite further (25.6%) but hurts LFM2 (16.3%, its CI reaching below 0; −36% and −45% on
  the two small-floor chats). Its held-out R² collapses (0.42 / 0.25 into Qwen3.8, 0.10 LFM2 →
  DeepSeek-V2-Lite). Paired against it, reverse is significantly worse on DeepSeek-V2-Lite and
  significantly better on LFM2:

| KL, reverse − reverse@z | DS-V2-Lite → Qwen3.8 | LFM2 → Qwen3.8 |
| --- | --- | --- |
| paired difference, lower on n of 8 | +0.0059 [+0.0016, +0.0104], 2/8 | −0.0048 [−0.0116, −0.0007], 7/8 |
| share of the floor's KL removed | 16.9% against 25.6% | 23.5% against 16.3% |

  Averaged over the two runs they are even (reverse 20.2%, reverse@z 21.0%).

**Choice: `reverse`**, now `hub.DEFAULT_ONBOARD_ENCODER`, fit_hub.py's `--onboard-encoder` default and
campaign.py's `ONBOARD_ENCODER`. It is a choice for robustness, not the best result in each run:
reverse@z removes as much on average but swings by source (9 points better on DeepSeek-V2-Lite, 7 worse
on LFM2, where it does worse than the floor on two chats by 36% and 45% of their small floors; reverse's
worst chat in either run is −14% of a floor of 0.017, where ridge and match are also below the floor),
and it gives up 0.19-0.27 of held-out K R² where reverse gives up 0.03-0.05 (K) and 0.03-0.04 (V) against
the best encoder. Against ridge, reverse removes about 6 times the floor's KL in both runs and closes most
of the gap to the jointly fitted Qwen3-4B (+0.0087 and +0.0043 KL, from +0.0182 and +0.0175). Its
per-dimension variance is 1.0-3.75 times the pooled pivot estimates' on the train spans (median
1.9-2.4; up to 4.2 on the test spans), which the cap below keeps it to. The decoders do not depend on
it, so translations into onboarded models and between the base models are unchanged. Translations
between two onboarded models lose some R² (LFM2 → DeepSeek-V2-Lite K 0.472 → 0.427); their end-to-end
effect was not measured. All of this was chosen and measured on the same eight chats, into Qwen3.8 only;
the review's follow-up below checks it on other chats.

### The pilots re-encoded, through campaign.py

`campaign.py reencode deepseek-v2-lite lfm2-2.6b qwen3.6-35b-a3b` fitted each encoder again with
`reverse` (`fit_hub.py --reencode` on the GPU: 137-175 s each), then `campaign.py run` evaluated the
directions out of each model again and wrote its report; the runs into each model read decoder sides
whose sha256 did not change and were kept. The previous bundles and runs are in each campaign's
`history/onboard-ridge-20261007T*/`. Against the pilots' own runs (same chats, identical floors):

| Direction | ridge (pilot) | reverse | KL, reverse − ridge | ΔNLL, reverse − ridge |
| --- | --- | --- | --- | --- |
| DeepSeek-V2-Lite → Qwen3.8 | 2.9% | 16.9% [9.0%, 22.7%] | −0.0095 [−0.0179, −0.0025], 7/8 | −0.0118 [−0.0211, −0.0034] |
| LFM2 → Qwen3.8 | 4.0% | 23.7% [13.0%, 32.2%] | −0.0133 [−0.0257, −0.0033], 7/8 | −0.0144 [−0.0266, −0.0035] |
| Qwen3.6 → Qwen3.8 (templates) | 11.3% | 41.0% [19.6%, 53.6%] | −0.0201 [−0.0387, −0.0046], 7/8 | −0.0143 [−0.0296, −0.0020] |
| Qwen3.6 → Qwen3.8 (as the servers render) | 17.5% | 49.8% [33.4%, 59.6%] | −0.0218 [−0.0394, −0.0069], 7/8 | −0.0164 [−0.0307, −0.0032] |

Against Qwen3-4B → Qwen3.8 (30%): DeepSeek-V2-Lite +0.0087 [+0.0020, +0.0166] KL, LFM2 +0.0042
[−0.0003, +0.0097] (not clearly different: within the numeric jitter below), Qwen3.6 −0.0076 [−0.0157,
−0.0008] (lower on 5 of 8 chats). With 8 chats the bootstrap interval excludes 0 more readily than the exact
sign-flip test (all 256 sign patterns, `review-20261009/stats.py`): its p is 0.055 for DeepSeek-V2-Lite, 0.242 for
LFM2 and 0.078 for Qwen3.6, so none of the three differs from Qwen3-4B at p < 0.05; each one's gain over its
`ridge` pilot does (p 0.023, 0.023, 0.039). The campaign's re-fitted encoders give the end-to-end comparison's
numbers to within 0.2 points (LFM2 23.7% against 23.5%: the GPU fit's float16 products).

**Numeric jitter.** The DeepSeek-V2-Lite and Qwen3.6 encoders campaign.py re-fitted are bit-identical to
the diagnosis's; LFM2's differs by 4.9e-6 (relative Frobenius norm). That alone moves its per-chat KL by
up to ±0.0021 (5% of a chat's KL), while the eight-chat pooled difference is −0.0002 [−0.0009, +0.0006]
KL (ΔNLL −0.0010 [−0.0024, +0.0004]): differences of a few 0.001 KL per chat, or a CI bound within
±0.001 of 0, are within what a refit's rounding moves. The directions into the models
are the pilots' runs: 47%, 37%, 39% (templates) and 45% (servers) of the floor's KL removed.

### GLM-4.7-Flash, onboarded with reverse, and hub5

GLM-4.7-Flash (MLA, 47 layers; captured whole on October 5) went through `campaign.py run` with the new
default: K R² 0.66-0.75 out of it and 0.60-0.65 into it. GLM-4.7-Flash → Qwen3.8 removes 38.1% [22.7%,
46.3%] of the floor's KL (lower on 8/8), better than Qwen3-4B (−0.0056 [−0.0107, −0.0014], 7/8, sign-flip p 0.031), and
48.1% [35.4%, 55.4%] as the servers render; Qwen3.8 → GLM-4.7-Flash removes 78.5% [74.9%, 82.4%] (13
chats, with the native prompt start: without it GLM's `[gMASK]` attention sink makes the translation
collapse) and 81.0% [77.6%, 84.7%] as the servers render.

`hub5-20261006` (`-v2` for the bridge) holds hub4-d4096's four models and the four onboarded ones, all
with `reverse` encoders. Between the onboarded models (8,973 test spans): K 0.43-0.71, V 0.41-0.61,
into DeepSeek-V2-Lite the weakest (K 0.43-0.49).

| Source → Qwen3.8 (8 fixed chats) | templates' rendering | as the servers render |
| --- | --- | --- |
| Qwen3-4B (jointly fitted) | 29.8% [17.0%, 36.2%] | |
| DeepSeek-V2-Lite | 16.9% [9.0%, 22.7%] | |
| LFM2-2.6B | 23.7% [13.0%, 32.2%] | |
| Qwen3.6-35B-A3B | 41.0% [19.6%, 53.6%] | 49.8% [33.4%, 59.6%] |
| GLM-4.7-Flash | 38.1% [22.7%, 46.3%] | 48.1% [35.4%, 55.4%] |

**The joint refit (all seven models) was not run on October 6.** `fit_hub.py --estimate` put its peak at
46.2 GiB of anonymous memory, so the guard's fit check needed 71.2 GiB of MemAvailable, and the
kvpark.service llama-server (the production Qwen3.8 with its 262,144-token cache, running since
2026-10-06 18:02) held 46 GiB of GPU memory all night (MemAvailable 30.3-72.9 GiB). That estimate was
too high, and its attempt on October 7 is in the review follow-up below. It would also fit every model on
the 87,426 train spans the seven share (DeepSeek-V2-Lite's template drops reasoning and tools) instead
of hub4's 185,672, and change every adapter, the base pairs' included.

On those seven models' common spans (8,884 test spans), hub4 with the three pilots assembled gives the
pairs out of the onboarded models K / V R² into Qwen3.8 of 0.604 / 0.454 (DeepSeek-V2-Lite), 0.619 / 0.503
(LFM2), 0.702 / 0.550 (Qwen3.6) with ridge and 0.571 / 0.425, 0.575 / 0.475, 0.684 / 0.543 with reverse
(`assembled/hub7-*.json`); the base pairs are the base's.

### Review follow-up (October 7)

A review of this section re-derived its numbers from the evidence and found the main result sound, the
choice's justification overstated (corrected above: reverse against reverse@z, the R² gaps, the variance
range) and a few gaps. Evidence: `onboard-encoders/` (`long/`, `e2e-c/`, `capcheck/`, `heldout/`,
`joint7/`, `review_tables.py` and its `review_tables.json`).

**A cap on reverse's scale.** reverse scales hub dimension d by var(target) / cov(target, encoding), about
1 / R²_d, which has no bound as R²_d → 0: a view that barely predicts a dimension (the layer-subset
encoder of a sliding-window model is the likely case) would be inflated a hundredfold, the regime where
reverse@z lost on LFM2's small-floor chats. Each dimension's variance is now capped at 4 times the pooled
pivot estimates' (`hub.ONBOARD_VARIANCE_CAP`; the scale at most twice match's; `fit_hub.py
--onboard-variance-cap`, 0 for none), the largest the end-to-end comparison covered. The fit logs each
encoder's smallest per-dimension R², its variance ratios and the dimensions capped, and warns when the cap
binds or a dimension's R² is below 0.05. Fitted again with it (`fit_hub.py --reencode`, as campaign.py
runs it; `capcheck/`), the four onboarded models' encoders reach at most 2.38-3.75 times the pooled
variance (DeepSeek-V2-Lite 2.38 / 2.45 K / V, LFM2 2.95 / 3.07, Qwen3.6 3.71 / 3.75, GLM-4.7-Flash 3.49 /
3.68; medians 1.9-2.3), with smallest per-dimension R² 0.09-0.14: the cap binds on no dimension, and the
DeepSeek-V2-Lite, Qwen3.6 and GLM-4.7-Flash encoders come out bit-identical to the campaigns' (LFM2's
again 4.9e-6 apart, its GPU fit's float16 accumulation order). Their bundles, hub5's included, are what
the capped default fits.

**The longer set.** Each pilot's NAME → Qwen3.8 ran again on the campaign's longer set (the `-long`
command: pair_eval's pick of every chat Qwen3.8 renders 6,144-16,384 tokens before the final message)
with its `reverse` bundle and its pilot `ridge` bundle as `--extra-hub`, paired. That set is the eight
fixed chats plus the three other chats of the corpus's two chat sources in that range (b6ca4d01, 6529e2d7
and a5add943, the reply that rewrites the earlier answer); the corpus holds no more. On the eight, every
condition reproduced the campaign's runs to the last digit, so the new information is in the three:

| Source → Qwen3.8, 11 chats | reverse | ridge | KL, reverse − ridge | ΔNLL, reverse − ridge | the three chats outside the eight: reverse, ridge, KL difference |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V2-Lite | 19.5% [14.9%, 22.6%] | 4.6% [2.9%, 6.1%] | −0.0146 [−0.0233, −0.0066], 10/11 | −0.0178 [−0.0287, −0.0081] | 22.2%, 6.4%, −0.0281 [−0.0449, −0.0193], 3/3 |
| LFM2-2.6B | 24.7% [19.1%, 29.8%] | 6.0% [4.0%, 8.4%] | −0.0183 [−0.0291, −0.0081], 10/11 | −0.0189 [−0.0293, −0.0085] | 25.7%, 8.1%, −0.0314 [−0.0462, −0.0140], 3/3 |
| Qwen3.6-35B-A3B | 39.5% [30.2%, 48.5%] | 11.9% [9.2%, 14.6%] | −0.0270 [−0.0424, −0.0121], 10/11 | −0.0217 [−0.0345, −0.0090] | 38.0%, 12.5%, −0.0455 [−0.0626, −0.0258], 3/3 |

On each of the three new chats reverse removes 2-5 times ridge's share (a5add943: 23%, 23% and 33% against
5-9%). The one chat per model where ridge does better is a small-floor chat (1749e42d, floor KL 0.017, for
DeepSeek-V2-Lite and LFM2, where both are below the floor; 08e28e45, 0.032, for Qwen3.6), differences
of 0.0007-0.0012 KL, within the jitter above. Qwen3-4B → Qwen3.8 was not run on the three new chats, so
"against Qwen3-4B" stays an eight-chat statement.

**Three more encoders on the eight fixed chats** (`e2e-c/`, `make_bundles.py`): `revK` (reverse on K, the
pilot's ridge on V: does the gain come from K, as the diagnosis's flattened attention says?), `cca` (run end
to end for the first time) and `ccam`, the CCA scaled like the base views' estimates (each hub dimension's
variance set to the pooled pivot estimates': the spec's option (b); the cca encoder times
sqrt(pooled_var / var(Z))). Each pilot's campaign command again, with them as `--extra-hub`; its own
conditions reproduced the campaign's runs to the last digit, and its floors are the pilot runs' and the
baseline's, so the table pairs them with ridge and reverse@z from the runs above:

| Encoder | DS-V2-Lite → Qwen3.8 | KL against reverse | LFM2 → Qwen3.8 | KL against reverse | held-out K / V R² into Qwen3.8, DS / LFM2 |
| --- | --- | --- | --- | --- | --- |
| reverse | 16.9% [9.0%, 22.7%] | | 23.7% [13.0%, 32.2%] | | 0.570 / 0.425, 0.526 / 0.422 |
| revK | 13.0% [7.1%, 19.0%] | +0.0027 [+0.0004, +0.0055], lower on 2/8 | 17.2% [9.2%, 25.2%] | +0.0044 [+0.0012, +0.0081], 1/8 | 0.570 / 0.454, 0.526 / 0.455 |
| cca | 12.4% [5.6%, 19.7%] | +0.0031 [+0.0004, +0.0063], 2/8 | 19.1% [11.6%, 27.5%] | +0.0031 [+0.0002, +0.0069], 3/8 | 0.542 / 0.375, 0.507 / 0.385 |
| ccam | 8.7% [3.8%, 13.5%] | +0.0056 [+0.0018, +0.0105], 2/8 | 14.5% [8.2%, 20.6%] | +0.0062 [+0.0014, +0.0118], 2/8 | 0.573 / 0.419, 0.541 / 0.426 |
| ridge (pilots) | 2.9% | +0.0095 [+0.0025, +0.0179], 1/8 | 4.0% | +0.0133 [+0.0033, +0.0257], 1/8 | 0.604 / 0.454, 0.577 / 0.455 |

- **K carries most of it, V the rest.** Rescaling K alone gives 72% (DeepSeek-V2-Lite) and 67% (LFM2) of
  reverse's gain over ridge; rescaling V as well adds a significant remainder (ΔNLL +0.0023 [+0.0001,
  +0.0050] and +0.0032 [+0.0001, +0.0068] for revK against reverse), although it costs V 0.03 of R². The
  spread, not R², decides on V too.
- **Neither CCA beats reverse**, in either run: at unit gain (Z's covariance) it removes 12% / 19%, scaled
  like the base views' estimates 9% / 15%, though ccam has the best held-out K R² of the rescaled encoders.
  The `reverse` default stands. ccam is not an option of `fit_hub.py`.

### The joint refit of the seven models (October 7-9)

**Track A's option (c).** A refit changes every adapter, the base pairs'
included, and it trains on the 87,426 train spans the seven models share (DeepSeek-V2-Lite's template drops
reasoning and tools) instead of hub4's 185,672, so whether it beats onboarding had to be measured. Getting it to
run took five tries (`joint7/attempt-*`):

- October 7, next to kvpark.service's llama-server (47 GiB of GPU memory), with the GPU-evaluation guard settings
  (85/65 °C, 4 threads): stopped 70% of the time, its pages went to swap until swap was 99% full, and the guard
  could not let it continue (that needed swap use below 80%); ended by hand after 61 minutes. Swap use now counts
  only while MemAvailable is below 24 GiB (`resource_guard` `swap_ignore_gib`, README "Running heavy jobs safely").
- October 8, with kvpark.service stopped (MemAvailable 104-112 GiB): four tries, each after 17-67 minutes of
  waiting for the GPU/memory lock behind other campaigns. The guard ended them after 7, 20, 91 and 17 minutes, each
  time on two samples at 95.0-96.5 °C in a row taken while the fit was already stopped, with other projects' jobs
  outside the locks (54-122 W). fit_hub.py cannot resume, so each try started over.
- October 8-9, the fifth, ended a job at 98 °C (the user's change of October 8) and ran once other projects' jobs
  were off the machine: both kinds in one try (it peaked at 97.0 °C, so it would have been ended under the earlier
  95 °C line). Its bundle is `joint7/hub7` (`-v2` for the bridge), and `joint7/comparison.json` holds every number
  below.

Held-out R² of the jointly fitted hub on the seven models' 8,884 common test spans; in brackets the difference
from hub4 plus the pilots' `reverse` encoders assembled on the same spans (`assembled/hub7-reverse.json`,
hub5-20261006's adapters):

| K, source → target | Qwen3.8 | Muse | Qwen3-4B | Ornith | LFM2 | DS-V2-Lite | Qwen3.6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8 | 0.867 (−0.002) | 0.756 (+0.006) | 0.772 (+0.006) | 0.768 (+0.003) | 0.755 (+0.011) | 0.501 (+0.003) | 0.767 (+0.003) |
| Muse | 0.684 (+0.006) | 0.851 (−0.004) | 0.738 (+0.004) | 0.702 (+0.007) | 0.723 (+0.007) | 0.465 (−0.001) | 0.709 (+0.005) |
| Qwen3-4B | 0.705 (+0.007) | 0.749 (+0.004) | 0.896 (−0.001) | 0.728 (+0.008) | 0.775 (+0.008) | 0.533 (+0.008) | 0.730 (+0.006) |
| Ornith | 0.732 (+0.004) | 0.741 (+0.006) | 0.759 (+0.006) | 0.895 (−0.003) | 0.747 (+0.012) | 0.491 (+0.001) | 0.748 (+0.004) |
| LFM2-2.6B | 0.639 (+0.064) | 0.687 (+0.065) | 0.745 (+0.044) | 0.666 (+0.067) | 0.915 (+0.002) | 0.484 (+0.057) | 0.668 (+0.065) |
| DS-V2-Lite | 0.619 (+0.049) | 0.669 (+0.046) | 0.719 (+0.029) | 0.642 (+0.049) | 0.717 (+0.035) | 0.736 (−0.001) | 0.652 (+0.045) |
| Qwen3.6 | 0.721 (+0.037) | 0.734 (+0.035) | 0.750 (+0.030) | 0.735 (+0.040) | 0.743 (+0.041) | 0.479 (+0.043) | 0.907 (−0.016) |

| V, source → target | Qwen3.8 | Muse | Qwen3-4B | Ornith | LFM2 | DS-V2-Lite | Qwen3.6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8 | 0.823 (+0.009) | 0.625 (+0.007) | 0.548 (+0.006) | 0.692 (+0.002) | 0.576 (+0.013) | 0.469 (+0.003) | 0.671 (+0.006) |
| Muse | 0.547 (+0.005) | 0.678 (+0.011) | 0.495 (+0.005) | 0.562 (+0.006) | 0.526 (+0.009) | 0.435 (−0.000) | 0.595 (+0.007) |
| Qwen3-4B | 0.566 (+0.007) | 0.604 (+0.005) | 0.804 (+0.018) | 0.596 (+0.007) | 0.580 (+0.011) | 0.493 (+0.006) | 0.605 (+0.006) |
| Ornith | 0.659 (+0.003) | 0.608 (+0.006) | 0.544 (+0.005) | 0.855 (+0.004) | 0.577 (+0.014) | 0.462 (+0.001) | 0.649 (+0.006) |
| LFM2-2.6B | 0.521 (+0.046) | 0.587 (+0.036) | 0.503 (+0.039) | 0.547 (+0.049) | 0.853 (−0.007) | 0.460 (+0.054) | 0.583 (+0.041) |
| DS-V2-Lite | 0.460 (+0.035) | 0.549 (+0.025) | 0.450 (+0.028) | 0.483 (+0.037) | 0.444 (+0.034) | 0.728 (−0.001) | 0.536 (+0.030) |
| Qwen3.6 | 0.574 (+0.032) | 0.598 (+0.029) | 0.492 (+0.031) | 0.580 (+0.036) | 0.534 (+0.038) | 0.453 (+0.043) | 0.864 (−0.018) |

- Out of the onboarded models the joint fit gains K R² +0.029 to +0.067 (mean +0.046) and V +0.025 to +0.054 (mean
  +0.037), into the base models and between the pilots alike: more than `reverse` gave up against `ridge`
  (0.03-0.05).
- Out of the base models, into any model, it gains −0.001 to +0.012 (K) and 0.000 to +0.014 (V), mean +0.006,
  although it sees 87,426 train spans instead of hub4's 185,672.
- The self pairs move by −0.016 to +0.002 (K) and −0.018 to +0.018 (V).

End to end, each campaign's own command ran again with the joint hub (`joint7/hub7-v2`) as `--extra-hub`, so the
two hubs are scored on the same chats against the same native outputs (`onboard-encoders/e2e-joint/`). The guard
ended two of these runs, one of them twice, and each finished on a retry (`--resume`); every rerun reproduced the
campaign runs' own conditions exactly (largest difference 0.0). Every decoder is refitted, Qwen3.8's
included, so the joint hub's floors (`floor_prefix.joint`) are its own: they differ from the onboarded hub's by up
to 0.0041 KL per chat into Qwen3.8, 0.0084 into DeepSeek-V2-Lite and 0.0125 into LFM2 (a5add943; into LFM2 the
joint floors are higher on average, +0.0017 [+0.0001, +0.0041]). The shares below are each hub's against its own
floor; the paired KL difference does not involve the floors. p is the exact sign-flip test's (two-sided; all 2^n
sign patterns), next to the document-bootstrap CI (`review-20261009/joint.py`, `joint.json`):

| Direction | chats | onboarded (`reverse`) | joint refit | KL, joint − onboarded | p | joint lower on |
| --- | --- | --- | --- | --- | --- | --- |
| DS-V2-Lite → Qwen3.8 | 8 | 16.9% [9.0, 22.7] | 6.3% [4.3, 8.7] | +0.0069 [+0.0011, +0.0137] | 0.109 | 3/8 |
| LFM2-2.6B → Qwen3.8 | 8 | 23.7% [13.0, 32.2] | 6.8% [2.9, 10.2] | +0.0112 [+0.0029, +0.0211] | 0.023 | 1/8 |
| Qwen3.6 → Qwen3.8 | 8 | 41.0% [19.6, 53.6] | 22.9% [13.4, 26.9] | +0.0120 [+0.0018, +0.0243] | 0.023 | 1/8 |
| Qwen3-4B → Qwen3.8 (hub4's own pair) | 8 | 29.8% [17.0, 36.2] | 22.8% [12.1, 29.3] | +0.0045 [+0.0011, +0.0084] | 0.055 | 1/8 |
| Qwen3.8 → DS-V2-Lite (fixed + longer set) | 13 | 47.4% [27.7, 48.5] | 55.9% [27.0, 57.5] | −0.0233 [−0.0708, +0.0005] | 0.372 | 8/13 |
| Qwen3.8 → DS-V2-Lite without a5add943 | 12 | | | −0.0001 [−0.0010, +0.0008] | 0.744 | 7/12 |
| Qwen3.8 → LFM2-2.6B | 13 | 37.3% [32.6, 55.4] | 35.2% [26.9, 52.3] | +0.0063 [+0.0006, +0.0173] | 0.002 | 1/13 |
| Qwen3.8 → Qwen3.6 | 12 | 39.2% [26.3, 48.9] | 30.0% [18.7, 36.2] | +0.0034 [+0.0007, +0.0072] | 0.006 | 2/12 |

- Out of the four sources into Qwen3.8 the joint hub is worse in all four, and the sign holds when any one chat is
  left out (leave-one-out means +0.0033 to +0.0137). Individually the sign-flip test says so for LFM2 and Qwen3.6
  (p 0.023), not for DeepSeek-V2-Lite (0.109) or Qwen3-4B (0.055): with 8 chats the bootstrap CI excludes 0 more
  readily than the exact test.
- Into LFM2 and Qwen3.6 it is worse (p 0.002 and 0.006). Into DS-V2-Lite the −0.0233 is entirely the one chat
  that rewrites the earlier answer (a5add943, 93% of the floor's KL there): without it the two hubs are equal
  (−0.0001, lower on 7 of 12, p 0.74).
- The joint K R² gains above do not show up as KL, as in section 9's diagnosis.

**What this does and does not show.** The comparison is of one configuration, which differs from onboarding in
three ways at once: it is fitted jointly; it trains on the seven models' 87,426 common spans, against 185,672 for
the base hub and its onboardings; and it refits Qwen3.8's own decoder (and every other). Its encoders have no
`reverse`-style variance matching, a likely cause for the onboarded sources, but not for Qwen3-4B → Qwen3.8, which
drops from 29.8% to 22.8% although hub4's own Qwen3-4B encoder never had variance matching either: the smaller
common span set or the refitted decoders may matter as much. Not run (each cheap next to a refit): §9's spread and
decoded-variance diagnostics for the joint encoders; the four base models refitted on the same 87,426 spans with
Qwen3-4B → Qwen3.8 run again; `reverse`-style scaling applied to the joint encoders.

Cost, measured (`joint7/comparison.json` `cost`; the Watch's package power covers all work on the package):

| | wall | running | package energy over the wall time | peak memory: anonymous / on the GPU |
| --- | --- | --- | --- | --- |
| joint seven-model refit, both kinds, on the GPU | 104 min (6,230 s) | 53 min (3,163 s) | 235 kJ (171 kJ while running) | 28.3 GiB / 16.6 GiB (torch's reserved peak: GTT, which RssAnon does not show); estimated then 35.6 GiB in all |
| one GPU onboarding: LFM2-2.6B (README "Fit placement") / GLM-4.7-Flash (campaign) | 5 / 45 min | 2.5 / 21 min | 10 / 97 kJ | — / 4.5 GiB; 19.2 / 12.1 GiB |
| `reverse` re-encode of one pilot (campaign) | 5-6 min | 2.3-2.9 min | 12.7-15.2 kJ | 8.9-10.2 GiB / not recorded |

One refit used the energy of 15-19 re-encodes, or of 2.4-24 full onboardings (from LFM2's 10 kJ to GLM-4.7-Flash's
97 kJ). It cannot resume, changes every adapter (the bridge then needs a new hub directory, and every campaign
evaluation would have to run again) and grows with the square of the summed view dimensions.

**Conclusion: keep onboarding with `reverse`.** A seven-model joint refit on the models' common spans translated
worse than onboarding out of every source into Qwen3.8 and into two of the three pilots, at the cost above. That
is one configuration: a refit with variance-matched encoders, or on more spans, is untested, so this does not rule
out joint refits in general; it rules out this one as a replacement for onboarding.

### Selection set

Everything in this section was chosen and measured on the same chats: the `reverse` encoder,
its variance cap of 4 and the native-prefix rule were chosen on the eight fixed chats (RESULTS.md section 2's),
which every comparison here uses again; the longer set adds the corpus's three other chats in range (section 9's
review follow-up). By October 9 all 18 chat-type test conversations of the long-chat corpus had been scored, the
fixed eight 48-88 times each. These are selection-set results: the method choices are frozen as they are, and a
fresh held-out set (`evidence/linear-transfer/corpus-heldout-20261009/`, from the local SmolTalk2 files, disjoint
from every corpus used so far; README "Onboarding campaign") is kept unscored for the next choice.

**Not tested.** A decoder-side calibration (a variance or temperature correction of the decoded K rows, which would
also help the jointly fitted Qwen3-4B, whose decoded K variance is 0.70-0.72 of the native) and every target other
than Qwen3.8 and the onboarded model itself (section 10 adds a few live hops between onboarded models).

## 10. hub6: eight more models onboarded (October 8-9, 2026)

Eight more local models went through `campaign.py run` with the `reverse` encoder and its 4x variance cap
(section 9), each onboarded against hub4-d4096 with its four models as pivots and evaluated with Qwen3.8 in
both directions under the campaign's settings (the eight fixed chats, the longer set where a target renders
some of them too short, and again as the servers render where the render stage found that they differ;
the Qwen3-4B → Qwen3.8 baseline, 29.8% [17.0%, 36.2%], under the same settings). Each report was checked
afterwards by rebuilding its numbers from the per-token score arrays and the held-out R² from the bundles and
captures; every headline number reproduced. Evidence: `evidence/universal-kv/campaign/NAME/report.md`.

A review of this section on October 9 found the numbers recorded faithfully and their reading too strong: the
shares into the models rest on chat sets that differ from model to model and largely on one chat; with 8 chats
the percentile bootstrap calls differences significant that an exact test does not; and hub6 shipped encoders
already measured to hurt. This version therefore gives the shares into each model on the chats all of them share
and without the dominant chat, the exact sign-flip test's p next to every bootstrap interval, and hub6 with `ridge`
encoders for the two models whose `reverse` encoders did not help (`hub6-20261009`). The evidence of the follow-up
is `evidence/universal-kv/review-20261009/` (`stats.py` and `stats.json`, `tables.py`, `joint.py`). All of it is
selection-set evidence (section 9, "Selection set").

| Model (llama.cpp arch) | Family | K columns | Source encoder in hub6: smallest / median per-dimension train R², dimensions the 4x cap binds on (K / V) | Notes |
| --- | --- | --- | --- | --- |
| Granite-4.0-H-Tiny Q8 (`granitehybrid`) | Mamba2 hybrid, 4 GQA layers, MoE | 2,048 | `ridge` since October 9 (`reverse` before: 0.044 / 0.103, 2,561 / 2,009 of 4,096; warnings) | |
| Falcon-H1-3B Q8 (`falcon-h1`) | Mamba2 + attention in every layer (32 × 2 × 128) | 8,192 | `reverse`: 0.123 / 0.252, none | 17 of 248 documents left out of the fit (its 192 user-defined LaTeX tokens: the fit keeps kv_probe's special flags, pair_eval and the engine flag control tokens only) |
| Jamba-Reasoning-3B Q8 (`jamba`) | Mamba hybrid, 2 attention layers of one KV head | 256 | `reverse`: 0.003 / 0.010, 3,926 / 3,876 (uncapped up to 110x / 131x; warnings) | held-out K R² out of it −0.35 to −0.87; a `ridge` encoder (`checks/encoder-ridge/`) is harmless but no better than the floor (−0.2% [−2.6%, 2.5%]); **left out of hub6** |
| Nemotron-3.5-Lightning-30B-A3B Q8 (`nemotron_h_moe`) | Mamba2 hybrid, 6 GQA layers (the 7th, the MTP block, skipped), MoE | 1,536 | `ridge` since October 9 (`reverse` before: 0.028 / 0.079, 2,915 / 2,952, uncapped up to 11.8x / 12.2x; warnings) | |
| Kimi-Linear-48B-A3B Q4_K_M (`kimi-linear`) | KDA + MLA hybrid (7 MLA layers of 27, K-only latents) | 4,032 | `reverse`: 0.097 / 0.201, none | |
| Ling-3.0-tiny Q8 (`bailingmoe3`) | MLA + linear-attention hybrid (6 MLA layers), MoE | 3,456 | `reverse`: 0.061 / 0.147, 1,509 / 1,538 (uncapped up to 5.5x; no R² warning) | its template keeps the earlier reasoning, so 95-99% of the translated cells come from the source, against Qwen3-4B's 10-88% on the same chats: its comparison with Qwen3-4B is not like for like |
| Gemma-4-26B-A4B QAT Q4 (`gemma4`) | iSWA: 25 sliding + 5 full-attention layers, MoE | 56,320 (source: the 5 full layers, 2,560) | `reverse` on the full-attention layers: 0.095 / 0.186, none | self R² K 0.68, V 0.52 (V equals K up to a scale there); the capture covers 41% of the training tokens (73,656 spans) |
| Gemma-4-31B QAT Q4 (`gemma4`) | iSWA: 50 sliding + 10 full-attention layers, dense | 225,280 (source: the 10 full layers) | `reverse` on the full-attention layers: 0.211 / 0.353, none | a 185 GB capture of 96 documents (48,746 train spans); its onboarding needed the corrected memory estimate (README "Fit memory"): 32.7 GiB anonymous and 23.7 GiB on the GPU, against 98.5 GiB estimated before, which no lock could grant |

The pilots and GLM-4.7-Flash (section 9) are in the tables below too. Into a model: `hub_prefix` against
`floor_prefix` on every chat its runs scored (the eight fixed chats, plus the longer set where the model renders
some of them too short; the reports' figure), the same without the chat that carries the largest part of the floor,
and on the six fixed chats every model's runs share (1749e42d, b19deab5, f2eef633, 08e28e45, a6c4726d, 3b211f08).
Each share with its document-bootstrap 95% CI, the exact sign-flip test's p of the paired KL difference and the
chats where the switch is below the floor (with six chats the smallest p possible is 0.031):

| Into the model | every chat scored (pooled) | the chat carrying most of the floor | without that chat | the six chats every run shares |
| --- | --- | --- | --- | --- |
| LFM2-2.6B | 37.3% [32.6%, 55.4%], p < 0.001, 13/13 | a5add943, 92% | 44.8% [26.9%, 59.6%], p < 0.001, 12/12 | 50.6% [45.0%, 65.2%], p 0.031, 6/6 |
| DeepSeek-V2-Lite | 47.4% [27.7%, 48.5%], p < 0.001, 13/13 | a5add943, 93% | 29.7% [26.9%, 35.5%], p < 0.001, 12/12 | 30.7% [28.1%, 34.1%], p 0.031, 6/6 |
| Qwen3.6-35B-A3B | 39.2% [26.3%, 48.9%], p 0.002, 11/12 | a5add943, 47% | 41.8% [21.2%, 50.9%], p 0.004, 10/11 | 50.4% [36.9%, 54.5%], p 0.031, 6/6 |
| Qwen3.6-35B-A3B, as the servers render | 44.6% [36.5%, 53.2%], p 0.008, 8/8 | 2ddc2137, 32% | 48.4% [36.9%, 54.9%], p 0.016, 7/7 | 49.3% [37.5%, 56.0%], p 0.031, 6/6 |
| GLM-4.7-Flash | 78.5% [74.9%, 82.4%], p < 0.001, 13/13 | a5add943, 22% | 80.4% [77.0%, 83.0%], p < 0.001, 12/12 | 82.9% [80.3%, 84.3%], p 0.031, 6/6 |
| GLM-4.7-Flash, as the servers render | 81.0% [77.6%, 84.7%], p 0.008, 8/8 | 2ddc2137, 22% | 82.5% [78.6%, 85.4%], p 0.016, 7/7 | 82.3% [78.0%, 85.5%], p 0.031, 6/6 |
| Granite-4.0-H-Tiny | 41.8% [5.4%, 43.3%], p 0.018, 9/13 | a5add943, 95% | 9.6% [3.5%, 17.3%], p 0.036, 8/12 | 6.3% [−2.0%, 8.3%], p 0.250, 3/6 |
| Falcon-H1-3B | 50.5% [−4.9%, 53.2%], p 0.184, 12/14 | a5add943, 93% | 8.1% [−13.1%, 21.0%], p 0.367, 11/13 | 9.5% [8.0%, 10.1%], p 0.094, 5/6 |
| Jamba-Reasoning-3B | 3.6% [−1.8%, 10.0%], p 0.234, 11/14 | db6f4e5f, 16% | 3.7% [−2.7%, 11.3%], p 0.299, 10/13 | 6.8% [−1.1%, 17.6%], p 0.219, 5/6 |
| Nemotron-3.5 | 22.5% [15.2%, 27.8%], p < 0.001, 13/13 | a5add943, 36% | 19.6% [14.2%, 28.2%], p < 0.001, 12/12 | 16.3% [11.6%, 30.1%], p 0.031, 6/6 |
| Nemotron-3.5, as the servers render | 33.1% [23.2%, 40.1%], p 0.008, 8/8 | 2ddc2137, 31% | 28.6% [21.4%, 36.3%], p 0.016, 7/7 | 26.9% [19.6%, 34.8%], p 0.031, 6/6 |
| Kimi-Linear-48B-A3B | 70.2% [11.9%, 72.4%], p 0.011, 11/13 | a5add943, 95% | 19.8% [8.0%, 29.0%], p 0.021, 10/12 | 20.6% [19.3%, 24.0%], p 0.031, 6/6 |
| Ling-3.0-tiny | 72.1% [66.3%, 75.0%], p 0.008, 8/8 | f2eef633, 33% | 71.6% [63.9%, 75.8%], p 0.016, 7/7 | 70.7% [63.8%, 74.5%], p 0.031, 6/6 |
| Gemma-4-26B-A4B | 43.5% [28.7%, 48.4%], p 0.001, 11/12 | a5add943, 33% | 39.3% [25.6%, 45.4%], p 0.003, 10/11 | 45.1% [30.5%, 48.1%], p 0.031, 6/6 |
| Gemma-4-26B-A4B, as the servers render | 39.6% [32.8%, 42.4%], p < 0.001, 12/12 | a5add943, 36% | 37.2% [30.4%, 40.7%], p < 0.001, 11/11 | 37.7% [28.8%, 41.5%], p 0.031, 6/6 |
| Gemma-4-31B | 27.3% [22.1%, 39.0%], p < 0.001, 11/12 | a5add943, 47% | 31.1% [21.5%, 42.3%], p 0.002, 10/11 | 34.5% [27.2%, 51.2%], p 0.031, 6/6 |
| Gemma-4-31B, as the servers render | 32.5% [28.7%, 42.2%], p < 0.001, 12/12 | a5add943, 48% | 35.4% [27.9%, 45.1%], p < 0.001, 11/11 | 35.6% [22.5%, 49.7%], p 0.031, 6/6 |

Out of a model, into Qwen3.8, on the eight fixed chats; against the floor, and the paired KL difference from Qwen3-4B →
Qwen3.8 on the same chats (positive: worse than Qwen3-4B; its floors are identical); for Granite and Nemotron the
`reverse` runs of October 8 next to the `ridge` runs of October 9 (same chats, identical floors):

| Out of the model, into Qwen3.8 (8 fixed chats) | against the floor | against Qwen3-4B → Qwen3.8 (KL difference) |
| --- | --- | --- |
| LFM2-2.6B | 23.7% [13.0%, 32.2%], p 0.023, 7/8 | +0.0042 [−0.0003, +0.0097], p 0.242, 3/8 |
| LFM2-2.6B, its earlier `ridge` encoder | 4.0% [2.6%, 5.3%], p 0.031, 7/8 | current − earlier: −0.0133 [−0.0257, −0.0033], p 0.023, 7/8 |
| DeepSeek-V2-Lite | 16.9% [9.0%, 22.7%], p 0.031, 7/8 | +0.0087 [+0.0020, +0.0166], p 0.055, 1/8 |
| DeepSeek-V2-Lite, its earlier `ridge` encoder | 2.9% [1.1%, 3.6%], p 0.078, 6/8 | current − earlier: −0.0095 [−0.0179, −0.0025], p 0.023, 7/8 |
| Qwen3.6-35B-A3B | 41.0% [19.6%, 53.6%], p 0.016, 7/8 | −0.0076 [−0.0157, −0.0008], p 0.078, 5/8 |
| Qwen3.6-35B-A3B, as the servers render | 49.8% [33.4%, 59.6%], p 0.016, 7/8 | — |
| Qwen3.6-35B-A3B, its earlier `ridge` encoder | 11.3% [7.3%, 13.9%], p 0.008, 8/8 | current − earlier: −0.0201 [−0.0387, −0.0046], p 0.039, 7/8 |
| Qwen3.6-35B-A3B, as the servers render, its earlier `ridge` encoder | 17.5% [7.5%, 24.0%], p 0.023, 7/8 | current − earlier: −0.0218 [−0.0394, −0.0069], p 0.016, 7/8 |
| GLM-4.7-Flash | 38.1% [22.7%, 46.3%], p 0.008, 8/8 | −0.0056 [−0.0107, −0.0014], p 0.031, 7/8 |
| GLM-4.7-Flash, as the servers render | 48.1% [35.4%, 55.4%], p 0.008, 8/8 | — |
| Granite-4.0-H-Tiny | 1.3% [0.6%, 2.3%], p 0.039, 7/8 | +0.0193 [+0.0063, +0.0360], p 0.016, 1/8 |
| Granite-4.0-H-Tiny, its earlier `reverse` encoder | 5.4% [−14.4%, 22.1%], p 0.734, 3/8 | current − earlier: +0.0028 [−0.0085, +0.0163], p 0.742, 5/8 |
| Falcon-H1-3B | 25.3% [13.1%, 32.7%], p 0.055, 6/8 | +0.0031 [−0.0002, +0.0067], p 0.164, 1/8 |
| Jamba-Reasoning-3B | −1030.5% [−2210.4%, −606.9%], p 0.008, 0/8 | +0.7176 [+0.5148, +0.9279], p 0.008, 0/8 |
| Nemotron-3.5 | 4.7% [2.6%, 7.7%], p 0.008, 8/8 | +0.0170 [+0.0041, +0.0325], p 0.016, 1/8 |
| Nemotron-3.5, as the servers render | −43.8% [−158.5%, −5.7%], p 0.062, 1/8 | — |
| Nemotron-3.5, its earlier `reverse` encoder | −16.1% [−91.6%, 17.2%], p 0.453, 2/8 | current − earlier: −0.0141 [−0.0382, +0.0127], p 0.328, 6/8 |
| Nemotron-3.5, as the servers render, its earlier `reverse` encoder | −115.4% [−298.6%, −29.4%], p 0.039, 2/8 | current − earlier: −0.0485 [−0.1048, −0.0114], p 0.031, 6/8 |
| Kimi-Linear-48B-A3B | 32.2% [15.1%, 44.5%], p 0.023, 7/8 | −0.0016 [−0.0073, +0.0041], p 0.594, 4/8 |
| Ling-3.0-tiny | 42.4% [29.3%, 48.4%], p 0.016, 7/8 | −0.0085 [−0.0154, −0.0020], p 0.078, 6/8 |
| Gemma-4-26B-A4B | 36.9% [16.6%, 49.7%], p 0.016, 7/8 | −0.0048 [−0.0118, +0.0008], p 0.336, 5/8 |
| Gemma-4-26B-A4B, as the servers render | 35.4% [14.2%, 49.1%], p 0.039, 6/8 | — |
| Gemma-4-31B | 33.0% [16.1%, 44.0%], p 0.008, 8/8 | −0.0021 [−0.0067, +0.0020], p 0.320, 6/8 |
| Gemma-4-31B, as the servers render | 34.0% [18.0%, 43.3%], p 0.008, 8/8 | — |

What the twelve show:

- **Into a new model the shares depend on the chats.** On the six chats every run shares a switch from Qwen3.8
  removes 6-83% of the floor's KL: GLM-4.7-Flash 83%, Ling 71%, LFM2 51%, Qwen3.6 50%, Gemma-4-26B 45%, Gemma-4-31B
  35%, DeepSeek-V2-Lite 31%, Kimi-Linear 21%, Nemotron 16% (27% as the servers render), and no measurable part into
  Falcon-H1 (9.5%, p 0.094), Jamba (6.8%, p 0.22) and Granite (6.3%, p 0.25). The reports' pooled figures differ
  because the chat sets differ: where a model renders two of the fixed chats too short, its runs add the longer set,
  which holds a5add943, the reply that rewrites the earlier answer. Its floor KL is 2.2-2.5 nats/token (0.001-0.18 on
  the other chats), so it carries 92-95% of the pooled floor into DeepSeek-V2-Lite, LFM2, Granite, Falcon and Kimi:
  into Granite (41.8% pooled), Falcon (50.5%) and Kimi (70.2%) the pooled share is mostly that one chat (without it
  9.6%, 8.1% with a CI reaching below 0, and 19.8%). Ling renders all eight long enough, so its runs never saw
  a5add943. The decoders, fitted on the pooled pivot estimates, do not depend on the encoder; they help into nine of
  the twelve on ordinary chats, and into Granite and Falcon only on the chat that copies earlier context.
- **Out of a new model the onboarding's encoder warnings decide, not the number of attention layers.** Where no
  dimension's train R² falls below 0.05 (the source encoder's smallest 0.06-0.21: Falcon, Kimi, Ling, whose cap warning
  fired on 37% of its dimensions, both Gemma 4 through their full-attention layers) the switch into Qwen3.8 removes 25-42%
  of the floor's KL, not distinguishable from the jointly fitted Qwen3-4B's 30% (sign-flip p 0.16-0.59; Ling, better
  at p 0.078, is not like for like). Where both warnings fire (a dimension below R² 0.05 and the cap binding on 49-96%
  of the dimensions: Granite 0.044, Nemotron 0.028, Jamba 0.003; medians 0.10, 0.08, 0.01), `reverse`, which scales each
  dimension by about 1 / R², did no measurable good (Granite 5.4%, p 0.73), harm as the servers render (Nemotron
  −115%, p 0.039) or great harm (Jamba −1031%). The attention layers do not explain it: Gemma-4-26B's source reads 5
  full-attention layers (2,560 columns) and removes 37%, and Ling, with 6 attention layers like Nemotron, 42%.
- **`ridge` where the warnings fire.** Re-encoded with `ridge` (October 9, `campaign.py reencode --encoder ridge`), a
  switch out of Granite removes 1.3% [0.6%, 2.3%] (p 0.039, lower on 7 of 8; every chat within 0.002 KL of the
  floor) and out of Nemotron 4.7% [2.6%, 7.7%] (p 0.008, lower on 8 of 8) with the templates' rendering, but as the servers
  render (Nemotron's server keeps the earlier reasoning, so 95-97% of the translated cells come from its cache,
  against 10-85% with the templates' rendering; what a live switch holds) −43.8% [−158.5%, −5.7%] (p 0.062, lower on 1 of 8; −115% with `reverse`, so `ridge` is less
  harmful, −0.0485 [−0.1048, −0.0114] KL, p 0.031, but not harmless). Out of either a switch keeps at best about what
  the floor keeps: the target gets little beyond its native tail. hub6 now carries both `ridge` encoders, and
  Nemotron is to be registered with `kvpark bridge add --no-source` (new: a target only; switches out of it are
  skipped, `source-refused`, and the next model prefills natively), which its campaign report's "Bridge settings"
  now recommend because a switch out of it did worse than the floor; Jamba (`ridge` −0.2% [−2.6%, 2.5%]) stays out.
  The warnings come from the onboarding itself, before any evaluation (state.json `onboard.result.warnings`). The
  `ridge` choice was made on the same eight chats (a selection-set result); the campaign's re-encodes reproduced the
  `checks/encoder-ridge/` runs exactly.
- **R² does not track this.** In hub6-20261009 (below) K R² out of Nemotron into the base models is 0.63-0.70 and
  out of Granite 0.56-0.67, as high as out of the healthy sources (0.53-0.76), yet switches out of them keep about the
  floor or less: `ridge` is the R²-optimal encoder and shrinks, which is what section 9's diagnosis found R² cannot
  see. With `reverse` those two read 0.36-0.55 (hub6-20261008), which overlapped LFM2's 0.53-0.68 already.

**hub6-20261009** (and `-v2`, for `kvpark bridge engine --hub`) is hub4-d4096's four models plus eleven of the
twelve onboarded models (all but Jamba): qwen38, muse, qwen3-4b, ornith, lfm2-2.6b, deepseek-v2-lite, qwen3.6-35b-a3b,
glm-4.7-flash, granite-4.0-h-tiny (`ridge`), falcon-h1-3b, nemotron-3.5-lightning-30b-a3b (`ridge`), kimi-linear-48b-a3b,
ling-3.0-tiny, gemma-4-26b-a4b, gemma-4-31b (`campaign.py assemble --evaluate`, October 9: hard links, 22 GB in format 1
and 34 GB in format 2 with no new disk). It replaces **hub6-20261008**, assembled the day before with the `reverse`
encoders of Granite and Nemotron, which `campaign.py status` now flags as built from an older onboarding. Its R²
evaluation ran on the GPU under the guard: 146 s running, 142 s paused (69 pauses, at most 95.9 °C), 36.1 GiB anonymous
and 12.5 GiB on the GPU at the same time at most (46.6 GiB), the longest timed GPU launch 0.0099 s, no GPU line in the
kernel log. Held-out R² (K / V) of every pair; between two onboarded models on the 8,699 test spans all eleven share
(`evaluation-onboarded.json`), with a base model on each onboarding's own test spans, between base models on hub4's
(`hub6-20261009-summary/r2-summary.json`; its script now leaves out hub4's dedicated pairwise map "direct qwen38->muse",
which until October 9 it read as a hub pair, so hub6-20261008's Muse row said 0.77 and 0.67 for 0.75 and 0.62):

| Model | K R² into it, from the 14 others | K R² out of it, into the 14 others | V into | V out | self K / V |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 | 0.53-0.71 | 0.50-0.77 | 0.38-0.63 | 0.46-0.67 | 0.87 / 0.81 |
| Muse | 0.59-0.75 | 0.47-0.75 | 0.47-0.62 | 0.43-0.62 | 0.86 / 0.67 |
| Qwen3-4B | 0.64-0.77 | 0.53-0.76 | 0.36-0.54 | 0.45-0.63 | 0.90 / 0.79 |
| Ornith | 0.55-0.75 | 0.49-0.76 | 0.40-0.67 | 0.45-0.65 | 0.90 / 0.85 |
| LFM2-2.6B | 0.60-0.76 | 0.42-0.68 | 0.38-0.56 | 0.36-0.60 | 0.91 / 0.86 |
| DeepSeek-V2-Lite | 0.34-0.53 | 0.42-0.69 | 0.31-0.49 | 0.36-0.55 | 0.74 / 0.73 |
| Qwen3.6-35B-A3B | 0.58-0.75 | 0.44-0.75 | 0.47-0.66 | 0.39-0.63 | 0.92 / 0.88 |
| GLM-4.7-Flash | 0.51-0.65 | 0.49-0.76 | 0.42-0.58 | 0.42-0.61 | 0.83 / 0.79 |
| Granite-4.0-H-Tiny (`ridge`) | 0.58-0.73 | 0.44-0.70 | 0.44-0.58 | 0.38-0.54 | 0.86 / 0.75 |
| Falcon-H1-3B | 0.57-0.72 | 0.44-0.72 | 0.36-0.53 | 0.38-0.59 | 0.88 / 0.83 |
| Nemotron-3.5 (`ridge`) | 0.62-0.76 | 0.42-0.71 | 0.48-0.65 | 0.39-0.54 | 0.88 / 0.80 |
| Kimi-Linear | 0.60-0.74 | 0.41-0.68 | 0.45-0.58 | 0.32-0.55 | 0.91 / 0.85 |
| Ling-3.0-tiny | 0.54-0.69 | 0.37-0.67 | 0.48-0.61 | 0.28-0.48 | 0.91 / 0.89 |
| Gemma-4-26B-A4B | 0.42-0.61 | 0.34-0.64 | 0.31-0.49 | 0.31-0.50 | 0.68 / 0.52 |
| Gemma-4-31B | 0.37-0.54 | 0.44-0.72 | 0.28-0.46 | 0.41-0.61 | 0.64 / 0.54 |

Between the onboarded models K R² is 0.35-0.75 and V 0.28-0.63 (110 pairs); the weakest pairs are into DeepSeek-V2-Lite
and Gemma-4-31B (Gemma-4-26B → DeepSeek-V2-Lite K 0.345, Ling → Gemma-4-31B 0.369, Ling → DeepSeek-V2-Lite 0.381). In
hub6-20261008 they were 0.24-0.75 and 0.17-0.63, the weakest out of Granite and Nemotron into the Gemma 4 models (K
0.24-0.32) and into DeepSeek-V2-Lite (Granite 0.262, Nemotron 0.274). No onboarded → onboarded pair has a seeded
end-to-end measurement; the live hops below are the first switches between two onboarded models.

**A joint hub6 instead?** No. The seven-model joint refit on its models' common spans was worse end to end than
onboarding (section 9), and a fifteen-model one does not fit this machine: its summed views are 55,296 wide, so the
GCCA phase alone needs 77.7 GiB of anonymous memory and the decoder phase as `fit_hub.joint_bytes` counts it 301 GiB
(106 GiB without Gemma-4-31B, whose decoder is 225,280 columns wide), against the 98 GiB a 123 GiB machine can grant
with the 25 GiB margin, before the GPU's share. The eleven onboarded models share only 25,441 train spans
(Gemma-4-31B's capture holds 52 training documents), against hub4's 185,672. A joint hub6 would need the decoder's dev
statistics streamed as the onboarding's are, a GCCA that does not hold two float64 P x P matrices (a float32 or GPU
subspace solver), fits that resume, and full captures of the large Gemma 4 models; whether it would then translate
better than onboarding is open (section 9's refit differed from onboarding in three ways at once).

### Live switches with hub6 (October 9)

Throwaway llama-servers of the repository's pin, llama.cpp a7b94df2c (its ROCm build in `runtime-next/build-rocm`;
`runtime/build` is a CPU-only build): every layer on the GPU, 36,864-token contexts, one slot, 4 threads. The bridge
configuration, slots and engine socket lived in scratch directories (`XDG_DATA_HOME`, `KVPARK_BRIDGE_CONFIG`), each
model registered with its campaign report's settings (native prompt start; the recurrent means of Qwen3.8, Granite
and Kimi). `live_bridge.py hops` took two held-out chats of the corpus's test split that are also among the campaigns'
eight (`multi_turn_concat_sessions:595` = 3b211f08, `multi_turn_reasoning_if_think:18676` = b19deab5): prime the
first model, then per hop a native prefill of the target (the reference), an engine prewarm,
`kvpark.bridge.switch.switch()` (save the source's slot, translate, restore, then the warm request that prefills
the 2,048-token native tail) and 32 generated tokens (temperature 0) compared with the native run's. Two models per
run, each run under `campaign.py exec --gpu-evaluation` (GPU/memory lock, memory check, the guard's Watch over the
servers and the engine daemon), with `--cool-c 65` (a wait until the package is at most 65 °C before each timed
step) and the ggml Vulkan engine (below: the torch daemon's busy thread).

Two series ran. The first (`live-hub6-20261009/`, hub6-20261008-v2) is superseded: its timings were wall times with
the guard's pauses inside, its "running" times took those pauses out (from the guard log), and both mislead. A
guard pause stops the processes, but the GPU finishes the work already queued, so a step's own time lies between
its wall time less the pauses inside it and its wall time; the identical 15,355-token Qwen3.8 prefill read 39.3 s
"running" in one run and 28.6 s in another. Its two Gemma-4-26B → Qwen3.8 hops were round trips as well: the
Gemma slot held a prefix translated from Qwen3.8 (6,197 of its 8,245 tokens). The second series
(`live-hub6-20261009b/`, hub6-20261009-v2, five runs) runs Gemma-4-26B → Qwen3.8 on its own route (Gemma primed
natively), and prefills every native reference once more in 1,024-token requests, each after the cooling wait
(`live_bridge.py --native-chunk`), so that no guard pause falls inside a timed request. No pause did, but the
requests carry an overhead of their own: on Qwen3.8 each took 4.1 s per 1,024 tokens wherever it sat in the
conversation, against about 3 s inside one request (on Granite they matched one request, on Gemma they were 10-27%
slower), so their sum bounds the native prefill from above rather than measuring it. The table gives, per step,
the wall time and the pauses inside it, and the speed-up as an interval: native in [wall − pauses, min(wall, the
chunked sum)] over the switch in [wall − pauses, wall] (`scripts/live_md.py`; evidence `live-table.json`, each run's
`hops.jsonl` with the generated text, guard logs, `scripts/`):

| Hop | Chat | T0 / P | Reuse | Engine s | Tail prefill s | Switch s (paused inside) | Native prefill, one request s (paused inside) | Native prefill, cooled 1,024-token requests s | Switch faster by | First 32 tokens agree |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Gemma-4-26B-A4B → Qwen3.8 | 595 (3b211f08) | 13,307 / 15,355 | yes | 1.46 | 8.5 | 11.3 (4) | 48.1 (4) | 62.7 | 3.9-6.8x | 0 |
| Gemma-4-26B-A4B → Qwen3.8 | 18676 (b19deab5) | 10,028 / 12,076 | yes | 1.09 | 8.9 | 10.4 (4) | 49.4 (29) | 48.3 | 1.9-7.7x | 2 |
| Kimi-Linear-48B-A3B → Gemma-4-26B-A4B | 595 (3b211f08) | 6,197 / 8,245 | yes | 0.47 | 1.9 | 3.0 (0) | 6.6 (0) | 8.4 | 2.2x | 32 |
| Kimi-Linear-48B-A3B → Gemma-4-26B-A4B | 18676 (b19deab5) | 7,845 / 9,893 | yes | 0.58 | 1.9 | 2.7 (0) | 13.4 (6) | 8.6 | 2.6-3.1x | 32 |
| Kimi-Linear-48B-A3B → Qwen3.8 | 595 (3b211f08) | 13,307 / 15,355 | yes | 1.09 | 10.3 | 12.5 (4) | 52.2 (10) | 61.6 | 3.3-6.4x | 0 |
| Kimi-Linear-48B-A3B → Qwen3.8 | 18676 (b19deab5) | 10,028 / 12,076 | yes | 1.04 | 8.4 | 9.7 (4) | 50.0 (29) | 49.4 | 2.1-9.0x | 2 |
| Qwen3.8 → Gemma-4-26B-A4B | 595 (3b211f08) | 6,197 / 8,245 | yes | 0.69 | 1.9 | 3.5 (0) | 6.8 (0) | 7.5 | 2.0x | 32 |
| Qwen3.8 → Gemma-4-26B-A4B | 18676 (b19deab5) | 7,845 / 9,893 | yes | 0.74 | 2.0 | 3.2 (0) | 13.7 (8) | 8.8 | 1.8-3.1x | 32 |
| Qwen3.8 → Granite-4.0-H-Tiny | 595 (3b211f08) | 6,098 / 8,146 | yes | 0.51 | 1.0 | 2.1 (0) | 3.8 (0) | 3.9 | 1.8x | 32 |
| Qwen3.8 → Granite-4.0-H-Tiny | 18676 (b19deab5) | 9,187 / 11,235 | yes | 0.45 | 1.0 | 1.8 (0) | 8.9 (4) | 5.3 | 2.8-3.0x | 0 |

- **Reuse: 10 of 10 switches restored and reused exactly T0**, into the Mamba2 hybrid Granite-4.0-H-Tiny (with its
  recurrent mean), into Gemma 4's iSWA cache (the sliding layers' last window) from Qwen3.8 and from Kimi, and into
  Qwen3.8 out of Kimi-Linear's MLA + KDA cache and out of Gemma-4-26B's. Kimi → Gemma-4-26B is a switch between two
  onboarded models. The first series reused exactly T0 in 10 of 10 as well.
- **Time.** The engine took 0.45-1.46 s. Into Qwen3.8 (12-15K tokens) a switch took 9.7-12.5 s of wall time with
  about 4 s of pauses inside each (5.7-8.5 s of its own at least), most of it the 2,048-token tail (8.4-10.3 s of
  wall time), against 48-52 s of native prefill. On the 15K-token chat, whose native prefill met 4-10 s of pauses,
  the switch was 3.3-6.8 times faster; on the 12K-token chat the native prefill met 29 s of pauses, so only
  1.9-2.1 times is certain there (up to 7.7-9.0). Into Gemma-4-26B (8-10K tokens) 2.7-3.5 s against 6.6-6.8 s of
  native prefill with no pause inside (2.0-2.2 times), and 1.8-3.1 times on the longer chat. Granite prefills 8-11K
  tokens in 3.8-5.3 s, so a switch into it (1.8-2.1 s) saves 1.7-3.5 s (1.8-3.0 times); the bridge's `min_handover`
  setting exists for such targets. The first series' "3.6-5.9 times" into Qwen3.8 rested on the pause correction
  above and is withdrawn.
- **Answers.** Into Gemma-4-26B the 32 generated tokens equal the native run's in all four switches (from Qwen3.8 and
  from Kimi). Into Qwen3.8, which opens with its reasoning, they part after 0-2 tokens in all four, but each
  restates the last user request correctly in other words. Into Granite one chat agrees 32 of 32; on the other
  (18676: the last turn asks to work two keywords into the previous four-section answer, which lies in the
  translated part) the switched Granite again wrote a new short essay on the keywords instead of revising its
  answer: section 9's weakness on replies that copy the earlier context (the campaign's teacher-forced score of this
  chat into Granite: KL 0.044 against the floor's 0.048). These agreements say little about the translation: the
  2,048-token native tail holds the last user message, which Gemma restates and which Granite's 595 answer copies as
  JSON, and the campaigns' teacher-forced scores are the measure of quality.
- **Guard.** The five runs took 3-8 minutes each, with 14-63 pauses (29-130 s), at most 96.4 °C, MemAvailable at
  least 50.2 GiB, no abort and no amdgpu line in the kernel log. Their memory (the guard's GTT reading, new): up to
  7.9 GiB anonymous and 57 GiB on the GPU, 63.7 GiB at the same time for Kimi-Linear and Qwen3.8 together (the
  command asked for 63.0 GiB: the models' files plus 8 GiB; `scripts/live.sh` now asks for 12 GiB on top). In the
  first series, the first attempt ran the torch engine as the bridge starts it, in a session of its own outside the
  guard's process group: the guard stopped the run at 90 °C after the 15K-token prime, the daemon's first load drew
  up to 59 W while it was stopped, Tctl reached 98.2 °C twice and the guard ended the run (70 s in; `aborted-1/`).
  With the torch daemon inside the group, its busy runtime thread alone held Tctl at 93 °C at 48 W, and the guard
  stopped and continued the run 170 times in 764 s; it was stopped by hand after one chat (`aborted-2/`). Since
  then `live_bridge.py` starts the daemon inside the run's group with an idle exit of a week (the run stops it; its
  default 600 s counts on a clock that runs on while the guard holds the run, and the bridge would then start a new
  daemon outside the group) and checks before every switch that the daemon answering is still that one.
