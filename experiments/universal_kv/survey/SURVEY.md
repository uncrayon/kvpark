# Top-20 open-weights models: architecture survey for a universal KV-cache translator

Retrieved 2026-10-02. Machine-readable version (all fields, formulas, per-model source URLs): `top20_open_models.json` in this directory.

## How the ranking was built

- Source: Artificial Analysis leaderboard, https://artificialanalysis.ai/leaderboards/models?open_weights=open_source . The page text could not be read through the fetch tool (it returned an unrelated sub-metric), so I parsed the `models` array embedded in the page's Next.js payload (689 models, 373 flagged `isOpenWeights`).
- Filter: `isOpenWeights = true` and `deprecated = false`. Reasoning-effort variants of the same weights (e.g. "GLM-5.3 (Max)" / "(Low)", "Qwen3.8 27B (Xhigh/Medium/Low/Non-reasoning)") are merged, and each model is ranked by its best variant.
- Index version: **Intelligence Index v4.3**. The page note says AutomationBench-AA joins, tau3-Banking leaves, and Terminal-Bench moves to 4.0. The newest release date in the payload is 2026-09-30. AA shows no "last updated" date for the scores.
- "(est.)" means AA flags the score as estimated (`intelligenceIndexIsEstimated`).
- Deprecated open-weights entries that would otherwise make the cut: DeepSeek V4 Flash 0731 (34.33), GLM-5.2 (33.71), DeepSeek V4 Pro 0424 (30.45), GLM-5 (27.91), Kimi K2.6 (26.98), GLM-5.1 (26.06), MiMo-V2.5-Pro (25.99), MiMo-V2.5 (25.17). Each is an older checkpoint of a family already in the list. Ranks 21 and 22 are Ling-3.0-flash-VL (24.57; KDA + gated MLA 5:1) and Nemotron 3 Ultra 550B (22.93; Mamba-2 + attention). Nemotron is the nearest pure-Mamba-2 hybrid, and no top-20 model has that design. For reference, Muse Glimmer (the kvpark target model) ranks #33 at 17.48 on the same deduplicated list.

## Conventions

- **KV B/token bf16** is the per-token cache growth at 2 bytes per element:
  - GQA: n_kv x (d_k + d_v) x 2 per KV-bearing layer.
  - MLA: (kv_lora_rank + qk_rope_head_dim) x 2 per layer.
  - DeepSeek V4: 512 x 2 / compress_ratio per compressed layer.
  - Sliding-window layers are bounded and do not count as growth. Their fixed size is under "Fixed state".
- **Extra per-token**: sparse-attention indexer keys (DSA/MSA/QSA/CSA). Production kernels often store these in FP8 or FP4.
- **Fixed state**: per-sequence bytes that do not grow with length.
  - SWA ring buffers.
  - Recurrent state for Gated DeltaNet (GDN) and Kimi Delta Attention (KDA): matrix state plus short-conv history, counted at bf16. Qwen configs set `mamba_ssm_dtype=float32`, so the real state is about 2x. The JSON has a `recurrent_state_bytes_fp32_state` field.
- `*` = inferred (see "Data I could not verify").
- **Q4_K_M est.** = total stored parameters x 4.85 bits / 8. These are estimates, not measured file sizes. "yes" means 100 GB or less, which leaves room for KV and compute buffers on a 128 GB Ryzen AI Max+ 395 (EVO-X2). "borderline" means 100 to 115 GB.
- **llama.cpp** = GGUF architecture name in `ggml-org/llama.cpp` master at commit 99b95488ca (2026-10-03 00:50 UTC). I checked `src/llama-arch.cpp` and the `conversion/` registrations.

## Summary table

| # | Model (HF repo) | AA II | Released | Total / active | Layers | Hidden | Vocab (cfg) | Attention (KV-bearing layers) | Q heads / KV heads x head_dim | Position | Max ctx (cfg) | KV B/token bf16 | Extra per-token | Fixed state | llama.cpp | Q4_K_M est. / fits 128 GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MiMo-V2.6-Pro (`XiaomiMiMo/MiMo-V2.6-Pro-RL`) | 46.32 | 2026-09-21 | 1.02T / 42B | 70 | 6144 | 152,576 | SWA(128)+global GQA: 10 global / 60 SWA | 128 / 8 x K192/V128 | 64-d RoPE, θ=10,000,000 (SWA θ=10,000) | 1,048,576 | 51,200 | - | SWA 39.3 MB | `mimo2` | 618 GB / no |
| 2 | GLM-5.3 (`zai-org/GLM-5.3`) | 44.78 | 2026-08-18 | 744B / 40B | 78 | 6144 | 154,880 | MLA + DSA indexer (IndexShare 1/4): 78 MLA | 64 / MLA latent 512+64 | 64-d RoPE, θ=8,000,000 | 1,048,576 | 89,856 | idx 5,376 | - | `glm-dsa` | 451 GB / no |
| 3 | Kimi K3 (`moonshotai/Kimi-K3`) | 43.59 | 2026-07-16 | 2.8T / 104B | 93 | 7168 | 163,840 | KDA + NoPE gated MLA: 24 MLA / 69 KDA | 96 / MLA latent 512+64 | NoPE | 1,048,576 | 27,648 | - | rec 232.3 MB | `kimi-k3` | 1698 GB / no |
| 4 | GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) | 41.81 | 2026-08-26 | 320B / 18B | 45 | 4096 | 154,880 | KDA + NoPE DSA-MLA: 11 MLA / 34 KDA | 64 / MLA latent 512+0 | NoPE | 1,048,576 | 11,264 | idx 704* | rec 76.3 MB | `glm5-next` | 194 GB / no |
| 5 | Qwen3.8-2.4T-A95B (`Qwen/Qwen3.8-2.4T-A95B`) | 39.89 | 2026-08-12 | 2.4T / 95B | 92 | 8192 | 248,320 | Gated DeltaNet + GQA 3:1: 23 GQA / 69 GDN | 64 / 4 x 256 | 64-d RoPE, θ=10,000,000 | 262,144 | 94,208 | - | rec 297.9 MB | `qwen35moe` | 1455 GB / no |
| 6 | Qwen3.8-Flash-Next (`Qwen/Qwen3.8-Flash-Next`) | 39.82 | 2026-08-26 | 125B (+51B n-gram, +4B MTP) / 6B | 48 | 2560 | 248,320 | Gated DeltaNet + QSA (sparse GQA) 3:1: 12 / 36 | 24 / 2 x 256 | 64-d RoPE, θ=10,000,000, IMRoPE | 262,144 | 24,576 | idx 3,072* | rec 58.8 MB | `qwen4exp` | 109 GB / borderline |
| 7 | DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash`) | 39.46 | 2026-09-10 | 552B (+196B Engram) / 8B prefill / 16B decode | 40 | 5120 | 129,280 | CSA2 latent MQA: SWA128 all 40 + compressed KV from 4 source layers (ratio 2/1), enc-dec | 64 / 1 x 512 | 64-d RoPE, θ=10,000 (compressed θ=160,000+YaRN16) | 1,048,576 | 2,560* | idx 640 | SWA 5.2 MB | no (PR/fork) | 454 GB / no |
| 8 | MiMo-V2.6-Flash (`XiaomiMiMo/MiMo-V2.6-Flash-RL`) | 37.88 | 2026-09-21 | 309B / 15B | 48 | 4096 | 152,576 | SWA(128)+global GQA: 9 global / 39 SWA | 64 / 4 x K192/V128 (SWA: 8 kv) | 64-d RoPE, θ=10,000,000 (SWA θ=10,000) | 1,048,576 | 23,040 | - | SWA 25.6 MB | `mimo2` | 187 GB / no |
| 9 | DeepSeek V4 Pro 0813 (`deepseek-ai/DeepSeek-V4-Pro-0813`) | 36.00 | 2026-08-13 | 1.6T / 49B | 61 | 7168 | 129,280 | latent MQA: SWA128 all 61 + CSA(4) x30 / HCA(128) x31 | 128 / 1 x 512 | 64-d RoPE, θ=10,000 (compressed θ=160,000+YaRN16) | 1,048,576 | 7,928 | idx 1,920 | SWA 8.0 MB | `deepseek4` | 970 GB / no |
| 10 | Qwen3.8-27B (`Qwen/Qwen3.8-27B`) | 33.70 | 2026-08-14 | 27B / 27B | 64 | 5120 | 248,320 | Gated DeltaNet + GQA 3:1: 16 GQA / 48 GDN | 24 / 4 x 256 | 64-d RoPE, θ=10,000,000, IMRoPE | 262,144 | 65,536 | - | rec 78.4 MB | `qwen35` | 16 GB / yes |
| 11 | Motif 3 (`Motif-Technologies/Motif-3`) | 33.57 (est.) | 2026-08-12 | 314B / 13.2B | 53 | 4096 | 220,160 | GDLA (MLA-like latent, diff. attn) : 14 global / 39 SWA(128) | 80 / 16 (latent 512+64; K 192, V 128) | 64-d RoPE, θ=10,000 (SWA θ=10,000) | 262,144 | 16,128 | - | SWA 5.8 MB | no (PR/fork) | 190 GB / no |
| 12 | K2 Horizon 375B A23B (`IFM/K2-Horizon-375B-A23B`) | 30.50 | 2026-09-03 | 375B / 23B | 61 | 6144 | 250,624 | full GQA: 61 | 48 / 8 x 128 | 64-d RoPE, θ=10,000,000 | 524,288 | 249,856 | - | - | no (PR/fork) | 227 GB / no |
| 13 | MiniMax-M3 (`MiniMaxAI/MiniMax-M3`) | 29.22 | 2026-06-01 | 428B / 23B | 60 | 6144 | 200,064 | GQA + MSA block-sparse: 60 (57 MSA) | 64 / 4 x 128 | 64-d RoPE, θ=5,000,000 | 1,048,576 | 122,880 | idx 14,592 | - | `minimax-m3` | 260 GB / no |
| 14 | Nex-N2-Pro (`nex-agi/Nex-N2-Pro`) | 28.20 (est.) | 2026-06-02 | 397B / 17B | 60 | 4096 | 248,320 | Gated DeltaNet + GQA 3:1: 15 GQA / 45 GDN | 32 / 2 x 256 | 64-d RoPE, θ=10,000,000, IMRoPE | 262,144 | 30,720 | - | rec 97.7 MB | `qwen35moe` | 241 GB / no |
| 15 | Kimi K2.7 Code (`moonshotai/Kimi-K2.7-Code`) | 25.81 | 2026-06-12 | 1T / 32B | 61 | 7168 | 163,840 | MLA (DeepSeek-V3): 61 | 64 / MLA latent 512+64 | 64-d RoPE, θ=50,000, YaRN x64 | 262,144 | 70,272 | - | - | `deepseek2` | 606 GB / no |
| 16 | Inkling Small (`thinkingmachines/Inkling-Small`) | 25.66 | 2026-07-30 | 276B / 12B | 42 | 4096 | 201,024 | SWA(512) local + global GQA, K/V short conv: 7 global / 35 local | 32 / 8 x 128 | NoPE + learned rel. bias | 1,048,576 | 28,672 | - | SWA 73.4 MB | no (PR/fork) | 167 GB / no |
| 17 | K2 Horizon MoVA 36B A4B (`IFM/K2-Horizon-MoVA-36B-A4B`) | 25.30 | 2026-09-03 | 36B / 4B | 48 | 2560 | 250,624 | full GQA, MoVA value experts: 48 | 32 / 8 x 128 | 128-d RoPE, θ=10,000,000 | 524,288 | 196,608 | - | - | no (PR/fork) | 22 GB / yes |
| 18 | Hy3 (`tencent/Hy3`) | 25.30 | 2026-07-06 | 295B / 21B | 80 | 4096 | 120,832 | full GQA + QK-norm: 80 | 64 / 8 x 128 | 128-d RoPE, θ=11,158,840 | 262,144 | 327,680 | - | - | `hy_v3` | 179 GB / no |
| 19 | Inkling (`thinkingmachines/Inkling`) | 24.98 | 2026-07-15 | 975B / 41B | 66 | 6144 | 201,024 | SWA(512) local + global GQA, K/V short conv: 11 global / 55 local | 64 / 8 x 128 (SWA: 16 kv) | NoPE + learned rel. bias | 1,048,576 | 45,056 | - | SWA 230.7 MB | no (PR/fork) | 591 GB / no |
| 20 | Solar Open2 250B (`upstage/Solar-Open2-250B`) | 24.74 (est.) | 2026-08-12 | 250B / 15B | 48 | 4096 | 196,608 | KDA-style linear + NoPE GQA 3:1: 12 GQA / 36 linear | 64 / 8 x 128 | NoPE | 1,048,576 | 49,152 | - | rec 80.8 MB* | no (PR/fork) | 152 GB / no |

Per-model notes that do not fit the table:

- **MiMo-V2.6**
  - Global and SWA layers use different KV head counts in Flash (4 vs 8). K is 192-d and V is 128-d.
  - RoPE covers 64 of the 192 dims (NeoX). Global and SWA layers use different rope thetas.
  - SWA layers have a learned attention-sink bias, and `attention_value_scale` is 0.612 (Pro) or 0.707 (Flash).
- **GLM-5.3**
  - Only 21 of 78 layers compute indexer keys. The other 57 reuse the previous layer's top-k (IndexShare).
- **Kimi K3 / GLM-5.3-Flash**
  - The MLA layers are NoPE. Kimi K3 keeps a 64-d decoupled key in the latent but never rotates it. GLM-5.3-Flash has `qk_rope_head_dim=0`.
  - Kimi K3 also has depth-wise Attention Residuals. These mix earlier block outputs, not tokens, so they add no cache.
- **DeepSeek V4 / V4.1**
  - There is no per-token K/V beyond the 128-token window. Global entries are learned gated pools of 4 or 128 tokens (V4), or 2 or 1 tokens (V4.1).
  - Each layer has one 512-d latent that serves as both K and V. RoPE is applied to its last 64 dims.
  - The compressors hold partial-group fp32 state while a group is filling.
  - V4.1 has a Causal Encoder-Decoder layout. Global KV is produced by only 4 "source" layers (2, 8, 14, 20) and shared by the rest. The decoder's global KV is projected from the final encoder hidden state. The official figure is 890 B/token with FP4 KV.
- **Qwen3.8-Flash-Next**
  - QSA selects micro-blocks through a 1-head 128-d indexer pooled over 4 tokens.
  - It also has 3-gram hashed embeddings (51B params), PLE with a conv state, and Gated Residual (hc_count=4). None of these add per-token cache, but PLE adds a small fixed conv state.
- **Motif 3**
  - GDLA = MLA-style latent (kv_lora 512 + 64 rope) up-projected to 16 KV heads, with differential (signal/noise) heads.
  - The table counts the latent cache (vLLM `FLASH_ATTN_MLA`, as the card recommends). The HF DynamicCache stores expanded K/V instead: 143,360 B/token.
- **MiniMax-M3**
  - MSA is block-sparse: 128-token blocks, top-16 blocks plus the local block.
  - Indexer keys are one 128-d key per token in each of 57 layers, with partial RoPE.
- **Inkling / Inkling Small**
  - There is no RoPE. Position comes from a query-dependent learned relative logit bias that is zero beyond 1024 tokens (512 in local layers), a 4-tap short convolution applied to K and V before caching, and log-length logit scaling.
  - Cached K/V carry no rotation, but each cached K/V already mixes its 3 predecessors.
- **K2 Horizon MoVA**
  - V comes from 4 of 64 routed "value experts". The cache is still plain GQA K/V.
- **Solar Open2**
  - Every layer is NoPE. The card says "positional encoding is removed entirely".
  - The linear-attention state shape is inferred from `linear_attn_config` (KDA-style naming). There is no modeling code in the repo.

## Architecture families (what a KV translator has to handle)

Grouped by the kind of state a translator must produce for the target model. Several models fall into two groups.

### F1 - Per-token GQA in every layer (RoPE)
"Classic" K/V tensors per layer and per token. The translator maps K/V per token, and K needs its rotation undone and redone.

| Model | KV heads x dim | RoPE | Notes |
|---|---|---|---|
| Hy3 | 8 x 128, 80 layers | full 128-d NeoX, θ 11,158,840 | QK-norm. Largest cache in the set (320 KiB/token) |
| K2 Horizon 375B | 8 x 128, 61 layers | partial 64/128 (interleaved-select), θ 1e7 | |
| K2 Horizon MoVA 36B | 8 x 128, 48 layers | full 128-d, θ 1e7 | V from routed value experts. Softplus output gate |
| MiniMax-M3 | 4 x 128, 60 layers | partial 64/128 NeoX, θ 5e6 | Also needs MSA indexer keys (one 128-d key per token per layer, 57 layers) |

### F2 - Sliding-window + global GQA hybrids
Per-token K/V only in the global layers. SWA layers need just the last W tokens, so the translator can rebuild those cheaply by replaying the last W tokens natively. The kvpark v3 "native tail" design already does this.

| Model | Global / SWA layers | Window | RoPE | Notes |
|---|---|---|---|---|
| MiMo-V2.6-Pro | 10 / 60 | 128 | 64/192 NeoX; θ 1e7 global, 1e4 SWA | K≠V dims (192/128). Sink bias on SWA |
| MiMo-V2.6-Flash | 9 / 39 | 128 | same | Global uses 4 KV heads, SWA uses 8 |
| Inkling | 11 / 55 | 512 | **none** (learned rel. bias) | SWA uses 16 KV heads, global uses 8. K/V short conv |
| Inkling Small | 7 / 35 | 512 | **none** | 8 KV heads everywhere |
| Motif 3 | 14 / 39 | 128(+1) | 64-d decoupled, θ 1e4 | Latent cache, so also in F3 |
| DeepSeek V4 / V4.1 | every layer has a 128-token window | 128 | | Also in F4 |

### F3 - MLA latent (DeepSeek-V2/V3 style)
One latent per token per layer, (512 + rope) x 2 bytes. The latent is shared by all heads, so the translator writes 576 numbers per layer instead of n_kv x (d_k + d_v). The decoupled rope part needs its own rotation handling.

| Model | Layers | Latent | RoPE | Extras |
|---|---|---|---|---|
| Kimi K2.7 Code | 61 | 512 + 64 | 64-d interleaved, θ 5e4, YaRN x64 (orig 4096) | - |
| GLM-5.3 | 78 | 512 + 64 | 64-d interleaved, θ 8e6 | DSA indexer keys (128-d) in 21 layers |
| Motif 3 | 14 global (+39 SWA) | 512 + 64 | 64-d, θ 1e4 | 16-head differential up-projection |
| Kimi K3 (MLA part) | 24 | 512 + 64 | **NoPE** | See F5 |
| GLM-5.3-Flash (MLA part) | 11 | 512 + 0 | **NoPE** | DSA k-pool indexer. See F5 |

### F4 - Compressed / pooled latent (DeepSeek V4 family)
There is no per-token K/V outside a 128-token window. Global memory is a sequence of learned gated pools of 4 or 128 tokens (V4), or 2 or 1 tokens (V4.1). Each entry is a 512-d latent used as both K and V, and RoPE on its last 64 dims runs at a separate theta (160,000) with YaRN.

A translator either has to synthesise pooled entries (and indexer keys), or prefill the target natively and only translate the window. Cache sizes in this family are the smallest in the set: 7.9 KB/token for V4 Pro and 2.5 KB/token for V4.1 Flash, plus indexer keys.

- DeepSeek V4 Pro 0813: 30 CSA(4) + 31 HCA(128) layers.
- DeepSeek V4.1 Flash: CSA2. Compressed KV comes from 4 source layers, shared across layers. Encoder-decoder layout, so the decoder's KV is a function of encoder outputs.

### F5 - Linear-attention / recurrent hybrids
A fixed recurrent state that cannot be written per token, plus per-token caches in roughly 1/4 of layers. A translator has to produce a recurrent state (e.g. by running the target's linear layers over the text, or by learning a state map), or fall back to re-prefilling those layers. The `recurrent_reset.py` path in this experiment is relevant here.

| Model | Linear type | Linear : full | Full-layer cache | Position | Recurrent state (bf16) |
|---|---|---|---|---|---|
| Qwen3.8-27B | Gated DeltaNet (16 K-heads / 48 V-heads, 128-d) | 48 : 16 | GQA 4 x 256 | partial 64/256 IMRoPE θ 1e7 | 78 MB |
| Qwen3.8-2.4T-A95B | GDN (16 / 128) | 69 : 23 | GQA 4 x 256 | partial 64/256 θ 1e7 | 298 MB |
| Nex-N2-Pro (Qwen3.5-397B) | GDN (16 / 64) | 45 : 15 | GQA 2 x 256 | partial 64/256 IMRoPE | 98 MB |
| Qwen3.8-Flash-Next | GDN (16 / 48) | 36 : 12 | QSA = GQA 2 x 256 + block indexer | partial 64/256 IMRoPE | 59 MB |
| Kimi K3 | KDA (96 heads x 128) | 69 : 24 | NoPE MLA latent 576 | **NoPE** | 232 MB |
| GLM-5.3-Flash | KDA (64 x 128) | 34 : 11 | NoPE MLA latent 512 + k-pool DSA | **NoPE** | 76 MB |
| Solar Open2 250B | KDA-style (64 x 128) | 36 : 12 | NoPE GQA 8 x 128 | **NoPE** | 81 MB* |

### Cross-cutting: positional encoding

The translator's RoPE handling falls into five cases:

- **NoPE caches (no phase to remove):**
  - Kimi K3 and GLM-5.3-Flash (MLA layers).
  - Solar Open2 (all layers).
  - Inkling and Inkling Small: learned relative bias plus K/V convolution. No rotary phase, but K/V are conv-mixed.
- **Partial NeoX RoPE:**
  - Qwen3.5/3.8 family: 64 of 256 dims, interleaved M-RoPE sections [11, 11, 10]. With text-only positions this reduces to 1-D.
  - MiMo: 64 of 192 dims, with two thetas.
  - MiniMax-M3: 64 of 128 dims.
  - K2 Horizon 375B: 64 of 128 dims, selected via an interleaved split.
- **Full NeoX RoPE:** Hy3 and K2 Horizon MoVA (128 dims).
- **Decoupled 64-d interleaved (GPT-J) RoPE on an MLA / latent key:**
  - Kimi K2.7 Code: θ 5e4 with YaRN x64.
  - GLM-5.3: θ 8e6.
  - Motif 3: θ 1e4. YaRN is listed in the config but not applied to the frequencies (`apply_yarn_scaling=false`); only the mscale term changes the softmax scale on global layers.
  - DeepSeek V4/V4.1: θ 1e4 on window-only paths, θ 1.6e5 with YaRN x16 on compressed layers.
- **Different theta per layer type within one model:** MiMo (SWA 1e4 vs global 1e7) and DeepSeek V4 (window vs compressed).

## Tokenizer families (byte-identical vocabularies)

All 20 use byte-level BPE. I compared the full vocab and merge lists in `tokenizer.json` (and `tiktoken.model` for Kimi).

| Group | Models | Base vocab | Identity evidence |
|---|---|---|---|
| Qwen3.5 / 3.8 | Qwen3.8-2.4T-A95B, Qwen3.8-Flash-Next, Qwen3.8-27B, Nex-N2-Pro | 248,044 (+33 added; config pads to 248,320) | The 3 Qwen `tokenizer.json` files are byte-identical. Nex has identical vocab, merges and added tokens (file differs only elsewhere). Same base as Qwen3.5-397B |
| Qwen2 / Qwen3 (151k) | MiMo-V2.6-Pro, MiMo-V2.6-Flash | 151,643 (+32 added; config 152,576) | Byte-identical file across Pro/Flash. Vocab and merges identical to `Qwen/Qwen3-8B` and MiMo-V2-Flash. 26 of the added tokens also match Qwen3's IDs |
| GLM-5 | GLM-5.3, GLM-5.3-Flash | 154,820 (+36; config 154,880) | Byte-identical file. Identical to GLM-5. Not the GLM-4.7 vocab (151,329) |
| DeepSeek | DeepSeek V4 Pro 0813, DeepSeek V4.1 Flash | 128,000 (+1,283; config 129,280) | Vocab and merges identical to each other and to DeepSeek-V3.2 / V4-Flash. Added tokens differ in 9 IDs between V4 and V4.1 |
| Kimi tiktoken | Kimi K3, Kimi K2.7 Code | 163,584 ranks (config 163,840) | `tiktoken.model` byte-identical, and identical to Kimi-K2-Instruct |
| K2 Horizon | K2 Horizon 375B, K2 Horizon MoVA 36B | 250,000 (+626; config 250,624) | Byte-identical file |
| o200k | Inkling, Inkling Small | 199,998 (+60; config 201,024) | Byte-identical file. Base vocab and merges identical to OpenAI o200k as shipped in `openai/gpt-oss-20b`. Special tokens differ |
| MiniMax | MiniMax-M3 | 200,000 (+61; config 200,064) | Identical vocab and merges to MiniMax-M2.7 |
| Motif | Motif 3 | 220,160 | Unique |
| Hy | Hy3 | 120,000 (+818; config 120,832) | Unique. Same as Hy3-preview. Uses the DeepSeek pre-tokenizer regex but a different vocab |
| Solar | Solar Open2 250B | 196,608 | Unique. Same merges as Solar-Open-100B, but 45 vocab entries differ |

So the 20 models use **11 distinct tokenizers**. Seven tokenizer groups contain 2-4 of the top-20 models each (pairs or quads), and 4 models have a tokenizer of their own.

Across groups, token IDs are essentially unrelated. The largest cross-group same-ID overlap is MiMo/Qwen3 vs GLM-5 at 7,952 tokens. Every other pair is under 300. Byte-string overlap between vocabularies is 44-92% (share of the smaller vocab); Qwen3.5 and GLM-5 share 92% of GLM-5's tokens as strings but only 281 at the same ID. The 248k Qwen3.5 vocab is **not** an ID-preserving superset of the 151k Qwen3 vocab (281 shared IDs). Any cross-family translation needs a token-alignment step.

Pairs where the KV translator needs no token remapping (same tokenizer, different architecture):

- Qwen3.8-27B ↔ Qwen3.8-Flash-Next ↔ Qwen3.8-2.4T ↔ Nex-N2-Pro. All GDN hybrids, with different head counts and depths.
- GLM-5.3 (MLA + DSA, 78 layers) ↔ GLM-5.3-Flash (KDA + NoPE MLA, 45 layers).
- Kimi K2.7 Code (MLA + RoPE) ↔ Kimi K3 (KDA + NoPE MLA).
- DeepSeek V4 Pro ↔ V4.1 Flash.
- MiMo Pro ↔ Flash.
- K2 Horizon 375B ↔ MoVA 36B.
- Inkling ↔ Inkling Small.

## KV bytes/token range (bf16, growing part)

- Smallest: **2,560 B** for DeepSeek V4.1 Flash, 3,200 B with indexer keys (inferred from the reference code; the official FP4 figure is 890 B). Next is DeepSeek V4 Pro at 7,928 B (+1,920 indexer).
- Largest: **327,680 B** for Hy3 (80 full-GQA layers). Next are K2 Horizon 375B at 249,856 B and K2 Horizon MoVA at 196,608 B.
- Median: about 47 KB/token. Inkling is 45,056 and Solar Open2 is 49,152.
- That is a 128x spread across the top 20, or 102x counting DeepSeek V4.1's indexer keys.
- At 128K context, the total cache including fixed state ranges from 0.40 GiB (DeepSeek V4.1 Flash) to 40 GiB (Hy3).
- Fixed per-sequence state ranges from 0 (pure GQA/MLA) up to 298 MB (Qwen3.8-2.4T GDN state at bf16, 2x at fp32), 232 MB (Kimi K3 KDA) and 231 MB (Inkling SWA buffers).

## What runs on the 128 GB EVO-X2 (llama.cpp, Q4_K_M)

- **Fits:** Qwen3.8-27B (~16 GB, `qwen35`, already the kvpark Qwen server). K2 Horizon MoVA 36B A4B (~22 GB, but it needs the MBZUAI-IFM llama.cpp fork or open PR #29535).
- **Borderline:** Qwen3.8-Flash-Next (~109 GB including the 51B-param n-gram table and MTP; `qwen4exp` is in master).
- **Too large at Q4_K_M:** the other 17 models. At about 2-2.5 bits/weight (IQ2/Q2_K class), the 250-320B models would land around 63-100 GB: Solar Open2, Inkling Small, Hy3, MiMo-V2.6-Flash, Motif 3, GLM-5.3-Flash. Of those, only Hy3, MiMo-V2.6-Flash and GLM-5.3-Flash have mainline llama.cpp support.
- **Not in llama.cpp master as of 2026-10-02:**
  - DeepSeek V4.1 Flash: PR #28696.
  - Motif 3: PR #26298.
  - K2 Horizon: PR #29535 / IFM fork.
  - Inkling: PR #25731.
  - Solar Open2: feature request #26115; a community patch ships in `prometheusAIR/Solar-Open2-250B-GGUF`.

## Data I could not verify (marked `*` / `inferred` in the JSON)

1. **Ranking date.** AA shows no "last updated" date. The scores are as served on 2026-10-02, with index v4.3. The open-weights flag and the dedup of effort variants follow the AA payload; I did not read AA's per-model pages.
2. **GLM-5.3 params (744B / 40B).** These are carried over from the GLM-5 card. The GLM-5.3 card states no count, and its config shapes match GLM-5. The HF safetensors total is 753.3B including MTP.
3. **Nex-N2-Pro params (397B / 17B)** are taken from its base, Qwen3.5-397B-A17B.
4. **DeepSeek V4.1 Flash bf16 KV (2,560 + 640 B/token)** is derived from `inference/model.py` (kv_source / index_source layers). The tech report PDF was not read. Only the official 890 B/token (FP4) is first-party.
5. **Indexer cache sizes** for GLM-5.3-Flash (k-pool, 704 B) and Qwen3.8-Flash-Next (QSA, 3,072 B) are inferred from configs and llama.cpp code. llama.cpp also stores pooled keys, so real storage differs.
6. **Solar Open2 linear-attention state** (64 heads x 128 x 128, KDA-style) is inferred from config field names. The repo has no modeling code, and I did not read the tech report PDF.
7. **Inkling positional scheme and layer roles** come from `transformers` main (`models/inkling/modeling_inkling.py`), not from the HF repo, which ships no code. The meaning of `dense_mlp_idx=2` is inferred.
8. **Motif 3 latent-cache size** assumes MLA-style latent caching, which matches the card's vLLM `FLASH_ATTN_MLA` recommendation. HF eager caching is 8.9x larger.
9. **Recurrent-state dtype.** Sizes are given at bf16. Kernels commonly keep the GDN/KDA matrix state in fp32, which doubles it.
10. **Q4_K_M sizes** are estimates (4.85 bpw x stored params), not measured GGUF file sizes. Whether a quant fits 128 GB depends on the iGPU memory carve-out and the context length.
11. **Release dates** are AA's release dates. The HF repo creation date is in the JSON (`hf_created_at`).
12. **Not read:** the technical-report PDFs (MiMo V2.6, DeepSeek V4.1, Motif 3, Solar Open 2). For the DeepSeek V4 family, DeepSeek-V4-Pro's own card only gives totals, so V4 numbers come from config.json and `inference/model.py`.
