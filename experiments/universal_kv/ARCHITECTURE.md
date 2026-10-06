# Universal KV translator: architecture (2026-10-05)

Goal: let a person switch a live conversation between open models on one machine (an
EVO-X2-class box: Ryzen AI Max+ 395, 128 GB unified memory, Radeon 8060S, llama.cpp on
Vulkan/ROCm) without paying a full prefill on the new model, and park/resume any
conversation on any model. kvpark already parks and restores a model's own slots; this adds
**translation** between models.

Evidence this builds on (experiments/linear_transfer): a pairwise ridge map
Qwen3.8-27B → Muse-Glimmer-30B, used as "mapped old region + native tail of 2048 tokens",
meets ΔNLL < 0.05 nats/token on chat at 4–13K tokens, fails on long documents and on agent
traces (two-thirds of the agent loss is in table-filled template/unmatched cells), and on a
real llama-server restores in ~30 ms and hands over in 21–28 s at any length (native prefill
34–188 s).

Measurements: [RESULTS.md](RESULTS.md). The product side: [docs/bridge.md](../../docs/bridge.md).

## 1. Why a hub, not pairs

N models need N(N−1) pairwise maps (380 for 20 models), each trained on paired captures of
both models. A hub ("KV interlingua") needs 2N adapters: an encoder E_m from model m's cache
into a shared per-token latent z, and a decoder D_n from z into model n's cache. Any pair is
D_n ∘ E_m. Onboarding a new model needs only its own captures on the **hub corpus** (public
text plus its stored hub latents) — no other model, no GPU beyond running the new model once.

## 2. What must be translated (state kinds)

The top-20 open-weights models (Artificial Analysis Intelligence Index v4.3, read
2026-10-02; [survey/SURVEY.md](survey/SURVEY.md)) reduce to five state families, plus
per-token side caches. A conversation's state in llama.cpp is a slot file: token ids plus
these sections.

| Family | Top-20 members | Per token? | Translation strategy (status) |
| --- | --- | --- | --- |
| F1 GQA K/V in every layer (RoPE full/partial) | Hy3, K2 Horizon 375B / MoVA 36B, MiniMax-M3; local: Qwen3-4B/8B, Llama, Mistral | yes | hub decode per token; K de-rotated into the hub, re-rotated with the target's RoPE (full, partial, NeoX, interleaved, M-RoPE). Implemented for `qwen3`; measured on Qwen3-4B |
| F2 sliding window + global GQA | MiMo V2.6 Pro/Flash (window 128), Inkling (512, no RoPE, conv on K/V), Motif 3; local: Muse (2048), Gemma 3, gpt-oss | global layers yes; window layers last W only | global layers from the hub; window layers decoded for their last n_swa + 1 positions only; as a **source**, only the global layers exist for old tokens, so the hub keeps a global-layers-only encoder per such model. Implemented for `muse-glimmer`; measured on Muse |
| F3 MLA latent (512 + 64 rope, or NoPE) | Kimi K2.7, GLM-5.3, Motif 3; NoPE MLA in Kimi K3, GLM-5.3-Flash | yes | decode the K row; the latent c_kv goes back on its RMS shell, the rope part is re-rotated (YaRN included); the up-projections stay in the target. Implemented; tiny fixtures only |
| F4 compressed / pooled latent | DeepSeek V4 Pro, V4.1 Flash | window only (pools of 2–128 tokens beyond) | as a target, pools synthesized from per-token compressor inputs the hub decodes ([section 10](#10-deepseek-v4-as-a-target)); as a source, native prefill. Implemented for `deepseek4` (not V4.1, which the pinned llama.cpp lacks); tiny fixture only |
| F5 linear-attention / recurrent hybrid | Qwen3.8-27B/2.4T/Flash-Next, Nex-N2 (gated DeltaNet), Kimi K3, GLM-5.3-Flash (KDA), Solar Open2 | per-token cache in ~1/4 of layers; a fixed recurrent state (59–298 MB) | per-token layers from the hub; the recurrent cell starts from zeros (R0) or the model's mean of native cells, and the native tail carries it on ([section 7](#7-recurrent-state-hybrid-targets)). Implemented; measured on Qwen3.8 and Ornith, other hybrids on fixtures |
| side caches | sparse-attention index keys (GLM-5.3 DSA, MiniMax-M3, Qwen3.8-Flash-Next) | yes | DSA indexer keys: zeros while the prompt fits in the indexer's top_k, else a per-model ridge from the decoded K row (the hub's indexer ridge). Implemented for the DSA family; fixtures only |

Cross-cutting facts that shape the hub:

- **Tokenizers:** 11 distinct tokenizers among the 20; token ids are unrelated across
  families (largest same-id overlap 7,952). Translation must align by text (byte spans),
  never by id. Families that share a tokenizer (the Qwen3.5/3.8 line, GLM-5.3 pair, Kimi
  pair, DeepSeek pair) translate token-for-token with no fill cells.
- **Positions:** NoPE (5 models), partial NeoX (Qwen 64/256, MiMo 64/192, MiniMax 64/128),
  full NeoX, decoupled 64-d interleaved on MLA keys, and per-layer-type thetas. The hub
  stores position-free K (de-rotated with the source's own scheme); the target re-rotates.
- **Cache size per token** spans 2.5 KB (DeepSeek V4.1) to 320 KB (Hy3), median ~47 KB.
- **On an EVO-X2 (128 GB):** of the top 20 only Qwen3.8-27B fits comfortably at Q4_K_M
  (~16 GB); K2 Horizon MoVA 36B (~22 GB) needs an unmerged llama.cpp PR; Qwen3.8-Flash-Next
  is borderline (~109 GB). The only DeepSeek V4 that fits is V4-Flash-0731 at UD-IQ2_XXS
  (~91 GB, with both resident servers stopped; size from the download page, not re-checked).
  Real local use mixes these with smaller dense and sliding-window models (Qwen3 4–32B,
  Gemma 3, Muse, gpt-oss-20b).

### What the translator supports today

Per llama.cpp architecture (the GGUF's `general.architecture`). The state codec is
`experiments/linear_transfer/kvstate.py` (layouts derived from GGUF metadata and checked
against a one-token native save of the target); the bridge's list is `TRANSLATABLE` in
`src/kvpark/bridge/capabilities.py`.

| Family | Architectures | What a translated state holds | Validated on |
| --- | --- | --- | --- |
| Dense GQA | `qwen3` | K/V of every layer for [0, T0), K rotated at the target positions | real model (Qwen3-4B-Instruct-2507), live switches |
| Sliding window + global | `muse-glimmer` | global layers for [0, T0) (NoPE); sliding layers for their last n_swa + 1 positions | real model (Muse-Glimmer-30B), live switches |
| Gated DeltaNet hybrid | `qwen35` | the attention layers (IMROPE, partial); one recurrent cell at T0 − 1 | real models (Qwen3.8-27B, Ornith-1.5-9B), live switches into the production Qwen3.8 build |
| GQA family | `llama`, `mistral3`, `qwen3moe`, `hy_v3`, `phi3` (none of them LongRoPE), `gemma3`, `gemma4`, `gpt-oss`, `laguna`, `mimo2` | K/V of every layer for [0, T0) with per-layer KV heads and head sizes; a sliding cache's last n_swa + 1 positions; per-layer-kind RoPE (the sliding layers' own base, scale and rotary width; Laguna's plain RoPE on its sliding layers; the GGUF's `rope_freqs.weight` frequency factors of Llama, Mistral 3 and Gemma 4's full layers, Gemma 4's proportional RoPE); Gemma 4's shared KV layers hold nothing of their own | test-llama-archs and gguf-py Gemma 3/4 fixtures on both pins (codec byte-identical, RoPE bit-exact against the CPU cache rows, identity-hub translation restores the native logits); real Gemma-4 26B-A4B and 31B saves parse and write back byte for byte, and a translated-shape state of the 26B restores token for token with its MTP head (CPU) |
| Other GQA hybrids | `qwen35moe`, `qwen3next`, `qwen4exp` (with its block indexer), `minimax-01`, `jamba`, `falcon-h1`, `granitehybrid`, `nemotron_h`, `nemotron_h_moe`, `plamo2`, `lfm2`, `lfm2moe` | the same | random test-llama-archs fixtures |
| MLA | `deepseek2` (DeepSeek V2/V3, Kimi K2.x, GLM-4.7-Flash), `mistral4`, `hy_v4` | K-only rows [c_kv on its RMS shell, RoPE(k_pe)], no V; nextn layers as the zero rows a native save holds | tiny gguf-py and test-llama-archs fixtures |
| DSA (MLA plus indexer keys) | `deepseek32`, `glm-dsa`, `dots3note`, `hy_v4` with an indexer | as MLA, plus indexer keys: zeros while the whole prompt fits in top_k, the hub's indexer ridge beyond; refused without a ridge | fixtures |
| K-only hybrids (MLA + KDA) | `kimi-linear`, `kimi-k3`, `bailingmoe3` (M-RoPE on a7b94df2c when its GGUF has rope sections); `glm5-next` (a7b94df2c only: NoPE latents, k-pool indexer rows `[key \| gate \| pooled]`) | MLA rows and one recurrent cell; GLM5-Next's indexer keys and per-token gates as DSA's (zeros within top_k, the indexer ridge beyond) | test-llama-archs fixtures |
| Pooled latent | `deepseek4`, target only | raw window of the last n_swa latents, CSA/HCA/LID pooled rows and compressor rings, synthesized; T0 rounded down to a multiple of 128 | test-llama-archs fixture |
| Pure recurrent | `mamba`, `mamba2`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` | nothing: there is no per-token cache, so the bridge skips (`no-per-token-cache`) and the target prefills natively | the codec parses and writes them back byte for byte |

What "fixtures" means here (`test_families.py`). Each fixture architecture but `dots3note` and
`deepseek4` ([section 10](#10-deepseek-v4-as-a-target)) was translated into itself through an
exact identity hub: the written state equals what kvstate's own operations make of the native
one (de-rotation, latent shell, re-rotation, indexer keys), and it restores in llama.cpp to the native logits (bit-identical with f16 caches, within 3.6e-6
with f32 caches and flash attention off; the hybrids with their native recurrent rows in place,
since R0 changes the logits by design). The logits of the qwen3next, qwen35moe and qwen4exp
fixtures barely depend on attention even when amplified, so only their states are checked.
dots3note's sliding layers keep too few source cells for an identity test; it is checked through
a hub fitted with `fit_hub.py` on probe captures of random-weight models, which translates `qwen3`
to and from `deepseek2`, `deepseek32` and `dots3note`: shapes, positions, the restore and
decoding are checked, quality is not. For the hybrids the tests prove layouts and acceptance,
not the cost of R0. Real MLA (DeepSeek-V2-Lite-Chat, GLM-4.7-Flash), KDA-hybrid (Kimi-Linear,
Ling-3.0-tiny), other hybrid (LFM2, granite-4.0-h, Falcon-H1, Jamba, Nemotron-3.5) and Gemma 4
models are now local and smoke-tested at a7b94df2c (evidence/models/README.md), and DeepSeek-V4-Flash
and Laguna are on disk but not loaded; none of them has been captured or translated yet, so
real-model quality for those families is unmeasured (no DSA model fits this machine).

Of the top 20, eleven are translatable (Qwen3.8-27B measured; Qwen3.8-2.4T and Nex-N2-Pro as
`qwen35moe`, Qwen3.8-Flash-Next as `qwen4exp`, GLM-5.3 as `glm-dsa`, Kimi K3 as `kimi-k3`, Kimi
K2.7 Code as `deepseek2`, MiMo-V2.6 Pro and Flash as `mimo2`, Hy3 as `hy_v3` and GLM-5.3-Flash as
`glm5-next`, a7b94df2c only, on fixtures only) and DeepSeek V4 Pro is a target only (fixture
only). MiniMax-M3 (`minimax-m3`) is in llama.cpp but has no state codec yet (its
`llama_kv_cache_msa`: a GQA cache plus a block-sparse indexer cache with K and V rows), and the
other seven (DeepSeek V4.1 Flash, Motif 3, both K2 Horizon models, both Inkling models, Solar
Open2) are not in either pinned llama.cpp. Locally, Gemma 3 and gpt-oss have codecs now too
(fixtures; no local weights).

A model can be ruled out by its GGUF beyond its architecture: LongRoPE (frequency factors
`rope_factors_long`/`_short`, chosen by the context size: the long-context Phi-3/4 models, and
Llama or Mistral 3 converted with a `longrope` scaling type) and embedding models (a vocabulary that appends EOS to
every text, such as gte-Qwen2) are skipped by the bridge (`rope-factors`, `embedding-model`).
`plamo2` caches are translated, but the engine does not read its PLaMo-2 vocabulary yet, so the
bridge skips switches with it (`tokenizer`).

### Both pinned runtimes

The repository's runtime moved from llama.cpp 85d5703a3 to a7b94df2c (`runtime/` is a
`scripts/build_runtime.py` build of a7b94df2c since October 5, 2026; the 85d5703a3 build is kept at
`runtime-85d5703a3/`), while the production Qwen3.8 server still runs 85d5703a3. Every layout
above is read and written on both pins
([STATE_FORMAT_a7b94df.md](../linear_transfer/STATE_FORMAT_a7b94df.md)). They differ in three
places: deepseek2/mistral4 nextn layers (kept by 85d5703a3's plain cache only), qwen4exp's indexer
rows (V rows on 85d5703a3, `[key | pooled]` on a7b94df2c) and bailingmoe3 with rope sections
(M-RoPE cells on a7b94df2c); GLM5-Next exists on a7b94df2c only. A state never names its pin, so
kvstate reads it with the spec's revision, then the other one, and the parsed state's spec is the
reading that fits; the translator takes the target layout from the template state's reading and
the source RoPE from the source state's. A job may name each server's `build_info` (`/props`,
which the bridge adds), and the engine's specs then take that pin (`engine.job_revision`); the
daemon's memory guard sizes a translated state by the larger of the two layouts. The capture and
state tools (`kv_probe.cpp`, `state_driver.cpp`, `vocab_tool.cpp`) build against either pin with
`build_collector.py`, whose manifest records the pin; every experiment suite passes with either
pin's binaries ([Tests on both pinned runtimes](../linear_transfer/README.md#tests-on-both-pinned-runtimes)).

### Vocabularies and chat templates

Translation aligns the two models by text, so the engine must spell each side's tokens exactly
as llama.cpp prints them and render each conversation exactly as llama-server does:

- **Vocabularies** (`engine.TOKENIZER_MODELS`): GPT-2 byte-level BPE (Qwen, Muse, DeepSeek, GLM,
  Kimi, ...), Gemma 4's BPE (U+2581 for a space, `<0xNN>` byte tokens) and SentencePiece (Jamba).
  A SentencePiece tokenizer with `add_space_prefix` puts a space before the text that starts a
  prompt and after each special token; `engine.spelled_pieces` drops it from both sides' pieces
  (and kv_probe's `tokenize` mode does the same), so the pieces join to the text. Every local
  vocabulary (19 models, about 3 million ids) matches llama.cpp's own (`vocab_tool.cpp`,
  `test_engine.RealVocabularyTest`) on both pins. Engines before the `vocabularies` feature read
  GPT-2 BPE only; the bridge skips switches into or out of other vocabularies for them
  (`tokenizer`).
- **Templates**: `chat.py` follows llama.cpp's jinja (printing, `tojson` floats, ASCII-only
  trimming, its filter set) and spans a segment inside the whitespace at its edges; a string
  argument a template writes JSON-escaped renders exactly without a span. `template_check.py`'s
  server mode compares it with `/apply-template` on 12 conversations × 8 template-option sets
  (96 cases) per model, on the real servers or on *tokenizer twins* (tiny models carrying the
  real tokenizer and template): parity 96/96 for all 19 local models on twins at both pins, and
  56/56 on the real servers on the earlier battery.
- **The source side**: the engine places the source's segments where the source's own template
  renders them in the slot's history (`source_spans`), and only the rest by search
  (`located_record`: the in-order occurrences that cover the most bytes); the report counts both
  (`source_segments_rendered`, `source_segments_found`).

## 3. The hub (implemented: hub.py, fit_hub.py)

- **Unit:** a *hub span*: a token that ends at the same byte of the same chat segment in
  every model's rendering (align.py's rule extended to N models; segments a template drops,
  such as earlier turns' reasoning, are simply not shared). With 4 models (3 tokenizers) 89%
  of the pairwise-aligned tokens remain hub spans (185,672 train spans).
- **Two hubs, one per kind:** keys and values are translated separately (K from K, V from V),
  each through its own latent of D = 2,048–4,096 dimensions per token.
- **Fitting (closed form, CPU, streaming float64 statistics over float32 chunks):**
  1. per model and layer: PCA of pre-RoPE K (or V) to ≤ 512 components;
  2. per model: PCA of the concatenated layers to the view dimension, whitened;
  3. multi-view CCA (MAXVAR) as a generalized eigenproblem: each view block is whitened with
     its own full-train covariance, so eigenvalues are sums of canonical correlations (≤ M).
     (A first version whitened on a subsample only; leftover view-specific variance then
     ranked as "shared" and cost 0.03 R². Regression test in test_hub.py.)
  4. encoders: ridge from each view to the consensus latent; models whose old tokens only
     keep their global layers (sliding-window sources) get a second encoder from those
     layers alone;
  5. decoders: ridge from the pooled encoder estimates of all *other* models to every KV
     layer of the model, so any source works at run time (ridge strength chosen on dev).
- **Onboarding without refitting (hub.add_model_kind, `fit_hub.py --base`):** a new model
  needs only its own captures on the hub corpus plus one or more pivot models' captures; its
  encoder regresses onto the pivots' latent estimates and its decoder from them. Existing
  adapters are untouched (test_hub.py checks bit-identity).
- **Growing the hub (campaign.py).** Every new model is onboarded against the same base
  (hub4-d4096, its four models as pivots), so its adapters depend on the base and its own
  captures only, never on another new model. Its bundle hard-links the base's files
  (`fit_hub.py --link-base`, after checking they hold the arrays in memory; `convert_hub.py
  --reuse`), and `campaign.py assemble` merges the base and any set of onboarded bundles into
  one hub per format (`hub.merge_bundles`: hard links, any order, no refit). Linked files are
  read-only (0444, every name of the inode, the base's included) and never cross a filesystem
  (a link there would be a copy). Two onboarded models translate between each other through the
  same latent; the onboarding tests each one against the pivots only, and `campaign.py assemble
  --evaluate` measures the pairs between onboarded models (they all capture the same dev and test
  documents). An assembled hub is never changed: onboarding a model again (`--redo onboard`)
  leaves it with the old adapters, which `campaign.py status` flags by sha256; the bridge moves
  to a newly assembled directory.
- **Other cache families in the same hub.** An MLA model's K view is its pre-RoPE row
  [c_kv | k_pe] and its V view is c_kv (the V that attention reads); into an MLA target only
  the K kind is decoded. A DSA model can carry an *indexer ridge* (`fit_hub.py --indexer-ridge
  NAME`): its pre-RoPE indexer key as a ridge regression on the same layer's decoded K row
  (R² 0.69–0.89 on fresh fixture prompts; above top_k its logits were closer to native than
  zero keys on every prompt). A DeepSeek V4 model carries *head kinds* `csa`, `hca`, `lid`
  (`fit_hub.py --heads NAME`): decoders from the hub latent to each compressed layer's
  compressor inputs [a | s], with dev and held-out test R² in the bundle (fitted on the
  fixture only). Models whose graphs name no pre-RoPE key (kimi-k3, bailingmoe3, hy_v4) get
  their views from f32 state saves (`collect_v2.py --mode states`).
- **Measured, Qwen3.8 → Muse, test tokens (2 views, D = 2,048):** K R² 0.750 vs 0.778 for the
  dedicated pairwise map (96%), V 0.604 vs 0.678 (89%); and the reverse direction, which no
  pairwise map existed for: Muse → Qwen3.8 K 0.673 / V 0.522 from Muse's 13 global layers.
  Muse's V spectrum is the bottleneck at D = 2,048 (87.6% of its variance kept); 4,096-dim
  views keep ~98%.
- **Measured, 4 models (D = 4,096):** all 12 directions at K R² 0.67–0.77 / V 0.49–0.67; for
  Qwen3.8 → Muse 97% / 93% of the pairwise map. The top canonical eigenvalues are 3.99 / 3.98 of
  a maximum of 4: the four models share most of their cache geometry. Onboarding a model into
  an existing hub matches a joint refit when decoding into it (±0.003) and loses 0.02–0.03
  when encoding from it.
- **Bundles.** Format 1 (`save_hub`: one npz per model and kind; hub4-d4096 is 4.9 GB) is what
  fitting writes; `convert_hub.py` turns it into format 2 (one flat C-contiguous `.npy` per
  model, kind and role, the encoder fused as `(basis2 * scale) @ encoder`, float16 copies for
  the GPU, sha256 per file; 7.9 GB for hub4-d4096, of which the `view` role that only refitting
  needs is about 2.2 GB). A switch reads only the two sides it needs: Qwen3.8 → Muse reads
  0.9 GB. Indexer ridges and head kinds are carried into format 2.

## 4. Cells the hub does not cover: native prefix and fill

Template, special and unmatched tokens differ between models' chat templates and tokenizers.
v3b showed they decide agent-trace quality, and the first tokens of a prompt matter most of all:
they are the attention sink every layer leans on.

1. **Native prefix (implemented; exact; opt-in per target).** K/V of a prompt's first H tokens
   depend only on those tokens, so the target can prefill them itself: the switch prefills the
   system message (at least the first 4 tokens) natively once, caches the slot file by a digest
   of the token ids (agents reuse one system prompt and tool block across conversations; cache
   capped at 2 GiB) and the translator copies those cells exactly
   (`translate(..., native_state, native_prefix)`). Measured effect depends on the target: on
   Ornith it removes a catastrophic failure of a mean-filled prompt start (information-free
   floor +0.74 → +0.05 nats/token) and is neutral for the hub (+0.016 → +0.027); on Muse it
   *hurts* (hub +0.029 → +0.083). So it is enabled per target model
   (`kvpark bridge add --native-prefix`), not by default. A DeepSeek V4 target takes none (its
   cache is synthesized whole; registration refuses the flag).
2. **Token tables (implemented, experiments only).** Per model, mean native K/V per token id
   over the rows the hub does not cover (token_tables.py, from the hub corpus captures);
   `fill="table"` in `translate` and `pair_eval.py`. The bridge fills with the decoder mean
   (`--fill mean` is the only choice there).
3. Structural map (template tokens aligned by message structure and decoded from the source's
   corresponding template tokens) and byte-overlap map (unmatched target tokens from the source
   tokens covering the same bytes): future work.

## 5. Runtime: a switch (implemented: `kvpark bridge switch`, Hermes `/kvpark switch`)

```
kvpark bridge switch --from A --to B     (or /kvpark switch B in Hermes, then /model B)
  checks: embedding models and LongRoPE, architectures, vocabularies vs the engine's features,
          media vs B's modalities, prompt length, DSA indexer keys, DeepSeek V4 head
          decoders (src/kvpark/bridge/capabilities.py): else skip
  B: /apply-template + /tokenize (what B will process, with the request's template options);
     layout template: a cached one-token native save (and its .resume on kvpark-patched
     servers, which tells the draft model's kind)
  A: save slot (0.3-0.8 s)
  B: native prompt start if enabled for B (cached by token digest)
  engine daemon: render B's prompt as B's server does, decode A's slot tokens with A's
     vocabulary, locate the messages, align, encode (E_A) → decode (D_B) → fill → B's slot
     file holding [0, T0)        0.45-1.1 s on the GPU for 10-15K tokens, warm and quiet
  bridge: B's .resume with a synthesized draft state (patched servers)
  B: erase → restore (0.02-0.3 s) → B prefills the native tail on the next request
     (2,048 tokens: 8-16 s on the 27-30B models, 4 s on Qwen3-4B)
```

T0 = min(P − tail, the first token of the newest message, the start of the first message
with media), P the target prompt length, tail 2,048 by default (per model `--tail`); a target
with a DFlash draft keeps at least window − 1 tail tokens ([section 9](#9-speculative-decoding-dflash-mtp)),
and a DeepSeek V4 target rounds T0 down to whole 128-token blocks.

- Short prompts (fewer than `min_handover`, default 1,024, tokens left to translate) are not
  translated; the target prefills (`short`).
- Correctness: llama-server reuses only cells whose token ids match the request, so the
  target always conditions on the right tokens; translation only changes old K/V values.
- Rendering: the request's `chat_template_kwargs` and `reasoning_effort` change what the target
  processes (Qwen3.8 writes the reasoning effort into its system prompt). The engine renders the
  target with them as llama-server builds its template context and checks that the server's
  tokens spell that rendering byte for byte; a server run with template flags the bridge was not
  told about ends in `template-mismatch`, a skip.
- Every switch the translator cannot handle is a skip with a reason code, never a forced
  translation; the target then prefills natively (the codes: docs/bridge.md).
- Undo: `kvpark bridge undo B` erases B's slot; the next turn prefills natively.
- Parking keeps working as before; the bridge reads a model's slot through the server.

## 6. The engine: GPU backends and a resident daemon

The first live switches spent 6.5–42 s in a one-shot numpy engine process, mostly loading and
hashing the whole hub, decoding sliding layers for cells they do not keep, and (once) running
on a reference-BLAS numpy. The engine is now a resident daemon with pluggable arithmetic.

**Translation stages** (`translate.py`). `prepare` aligns the two renderings and decides which
source row feeds which target cell and which positions each target section keeps (index
arithmetic only, 3–8 ms); `compute` encodes the mapped source rows with the fused encoder and
decodes each section only for the positions it keeps (a sliding section its last n_swa + 1);
`assemble` puts MLA latents back on their shell, fills indexer keys, rotates K at the target
positions, casts to the cache type and copies the native prefix. compute and the per-row part
of assemble run window by window of target positions (4,096 rows on a GPU, 16,384 on numpy), so
memory beyond the output does not grow with T0. `state_writer.py` writes the result part by
part: the bytes of kvstate's own serializer, 0600, atomic, no fsync (1.03 GB in 0.17 s).

**Backends** (`backends/`).

- `numpy`: float32, the reference and always available. RoPE is kvstate's own (bit-identical
  to llama.cpp's CPU build); per-layer gathers, rotations and casts run on as many threads as
  BLAS. BLAS threads default to one per physical core, at most 16 (the servers are idle during
  a switch; 16 threads translate 24–30% faster than 8 on this machine).
- `torch`: float16 weights and activations with float32 accumulation (hipBLASLt/cuBLAS;
  bfloat16 is never used, 8× the error), 25–35 TFLOPS on the 8060S. RoPE is computed on the
  device from positions, never from tables (production contexts reach 262K): the forward
  rotation in float32 with `torch.addcmul`, one fused multiply-add as in the CPU build (on this
  ROCm build exactly the bits of the float64 emulation it replaced), the inverse in float64.
  Every rotation is checked against kvstate on first use; one that disagrees is refused and the
  job runs on numpy.
  Against numpy, decoded K/V are within 9.2e-4 of each layer's largest value (relative
  Frobenius 5.1e-4) and the seed-scored loss differs by −0.0014 [−0.0065, +0.0034] nats/token.
  While the GPU computes, four threads touch the fresh output pages (`translate.Prefault`).
- `backends.select("auto" | "cpu" | "gpu")`: auto takes torch when it imports and sees a
  device. A GPU failure redoes the job on numpy; one that looks like a broken device moves the
  daemon to the CPU until it exits. A DeepSeek V4 target always runs on numpy (it pools on the
  CPU).

**The daemon** (`engine_daemon.py`, client `src/kvpark/bridge/engine_client.py`, standard
library only).

- One process per bridge directory, started by the first switch (or by staging one in Hermes)
  with the configured engine Python. An AF_UNIX socket in the 0700 bridge directory, the peer's
  uid checked (SO_PEERCRED on Linux, LOCAL_PEERCRED on macOS), a flock against a second
  instance, one JSON object per line: `ping`, `load`, `translate`, `unload`, `stats`,
  `shutdown`. The job travels over the socket, so no conversation text is written to disk;
  the log holds timings and sizes only.
- One FIFO worker, so two switches never compete for the GPU; resident hub sides in an LRU per
  backend; recurrent mean groups kept per file identity ([section 7](#7-recurrent-state-hybrid-targets)).
- A memory guard compares each job's estimated peak, calibrated by the largest peak per moved
  byte seen so far, with MemAvailable and the GPU's free memory: it drops sides the job does
  not need, then moves a GPU job to the CPU, then refuses it (the switch is skipped,
  `engine-memory`). After each job the GPU allocator's cached blocks are returned (on unified
  memory they are the servers' memory).
- Idle: the sides are dropped after 120 s and the daemon exits after 600 s.
- Prewarm: staging a switch loads the pair's sides and runs the switch's products once on zero
  rows of its shapes, so the first job does not pay the GPU's kernel selection.
- Errors carry a kind (`job`, `template`, `memory`, `busy`, `internal`, `request`); only a
  daemon that is unavailable or at fault of its own makes the client rerun the job in a
  one-shot `engine.py JOB.json`. Windows has no daemon and always runs one-shot.
- `FEATURES` (`media-source`, `template-options`, `recurrent-mean`): what this engine handles
  beyond earlier ones; the bridge keeps the matching skips for an older engine.

Measured on a 13K-token conversation (run-v3b document 3b211f08, T0 12,962 / 13,307, hub4-d4096
in format 2; evidence/universal-kv/bench-engine-fix-20261003 and RESULTS.md section 6), engine
part only, seconds:

| Engine | Qwen3.8 → Muse | Muse → Qwen3.8 |
| --- | --- | --- |
| before this work (one-shot, numpy, format 1 hub, whole bundle loaded) | 13.3 | |
| one-shot process, numpy | 3.2 | 4.2 |
| one-shot process, torch (0.8–0.9 s of it GPU start) | 2.2 | 2.4 |
| numpy, 16 BLAS threads, warm (in process, as the daemon runs it) | 2.1–2.2 | 2.75–2.8 |
| daemon, torch, first job after a prewarm with warm-up | 0.51–0.52 | 0.76 |
| daemon, torch, warm | 0.45–0.47 | 0.72–0.78 |

In live switches under load the GPU engine took 0.5–3.7 s (stage 2, every ordered pair of
Qwen3.8, Muse and Qwen3-4B); into the dense Qwen3-4B, whose 1.3–1.6 GB states made the assemble
stage cost 0.5–2 s then, it now takes 0.77–1.10 s with assemble at 0.13–0.16 s (RESULTS.md
section 8).

## 7. Recurrent state (hybrid targets)

A hybrid target keeps one recurrent cell per sequence: per recurrent layer a short conv row R
(the last kernel − 1 projections), a long-memory row S (per head the delta-rule matrix for
GDN, the SSM state for Mamba) and, on qwen4exp, a PLE row. They summarize the whole history and
have no per-token form, so the translator writes one cell at T0 − 1 and the native tail
[T0, P) carries it on (`recurrent_init.py`):

| Strategy | The cell at T0 − 1 |
| --- | --- |
| zero (R0) | every row zero: exactly llama.cpp's fresh-sequence state. The default |
| mean | the model's mean of native cells over training conversations, from an npz built once per model (`r0_experiments.py calibrate`: native saves every 1,024 tokens, 227K tokens of prefill, 18–24 min for Qwen3.8 on the 8060S) |
| mean-match | the mean of the conversation's group: `chat` without tools; with tools the pooled `all`, or `tools` when that group was calibrated on the deployment's own agent prompts |

Measured on Qwen3.8-27B and Ornith-1.5-9B with the target's *native* attention cache, so only
the recurrent cell differs (12 held-out conversations of 8.5–16.8K tokens; RESULTS.md section
7): on chats R0 costs Qwen3.8 +0.053 / +0.039 / +0.029 nats/token at tails of ≈1.4K / 2.4K /
4.4K, mean-match +0.008 / +0.002 / +0.003; Ornith goes from +0.065 / +0.035 / +0.021 to
−0.006 / −0.005 / −0.005. Mean-match's KL to the native prefill is below R0's on 12 of 12
documents at every tail on both models, and with the mean a 1,024-token tail beats R0 with
≈4.4K. On agent traces the group matters: the pooled mean does nothing on Qwen3.8 (KL 0.063 vs
0.063), a `tools` mean calibrated on the traces' own prompt and tools cuts KL to 0.026 / 0.019 /
0.015. With translated attention (Muse → Qwen3.8) the mean takes 0.032–0.038 nats/token off
chats (+0.129 / +0.102 / +0.061 → +0.097 / +0.064 / +0.027); agent traces stay broken by the
translated cells (KL 0.40–0.50). The mean leaves 12–43% of the S energy unexplained, and the
native prompt start's own cell (prefix) gains little.

In the bridge: `kvpark bridge add NAME --recurrent-mean NPZ` (checked against the GGUF: the
architecture, the layer count and, for `qwen35`, `qwen35moe` and `qwen3next`, the recurrent
layers and row sizes; the engine checks the whole layout against the target's own save)
makes every switch into that hybrid start from the mean, with the group above;
`--recurrent-tools` allows the `tools` group. A file that lacks the group falls back to `all`,
and one that lacks that too, or cannot be read any more, gives zero with a note
(`report["recurrent"]`). Without a mean the cell is zero, as before. Tails are not shortened
automatically: a DFlash draft keeps its window rule, and a shorter `--tail` holds for chats (and
for tool conversations only with a matched `tools` group). Calibration files exist for Qwen3.8
and Ornith only; the other hybrids are translated on fixtures only.

Next lever (not built): R2, replaying the recurrence over per-token inputs decoded from the hub
(about 0.05 s per 1K tokens for Qwen3.8's 48 GDN layers on the GPU, inferred; the cost is in
decoding 8,288 floats per layer and token).

## 8. Images and audio

Cross-model image KV is not translated: the vision encoders differ (Qwen's ViT has 27 layers,
Muse's 50), and so do token counts and resize policies. Instead the translated cache
stays text-only and stops before the first message that holds media:

- The bridge renders each media part as the target's media marker (from `/props`; random per
  server process), exactly as llama-server renders the real request, and passes
  `max_t0_message` (the first message with media) to the engine; T0 stops at that message's
  start, and the target prefills it, the media and everything after natively at the right
  positions (M-RoPE included). The markers count on top of the tail, since the server puts the
  media's own cells there. No warm request is sent.
- Targets without the modality skip (`target-no-vision`, `-audio`, `-video`), as does a server
  that does not report its marker.
- A source whose server encoded the media (a vision source that answered after the image) is
  read up to its first medium (`kvstate.media_cut`, `truncate_state`); T0 also stops after the
  last message found in that text. kvstate parses media slot files (token words with the mtmd
  chunk table: grid, position type, the sha256 of the upload; never pixels).
- Measured live, Muse (no vision) → the production Qwen3.8 with its mmproj (RESULTS.md section
  6): every switch reused exactly T0 and the logs show no position or mtmd error. With the
  image in the newest message the answer equalled the native one; with the image in an earlier
  message, the translated prefix in front of it changed the outcome in 3 of 3 conversations,
  while the target's own cache restored the same way answered like the native prefill. The
  restore places media correctly; what follows depends on the translated prefix's quality.
- Audio in the local setup is transcribed to text before llama-server sees it. Audio, video and
  HunyuanVL chunks in slot files are covered by hand-built blobs only; a source-media switch
  between two vision models has not run live (only Qwen3.8 has vision locally).
- Later: "native islands" (the target prefills each image natively and translated text resumes
  after it at shifted positions).

## 9. Speculative decoding (DFlash, MTP)

kvpark-patched servers (and the production Qwen3.8 build) save `<slot>.resume` with the draft
model's state and refuse a restore without it. The bridge reads the draft kind from the target
template's own `.resume` (no draft blob; an iSWA blob with an empty base: DFlash/DFlash2/DSpark;
a one-layer plain KV blob: MTP), on upstream servers from `/slots` (`/props` always reports
none), and writes a `.resume` for the translated slot: the synthesized draft state, an empty
speculative state, no checkpoints.

- **DFlash/DFlash2.** The draft's K/V at a position depend only on the target's hidden states
  there, and every layer slides over a 2,048-token window. The bridge keeps the native tail at
  least window − 1 long (`--draft-window` for other windows), so the draft's visible window is
  native at the first draft, and writes an "aligned dummy": two zero cells at T0 − 2 and T0 − 1
  in the template's layout (41,216 bytes for the production DFlash2 draft), which keeps the
  draft's position counter equal to the target's. On the production build: 37% of drafted
  tokens accepted after a translated restore against 41% natively; with the target's own cache
  restored the same way, 43.4% with the dummy against 43.7% with the server's real draft state,
  so the dummy itself costs nothing measurable. No draft-position warning in any run.
- **MTP.** The head keeps its own full-attention cache; zero cells would be attended forever, so
  the bridge restores it empty (a 16-byte marker) and the head sees only the native tail and
  what follows. Acceptance 48% against 61% natively; the empty state alone costs about 5
  points (55% against 60% with the real state, same text). Translating the nextn layer as one
  more hub layer would recover that (not built).
- **Shared** (Gemma 4's `gemma4-assistant` MTP head): its cache views the target's last sliding
  and full layers and writes no cells of its own, so its draft blob is the 8-byte header
  (`resume.classify_draft` kind `shared`) and a translated restore keeps the same 8 bytes; the
  head then reads the restored target cache. Only `draft-mtp` runs it (`switch.DRAFT_TYPES`): a
  server naming another speculative type with that blob is skipped. On the real 26B (CPU) a
  translated-shape state continued token for token like the resident slot with the head
  (acceptance as resident).
- **Skipped** (`draft-unsupported`): full-attention DFlash drafts, plain and EAGLE3 draft models
  whose state looks like MTP's, hybrid drafts, and any draft on a DeepSeek V4 target.
- Upstream servers neither save nor restore draft state: the draft rebuilds from the native
  tail. The bridge always erases the target slot before restoring, which also clears the
  draft's cache.

## 10. DeepSeek V4 as a target

A DeepSeek V4 (`deepseek4`) slot keeps per-token latents only for its last n_swa (128) tokens.
Older tokens live in pooled rows: CSA (ratio 4, overlapping 8-token windows) with its
lightning indexer (LID, Hadamard-rotated), HCA (ratio 128), plus float32 rings of the unfinished
block's compressor inputs. A pooled row is a per-channel softmax-gated sum of per-token a_t over
its block, then RMSNorm and RoPE at the block start. The pools are functions of every old
token's (a, s), so they cannot be rebuilt from the window, but they can be synthesized:

- `dsv4_pool.py` pools rows from per-token (a, s) with the GGUF's APE and norm tensors;
  `dsv4_target.py` builds the state for [0, T0) with T0 rounded down to a multiple of 128 (no
  HCA block is then open, so the HCA ring is never read and the CSA/LID rings need only half
  their channels). The hub's head decoders give (a, s); the K kind gives the window latents.
- Exactness on the fixture (test_dsv4.py): a state synthesized from the native run's own
  per-token inputs restores within 9.3e-10 of a restored native save (logit scale 9e-3; q8_0
  caches identical); zeroing the window, any pooled rows or the CSA/LID rings instead moves them
  by 2.1e-7 to 1.5e-5 (the HCA ring is never read at a 128-aligned split). The codec
  round-trips every save layout (f16, f32, q8_0, `-np 2`, rollback planes) byte for byte.
- In the bridge: target only (a DeepSeek V4 source is skipped, `dsv4-source`); the hub needs
  all three head kinds for it (`no-head-decoders`); only a server without a draft (DSpark and
  MTP drafts are skipped); no native prefix; always numpy. A switch from a qwen3 fixture at
  T0 = P − 40 = 332 restored 256 cells and reused 256.
- Not validated on a real model: head quality, the HCA gate's sensitivity to decode error, and
  native V4-Flash prefill speed (which decides when synthesis pays) are unmeasured. Pooling in
  numpy took about 21 s for 32K tokens at V4-Flash geometry, before the hub decode; real use
  needs a float32 GPU path. DeepSeek V4 with images is untested; V4.1 is not in the pinned
  llama.cpp.

## 11. Machine budget (EVO-X2 class, measured)

- **Hub bundle:** 4.9 GB for four models at 4,096 dimensions in format 1 (per model and kind:
  view transforms, encoder, one decoder per KV layer; 0.29–1.05 GB per file, largest for the
  36-layer Qwen3-4B), 7.9 GB in format 2; 1.1 GB for two models at 2,048. A switch reads one
  pair: 0.40 s on first use (sha256 checked once per file identity), 0.03–0.05 s afterwards.
  Token tables: 0.3–0.9 GB per model. Recurrent means: 78 MB (Qwen3.8) and 26 MB (Ornith) per
  group in float16.
- **Translation:** 0.45–1.1 s per switch on the iGPU for 10–15K-token conversations with a warm
  daemon on a quiet machine (up to 3.7 s in stage 2's live runs under load), 2.1–2.8 s on the
  CPU with 16 BLAS threads (2.6–13.6 s live, under load and with Qwen3-4B on either side). A
  cold daemon (start, torch import, first job) took 2.5–2.7 s, and the first prewarm of a new
  bridge directory 20–22 s (GGUF profiles and hub hashes, once).
- **Memory:** the resident hub pair takes 0.39 GiB; a warm GPU switch of 13K cells peaks at
  1.3 GiB of host RSS and 1.5–3.1 GiB of GPU memory, and the daemon holds 1.2–1.5 GiB between
  switches. The output itself is written whole from host memory (1.0–1.6 GB at 13K; about 19 GB
  for Qwen3.8 at 262K, estimated). Qwen3.8 (production, context 65,536), Muse and Qwen3-4B plus
  two engine daemons left 20–27 GB available.
- **Fitting:** a 4-model hub takes ~2 h on the CPU (185K training spans), onboarding one model
  ~15 min; capturing a model on the hub corpus takes ~15 min on the GPU for 4–9B models.

## 12. Implementation map

| Piece | Where |
| --- | --- |
| captures, state codec (every layout above, both pins, media slots, cuts), RoPE, chat rendering with llama-server's template context and jinja, template parity, llama.cpp's vocabularies, probe and state driver | experiments/linear_transfer (kv_probe.cpp, state_driver.cpp, vocab_tool.cpp, kvstate.py, chat.py, template_check.py, collect_v2.py, check_model_support.py) |
| hub fit, onboarding, indexer ridges, DeepSeek V4 head kinds, evaluation; bundles (format 1 and 2) | experiments/universal_kv/hub.py, fit_hub.py, convert_hub.py, token_tables.py |
| generic translation (any supported source → any supported target, layout from a target template state) | experiments/universal_kv/translate.py, backends/, state_writer.py |
| DeepSeek V4 targets | experiments/universal_kv/dsv4_pool.py, dsv4_target.py |
| recurrent cell strategies and their measurement | experiments/universal_kv/recurrent_init.py, r0_experiments.py |
| engine (live conversation: source slot + target server tokens → target slot) and its daemon | experiments/universal_kv/engine.py, engine_daemon.py |
| product: model registry, capability rules, `.resume` draft states, llama-server client, engine client, switch orchestration, CLI | src/kvpark/bridge (`kvpark bridge add / engine / engine-status / engine-stop / switch / undo / status`), docs/bridge.md |
| evaluation: end-to-end ΔNLL for any pair; engine benchmarks; live multi-server tests | experiments/universal_kv/pair_eval.py, score_translate.py, bench_engine.py, live_bridge.py |

## 13. Status and open problems (2026-10-05)

Works today on an EVO-X2-class box, measured end to end on real models: Qwen3.8-27B →
Muse-Glimmer-30B, → Ornith-1.5-9B and → Qwen3-4B are within 0.05 nats/token of native on
held-out chats (+0.029, +0.016, +0.008); live switches among Qwen3.8 (the production DFlash2
build), Muse and Qwen3-4B restored and reused exactly the translated cells in every switch that
was not skipped (24 of 24 in the hop table, and every later run), in 8–18 s against 15–86 s of
native prefill, with DFlash2, MTP, images and Hermes' template options. The engine is no longer
the cost of a switch: the native tail is. The other cache families of the pinned llama.cpp
listed in section 2 are translated on fixtures. Open:

- **Agent and document workloads:** tool output and long documents translate much worse than
  chat (agent traces +0.295 nats/token Qwen3.8 → Muse; KL 0.40–0.50 into Qwen3.8);
  template-aware fill (structural and byte-overlap maps) and task-aware (attention- or
  loss-trained) maps are the next levers.
- **Sources matter:** a small model's cache (Qwen3-4B) carries little of what a larger model
  needs (0–15% of the gap recovered); prefer the larger model as the source.
- **Real models for the new families:** MLA, DSA, KDA hybrids, the other GQA hybrids, the GQA
  family (but Gemma 4's real saves) and DeepSeek V4 are validated on random-weight fixtures
  only. The local models are downloaded and smoke-tested (DeepSeek-V2-Lite, GLM-4.7-Flash,
  Kimi-Linear-48B, Ling-3.0-tiny, LFM2, granite, Falcon-H1, Jamba, Nemotron-3.5, Gemma 4 26B/31B;
  DeepSeek-V4-Flash and Laguna not loaded, about 100 GB each); capturing them and onboarding them
  into the hub is the next step. No DSA model fits this machine; MiniMax-M3's cache is not
  modelled.
- **DeepSeek V4:** target only, no drafts, numpy pooling (~21 s per 32K tokens at V4-Flash
  geometry); V4.1 waits for llama.cpp; as a source it needs a native prefill.
- **Recurrent state:** the mean is calibrated for Qwen3.8 and Ornith; their `tools` groups come
  from smolagents sessions, so Hermes' tool traffic gets the pooled mean, which on Qwen3.8 did
  no better than zero on agent traces. Calibrating on the deployment's own traffic (the bridge
  saves each source's native slot on every switch, so a running mean costs no prefill) and R2
  replay are the next steps.
- **Images:** positions are exact, but a translated prefix in front of an image degraded the
  answer in 3 of 3 conversations. Native islands and better translation are the levers.
- **Drafts:** MTP loses about 5 points of acceptance to its empty head cache; translating the
  nextn layer is not built. Draft kinds other than DFlash/DSpark, MTP and the shared kind
  (Gemma 4) are skipped.
- **Engine portability:** CUDA, Apple MPS and macOS's peer check are untested; Windows runs the
  one-shot engine; float32 RoPE bit-identity relies on `addcmul` fusing the multiply-add, as it
  does on this ROCm build. The ggml Vulkan backend (`backends/ggml_backend.py`, no torch) runs the
  same arithmetic on any Vulkan GPU.
- **Template settings:** a target server's own template flags (`--reasoning-effort`,
  `--chat-template-kwargs`, ...) must be registered with the model, or switches into it end in
  `template-mismatch`; a tool-calling trace into Qwen3-4B was skipped the same way (its tool-call
  rendering fails chat.py's check that message content renders unchanged).
- **Hub corpus:** 248 SmolTalk2 chats (~320K tokens); more and more varied text (code, tool
  use, documents) should raise quality, especially for values.
- **Hermes:** `/kvpark switch` and the middleware path are covered by integration tests with a
  fake engine; a live Hermes session has not been run yet.
