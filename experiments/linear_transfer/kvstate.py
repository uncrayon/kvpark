"""Read and write llama.cpp sequence-state data for the pinned runtime (85d5703a3).

Two containers carry the same memory payload:

* the raw blob of ``llama_state_seq_get_data`` (``parse_blob``/``StateBlob``), and
* the file of ``llama_state_seq_save_file`` (``parse_file``/``StateFile``), which is what
  llama-server's slot save writes and slot restore reads.

The payload is whatever ``llama_memory_i::state_write(io, seq_id, flags)`` emits, for one
sequence. A ``ModelSpec`` built from GGUF metadata (mirroring the loader and
``llama_model::create_memory``, src/llama-model.cpp:2230-2709) names the sections in write
order (``ModelSpec.layouts``):

    memory       GGUF architectures                            sections (flags 0)
    kv           qwen3, deepseek2, mistral4, hy_v4 without an   kv
                 indexer
    iswa         muse-glimmer                                   base, swa
    hybrid       qwen35 (Qwen3.5/3.8, Ornith), qwen35moe,       attn, recurrent
                 qwen3next, minimax-01, jamba, falcon-h1,
                 granitehybrid, nemotron_h(_moe), plamo2,
                 lfm2moe, lfm2 without a sliding window,
                 kimi-k3, kimi-linear, bailingmoe3
    hybrid_iswa  lfm2 with a sliding window                     base, swa, recurrent
    hybrid_idx   qwen4exp                                       attn, recurrent, indexer
    recurrent    mamba, mamba2, rwkv6, rwkv6qwen2, rwkv7,       recurrent
                 arwkv7
    dsa          deepseek32, glm-dsa, hy_v4 with an indexer     kv, indexer
    dsa_iswa     dots3note                                      base, indexer, swa
    dsv4         deepseek4 (DeepSeek V4, V4-Pro, V4-Flash)      base, swa, csa, hca, lid,
                                                                csa_state, hca_state, lid_state

Recurrent rows stay opaque bytes; their structure and lengths are checked. Everything else
(other types, flags, versions, layer counts, cell extensions, trailing bytes) is refused with
``StateFormatError``. Unmodified parsed objects serialize back to the identical bytes. Slot
files whose token words hold media chunks parse too (``parse_media_tokens``, ``MediaTokens``);
``truncate_state`` cuts a state before a token index (recurrent sections only by R0).

Layout, with the pinned source it mirrors (paths under runtime/source):

File, ``llama_context::state_seq_save_file`` (src/llama-context.cpp:3252-3270) and
``state_seq_load_file`` (3198-3250)::

    u32 magic = LLAMA_STATE_SEQ_MAGIC 0x67677371 'ggsq'   (include/llama.h:43,48)
    u32 version = LLAMA_STATE_SEQ_VERSION 3                (include/llama.h:49)
    u32 n_token_count
    i32 token words [n_token_count]
    memory payload (state_seq_write_data, src/llama-context.cpp:3318-3324, flags 0)

  The loader does not check for bytes after the payload; this module refuses them.
  llama-server stores ``server_tokens::serialize()`` as the token words
  (tools/server/server-common.cpp:553-573, writer 278-316): LLAMA_TOKEN_NULL (-1),
  SERVER_TOKENS_STATE_VERSION 1 (269), u32 n, n tokens, u32 n_media, media chunks.
  ``server_tokens::deserialize`` (575-617) also accepts a plain token list whose first
  word is not -1, which is what llama_state_seq_save_file with raw tokens produces.
  Media chunks follow as u32 start indices and the mtmd_input_chunk_save bytes of each
  chunk (``MediaChunk``; metadata only); a medium occupies n_tokens LLAMA_TOKEN_NULL
  entries and advances the position by n_pos (``MediaTokens``). Slot restore also requires
  ``<file>.resume`` (tools/server/server-context.cpp:67-121, see ``ResumeFile``).

Blob, ``llama_context::state_seq_get_data`` (src/llama-context.cpp:3080-3097) and
``state_seq_set_data`` (3099-3134)::

    u32 io_magic 0xaf143cd8 (3065), i32 seq_id, memory payload (flags as passed)

  Flags (include/llama.h:907-915): LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY (1) skips the
  base cache of iSWA memory (src/llama-kv-cache-iswa.cpp:259-265), the attention
  cache of hybrid memory (src/llama-memory-hybrid.cpp:190-195) and the DSA cache of
  DSA-iSWA memory (src/llama-kv-cache-dsa-iswa.cpp:236-242); server checkpoints use it.
  LLAMA_STATE_SEQ_FLAGS_ON_DEVICE (2) keeps data on the device and is refused.

Memory payload order:
  iSWA (llama_kv_cache_iswa::state_write, src/llama-kv-cache-iswa.cpp:259-265):
      [base KV section unless PARTIAL_ONLY] SWA KV section
      base = layers with !is_swa(il), SWA = is_swa(il) (filters 52-66), each in layer order.
  hybrid (llama_memory_hybrid::state_write, src/llama-memory-hybrid.cpp:190-195):
      [attention KV section unless PARTIAL_ONLY] recurrent section
      attention = !is_recr(il), recurrent = is_recr(il) (Qwen, qwen3next, MiniMax-01:
      il < n_layer() only, src/llama-model.cpp:2532-2538; falcon-h1: every layer in both;
      nemotron-h: attention = !is_recr && n_ff == 0, recurrent = is_recr && n_ff == 0, so
      MLP-only layers sit in neither; the KDA hybrids use the defaults of
      src/llama-memory-hybrid.cpp:46-64, so their attention section also holds the nextn
      layers, which the trunk never writes).
  hybrid-iSWA (lfm2 with a sliding window, src/llama-memory-hybrid-iswa.cpp:195-198):
      [base KV section unless PARTIAL_ONLY] SWA KV section, recurrent section
  hybrid-idx (qwen4exp, src/llama-memory-hybrid-idx.cpp:204-214):
      [attention KV section] recurrent section [indexer KV section] (PARTIAL_ONLY skips both
      KV sections); the indexer cache holds raw keys of the attention layers plus V rows that
      are never written (its hparams copy keeps the model's V width with one KV head).
  recurrent (mamba, mamba2, rwkv*: llama_memory_recurrent alone): recurrent section, any flags.
  plain KV (one llama_kv_cache over every layer of n_layer_all that passes the filter,
  src/llama-model.cpp:2630-2709; llama_kv_cache::state_write ignores the flags,
  src/llama-kv-cache.cpp:2053-2121):
      KV section ("kv"), with or without PARTIAL_ONLY. deepseek2 and mistral4 have no
      filter, so their nextn layers stay in the section as zero rows (src/llama-kv-cache.cpp
      clears the buffers, 293; only the MTP context writes them).
  DSA (llama_kv_cache_dsa::state_write, src/llama-kv-cache-dsa.cpp:162-165, flags ignored):
      MLA KV section ("kv": trunk layers) then the lightning-indexer key section ("indexer":
      deepseek32 every trunk layer, glm-dsa/hy_v4/dots3note the attention.indexer.types
      layers, src/llama-model.cpp:2276-2345).
  DSA-iSWA (dots3note, src/llama-kv-cache-dsa-iswa.cpp:236-242):
      [DSA sections over the non-sliding trunk layers ("base", "indexer") unless
      PARTIAL_ONLY] SWA KV section ("swa").
  DeepSeek V4 (llama_kv_cache_dsv4::state_write, src/llama-kv-cache-dsv4.cpp:1594-1626):
      u32 magic 0x34565344 'DSV4', u32 version 1, u32 mode (0 full, 1 PARTIAL_ONLY), then the
      raw K-only iSWA cache ([base KV section unless PARTIAL_ONLY] SWA KV section), [the CSA, HCA
      and lightning-indexer row sections ("csa", "hca", "lid") unless PARTIAL_ONLY], and the
      three compressor rings ("csa_state", "hca_state", "lid_state"). See ``CompressSpec``.

KV section, ``llama_kv_cache::state_write`` (src/llama-kv-cache.cpp:2053-2121)::

    u32 n_stream
    per stream: u32 cell_count; if 0 nothing else for this stream, else
      meta (state_write_meta 2203-2234), per written cell in cell-index order:
        i32 pos, u32 n_seq_id, [llama_kv_cell_ext if has_cell_ext()], i32 seq_id[n_seq_id]
      data (state_write_data 2236-2333):
        u32 v_trans, u32 n_layer
        per layer: i32 k_type, u64 k_row_bytes, k rows [cell_count][k_row_bytes]
        v_trans == 0: per layer with V: i32 v_type, u64 v_row_bytes, v rows [cell_count][v_row_bytes]
        v_trans == 1: per layer with V: i32 v_type, u32 v_element_bytes, u32 n_embd_v_gqa,
                      then for each of the n_embd_v_gqa elements its value for every cell
                      (the transposed V cache, stored element-major)

  Cells with no data for the sequence are skipped, and so are SWA-masked cells
  (2080-2085; ``llama_hparams::is_masked_swa`` src/llama-hparams.h:466-478: masked iff
  pos_max - pos >= n_swa), so a SWA section holds only the last n_swa positions.
  Rows are ggml rows: n_embd_k_gqa(il) = n_head_kv * head_dim_k elements; only F32 (0)
  and F16 (1) are accepted (quantized caches also enable a Hadamard rotation,
  src/llama-kv-cache.cpp:314-338), and q8_0 (8) in DeepSeek V4 caches, whose rotation is modelled. v_trans is ``!flash_attn`` (src/llama-model.cpp,
  memory creation). ``llama_kv_cell_ext`` (src/llama-kv-cells.h:13-32) is i32 x, i32 y,
  i32 tok, present iff n_pos_per_embd() > 1 (M-RoPE/IMROPE, src/llama-hparams.cpp:285-287)
  or PLE (src/llama-kv-cache.cpp:1831-1834); for Qwen text tokens x = y = pos and tok is
  the token id (apply_ubatch 1133-1151; text positions are broadcast to every M-RoPE
  component in src/llama-batch.cpp:781-788).
  MLA caches (key_length_mla and value_length_mla set, src/llama-hparams.cpp:297-302) are
  K-only: no V tensor is allocated (src/llama-kv-cache.cpp:229-233), the V blocks are
  skipped, and v_trans is still written and checked (2277, 2306, 2545-2548). Each K row is
  ``[RMSNorm(c_kv) * attn_kv_a_norm (kv_lora) | k_pe]`` (src/models/deepseek2.cpp:526-590):
  k_pe carries the architecture's RoPE (NORMAL pairing, YaRN included) except on the NoPE
  kimi-k3/kimi-linear, and attention reads V as the first kv_lora elements of K
  (src/llama-graph.cpp:2891-2926), so attention.value_length plays no part. The DSA indexer section is a second K-only cache of one
  ``indexer.key_length`` row per cell: H . RoPE(LayerNorm(W_k x)) (src/models/deepseek32.cpp:
  238-260; hparams copy in src/llama-kv-cache-dsa.cpp:39-46).
  On restore of one sequence (state_read_meta 2335-2436) the sequence is erased first,
  n_seq_id must be 1, and cells go wherever find_slot puts them; state_read_data
  (2508-2656) requires equal n_layer, types, row sizes and v_trans. The next batch is
  validated against llama_memory_seq_pos_max (for iSWA the SWA cache's maximum,
  src/llama-kv-cache-iswa.cpp:142-149; for hybrid memory the smaller of the attention
  and recurrent maxima, src/llama-memory-hybrid.cpp:172-180; for DSA the MLA cache's
  maximum; for a plain cache its maximum). With one position per token it must start at
  exactly pos_max + 1 (src/llama-batch.cpp:288-318); with M-RoPE/IMROPE (Qwen3.5,
  n_pos_per_embd 4) only start > pos_max is required (src/llama-batch.cpp:255-287), so jumps
  pass. Those checks do not see every section's end, so ``StateFile.to_bytes`` refuses a
  state whose sections do not all end at the last token (``StateFile.check_positions``).

Recurrent section, ``llama_memory_recurrent::state_write`` (src/llama-memory-recurrent.cpp:766-845)::

    u32 cell_count
    meta (878-895) per cell: i32 pos, u32 n_seq_id (0 when one sequence is saved), i32 seq_id[n]
    data (897-990): u32 s_trans (always 0), u32 n_layer (= hparams.n_layer(), the trunk)
      per recurrent layer (r_l non-null): i32 r_type, u64 r_row_bytes, rows [cell_count][r_row_bytes]
        and on a PLE layer (p_l non-null): u64 p_row_bytes, rows [cell_count][p_row_bytes] (no type)
      per recurrent layer: i32 s_type, u64 s_row_bytes, rows [cell_count][s_row_bytes]

  Rows are always F32 (src/llama-model.cpp:2503-2599). Their sizes come from
  n_embd_r()/n_embd_s()/ple_conv_state() (src/llama-hparams.cpp:208-275, see
  ``recurrent_row_floats``); a family can have 0-byte rows (MiniMax-01 R, LFM2 S).

RoPE helpers mirror ggml-cpu (ggml/src/ggml-cpu/ops.cpp): angles as in
ggml_rope_cache_init (5975-5989) and ggml_mrope_cache_init (5991-6056, interleaved
sectors for IMROPE), the YaRN ramp/mix and magnitude of rope_yarn (5951-5973) with the
correction dims of ggml_rope_yarn_corr_dims (ggml/src/ggml.c:4418-4430), pairs as in
rotate_pairs (6063-6080) with NORMAL (adjacent) and NEOX/IMROPE (half-split) pairing
(6206-6215) inside a window [offset, offset + n_rot) of the row (the MLA k_pe tail, the
hy_v4 indexer tail); other dims pass through. ``Rope`` carries one call's parameters and
``ModelSpec.rope`` picks them per layer; ``rope_runtime`` derives the runtime values
(freq_base, freq_scale, ext_factor, the overwritten yarn_attn_factor, beta, n_ctx_orig)
as llama_context does (src/llama-context.cpp:113-210) from the GGUF and llama-server's
--rope-*/--yarn-* settings. Qwen3 (dense) rotates every key after k_norm with NEOX over the
whole head (src/models/qwen3.cpp:57 asserts n_rot == head size, 96-104;
LLAMA_ROPE_TYPE_NEOX, src/llama-model.cpp:2950), so its cached keys are
cache_keys(Kcur_normed); without M-RoPE its cells carry no ext. Muse rotates keys of
sliding layers with NORMAL pairing and rope.freq_base_swa (src/models/muse-glimmer.cpp:9-10,
117-129); its full layers are NoPE, so their cached keys are Kcur_normed. Qwen rotates the
first n_rot dims of every full-attention key with IMROPE (src/models/qwen35.cpp:296-309); for
text the graph feeds positions (p, p, p, 0) (src/llama-graph.cpp:131-141), so pairs whose
sector selects the fourth component stay unrotated (none for Qwen3.8's sections
[11, 11, 10, 0] over 32 pairs). The forward rotation takes theta_scale from libm powf, cos/sin
from libm cosf/sinf and logf for the YaRN magnitude, and emulates the fused multiply-adds of
the pinned CPU build (the YaRN mix theta_interp * (1 - mix) + theta_extrap * mix, the
magnitude 1 + 0.1 * logf(1 / freq_scale) and the pair rotation); it reproduces that build's
cached keys bit for bit on the tiny fixtures (YaRN included) and matches
ggml_rope_ext/ggml_rope_multi of the pinned libggml-cpu at the real Muse (NORM 128, base 5e5)
and Qwen (IMROPE 64 of 256, base 1e7) geometries. Without libm it falls back to float64 math
rounded to float32, and other backends (the live Vulkan/ROCm servers) round differently. The
inverse is computed in float64.
"""

import dataclasses
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


LLAMA_STATE_SEQ_MAGIC = 0x67677371
LLAMA_STATE_SEQ_VERSION = 3
IO_MAGIC = 0xAF143CD8
LLAMA_TOKEN_NULL = -1
SERVER_TOKENS_STATE_VERSION = 1
FLAGS_NONE = 0
FLAGS_PARTIAL_ONLY = 1
FLAGS_ON_DEVICE = 2
RESUME_MAGIC = 0x51534C54
RESUME_VERSION = 1
MTMD_SERIALIZATION_VERSION = 2
MTMD_CHUNK_TYPES = {1: "image", 2: "audio"}    # mtmd_input_chunk_type (tools/mtmd/mtmd.h:54-59); 0 is text
MTMD_POS_TYPES = ("normal", "mrope", "hunyuanvl")   # mtmd_pos_type (tools/mtmd/mtmd.cpp:202-208)

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_DTYPES = {GGML_TYPE_F32: np.dtype("<f4"), GGML_TYPE_F16: np.dtype("<f2")}
DTYPE_NAMES = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16}
# block_q8_0 (ggml/src/ggml-common.h): an f16 scale and 32 int8 values; a q8_0 row of n elements
# is n / 32 blocks. Only the DeepSeek V4 sections accept it (their Hadamard rotation is modelled).
Q8_0_BLOCK = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
QUANT_BLOCKS = {GGML_TYPE_Q8_0: (Q8_0_BLOCK, 32)}
CACHE_TYPES = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0}
DSV4_STATE_MAGIC = 0x34565344        # 'DSV4' (src/llama-kv-cache-dsv4.cpp:21-26)
DSV4_STATE_VERSION = 1
DSV4_K_ROWS_VERSION = 2
DSV4_RING_VERSION = 1
DSV4_RATIOS = (0, 4, 128)            # window only, CSA (+ lightning indexer), HCA (src/models/deepseek4.cpp:132-153)
MLA_ARCHITECTURES = ("deepseek2", "mistral4", "hy_v4", "deepseek32", "glm-dsa", "dots3note",
                     "kimi-k3", "kimi-linear", "bailingmoe3")
# hybrids with plain (GQA) attention next to GatedDeltaNet, Mamba, Mamba-2, short-conv or lightning layers
HYBRID_ARCHITECTURES = ("qwen3next", "qwen35moe", "qwen4exp", "minimax-01", "jamba", "falcon-h1", "granitehybrid",
                        "nemotron_h", "nemotron_h_moe", "plamo2", "lfm2", "lfm2moe")
# recurrent-only models (llm_arch_is_recurrent, src/llama-arch.cpp:1052-1064): no per-token cache at all
RECURRENT_ARCHITECTURES = ("mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7")
ARCHITECTURES = (("muse-glimmer", "qwen35", "qwen3") + MLA_ARCHITECTURES + HYBRID_ARCHITECTURES
                 + RECURRENT_ARCHITECTURES + ("deepseek4",))
# memory layout per architecture: iSWA (base + SWA KV), hybrid (attention KV + recurrent), plain KV,
# DSA (MLA KV + indexer keys), DSA-iSWA (MLA KV + indexer keys + SWA KV), hybrid-iSWA (base + SWA KV +
# recurrent), hybrid-idx (attention KV + recurrent + indexer keys), recurrent (recurrent rows only);
# hy_v4 is "dsa" when the GGUF has an indexer (attention.indexer.top_k > 0, src/llama-model.cpp:2322-2345)
# and lfm2 "hybrid_iswa" when it has a sliding window (src/models/lfm2.cpp, src/llama-model.cpp:2541-2560)
MEMORY = {"muse-glimmer": "iswa", "qwen35": "hybrid", "qwen3": "kv", "deepseek2": "kv", "mistral4": "kv",
          "hy_v4": "kv", "deepseek32": "dsa", "glm-dsa": "dsa", "dots3note": "dsa_iswa", "kimi-k3": "hybrid",
          "kimi-linear": "hybrid", "bailingmoe3": "hybrid", **{a: "hybrid" for a in HYBRID_ARCHITECTURES},
          "qwen4exp": "hybrid_idx", **{a: "recurrent" for a in RECURRENT_ARCHITECTURES}, "deepseek4": "dsv4"}
MEMORY_LAYOUTS = ("kv", "iswa", "hybrid", "dsa", "dsa_iswa", "hybrid_iswa", "hybrid_idx", "recurrent", "dsv4")
# "nextn": an attention layer the trunk never writes, kept in a KV section (zero rows); "parallel": a
# layer in both the attention and the recurrent memory (falcon-h1); "mlp": a layer in neither (the
# MLP-only layers of nemotron-h)
LAYER_KINDS = {"muse-glimmer": {"full", "sliding"}, "qwen35": {"full", "recurrent"}, "qwen3": {"full"},
               "deepseek2": {"full", "nextn"}, "mistral4": {"full", "nextn"}, "hy_v4": {"full"},
               "deepseek32": {"full"}, "glm-dsa": {"full"}, "dots3note": {"full", "sliding"},
               "kimi-k3": {"full", "recurrent", "nextn"}, "kimi-linear": {"full", "recurrent", "nextn"},
               "bailingmoe3": {"full", "recurrent", "nextn"},
               **{a: {"full", "recurrent"} for a in HYBRID_ARCHITECTURES}, "falcon-h1": {"parallel"},
               "nemotron_h": {"full", "recurrent", "mlp"}, "nemotron_h_moe": {"full", "recurrent", "mlp"},
               "lfm2": {"full", "sliding", "recurrent"}, **{a: {"recurrent"} for a in RECURRENT_ARCHITECTURES},
               "deepseek4": {"sliding", "nextn"}}
# llama_model_rope_type (src/llama-model.cpp:2858-3035); kimi-k3/kimi-linear store k_pe unrotated;
# granitehybrid rotates only when rope.scaling.finetuned (default true, src/models/granite-hybrid.cpp:16-20)
ROPE_TYPES = {"muse-glimmer": "norm", "qwen35": "imrope", "qwen3": "neox", "deepseek2": "norm", "mistral4": "norm",
              "hy_v4": "norm", "deepseek32": "norm", "glm-dsa": "norm", "dots3note": "norm", "kimi-k3": "none",
              "kimi-linear": "none", "bailingmoe3": "norm", "qwen3next": "neox", "qwen35moe": "imrope",
              "qwen4exp": "imrope", "minimax-01": "neox", "jamba": "none", "falcon-h1": "neox", "granitehybrid": "norm",
              "nemotron_h": "none", "nemotron_h_moe": "none", "plamo2": "neox", "lfm2": "neox", "lfm2moe": "neox",
              **{a: "none" for a in RECURRENT_ARCHITECTURES}, "deepseek4": "norm"}
# loaders that read rope.scaling.yarn_log_multiplier and divide it by 0.1 (src/models/deepseek2.cpp:34-38,
# deepseek32.cpp:34-38; mistral4 reuses deepseek2's loader, models.h:1373-1377)
YARN_LOG_MUL_ARCHITECTURES = ("deepseek2", "mistral4", "deepseek32")
# DSA indexer keys: (RoPE pairing, rotated window, Hadamard, LayerNorm eps) per architecture.
# deepseek32/dots3note: NEOX over the leading n_rot dims ([rope | nope]) then H, eps 1e-6
# (src/models/deepseek32.cpp:9, 238-260; dots3note.cpp:10, 219-236); glm-dsa: NORM, H, f_norm_eps
# never set, so 0 (glm-dsa.cpp:276-293); hy_v4: NORM over the trailing n_rot dims ([nope | rope]), no
# Hadamard, eps = the RMS eps (hy-v4.cpp:53, 358-375).
INDEXER_CONVENTIONS = {"deepseek32": ("neox", "head", True, 1e-6), "dots3note": ("neox", "head", True, 1e-6),
                       "glm-dsa": ("norm", "head", True, 0.0), "hy_v4": ("norm", "tail", False, "rms")}
# recurrent row family per hybrid architecture (llama_hparams::n_embd_r/n_embd_s)
RECURRENT_FAMILY = {"qwen35": "ssm", "kimi-k3": "kda", "kimi-linear": "kda", "bailingmoe3": "kda",
                    **{a: "ssm" for a in HYBRID_ARCHITECTURES + ("mamba", "mamba2")}, "minimax-01": "lightning",
                    "lfm2": "shortconv", "lfm2moe": "shortconv",
                    **{a: "rwkv" for a in ("rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7")}}
# loaders that never read ssm.group_count (Mamba-1 style: n_group stays 0, src/models/mamba.cpp, jamba.cpp)
SSM_WITHOUT_GROUPS = ("mamba", "jamba")
# hybrids whose filters keep both caches to il < n_layer() (src/llama-model.cpp:2532-2538): their nextn
# (MTP) layers sit in neither section of the main context
NEXTN_FILTERED = ("qwen35", "qwen3next", "qwen35moe", "qwen4exp", "minimax-01")
ROPE_SCALING_TYPES = ("none", "linear", "yarn", "longrope")
RUNTIME_ROPE_KEYS = ("rope_scaling_type", "rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor",
                     "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx")


class StateFormatError(ValueError):
    """The bytes are not a layout this module understands for the given model."""


# -- byte reader ---------------------------------------------------------------

class _Reader:
    def __init__(self, data, offset=0):
        self.view = memoryview(data).cast("B")
        self.pos = offset

    def take(self, size, what):
        if size < 0 or self.pos + size > len(self.view):
            raise StateFormatError(f"truncated state: {what} needs {size} bytes at offset {self.pos}, "
                                   f"{len(self.view) - self.pos} left")
        start = self.pos
        self.pos += size
        return start

    def scalar(self, fmt, what):
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.view, self.take(size, what))[0]

    def u32(self, what):
        return self.scalar("<I", what)

    def i32(self, what):
        return self.scalar("<i", what)

    def u64(self, what):
        return self.scalar("<Q", what)

    def i64(self, what):
        return self.scalar("<q", what)

    def array(self, dtype, count, what):
        dtype = np.dtype(dtype)
        start = self.take(count * dtype.itemsize, what)
        return np.frombuffer(self.view, dtype=dtype, count=count, offset=start)

    def raw(self, size, what):
        start = self.take(size, what)
        return bytes(self.view[start:start + size])

    def finish(self, what):
        if self.pos != len(self.view):
            raise StateFormatError(f"{len(self.view) - self.pos} trailing bytes after the {what}")


def _pack(fmt, *values):
    return struct.pack(fmt, *values)


# -- model description -----------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IndexerSpec:
    """An indexer key cache: the DSA lightning indexer (llama_kv_cache_dsa::kv_lid,
    src/llama-kv-cache-dsa.cpp:39-53) or qwen4exp's QSA indexer (llama_memory_hybrid_idx::mem_idx)."""

    layers: tuple                 # full-indexer layers: the indexer section's layers, in order
    head_size: int                # attention.indexer.key_length: one K row of head_size per cell
    n_rot: int                    # rope.dimension_count of the main attention (0: raw keys)
    rope_type: str                # "neox" | "norm" | "none" (qwen4exp caches raw keys)
    rope_offset: int = 0          # first rotated dim: 0 ([rope | nope]) or head_size - n_rot ([nope | rope])
    hadamard: bool = True         # keys are stored as H @ rope(k), H the orthonormal Sylvester matrix
    top_k: int = 0                # indexer keys only change outputs once a query sees more than top_k cells
    n_head: int = 0
    norm_eps: float = 0.0         # LayerNorm eps before the k_norm weight and bias
    value_size: int = 0           # elements of the V row the section also stores (qwen4exp: never written,
                                  # zeros); 0 for the K-only DSA caches


@dataclasses.dataclass(frozen=True)
class CompressSpec:
    """DeepSeek V4's compressed caches (llama_kv_cache_dsv4, src/llama-kv-cache-dsv4.h:155-175).

    Every trunk layer slides: its raw cache (an iSWA cache that is K-only, src/llama-kv-cache-dsv4.cpp:
    1210-1260) keeps one latent per token for the last n_swa tokens, RoPE'd on its last n_rot dims
    and read as both K and V (src/models/deepseek4.cpp:928-936). A layer of compress ratio 4 (CSA)
    or 128 (HCA) also keeps one pooled row per completed block of ratio tokens, and CSA layers a
    lightning-indexer (LID) row per block; the tokens of an unfinished block live in an F32
    compressor ring (state_size rows of a and score, src/llama-kv-cache-dsv4.cpp:1313-1329). Per
    token, a = W_comp_kv x and s = W_comp_gate x + APE[pos mod ratio] (x the layer's attention
    input; the LID uses its own weights). A row is a per-channel softmax(s)-weighted sum of a over
    its block (HCA: the block's 128 tokens; CSA/LID: the previous block's 4 tokens in channels
    [0, D) and the block's own 4 in channels [D, 2D)), RMSNorm'd, RoPE'd at the block start with
    the compress base, and Hadamard-rotated for the LID (src/models/deepseek4.cpp:471-589).

    Sections of a sequence save (state_write, src/llama-kv-cache-dsv4.cpp:1594-1626):

        u32 'DSV4', u32 version 1, u32 mode (0 full, 1 PARTIAL_ONLY)
        raw iSWA cache:  [base KV section: every cell's metadata, no layers] SWA KV section
                         (K-only rows of every layer, nextn ones included, for the last n_swa cells);
                         both write n_stream = n_seq_max (-np): the raw cache is never unified
        [K-row sections "csa", "hca", "lid" unless PARTIAL_ONLY] (dsv4_state_write_k_cache 332-362):
            u32 version 2, u32 n_rows = floor((pos_max + 1) / ratio), u32 ns (1 for a sequence),
            u32 n_layer; per layer: u32 il, i32 type, u64 ne0, u64 rows, u64 row_size, rows
        ring sections "csa_state", "hca_state", "lid_state" (llama_dsv4_comp_state::state_write
        1080-1116): u32 version 1, u32 ratio, u32 state_size, u32 n_embd_state, u32 ns, u32 n_layer;
            per layer: u32 il, then kv and score each as (i32 type F32, u64 ne0, u64 rows,
            u64 row_size, state_size rows); a token's (a, s) sits at row pos % state_size

    Raw, CSA and HCA rows are Hadamard-rotated only in a quantized cache; LID rows always
    (src/llama-kv-cache.cpp:315-331). The rings are always F32; type_k must equal type_v."""

    ratios: tuple                 # per trunk layer: 0 (window only), 4 (CSA and LID), 128 (HCA)
    indexer_head_size: int        # LID row width (attention.indexer.key_length)
    indexer_n_head: int = 0
    indexer_top_k: int = 0        # the indexer only changes results past 4 * top_k tokens
    rope_base: float = 160000.0   # attention.compress_rope_freq_base: compressed layers' RoPE base
    norm_eps: float = 1e-6        # attention.layer_norm_rms_epsilon (the compressor norms)

    def layers(self, ratio):
        return tuple(il for il, r in enumerate(self.ratios) if r == ratio)


@dataclasses.dataclass(frozen=True)
class PoolGeometry:
    """One pooling type of a DeepSeek V4 model: its K-row section, ring section and widths."""

    name: str                     # "csa" | "hca" | "lid"
    ratio: int
    layers: tuple
    width: int                    # stored row width: head size (CSA, HCA) or indexer key length (LID)
    state_size: int               # ring rows: 2 * ratio (overlapping CSA/LID windows) or ratio (HCA)
    overlap: bool

    @property
    def state_width(self):
        """Ring row width (n_embd_state): a and score hold both halves of an overlapping window."""
        return 2 * self.width if self.overlap else self.width


@dataclasses.dataclass(frozen=True)
class SectionLayout:
    """One section of a memory payload, in write order."""

    name: str                     # kv | base | swa | attn | indexer | recurrent | csa | hca | lid | *_state
    kind: str                     # "kv" | "recurrent" | "rows" (DSV4 K rows) | "ring" (DSV4 compressor state)
    layers: tuple
    role: str = "attention"       # "attention" | "indexer" | "recurrent" | "pooled" | "ring"
    sliding: bool = False         # SWA section: a native save holds only the last n_swa positions
    ext: bool = False             # KV cells carry llama_kv_cell_ext (ModelSpec.has_cell_ext)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """What the state layout depends on, per layer (index = model layer).

    layer_types lists the trunk layers plus the nextn layers a KV section keeps; nextn layers
    that every cache filters out are left out (as qwen35's)."""

    architecture: str
    layer_types: tuple            # "full" | "sliding" | "recurrent" | "nextn" | "parallel" | "mlp"
    n_head_kv: tuple              # 0 for recurrent and MLP-only layers
    head_dim_k: tuple
    head_dim_v: tuple             # MLA: kv_lora (V is the first kv_lora elements of the K row)
    n_swa: int = 0
    rope_type: str = "none"       # "norm" (Muse sliding layers, MLA k_pe) | "imrope" (Qwen3.5) | "neox" (Qwen3)
    n_rot: int = 0
    rope_freq_base: float = 10000.0
    rope_freq_base_swa: float = 10000.0
    rope_sections: tuple = (0, 0, 0, 0)
    recurrent_row_bytes: tuple = None   # (R, S) bytes per F32 row, when known
    kv_lora: tuple = None         # MLA: per-layer latent width; the attention cache is K-only
    memory_layout: str = None     # default MEMORY[architecture]
    rope_freq_scale: float = 1.0  # cparams.rope_freq_scale
    rope_freq_scale_swa: float = 1.0
    rope_ext_factor: float = 0.0  # cparams.yarn_ext_factor
    rope_attn_factor: float = 1.0  # cparams.yarn_attn_factor, the attn_factor every ggml_rope_ext call gets
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_n_ctx_orig: int = 0
    indexer: IndexerSpec = None
    recurrent_p_bytes: int = 0    # PLE conv row bytes (llama_hparams::ple_conv_state, F32)
    ple_layers: tuple = ()        # recurrent layers that carry a PLE row
    # RoPE frequency factors per rotated pair (granitehybrid's optional rope_freqs.weight): () when the
    # GGUF has none; None leaves a rotating granitehybrid's factors unknown, and rope() refuses it
    rope_freq_factors: tuple = None
    compress: CompressSpec = None  # deepseek4: the compressed caches (memory "dsv4")

    def __post_init__(self):
        n = len(self.layer_types)
        if self.architecture not in ARCHITECTURES:
            raise StateFormatError(f"unsupported architecture {self.architecture!r}")
        if self.memory_layout is None:
            object.__setattr__(self, "memory_layout", MEMORY[self.architecture])
        if self.memory_layout not in MEMORY_LAYOUTS:
            raise StateFormatError(f"unknown memory layout {self.memory_layout!r}")
        if n == 0 or not all(len(x) == n for x in (self.n_head_kv, self.head_dim_k, self.head_dim_v)):
            raise StateFormatError("per-layer spec arrays must be nonempty and of equal length")
        allowed = LAYER_KINDS[self.architecture]
        if not set(self.layer_types) <= allowed:
            raise StateFormatError(f"{self.architecture} layers must be {sorted(allowed)}")
        for il, kind in enumerate(self.layer_types):
            if kind not in ("recurrent", "mlp") and min(self.n_head_kv[il], self.head_dim_k[il],
                                                        self.head_dim_v[il]) <= 0:
                raise StateFormatError(f"invalid attention dimensions at layer {il}")
        nextn = [il for il, kind in enumerate(self.layer_types) if kind == "nextn"]
        if nextn and nextn != list(range(n - len(nextn), n)):
            raise StateFormatError("nextn layers must follow every trunk layer")
        if self.architecture == "muse-glimmer":
            if self.rope_type != "norm" or self.n_swa <= 0:
                raise StateFormatError("Muse needs NORM RoPE and a positive sliding window")
            if "sliding" not in self.layer_types:
                raise StateFormatError("Muse needs at least one sliding layer")
        elif self.architecture == "qwen35":
            if self.rope_type != "imrope" or len(self.rope_sections) != 4 or self.n_swa != 0:
                raise StateFormatError("Qwen needs IMROPE with four sections and no sliding window")
        elif self.architecture == "qwen3":
            if self.rope_type != "neox" or self.n_swa != 0:
                raise StateFormatError("Qwen3 needs NEOX RoPE and no sliding window")
            if any(self.n_rot > dim for dim in self.head_dim_k):
                raise StateFormatError("n_rot exceeds the key head size")
        elif self.architecture in MLA_ARCHITECTURES:
            self._check_mla()
        elif self.architecture == "deepseek4":
            self._check_dsv4()
        else:
            self._check_generic()
        if (self.compress is not None) != (self.memory_layout == "dsv4"):
            raise StateFormatError("only the dsv4 memory (deepseek4) has compressed caches")
        if self.architecture in ("muse-glimmer", "qwen35", "qwen3") and (
                self.memory_layout != MEMORY[self.architecture] or self.kv_lora is not None or self.indexer is not None):
            raise StateFormatError(f"{self.architecture} has {MEMORY[self.architecture]} memory without MLA or indexer")
        if self.rope_type != "none" and (self.n_rot <= 0 or self.n_rot % 2):
            raise StateFormatError("n_rot must be a positive even number")
        p_bytes = self.recurrent_p_bytes
        if p_bytes < 0 or p_bytes % 4 or (p_bytes > 0) != bool(self.ple_layers):
            raise StateFormatError("PLE rows need a positive multiple of 4 bytes and at least one PLE layer")
        if not set(self.ple_layers) <= set(self.recurrent_layers):
            raise StateFormatError("PLE rows sit on recurrent layers only")
        if self.rope_freq_factors and (self.architecture != "granitehybrid" or self.rope_type == "none"
                                       or len(self.rope_freq_factors) != self.n_rot // 2):
            raise StateFormatError("RoPE frequency factors: granitehybrid only, one per rotated pair")

    def _check_generic(self):
        """The plain-attention hybrids and the recurrent-only models (HYBRID_ARCHITECTURES,
        RECURRENT_ARCHITECTURES): GQA K/V rows, the architecture's RoPE, no MLA."""
        expected = ROPE_TYPES[self.architecture]
        # "none" also covers granitehybrid without rope.scaling.finetuned (has_rope,
        # src/llama-hparams.cpp:333-345) and models whose n_rot comes out 0
        if self.rope_type not in (expected, "none"):
            raise StateFormatError(f"{self.architecture} needs {expected} RoPE (or none)")
        if self.kv_lora is not None:
            raise StateFormatError(f"{self.architecture} has no MLA cache")
        if self.rope_type != "none" and any(self.n_rot > self.head_dim_k[il] for il in self.attention_layers):
            raise StateFormatError("n_rot exceeds the key head size")
        if self.rope_type == "imrope" and len(self.rope_sections) != 4:
            raise StateFormatError(f"{self.architecture} needs IMROPE with four sections")
        sliding = "sliding" in self.layer_types
        allowed = {"lfm2": ("hybrid", "hybrid_iswa")}.get(self.architecture, (MEMORY[self.architecture],))
        if self.memory_layout not in allowed or (self.memory_layout == "hybrid_iswa") != sliding \
                or (sliding and self.n_swa <= 0) or (not sliding and self.n_swa != 0):
            raise StateFormatError(f"{self.architecture} cannot use {self.memory_layout} memory with these layers "
                                   "(sliding layers need hybrid_iswa memory and a positive window)")
        if (self.indexer is not None) != (self.memory_layout == "hybrid_idx"):
            raise StateFormatError("only the hybrid_idx memory (qwen4exp) has an indexer section")
        ix = self.indexer
        if ix is not None:
            if tuple(ix.layers) != self.layers_of("full") or ix.rope_type != "none" or ix.n_rot != 0 or ix.hadamard \
                    or ix.head_size <= 0 or ix.value_size < 0:
                raise StateFormatError("the qwen4exp indexer caches raw keys (no RoPE, no Hadamard) on every "
                                       "attention layer")
        if self.recurrent_layers and self.recurrent_row_bytes is None:
            raise StateFormatError(f"{self.architecture} needs the recurrent row sizes")
        if self.memory_layout == "recurrent" and self.attention_layers:
            raise StateFormatError(f"{self.architecture} has recurrent layers only")

    def _check_mla(self):
        if self.rope_type != ROPE_TYPES[self.architecture]:
            raise StateFormatError(f"{self.architecture} needs {ROPE_TYPES[self.architecture]} RoPE")
        if self.kv_lora is None or len(self.kv_lora) != len(self.layer_types):
            raise StateFormatError(f"{self.architecture} needs a per-layer kv_lora (MLA latent width)")
        for il, kind in enumerate(self.layer_types):
            if kind == "recurrent":
                continue
            lora, dim_k = self.kv_lora[il], self.head_dim_k[il]
            if lora <= 0 or self.n_head_kv[il] != 1 or self.head_dim_v[il] != lora or dim_k <= lora:
                raise StateFormatError(f"layer {il}: an MLA row is one KV head [c_kv (kv_lora) | k_pe] whose "
                                       f"V view is c_kv; got heads {self.n_head_kv[il]}, K {dim_k}, "
                                       f"V {self.head_dim_v[il]}, kv_lora {lora}")
            if self.rope_type != "none" and dim_k - lora != self.n_rot:
                raise StateFormatError(f"layer {il}: k_pe has {dim_k - lora} dims, RoPE rotates {self.n_rot}")
        sliding = "sliding" in self.layer_types
        if (self.memory_layout == "dsa_iswa") != sliding or (sliding and self.n_swa <= 0):
            raise StateFormatError("sliding MLA layers need the dsa_iswa memory and a positive window")
        if self.memory_layout not in ("kv", "hybrid", "dsa", "dsa_iswa"):
            raise StateFormatError(f"{self.architecture} cannot use {self.memory_layout} memory")
        if (self.memory_layout == "hybrid") != (self.architecture in RECURRENT_FAMILY):
            raise StateFormatError(f"{self.architecture} cannot use {self.memory_layout} memory")
        if "nextn" in self.layer_types and self.memory_layout not in ("kv", "hybrid"):
            raise StateFormatError("only plain-KV and hybrid caches keep nextn layers")
        ix = self.indexer
        if (ix is not None) != (self.memory_layout in ("dsa", "dsa_iswa")):
            raise StateFormatError("DSA memories need an indexer spec, others none")
        if ix is not None:
            if not ix.layers or not set(ix.layers) <= set(self.layers_of("full", "sliding")) \
                    or list(ix.layers) != sorted(set(ix.layers)):
                raise StateFormatError("indexer layers must be increasing attention layers")
            if ix.rope_type not in ("neox", "norm") or ix.n_rot <= 0 or ix.n_rot % 2 \
                    or not 0 <= ix.rope_offset <= ix.head_size - ix.n_rot:
                raise StateFormatError("invalid indexer RoPE window")
            if ix.hadamard and (ix.head_size < 64 or ix.head_size & (ix.head_size - 1)):
                # self_k_rot_lid is applied with a plain mul_mat, so H must span the whole key
                raise StateFormatError("Hadamard-rotated indexer keys need a power-of-two head size >= 64")

    def _check_dsv4(self):
        """deepseek4: every trunk layer slides (one K-only latent row per token, NORM RoPE on its
        last n_rot dims), nextn (MTP) layers follow them in the raw cache, and the compress ratios
        give each trunk layer's pooled caches (src/models/deepseek4.cpp:55-74, 132-153)."""
        cs = self.compress
        trunk = self.layers_of("sliding")
        if self.memory_layout != "dsv4" or cs is None or self.rope_type != "norm" or self.n_swa <= 0:
            raise StateFormatError("deepseek4 needs the dsv4 memory, a compress spec, NORM RoPE and a positive "
                                   "sliding window")
        if trunk != tuple(range(len(trunk))) or len(cs.ratios) != len(trunk):
            raise StateFormatError("deepseek4 needs one compress ratio per trunk layer, trunk layers first")
        if any(r not in DSV4_RATIOS for r in cs.ratios):
            raise StateFormatError(f"deepseek4 compress ratios must be in {DSV4_RATIOS}")
        if self.kv_lora is not None or self.indexer is not None:
            raise StateFormatError("deepseek4 has no MLA latent and no DSA indexer")
        dims = {(self.n_head_kv[il], self.head_dim_k[il], self.head_dim_v[il]) for il in range(len(self.layer_types))}
        heads, head, value = next(iter(dims))
        if len(dims) != 1 or heads != 1 or head != value:
            raise StateFormatError("deepseek4 layers store one latent head, read as both K and V")
        if not 0 < self.n_rot <= min(head, cs.indexer_head_size if cs.layers(4) else head):
            raise StateFormatError("n_rot exceeds the latent or indexer key length")

    @property
    def memory(self):
        return self.memory_layout

    @property
    def cell_ext(self):
        """Whether the attention cells carry llama_kv_cell_ext (has_cell_ext of the attention cache)."""
        return self.has_cell_ext("attention")

    def has_cell_ext(self, role="attention"):
        """llama_kv_cache::has_cell_ext (src/llama-kv-cache.cpp:1831-1834) of a section's cache:
        M-RoPE/IMROPE (n_pos_per_embd > 1) or a PLE model (the n-gram hash needs the token id).
        The qwen4exp indexer cache copies the hparams with rope type NONE (src/llama-memory-hybrid-idx.cpp:
        44-56), so only PLE gives its cells an ext."""
        if role == "recurrent":
            return False
        mrope = self.rope_type == "imrope" and role == "attention"
        return mrope or bool(self.ple_layers)

    @property
    def n_layer_trunk(self):
        """hparams.n_layer(): every layer but the nextn ones (the recurrent section's n_layer)."""
        return sum(kind != "nextn" for kind in self.layer_types)

    def layers_of(self, *kinds):
        return tuple(il for il, kind in enumerate(self.layer_types) if kind in kinds)

    @property
    def attention_layers(self):
        """Layers with an attention KV cache (falcon-h1's parallel layers included)."""
        return self.layers_of("full", "sliding", "nextn", "parallel")

    @property
    def recurrent_layers(self):
        """Layers with recurrent rows (falcon-h1's parallel layers included)."""
        return self.layers_of("recurrent", "parallel")

    def k_only(self, layer):
        """True when the layer's attention cache has no V tensor (MLA, DeepSeek V4's raw latents)."""
        return (self.kv_lora is not None and self.kv_lora[layer] > 0) or self.memory_layout == "dsv4"

    def pools(self):
        """PoolGeometry of every pooling type of a deepseek4 model, in section order (csa, hca, lid)."""
        cs = self.compress
        if cs is None:
            return []
        head = self.head_dim_k[0]
        return [PoolGeometry("csa", 4, cs.layers(4), head, 8, True),
                PoolGeometry("hca", 128, cs.layers(128), head, 128, False),
                PoolGeometry("lid", 4, cs.layers(4), cs.indexer_head_size, 8, True)]

    def pool(self, name):
        for geometry in self.pools():
            if geometry.name == name:
                return geometry
        raise KeyError(f"{self.architecture} has no {name} pool")

    def layouts(self, flags):
        """SectionLayout list in the order state_write emits it for these flags."""
        if flags not in (FLAGS_NONE, FLAGS_PARTIAL_ONLY):
            raise StateFormatError(f"unsupported state flags {flags}")
        partial = flags == FLAGS_PARTIAL_ONLY
        memory = self.memory

        def kv(name, layers, **options):
            role = options.get("role", "attention")
            return SectionLayout(name, "kv", tuple(layers), ext=self.has_cell_ext(role), **options)
        recurrent = SectionLayout("recurrent", "recurrent", self.recurrent_layers, role="recurrent")
        if memory == "kv":
            # llama_kv_cache::state_write writes the whole cache whatever the flags
            return [kv("kv", self.layers_of("full", "nextn"))]
        if memory == "dsa":
            # llama_kv_cache_dsa::state_write passes the flags to two plain caches, which ignore them
            return [kv("kv", self.layers_of("full")), kv("indexer", self.indexer.layers, role="indexer")]
        if memory == "iswa":
            out = [kv("base", self.layers_of("full")), kv("swa", self.layers_of("sliding"), sliding=True)]
            return out[1:] if partial else out
        if memory == "dsa_iswa":
            out = [kv("base", self.layers_of("full")), kv("indexer", self.indexer.layers, role="indexer"),
                   kv("swa", self.layers_of("sliding"), sliding=True)]
            return out[2:] if partial else out
        if memory == "recurrent":
            # llama_memory_recurrent::state_write ignores the flags (src/llama-memory-recurrent.cpp:766-845)
            return [recurrent]
        if memory == "dsv4":
            # the raw cache's base part keeps cell metadata only (every layer slides, the nextn ones
            # too), then the pooled rows (full saves only) and the rings (src/llama-kv-cache-dsv4.cpp:
            # 1594-1626)
            raw = [kv("base", ()), kv("swa", self.layers_of("sliding", "nextn"), sliding=True)]
            rows = [SectionLayout(p.name, "rows", p.layers, role="pooled") for p in self.pools()]
            rings = [SectionLayout(p.name + "_state", "ring", p.layers, role="ring") for p in self.pools()]
            return raw[1:] + rings if partial else raw + rows + rings
        if memory == "hybrid_iswa":
            # an iSWA attention cache (PARTIAL_ONLY still writes its SWA part) then the recurrent
            # rows (src/llama-memory-hybrid-iswa.cpp:195-198, src/llama-kv-cache-iswa.cpp:259-265)
            out = [kv("base", self.layers_of("full")), kv("swa", self.layers_of("sliding"), sliding=True), recurrent]
            return out[1:] if partial else out
        attention = kv("attn", self.layers_of("full", "nextn", "parallel"))
        if memory == "hybrid_idx":
            # attention, recurrent, then the indexer keys, which share the PARTIAL_ONLY gate
            # (src/llama-memory-hybrid-idx.cpp:204-214)
            return [recurrent] if partial else [attention, recurrent, kv("indexer", self.indexer.layers,
                                                                          role="indexer")]
        return [recurrent] if partial else [attention, recurrent]

    def sections(self, flags):
        """(name, kind, layers) in the order state_write emits them for these flags."""
        return [(layout.name, layout.kind, layout.layers) for layout in self.layouts(flags)]

    def layout(self, name, flags=FLAGS_NONE):
        for layout in self.layouts(flags):
            if layout.name == name:
                return layout
        raise KeyError(f"{self.architecture} state has no {name} section (flags {flags})")

    def row_elements(self, layer, key):
        """Elements of the logical attention row (MLA V: the c_kv view of K)."""
        head_dim = self.head_dim_k[layer] if key == "k" else self.head_dim_v[layer]
        return self.n_head_kv[layer] * head_dim

    def head_shape(self, layer, key):
        """Logical (heads, dim) of a layer's attention K or V (MLA V: the c_kv view of K)."""
        return self.n_head_kv[layer], (self.head_dim_k[layer] if key == "k" else self.head_dim_v[layer])

    def kv_shape(self, layer, key, role="attention"):
        """(heads, dim) of the rows a section stores for this layer, or None for the V of a K-only cache."""
        if role == "indexer":
            if self.indexer is None or layer not in self.indexer.layers:
                raise StateFormatError(f"layer {layer} has no indexer keys")
            if key == "k":
                return 1, self.indexer.head_size
            return (1, self.indexer.value_size) if self.indexer.value_size else None
        if key == "v" and self.k_only(layer):
            return None
        return self.head_shape(layer, key)

    def rope(self, layer, role="attention"):
        """The Rope that turned this layer's pre-RoPE keys into cached keys, or None (NoPE)."""
        kind = self.layer_types[layer]
        if kind in ("recurrent", "mlp"):
            raise ValueError(f"layer {layer} has no attention cache ({kind})")
        if self.memory_layout == "dsv4":
            return self._dsv4_rope(layer, role)
        common = dict(ext_factor=self.rope_ext_factor, attn_factor=self.rope_attn_factor, beta_fast=self.rope_beta_fast,
                      beta_slow=self.rope_beta_slow, n_ctx_orig=self.rope_n_ctx_orig)
        if role == "indexer":
            ix = self.indexer
            if ix is None or layer not in ix.layers:
                raise ValueError(f"layer {layer} has no indexer keys")
            if ix.rope_type == "none":
                return None
            return Rope(ix.rope_type, ix.n_rot, self.rope_freq_base, offset=ix.rope_offset,
                        freq_scale=self.rope_freq_scale, **common)
        if self.rope_type == "none" or (self.architecture == "muse-glimmer" and kind == "full"):
            return None
        if self.architecture == "granitehybrid" and self.rope_freq_factors is None:
            raise StateFormatError("granitehybrid RoPE frequency factors are unknown (rope_freqs.weight is an optional "
                                   "tensor): build the spec with ModelSpec.from_gguf, or pass rope_freq_factors= to "
                                   "from_metadata (() when the GGUF has none)")
        sliding = kind == "sliding"
        base = self.rope_freq_base_swa if sliding else self.rope_freq_base
        scale = self.rope_freq_scale_swa if sliding else self.rope_freq_scale
        offset = self.kv_lora[layer] if self.k_only(layer) else 0
        sections = self.rope_sections if self.rope_type == "imrope" else None
        return Rope(self.rope_type, self.n_rot, base, offset=offset, sections=sections, freq_scale=scale,
                    freq_factors=self.rope_freq_factors or None, **common)

    def _dsv4_rope(self, layer, role):
        """DeepSeek V4: NORM RoPE on the last n_rot dims of a row (rope_set_offset). Window-only layers
        (ratio 0, and the nextn ones) rotate their latents with freq_base and no YaRN; compressed
        layers rotate their latents ("attention") and their pooled rows ("pooled": CSA/HCA rows at the
        block start, "lid": indexer rows) with the compress base, the context's YaRN and an attn_factor
        that cancels ggml's YaRN magnitude (src/models/deepseek4.cpp:11-17, 903-910, 509-512, 582-585)."""
        cs = self.compress
        ratio = cs.ratios[layer] if layer < len(cs.ratios) else 0
        width = {"attention": self.head_dim_k[layer], "pooled": self.head_dim_k[layer],
                 "lid": cs.indexer_head_size}.get(role)
        if width is None or (role != "attention" and ratio == 0) or (role == "lid" and ratio != 4):
            raise ValueError(f"layer {layer} (ratio {ratio}) has no {role} rows")
        offset = width - self.n_rot
        if ratio == 0:
            return Rope("norm", self.n_rot, self.rope_freq_base, offset=offset)
        return Rope("norm", self.n_rot, cs.rope_base, offset=offset, freq_scale=self.rope_freq_scale,
                    ext_factor=self.rope_ext_factor,
                    attn_factor=dsv4_rope_attn_factor(self.rope_freq_scale, self.rope_ext_factor),
                    beta_fast=self.rope_beta_fast, beta_slow=self.rope_beta_slow, n_ctx_orig=self.rope_n_ctx_orig)

    @classmethod
    def from_metadata(cls, meta, *, runtime=None, rope_freq_factors=None):
        """Spec from a GGUF metadata mapping (key -> value, arrays as lists).

        runtime: the receiving context's RoPE settings where they differ from the GGUF, as
        llama_context_params names them (RUNTIME_ROPE_KEYS; llama-server: --rope-scaling,
        --rope-freq-base, --rope-freq-scale or 1 / --rope-scale, --yarn-ext-factor,
        --yarn-attn-factor, --yarn-beta-fast, --yarn-beta-slow, --yarn-orig-ctx); a None
        value means unset.

        rope_freq_factors: the rope_freqs.weight tensor, which metadata does not show; () when
        the GGUF has none. Only granitehybrid uses it, and a rotating granitehybrid spec built
        without it refuses rope() (from_gguf reads it)."""
        architecture = meta.get("general.architecture")
        if architecture not in ARCHITECTURES:
            raise StateFormatError(f"unsupported architecture {architecture!r}")
        get = _metadata_getter(meta, architecture + ".")
        if architecture == "deepseek4":
            spec = cls._dsv4_from_metadata(architecture, meta, get, runtime)
        elif architecture in MLA_ARCHITECTURES:
            spec = cls._mla_from_metadata(architecture, meta, get, runtime)
        elif architecture in HYBRID_ARCHITECTURES + RECURRENT_ARCHITECTURES:
            spec = cls._generic_from_metadata(architecture, meta, get, runtime)
        else:
            spec = cls._dense_from_metadata(architecture, meta, get, runtime)
        if rope_freq_factors is not None:
            spec = dataclasses.replace(spec, rope_freq_factors=tuple(float(x) for x in rope_freq_factors))
        return spec

    @classmethod
    def _dense_from_metadata(cls, architecture, meta, get, runtime):
        """muse-glimmer, qwen35 and qwen3."""
        if runtime and any(runtime.get(key) is not None for key in RUNTIME_ROPE_KEYS):
            raise StateFormatError(f"runtime RoPE overrides are not supported for {architecture}")
        prefix = architecture + "."
        nextn = int(get("nextn_predict_layers", 0))
        n_layer = int(get("block_count")) - nextn
        if n_layer <= 0 or nextn < 0:
            raise StateFormatError("invalid block count")
        if architecture in ("muse-glimmer", "qwen3") and nextn:
            # Muse's iSWA cache and a dense model's plain cache have no layer filter
            # (src/llama-model.cpp:2610-2709), so nextn layers would sit in the (base) cache
            # (is_swa is false past n_layer(), src/llama-hparams.cpp:19-21); not modelled.
            raise StateFormatError(f"{architecture} with nextn_predict_layers > 0 is not supported")
        embd = int(get("embedding_length"))
        types = []
        for il in range(n_layer):
            if architecture == "qwen35":
                flags = meta.get(prefix + "attention.recurrent_layers")
                if flags is not None:
                    recurrent = bool(get("attention.recurrent_layers", layer=il))
                else:
                    interval = int(get("full_attention_interval", 4))
                    if interval <= 0:
                        raise StateFormatError("invalid full-attention interval")
                    recurrent = (il + 1) % interval != 0
                types.append("recurrent" if recurrent else "full")
            elif architecture == "qwen3":
                types.append("full")
            else:
                pattern = meta.get(prefix + "attention.sliding_window_pattern", 4)
                if isinstance(pattern, (list, tuple, np.ndarray)):
                    sliding = bool(get("attention.sliding_window_pattern", layer=il))
                else:
                    if int(pattern) <= 0:
                        raise StateFormatError("invalid sliding-window period")
                    sliding = il % int(pattern) < int(pattern) - 1
                types.append("sliding" if sliding else "full")
        # head sizes as llama_model::load_hparams sets them (src/llama-model.cpp:1334-1370):
        # full layers key_length/value_length (default n_embd / n_head of layer 0), sliding
        # layers key_length_swa/value_length_swa defaulting to the full sizes, per
        # llama_hparams::n_embd_head_k/v(il) (src/llama-hparams.cpp:140-154).
        n_head0 = int(get("attention.head_count", layer=0))
        default_dim = embd // n_head0 if n_head0 > 0 else None
        full_k = int(get("attention.key_length", default_dim))
        full_v = int(get("attention.value_length", default_dim))
        swa_k = int(get("attention.key_length_swa", full_k))
        swa_v = int(get("attention.value_length_swa", full_v))
        heads_kv, dim_k, dim_v = [], [], []
        for il, kind in enumerate(types):
            if kind == "recurrent":
                heads_kv.append(0), dim_k.append(0), dim_v.append(0)
                continue
            n_head = int(get("attention.head_count", layer=il))
            heads_kv.append(int(get("attention.head_count_kv", n_head, layer=il)))
            dim_k.append(swa_k if kind == "sliding" else full_k)
            dim_v.append(swa_v if kind == "sliding" else full_v)
        scaling = meta.get(prefix + "rope.scaling.type")
        factor = float(meta.get(prefix + "rope.scaling.factor", meta.get(prefix + "rope.scale_linear", 0.0)) or 0.0)
        if scaling not in (None, "none", "linear") or factor not in (0.0, 1.0):
            raise StateFormatError(f"RoPE scaling {scaling!r} with factor {factor} is not supported")
        base = float(get("rope.freq_base", 10000.0))
        # n_rot_full defaults to the full head size, n_rot_swa to n_rot_full (src/llama-model.cpp:1344-1370)
        n_rot_full = int(get("rope.dimension_count", full_k))
        if architecture == "qwen3":
            # src/models/qwen3.cpp asserts n_rot == the head size (rope.dimension_count is optional)
            if n_rot_full != full_k:
                raise StateFormatError(f"qwen3 rotates the whole head: n_rot {n_rot_full} != head size {full_k}")
            return cls(architecture, tuple(types), tuple(heads_kv), tuple(dim_k), tuple(dim_v),
                       rope_type="neox", n_rot=n_rot_full, rope_freq_base=base, rope_freq_base_swa=base)
        if architecture == "muse-glimmer":
            return cls(architecture, tuple(types), tuple(heads_kv), tuple(dim_k), tuple(dim_v),
                       n_swa=int(get("attention.sliding_window")), rope_type="norm",
                       n_rot=int(get("rope.dimension_count_swa", n_rot_full)),
                       rope_freq_base=base, rope_freq_base_swa=float(get("rope.freq_base_swa", base)))
        r_floats, s_floats = recurrent_row_floats(meta, "ssm")
        sections = tuple(int(x) for x in get("rope.dimension_sections"))
        sections = (sections + (0, 0, 0, 0))[:4]
        return cls(architecture, tuple(types), tuple(heads_kv), tuple(dim_k), tuple(dim_v),
                   rope_type="imrope", n_rot=n_rot_full,
                   rope_freq_base=base, rope_freq_base_swa=base, rope_sections=sections,
                   recurrent_row_bytes=(4 * r_floats, 4 * s_floats))

    @classmethod
    def _mla_from_metadata(cls, architecture, meta, get, runtime):
        prefix = architecture + "."
        nextn = int(get("nextn_predict_layers", 0))
        n_all = int(get("block_count"))
        n_layer = n_all - nextn
        if n_layer <= 0 or nextn < 0:
            raise StateFormatError("invalid block count")
        if nextn and architecture == "hy_v4":
            # no nextn filter and is_indexer_full() aborts past n_layer() (src/llama-hparams.cpp)
            raise StateFormatError("hy_v4 with nextn_predict_layers > 0 is not supported")
        # deepseek2/mistral4 (plain cache, no filter) and the KDA hybrids (default !is_recr filter)
        # keep nextn layers in the trunk cache; deepseek32, glm-dsa and dots3note filter them out
        keep_nextn = architecture in ("deepseek2", "mistral4", "kimi-k3", "kimi-linear", "bailingmoe3")
        if not (meta.get(prefix + "attention.key_length_mla") and meta.get(prefix + "attention.value_length_mla")):
            # without them llama.cpp builds the expanded MHA cache of legacy deepseek2 GGUFs (deepseek2.cpp:602-643)
            raise StateFormatError(f"{architecture} without attention.key_length_mla/value_length_mla "
                                   "(an expanded MHA cache) is not supported")
        embd = int(get("embedding_length"))
        kv_lora = int(get("attention.kv_lora_rank"))
        n_head0 = int(get("attention.head_count", layer=0))
        default_dim = embd // n_head0 if n_head0 > 0 else None
        key_length = int(get("attention.key_length", default_dim))
        # An MLA cache stores no V (has_v = !is_mla, src/llama-kv-cache.cpp:231-234): attention
        # reads the c_kv view of K as V, so the logical V width is kv_lora whatever
        # attention.value_length says. The kimi-linear and bailingmoe3 converters write the base
        # class's head_dim or nothing there (conversion/kimi_linear.py:119, bailingmoe3.py:71,
        # base.py:1379-1381); kimi-k3 overwrites n_embd_head_v_full (src/models/kimi-k3.cpp:24-26).
        value_length = kv_lora
        n_rot = int(get("rope.dimension_count", key_length))
        memory = MEMORY[architecture]
        swa_pattern, n_swa = None, 0
        if architecture == "dots3note":
            swa_pattern = meta.get(prefix + "attention.sliding_window_pattern")
            if not isinstance(swa_pattern, (list, tuple, np.ndarray)):
                raise StateFormatError("dots3note needs an attention.sliding_window_pattern array")
            n_swa = int(get("attention.sliding_window"))
            swa_lora = int(get("attention.kv_lora_rank_swa"))
            swa_dims = (int(get("attention.key_length_swa", key_length)), swa_lora, swa_lora)
        types, heads_kv, dim_k, dim_v, lora = [], [], [], [], []
        for il in range(n_all if keep_nextn else n_layer):
            heads = int(get("attention.head_count_kv", get("attention.head_count", layer=il), layer=il))
            if il >= n_layer:
                kind = "nextn"
            elif memory == "hybrid":
                kind = "recurrent" if heads == 0 else "full"
            elif swa_pattern is not None and il < len(swa_pattern) and bool(swa_pattern[il]):
                kind = "sliding"
            else:
                kind = "full"
            types.append(kind)
            if kind == "recurrent":
                heads_kv.append(0), dim_k.append(0), dim_v.append(0), lora.append(0)
                continue
            k, v, rank = swa_dims if kind == "sliding" else (key_length, value_length, kv_lora)
            heads_kv.append(heads), dim_k.append(k), dim_v.append(v), lora.append(rank)
        rope = rope_runtime(meta, runtime=runtime)
        indexer = None
        top_k = int(meta.get(prefix + "attention.indexer.top_k", 0) or 0)
        if architecture == "hy_v4" and top_k > 0:
            memory = "dsa"
        if memory in ("dsa", "dsa_iswa"):
            indexer = _indexer_from_metadata(architecture, meta, get, n_layer, n_rot)
        recurrent_bytes = None
        if memory == "hybrid":
            r_floats, s_floats = recurrent_row_floats(meta, RECURRENT_FAMILY[architecture])
            recurrent_bytes = (4 * r_floats, 4 * s_floats)
        base_swa = float(get("rope.freq_base_swa")) if architecture == "dots3note" else rope["freq_base"]
        return cls(architecture, tuple(types), tuple(heads_kv), tuple(dim_k), tuple(dim_v), n_swa=n_swa,
                   rope_type=ROPE_TYPES[architecture], n_rot=n_rot, rope_freq_base=rope["freq_base"],
                   # dots3note's sliding layers take freq_base_swa but the context's freq_scale
                   # (src/models/dots3note.cpp:189, 335-340)
                   rope_freq_base_swa=base_swa, recurrent_row_bytes=recurrent_bytes, kv_lora=tuple(lora),
                   memory_layout=memory, rope_freq_scale=rope["freq_scale"], rope_freq_scale_swa=rope["freq_scale"],
                   rope_ext_factor=rope["ext_factor"], rope_attn_factor=rope["attn_factor"],
                   rope_beta_fast=rope["beta_fast"], rope_beta_slow=rope["beta_slow"],
                   rope_n_ctx_orig=rope["n_ctx_orig"], indexer=indexer)

    @classmethod
    def _dsv4_from_metadata(cls, architecture, meta, get, runtime):
        """deepseek4 as llama_model_deepseek4::load_arch_hparams (src/models/deepseek4.cpp:19-79) and
        llama_kv_cache_dsv4 (src/llama-kv-cache-dsv4.cpp:1210-1336) set it up: every trunk layer
        slides, and the main context's raw cache also keeps the nextn (MTP) layers (it zeroes
        n_layer_nextn in its hparams copy, 1246, and MTP layers are flagged sliding, 72-74), whose
        rows the trunk never writes. The loader drops nextn_predict_layers when the GGUF lacks
        blk.<n_layer>.nextn.eh_proj.weight, which metadata does not show (from_gguf checks it)."""
        nextn = int(get("nextn_predict_layers", 0))
        n_all = int(get("block_count"))
        n_layer = n_all - nextn
        if n_layer <= 0 or nextn < 0:
            raise StateFormatError("invalid block count")
        ratios = [int(x) for x in get("attention.compress_ratios")]
        if len(ratios) < n_all:
            raise StateFormatError("deepseek4 compress_ratios is shorter than block_count")
        if any(ratios[il] != 0 for il in range(n_layer, n_all)):
            raise StateFormatError("deepseek4 nextn (MTP) layers must have compress ratio 0")
        embd = int(get("embedding_length"))
        n_head0 = int(get("attention.head_count", layer=0))
        head = int(get("attention.key_length", embd // n_head0 if n_head0 > 0 else None))
        value = int(get("attention.value_length", head))
        rope = rope_runtime(meta, runtime=runtime)
        compress = CompressSpec(tuple(ratios[:n_layer]), int(get("attention.indexer.key_length")),
                                int(get("attention.indexer.head_count", 0)), int(get("attention.indexer.top_k", 0)),
                                float(get("attention.compress_rope_freq_base")),
                                float(get("attention.layer_norm_rms_epsilon")))
        return cls(architecture, ("sliding",) * n_layer + ("nextn",) * nextn, (1,) * n_all, (head,) * n_all,
                   (value,) * n_all, n_swa=int(get("attention.sliding_window")), rope_type="norm",
                   n_rot=int(get("rope.dimension_count", head)), rope_freq_base=rope["freq_base"],
                   rope_freq_base_swa=rope["freq_base"], memory_layout="dsv4", rope_freq_scale=rope["freq_scale"],
                   rope_freq_scale_swa=rope["freq_scale"], rope_ext_factor=rope["ext_factor"],
                   rope_attn_factor=rope["attn_factor"], rope_beta_fast=rope["beta_fast"],
                   rope_beta_slow=rope["beta_slow"], rope_n_ctx_orig=rope["n_ctx_orig"], compress=compress)

    @classmethod
    def _generic_from_metadata(cls, architecture, meta, get, runtime):
        """Plain-attention hybrids and recurrent-only models, as llama_model::load_hparams
        (src/llama-model.cpp:1270-1366), the architecture's load_arch_hparams (src/models/<arch>.cpp)
        and create_memory's layer filters (src/llama-model.cpp:2493-2600) set them up."""
        prefix = architecture + "."
        nextn = int(get("nextn_predict_layers", 0))
        n_all = int(get("block_count"))
        n_layer = n_all - nextn
        if n_layer <= 0 or nextn < 0:
            raise StateFormatError("invalid block count")

        def per_layer(key, default):
            # get_key_or_arr: a scalar for every layer or an array of exactly n_layer_all values
            value = meta.get(prefix + key, default)
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != n_all:
                    raise StateFormatError(f"{prefix + key} holds {len(value)} values for {n_all} layers")
                return [int(x) for x in value]
            return [int(value)] * n_all

        embd = int(get("embedding_length"))
        heads = per_layer("attention.head_count", 0)
        # head_count_kv defaults to head_count, layer by layer
        heads_kv = per_layer("attention.head_count_kv", 0) if prefix + "attention.head_count_kv" in meta else heads
        n_ff = per_layer("feed_forward_length", 0)
        if nextn and architecture not in NEXTN_FILTERED and not (
                architecture in ("nemotron_h", "nemotron_h_moe") and all(n_ff[il] for il in range(n_layer, n_all))):
            # nemotron's filters take attention layers with n_ff == 0, and nextn layers are never
            # recurrent (src/llama-model.cpp:2525-2531, src/models/nemotron-h.cpp:12-16): the MTP block
            # the converter writes (an attention and an MLP layer, n_ff = the MoE width,
            # conversion/nemotron.py:262-345) sits in neither section. With n_ff == 0, and under the
            # default filters, nextn layers would sit in the main attention cache; not modelled
            raise StateFormatError(f"{architecture} with nextn_predict_layers > 0 is not supported "
                                   "(nextn layers would sit in the attention cache)")
        # head sizes and n_rot follow layer 0's head count (src/llama-model.cpp:1335-1366); plamo2 reads
        # the head sizes again whatever layer 0 holds
        full_k = full_v = n_rot_full = 0
        if heads[0] > 0:
            full_k = int(get("attention.key_length", embd // heads[0]))
            full_v = int(get("attention.value_length", embd // heads[0]))
            n_rot_full = int(get("rope.dimension_count", full_k))
        if architecture == "plamo2":
            full_k, full_v = int(get("attention.key_length", full_k)), int(get("attention.value_length", full_v))
        swa_k, swa_v = int(get("attention.key_length_swa", full_k)), int(get("attention.value_length_swa", full_v))
        n_rot_swa = int(get("rope.dimension_count_swa", n_rot_full))
        if architecture in RECURRENT_ARCHITECTURES:
            types = ["recurrent"] * n_layer
        elif architecture in NEXTN_FILTERED:
            # attention.recurrent_layers, else every full_attention_interval-th layer is attention
            # (src/models/qwen3next.cpp, qwen35moe.cpp, qwen4exp.cpp; minimax-01.cpp with interval 8)
            if meta.get(prefix + "attention.recurrent_layers") is not None:
                recurrent = per_layer("attention.recurrent_layers", 0)
            else:
                interval = int(get("full_attention_interval", 8 if architecture == "minimax-01" else 4))
                if interval <= 0:
                    raise StateFormatError("invalid full-attention interval")
                recurrent = [(il + 1) % interval != 0 for il in range(n_all)]
            types = ["recurrent" if recurrent[il] else "full" for il in range(n_layer)]
        elif architecture == "falcon-h1":
            # every layer runs attention and an SSM in parallel and sits in both caches
            types = ["parallel"] * n_layer
        elif architecture in ("nemotron_h", "nemotron_h_moe"):
            # recurrent: no KV heads and no FFN; attention: KV heads and no FFN; MLP-only layers: neither
            types = ["mlp" if n_ff[il] else "recurrent" if heads_kv[il] == 0 else "full" for il in range(n_layer)]
        else:
            # jamba, granitehybrid, plamo2, lfm2, lfm2moe: a layer is recurrent iff it has no KV heads
            types = ["recurrent" if heads_kv[il] == 0 else "full" for il in range(n_layer)]
        n_swa = 0
        if architecture == "lfm2" and int(meta.get(prefix + "attention.sliding_window", 0) or 0) > 0:
            # every attention layer slides (src/models/lfm2.cpp: is_swa = !is_recr)
            n_swa = int(get("attention.sliding_window"))
            types = ["sliding" if kind == "full" else kind for kind in types]
        dims = {"full": (full_k, full_v), "parallel": (full_k, full_v), "sliding": (swa_k, swa_v)}
        dim_k = [dims[kind][0] if kind in dims else 0 for kind in types]
        dim_v = [dims[kind][1] if kind in dims else 0 for kind in types]
        kv_heads = [heads_kv[il] if types[il] in dims else 0 for il in range(n_layer)]
        rope_type = ROPE_TYPES[architecture]
        if architecture == "granitehybrid" and not meta.get(prefix + "rope.scaling.finetuned", True):
            rope_type = "none"
        rope = {"freq_base": 10000.0, "freq_scale": 1.0, "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0,
                "beta_slow": 1.0, "n_ctx_orig": 0}
        sections = (0, 0, 0, 0)
        # the graphs rotate every layer with hparams.n_rot() = n_rot(0) (src/llama-graph.cpp:1463); a
        # model whose layer 0 has no heads gets n_rot 0 and so no rotation at all (inferred: a
        # converted PLaMo2, whose layer 0 is Mamba and which writes no rope.dimension_count)
        n_rot = n_rot_swa if types[0] == "sliding" else n_rot_full
        if n_rot == 0:
            rope_type = "none"
        if rope_type == "none":
            n_rot = 0
        else:
            rope = rope_runtime(meta, runtime=runtime)
            if rope_type == "imrope":
                sections = (tuple(int(x) for x in get("rope.dimension_sections")) + (0, 0, 0, 0))[:4]
        recurrent_bytes, p_floats, ple_layers = None, 0, ()
        if "recurrent" in types or "parallel" in types:
            r_floats, s_floats = recurrent_row_floats(meta, RECURRENT_FAMILY[architecture])
            recurrent_bytes = (4 * r_floats, 4 * s_floats)
            if architecture == "qwen4exp":
                p_floats, ple_layers = ple_rows_from_metadata(meta)
        memory = "hybrid_iswa" if n_swa else MEMORY[architecture]
        indexer = None
        if memory == "hybrid_idx":
            # raw W_k x keys on every attention layer; the copied hparams keep the model's V width
            # with one KV head, so the section also stores (never written) V rows
            # (src/llama-memory-hybrid-idx.cpp:44-56, src/models/qwen4exp.cpp:575-590)
            indexer = IndexerSpec(tuple(il for il, kind in enumerate(types) if kind == "full"),
                                  int(get("attention.indexer.key_length")), 0, "none", hadamard=False,
                                  top_k=int(get("attention.indexer.top_k")),
                                  n_head=int(get("attention.indexer.head_count")), value_size=full_v)
        return cls(architecture, tuple(types), tuple(kv_heads), tuple(dim_k), tuple(dim_v), n_swa=n_swa,
                   rope_type=rope_type, n_rot=n_rot, rope_freq_base=rope["freq_base"],
                   rope_freq_base_swa=rope["freq_base"], rope_sections=sections, recurrent_row_bytes=recurrent_bytes,
                   memory_layout=memory, rope_freq_scale=rope["freq_scale"], rope_freq_scale_swa=rope["freq_scale"],
                   rope_ext_factor=rope["ext_factor"], rope_attn_factor=rope["attn_factor"],
                   rope_beta_fast=rope["beta_fast"], rope_beta_slow=rope["beta_slow"],
                   rope_n_ctx_orig=rope["n_ctx_orig"], indexer=indexer, recurrent_p_bytes=4 * p_floats,
                   ple_layers=ple_layers)

    @classmethod
    def from_gguf(cls, path, *, runtime=None):
        """Spec from a GGUF file's metadata (reads the header only, via gguf_tensors / gguf-py).

        granitehybrid passes its optional rope_freqs.weight tensor to ggml_rope_ext as frequency
        factors (model.get_rope_factors, src/models/granite-hybrid.cpp:106-120, 210-216), so the
        spec carries it, or () when the GGUF has none."""
        import gguf_tensors
        meta = gguf_tensors.read_metadata(path, max_array=1 << 16)
        if meta.get("general.architecture") == "deepseek4":
            # the loader keeps nextn layers only when the GGUF has the MTP probe tensor
            # (src/models/deepseek4.cpp:20-26); without it every block is a trunk layer
            n_all, nextn = int(meta.get("deepseek4.block_count", 0)), int(meta.get("deepseek4.nextn_predict_layers", 0))
            probe = f"blk.{n_all - nextn}.nextn.eh_proj.weight"
            if nextn > 0 and probe not in gguf_tensors.reader(path)[1]:
                meta = dict(meta, **{"deepseek4.nextn_predict_layers": 0})
        spec = cls.from_metadata(meta, runtime=runtime)
        if spec.architecture == "granitehybrid" and spec.rope_type != "none":
            factors = ()
            if "rope_freqs.weight" in gguf_tensors.reader(path)[1]:
                factors = np.asarray(gguf_tensors.read_tensor(path, "rope_freqs.weight"), dtype=np.float32).ravel()
            spec = dataclasses.replace(spec, rope_freq_factors=tuple(float(x) for x in factors))
        return spec


def _metadata_getter(meta, prefix):
    def get(key, default=None, layer=None):
        value = meta.get(prefix + key, default)
        if value is None:
            raise StateFormatError(f"missing GGUF field {prefix + key}")
        if isinstance(value, (list, tuple, np.ndarray)):
            if layer is None:
                return list(value)
            if layer >= len(value):
                raise StateFormatError(f"invalid layer array {prefix + key}")
            value = value[layer]
        return value
    return get


def _indexer_from_metadata(architecture, meta, get, n_layer, n_rot):
    prefix = architecture + "."
    head_size = int(get("attention.indexer.key_length"))
    types = meta.get(prefix + "attention.indexer.types")
    if architecture == "deepseek32":
        full = [True] * n_layer
    elif types is not None:
        full = [bool(get("attention.indexer.types", layer=il)) if isinstance(types, (list, tuple, np.ndarray))
                else bool(types) for il in range(n_layer)]
    elif architecture == "glm-dsa" and int(get("context_length")) < 1048576:
        # GLM 5 / 5.1 without indexer types: every layer owns an indexer (src/models/glm-dsa.cpp:59-66)
        full = [True] * n_layer
    else:
        raise StateFormatError(f"{architecture} needs attention.indexer.types (the GLM-5.2 default pattern "
                               "and missing types are not modelled)")
    if architecture == "hy_v4" and not full[0]:
        raise StateFormatError("hy_v4: layer 0 must own an indexer")
    rope_type, window, hadamard, eps = INDEXER_CONVENTIONS[architecture]
    if eps == "rms":
        eps = float(get("attention.layer_norm_rms_epsilon"))
    return IndexerSpec(tuple(il for il in range(n_layer) if full[il]), head_size, n_rot, rope_type,
                       rope_offset=head_size - n_rot if window == "tail" else 0, hadamard=hadamard,
                       top_k=int(get("attention.indexer.top_k")), n_head=int(get("attention.indexer.head_count", 0)),
                       norm_eps=float(eps))


def recurrent_row_floats(meta, family):
    """(R, S) F32 elements per recurrent row, as llama_hparams::n_embd_r()/n_embd_s() compute them
    (src/llama-hparams.cpp:208-258) from the keys the family's loaders read.

    ssm (Mamba, Mamba-2, GatedDeltaNet: qwen35/qwen3next, jamba, falcon-h1, granite-hybrid,
    nemotron-h, plamo2): R = (d_conv - 1) * (d_inner + 2 * n_group * d_state), S = d_state * d_inner,
    with n_group 0 for the Mamba-1 loaders (mamba, jamba), which never read ssm.group_count;
    kda (kimi-linear, kimi-k3, bailingmoe3): R = 3 * (d_conv - 1, or 3) * n_head * head_dim,
    S = head_dim^2 * n_head with n_head the first layer's head count;
    rwkv: R = token_shift_count (default 2) * n_embd, S = n_embd * wkv.head_size;
    shortconv (lfm2): R = n_embd * (l_cache - 1), S = 0 (no ssm keys are loaded);
    lightning (minimax-01): R = 0 (no conv state), S = head_dim^2 * n_head with head_dim the key length.
    ssm, kda, shortconv and lightning match the saves of the pinned test-llama-archs models and
    rwkv those of the gguf-py RWKV fixtures (tiny_fixture.create_rwkv_fixture; test_layouts.py)."""
    architecture = meta.get("general.architecture")
    get = _metadata_getter(meta, f"{architecture}.")
    if family == "ssm":
        conv, inner = int(get("ssm.conv_kernel")), int(get("ssm.inner_size"))
        state = int(get("ssm.state_size"))
        groups = 0 if architecture in SSM_WITHOUT_GROUPS else int(get("ssm.group_count"))
        return max(conv - 1, 0) * (inner + 2 * groups * state), state * inner
    if family == "kda":
        conv, head_dim = int(get("ssm.conv_kernel")), int(get("kda.head_dim"))
        n_head = int(get("attention.head_count", layer=0))
        return 3 * (conv - 1 if conv > 0 else 3) * n_head * head_dim, head_dim * head_dim * n_head
    embd = int(get("embedding_length"))
    if family == "rwkv":
        return int(get("token_shift_count", 2)) * embd, embd * int(get("wkv.head_size"))
    if family == "shortconv":
        return embd * (int(get("shortconv.l_cache")) - 1), 0
    if family == "lightning":
        n_head = int(get("attention.head_count", layer=0))
        head_dim = int(get("attention.key_length", embd // n_head))
        return 0, head_dim * head_dim * n_head
    raise StateFormatError(f"unknown recurrent family {family!r}")


def ple_rows_from_metadata(meta):
    """(P F32 elements, PLE layers) of the qwen4exp PLE conv row (llama_hparams::ple_conv_state,
    src/llama-hparams.cpp:268-275): (conv_kernel - 1) * ngram_size * hyper_connection.count * n_embd
    on the ple.layers, or (0, ()) when the model has none (no ple.layers, which leaves ple_n_heads 0
    in src/models/qwen4exp.cpp, no PLE heads, no kernel, or hc_mult 0)."""
    architecture = meta.get("general.architecture")
    get = _metadata_getter(meta, f"{architecture}.")
    ngram, per_ngram = int(get("ple.ngram_size", 0)), int(get("ple.heads_per_ngram", 0))
    kernel, hc_mult = int(get("ple.conv_kernel", 0)), int(get("hyper_connection.count", 0))
    layers = tuple(sorted(int(x) for x in get("ple.layers", [])))
    n_heads = max(ngram - 1, 0) * per_ngram
    floats = (kernel - 1) * ngram * hc_mult * int(get("embedding_length")) if layers and n_heads and kernel else 0
    return floats, layers if floats else ()


def rope_runtime(meta, *, runtime=None):
    """The RoPE values a llama_context hands to every ggml_rope_ext call, from GGUF metadata and
    the context's settings (llama_context_params names, see ModelSpec.from_metadata), as
    llama_model::load_hparams (src/llama-model.cpp:1293-1330) and llama_context's constructor
    (src/llama-context.cpp:113-210) derive them, in float32 like the C code:

    freq_base, freq_scale (1 / rope.scaling.factor; 1 with scaling none), ext_factor (1 for yarn
    unless set), n_ctx_orig (yarn_orig_ctx, rope.scaling.original_context_length, context_length),
    beta_fast/beta_slow (32/1: these loaders do not read the GGUF beta keys) and attn_factor:
    with ext_factor != 0 it is overwritten with get_mscale(f, m) / get_mscale(f, m_all)
    / (1 + 0.1 ln f), where m_all = yarn_log_multiplier / 0.1 and m = m_all for arch deepseek2
    only (1 otherwise), or get_mscale(f, 1) / (1 + 0.1 ln f) without the key; then times
    rope.scaling.attn_factor. ggml multiplies cos/sin by attn_factor * (1 + 0.1 ln(1 / freq_scale))
    when ext_factor != 0 (Rope.magnitude), so the effective k_pe magnitude is 1 for DeepSeek-V2
    style mscale == mscale_all_dim and for m_all = 1, and 1 + 0.1 ln f without the key (up to
    float rounding)."""
    f32 = np.float32
    architecture = meta.get("general.architecture")
    prefix = f"{architecture}."
    runtime = dict(runtime or {})
    unknown = set(runtime) - set(RUNTIME_ROPE_KEYS)
    if unknown:
        raise StateFormatError(f"unknown runtime RoPE settings {sorted(unknown)}")
    # None means unset (a flag the receiving server was not given); numbers must be finite
    runtime = {key: value for key, value in runtime.items() if value is not None}
    for key, value in runtime.items():
        if key == "rope_scaling_type":
            continue
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise StateFormatError(f"runtime RoPE setting {key} = {value!r} is not a finite number")
    n_ctx_train = int(meta.get(prefix + "context_length", 0) or 0)
    n_ctx_orig_yarn = int(meta.get(prefix + "rope.scaling.original_context_length", n_ctx_train) or 0)
    base_train = f32(meta.get(prefix + "rope.freq_base", 10000.0))
    scaling = meta.get(prefix + "rope.scaling.type", "linear")
    factor = f32(meta.get(prefix + "rope.scaling.factor", meta.get(prefix + "rope.scale_linear", 0.0)) or 0.0)
    scale_train = f32(1.0) if factor == 0 else f32(f32(1.0) / factor)
    rope_attn_factor = f32(meta.get(prefix + "rope.scaling.attn_factor", 1.0))
    log_mul = f32(0.0)
    if architecture in YARN_LOG_MUL_ARCHITECTURES and meta.get(prefix + "rope.scaling.yarn_log_multiplier"):
        log_mul = f32(f32(meta[prefix + "rope.scaling.yarn_log_multiplier"]) / f32(0.1))
    scaling = runtime.get("rope_scaling_type") or scaling
    if scaling not in ROPE_SCALING_TYPES or scaling == "longrope":
        raise StateFormatError(f"RoPE scaling {scaling!r} is not supported")
    freq_base = f32(runtime.get("rope_freq_base") or base_train)
    freq_scale = f32(runtime.get("rope_freq_scale") or scale_train)
    n_ctx_orig = int(runtime.get("yarn_orig_ctx") or n_ctx_orig_yarn or n_ctx_train)
    if scaling == "none":
        freq_scale = f32(1.0)
    ext_factor = f32(runtime.get("yarn_ext_factor", -1.0))
    if ext_factor < 0:
        ext_factor = f32(1.0 if scaling == "yarn" else 0.0)
    attn_factor = f32(runtime.get("yarn_attn_factor", -1.0))
    if attn_factor < 0:
        attn_factor = f32(1.0)
    if ext_factor != 0:
        logf = _logf

        def get_mscale(scale, mscale):
            return f32(1.0) if scale <= 1 else f32(f32(f32(f32(0.1) * mscale) * logf(scale)) + f32(1.0))

        scale = f32(f32(1.0) / freq_scale)
        if log_mul != 0:
            mscale = log_mul if architecture == "deepseek2" and log_mul != 1 else f32(1.0)
            attn_factor = f32(get_mscale(scale, mscale) / get_mscale(scale, log_mul))
        else:
            attn_factor = get_mscale(scale, f32(1.0))
        attn_factor = f32(attn_factor * f32(f32(1.0) / f32(f32(1.0) + f32(f32(0.1) * logf(scale)))))
    attn_factor = f32(attn_factor * rope_attn_factor)
    beta_fast = f32(runtime.get("yarn_beta_fast", -1.0))
    beta_slow = f32(runtime.get("yarn_beta_slow", -1.0))
    return {"freq_base": float(freq_base), "freq_scale": float(freq_scale), "ext_factor": float(ext_factor),
            "attn_factor": float(attn_factor), "beta_fast": float(beta_fast if beta_fast >= 0 else 32.0),
            "beta_slow": float(beta_slow if beta_slow >= 0 else 1.0), "n_ctx_orig": n_ctx_orig,
            "scaling": scaling, "yarn_log_mul": float(log_mul)}


# -- parsed objects ----------------------------------------------------------------

@dataclasses.dataclass
class KVStream:
    """Cells of one stream in written (cell-index) order; k/v are logical [cells, heads, dim].

    K-only layers (MLA, indexer keys) have no entry in v (see value_rows)."""

    positions: np.ndarray                 # int32 [cells]
    seq_ids: tuple                        # per cell, tuple of seq ids
    ext: object                           # int32 [cells, 3] (x, y, tok) or None
    v_trans: bool
    k: dict                               # model layer -> float16/float32 [cells, heads, dim]
    v: dict
    k_types: dict = dataclasses.field(default_factory=dict)   # layer -> ggml type
    v_types: dict = dataclasses.field(default_factory=dict)

    @property
    def n_cells(self):
        return len(self.positions)


@dataclasses.dataclass
class KVCache:
    name: str                             # kv | base | swa | attn | indexer
    layers: tuple
    streams: list                         # KVStream or None (empty stream), len = n_stream

    def stream(self):
        """The only nonempty stream (sequence saves write one)."""
        full = [s for s in self.streams if s is not None]
        if len(full) != 1:
            raise StateFormatError(f"{self.name} cache has {len(full)} nonempty streams, expected 1")
        return full[0]


@dataclasses.dataclass
class OpaqueRows:
    ggml_type: int
    row_bytes: int
    data: bytes                           # cell_count * row_bytes


@dataclasses.dataclass
class RecurrentState:
    name: str
    layers: tuple
    positions: np.ndarray
    seq_ids: tuple
    n_layer: int
    r: dict                               # layer -> OpaqueRows
    s: dict
    p: dict = dataclasses.field(default_factory=dict)   # PLE layer -> OpaqueRows (ggml_type None: not written)

    @property
    def n_cells(self):
        return len(self.positions)

    @property
    def nbytes(self):
        return sum(len(rows.data) for part in (self.r, self.s, self.p) for rows in part.values())


@dataclasses.dataclass
class PooledRows:
    """A DeepSeek V4 K-row section (csa, hca or lid): the first n_rows pooled rows of every layer.

    rows[layer] is [n_stream, n_rows, width] float16/float32, or [n_stream, n_rows, width / 32]
    Q8_0_BLOCK in a q8_0 cache; n_stream is 1 for a sequence save."""

    name: str
    layers: tuple
    n_stream: int
    n_rows: int
    rows: dict
    types: dict = dataclasses.field(default_factory=dict)   # layer -> ggml type


@dataclasses.dataclass
class CompressorRing:
    """A DeepSeek V4 compressor ring section (csa_state, hca_state or lid_state): per layer the F32
    a (kv) and score (APE included) rows, [n_stream, state_size, state_width]; the a and score of
    the token at position p sit in row p % state_size."""

    name: str
    layers: tuple
    n_stream: int
    kv: dict
    score: dict


@dataclasses.dataclass
class SequenceState:
    """The memory payload of one sequence: sections in state_write order."""

    spec: ModelSpec
    flags: int
    parts: list

    def part(self, name):
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"state has no {name} section (flags {self.flags})")

    def to_bytes(self):
        out = []
        layouts = self.spec.layouts(self.flags)
        if [p.name for p in self.parts] != [layout.name for layout in layouts]:
            raise StateFormatError("state sections do not match the model spec")
        if self.spec.memory == "dsv4":
            out.append(_pack("<III", DSV4_STATE_MAGIC, DSV4_STATE_VERSION, int(self.flags == FLAGS_PARTIAL_ONLY)))
        writers = {"kv": _write_kv, "rows": _write_pooled_rows, "ring": _write_compressor_ring}
        for part, layout in zip(self.parts, layouts):
            if tuple(part.layers) != layout.layers:
                raise StateFormatError(f"{part.name} layers {part.layers} differ from the spec {layout.layers}")
            out.append(writers[layout.kind](self.spec, part, layout) if layout.kind in writers
                       else _write_recurrent(self.spec, part))
        return b"".join(out)


@dataclasses.dataclass
class StateFile:
    """A llama_state_seq_save_file file: token words then the memory payload."""

    token_words: np.ndarray               # int32, as stored (plain or server-packed)
    state: SequenceState

    @property
    def tokens(self):
        """Token ids of a text-only state; a state holding media is refused (see media)."""
        return decode_token_words(self.token_words)[0]

    @property
    def media(self):
        """MediaTokens of the token words: the token list (LLAMA_TOKEN_NULL inside media), the
        media table and the positions (parse_media_tokens)."""
        return parse_media_tokens(self.token_words)

    @property
    def server_packed(self):
        return self.media.server_packed

    def check_positions(self):
        """Refuse a state whose sections do not all end at the last token's cell.

        A native save of n text tokens holds cells up to position n - 1 in every KV section
        (base and SWA, attention, MLA and indexer) and its recurrent cell at n - 1; decoding
        resumes at position n. With media the last cell sits at MediaTokens.cell_positions()[-1]
        (an M-RoPE image's cells all sit at its first position) and decoding resumes at
        MediaTokens.pos_next(). After a restore llama.cpp only sees one section's end (the SWA
        maximum for iSWA, the smaller of the attention and recurrent maxima for hybrid
        memory, the MLA cache for DSA), so a section reaching past the end would get duplicate
        positions without any error.
        """
        media = self.media
        n = len(media.tokens)
        last = int(media.cell_positions()[-1]) if n else -1
        for part in self.state.parts:
            if isinstance(part, CompressorRing):
                continue
            if isinstance(part, PooledRows):
                # one row per completed block (dsv4_state_n_used_k_rows, src/llama-kv-cache-dsv4.cpp:40-49)
                ratio = self.state.spec.pool(part.name).ratio
                if part.n_rows != (last + 1) // ratio:
                    raise StateFormatError(f"{part.name} section holds {part.n_rows} rows; with the last cell at "
                                           f"{last} it must hold {(last + 1) // ratio}")
                continue
            if isinstance(part, KVCache):
                positions = [np.asarray(s.positions) for s in part.streams if s is not None]
                positions = np.concatenate(positions) if positions else np.zeros(0, dtype=np.int32)
            else:
                positions = np.asarray(part.positions)
                if len(positions) and (positions != last).any():
                    raise StateFormatError(f"{part.name} cells sit at positions {sorted(set(positions.tolist()))}; "
                                           f"with {n} tokens the recurrent state must sit at {last}")
            if n == 0 and len(positions):
                raise StateFormatError(f"{part.name} section holds {len(positions)} cells for zero tokens")
            if n and (len(positions) == 0 or int(positions.max()) != last):
                end = int(positions.max()) if len(positions) else None
                raise StateFormatError(f"{part.name} section ends at position {end}; with {n} tokens every "
                                       f"section must end at {last} (the last token's cell)")

    def to_bytes(self, *, check_positions=True):
        """Serialize; check_positions=False skips check_positions (only for deliberately bad test files)."""
        words = np.asarray(self.token_words, dtype="<i4")
        parse_media_tokens(words)
        if check_positions:
            self.check_positions()
        return b"".join([_pack("<III", LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, len(words)),
                         words.tobytes(), self.state.to_bytes()])


@dataclasses.dataclass
class StateBlob:
    """A llama_state_seq_get_data blob: io magic, source seq id, memory payload."""

    seq_id: int
    state: SequenceState

    def to_bytes(self):
        return _pack("<Ii", IO_MAGIC, self.seq_id) + self.state.to_bytes()


# -- token words --------------------------------------------------------------------

def decode_token_words(words):
    """(tokens int32 array, server_packed) from a text-only state file's token words.

    Media chunks are refused here (a token list with LLAMA_TOKEN_NULL runs is no input to a text
    model); parse_media_tokens reads them."""
    media = parse_media_tokens(words)
    if media.media:
        raise StateFormatError("server tokens state holds media chunks; parse_media_tokens reads them and "
                               "truncate_state cuts the state before them")
    return media.tokens, media.server_packed


def token_words(tokens, *, server_packed=False):
    """Token words for a state file: plain ids, or server_tokens::serialize() of text tokens."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or (tokens < 0).any() or (tokens > 2**31 - 1).any():
        raise StateFormatError("tokens must be nonnegative int32 ids")
    tokens = tokens.astype("<i4")
    if not server_packed:
        return tokens
    head = np.asarray([LLAMA_TOKEN_NULL, SERVER_TOKENS_STATE_VERSION, len(tokens)], dtype="<i4")
    return np.concatenate([head, tokens, np.zeros(1, dtype="<i4")])


@dataclasses.dataclass(frozen=True)
class MediaChunk:
    """One image or audio chunk of llama-server's token words: an mtmd_input_chunk as
    mtmd_input_chunk_save writes it (metadata only, never pixels or samples; tools/mtmd/mtmd.cpp:
    29-136, 262-411, 2399-2450). It occupies n_tokens cells (LLAMA_TOKEN_NULL entries of the token
    list) and advances the position by n_pos (mtmd_input_chunk_get_n_pos, mtmd.cpp:2351-2361,
    2525-2537)."""

    start: int                    # token-list index of its first cell
    n_tokens: int                 # cells (one KV cell per attention layer each)
    n_pos: int                    # positions: max(nx, ny) for M-RoPE images, else n_tokens
    modality: str                 # "image" | "audio"
    id: str                       # mtmd chunk id: sha256 hex of the uploaded file bytes ("" without one)
    pos_type: str = None          # images: "normal" | "mrope" | "hunyuanvl" (MTMD_POS_TYPES)
    nx: int = 0                   # images: tokens per row and rows (n_tokens = nx * ny * frames)
    ny: int = 0
    blob: bytes = b""             # the saved chunk, written back verbatim

    @property
    def end(self):
        return self.start + self.n_tokens


@dataclasses.dataclass(frozen=True, eq=False)
class MediaTokens:
    """server_tokens as a slot file stores it (tools/server/server-common.cpp:553-617): the token
    list (one entry per cell; LLAMA_TOKEN_NULL inside a media chunk) and the media chunks by start."""

    tokens: np.ndarray            # int32 [cells]
    media: tuple                  # MediaChunk, increasing start
    server_packed: bool           # server_tokens::serialize() words (else a plain token list)

    def pos_next(self, n_tokens=None):
        """server_tokens::pos_next (server-common.cpp:372-418): the position after the first n_tokens
        entries (all by default), media counting n_pos each; n_tokens must not cut a chunk."""
        n = len(self.tokens) if n_tokens is None else int(n_tokens)
        if not 0 <= n <= len(self.tokens):
            raise StateFormatError(f"{n} tokens out of {len(self.tokens)}")
        pos = n
        for chunk in self.media:
            if chunk.start < n < chunk.end:
                raise StateFormatError(f"token {n} lies inside the media chunk at {chunk.start}..{chunk.end}")
            if chunk.end <= n:
                pos += chunk.n_pos - chunk.n_tokens
        return pos

    def cell_positions(self):
        """int64 [cells]: the position of each token-list entry's cell (the time component for
        M-RoPE: every cell of an M-RoPE image sits at the image's first position; HunyuanVL and
        NORMAL images and audio count up, mtmd_image_tokens_get_decoder_pos, mtmd.cpp:2466-2520)."""
        out = np.arange(len(self.tokens), dtype=np.int64)
        for chunk in self.media:
            if chunk.pos_type == "mrope":
                out[chunk.start:chunk.end] = out[chunk.start]
            # entries after the chunk move by what it adds beyond one position per cell
            out[chunk.end:] += chunk.n_pos - chunk.n_tokens
        return out

    def to_words(self):
        """The token words (server_tokens::serialize, server-common.cpp:278-316, 553-573): the
        marker, version, u32 n, the n entries, u32 n_media, the start indices, then each chunk as
        u32 size and its bytes, zero-padded to whole words; or the plain ids."""
        tokens = np.asarray(self.tokens, dtype="<i4")
        if not self.server_packed:
            if self.media:
                raise StateFormatError("a plain token list cannot hold media")
            return token_words(tokens)
        parts = [_pack("<iII", LLAMA_TOKEN_NULL, SERVER_TOKENS_STATE_VERSION, len(tokens)), tokens.tobytes(),
                 _pack("<I", len(self.media))]
        parts += [_pack("<I", chunk.start) for chunk in self.media]
        parts += [_pack("<I", len(chunk.blob)) + bytes(chunk.blob) for chunk in self.media]
        data = b"".join(parts)
        data += bytes(-len(data) % 4)
        words = np.frombuffer(data, dtype="<i4").copy()
        parse_media_tokens(words)
        return words

    def truncated(self, cut):
        """The first `cut` entries (server_tokens::keep_first, server-common.cpp:636-667: a chunk
        is kept whole or dropped, never cut)."""
        cut = int(cut)
        self.pos_next(cut)
        return MediaTokens(np.asarray(self.tokens[:cut], dtype=np.int32).copy(),
                           tuple(chunk for chunk in self.media if chunk.end <= cut), self.server_packed)


def parse_media_chunk(blob, start=0):
    """MediaChunk of one mtmd_input_chunk_save blob (MTMD_SERIALIZATION_VERSION 2): u64 version,
    u32 type, u64 n_text + i32 text tokens, u8 has_image [u32 nx, ny, pos_type, image_idx,
    n_temporal_merge, string id, u8 is_audio, u64 n_entries, per entry u8 viewsep, u8 newline,
    i32 lead_pad, i32 nx, i32 ny], u8 has_audio [u32 n_tokens, string id, the same entry batch];
    strings are u64 length + bytes (tools/mtmd/mtmd.cpp:29-136, 262-322, 356-411)."""
    blob = bytes(blob)
    reader = _Reader(blob)
    version = reader.u64("media chunk version")
    if version != MTMD_SERIALIZATION_VERSION:
        raise StateFormatError(f"media chunk serialization version {version}; only "
                               f"{MTMD_SERIALIZATION_VERSION} is supported")
    chunk_type = reader.u32("media chunk type")
    if chunk_type not in MTMD_CHUNK_TYPES:
        raise StateFormatError(f"media chunk type {chunk_type} (a text chunk or unknown) in the media table")
    n_text = reader.u64("media chunk text token count")
    reader.array("<i4", n_text, "media chunk text tokens")

    def flag(what):
        value = reader.scalar("<B", what)
        if value > 1:
            raise StateFormatError(f"{what}: invalid flag {value}")
        return bool(value)

    def string(what):
        size = reader.u64(what + " length")
        return reader.raw(size, what).decode("utf-8", errors="surrogateescape")

    def entries(what):
        flag(what + " is_audio")
        count = reader.u64(what + " entry count")
        sizes = []
        for i in range(count):
            flag(f"{what} entry {i} viewsep")
            flag(f"{what} entry {i} newline")
            reader.i32(f"{what} entry {i} lead_pad")
            sizes.append((reader.i32(f"{what} entry {i} nx"), reader.i32(f"{what} entry {i} ny")))
        return sizes

    image = audio = None
    if flag("media chunk has_image"):
        nx, ny = reader.u32("image nx"), reader.u32("image ny")
        pos_type, _ = reader.u32("image pos type"), reader.u32("image index")
        merge = reader.u32("image temporal merge")
        if pos_type >= len(MTMD_POS_TYPES):
            raise StateFormatError(f"invalid image position type {pos_type}")
        image = (nx, ny, MTMD_POS_TYPES[pos_type], merge, string("image id"), len(entries("image")))
    if flag("media chunk has_audio"):
        audio = (reader.u32("audio token count"), string("audio id"))
        entries("audio")
    reader.finish("media chunk")
    modality = MTMD_CHUNK_TYPES[chunk_type]
    if modality == "image":
        if image is None:
            raise StateFormatError("image chunk without image tokens")
        nx, ny, pos_type, merge, chunk_id, n_frames = image
        if pos_type == "hunyuanvl":
            # [BOI] [row + newline] * ny [EOI] (mtmd_image_tokens::n_tokens, mtmd.cpp:216-232)
            n_tokens = (nx + 1) * ny + 2
        else:
            frames = n_frames if merge <= 1 else -(-n_frames // merge)
            n_tokens = nx * ny * frames
        n_pos = max(nx, ny) if pos_type == "mrope" else n_tokens
        return MediaChunk(int(start), int(n_tokens), int(n_pos), "image", chunk_id, pos_type, int(nx), int(ny), blob)
    if audio is None:
        raise StateFormatError("audio chunk without audio tokens")
    return MediaChunk(int(start), int(audio[0]), int(audio[0]), "audio", audio[1], None, 0, 0, blob)


def parse_media_tokens(words):
    """MediaTokens of a state file's token words, with server_tokens::deserialize's checks
    (server-common.cpp:575-617) and validate's (738-772): every LLAMA_TOKEN_NULL run is exactly a
    media chunk starting at its first entry, chunks have n_tokens > 0 and n_pos > 0, nothing but
    zero padding follows. Plain token lists (no -1 marker) hold no media."""
    words = np.asarray(words)
    if words.ndim != 1 or words.dtype.kind not in "iu":
        raise StateFormatError("token words must be a 1-D integer array")
    words = words.astype("<i4")
    if len(words) == 0 or words[0] != LLAMA_TOKEN_NULL:
        if (words < 0).any():
            raise StateFormatError("plain token list holds negative ids")
        return MediaTokens(words, (), False)
    if len(words) < 4 or words[1] != SERVER_TOKENS_STATE_VERSION:
        raise StateFormatError("unsupported server tokens state (marker, version)")
    data = words.tobytes()
    reader = _Reader(data, 8)
    n = reader.u32("server tokens count")
    tokens = reader.array("<i4", n, "server tokens").copy()
    n_media = reader.u32("media count")
    starts = [reader.u32(f"media {i} start") for i in range(n_media)]
    chunks = []
    for i, start in enumerate(starts):
        size = reader.u32(f"media {i} size")
        if size == 0:
            raise StateFormatError(f"media {i} is empty")
        chunks.append(parse_media_chunk(reader.raw(size, f"media {i} chunk"), start))
    padding = data[reader.pos:]
    if len(padding) >= 4 or any(padding):
        raise StateFormatError("trailing data in server tokens state")
    if starts != sorted(set(starts)):
        raise StateFormatError("media start indices must increase")
    covered = np.zeros(n, dtype=bool)
    for chunk in chunks:
        if chunk.n_tokens <= 0 or chunk.n_pos <= 0 or chunk.end > n:
            raise StateFormatError(f"media chunk at {chunk.start} with {chunk.n_tokens} tokens and {chunk.n_pos} "
                                   f"positions does not fit {n} tokens")
        if covered[chunk.start:chunk.end].any() or (tokens[chunk.start:chunk.end] != LLAMA_TOKEN_NULL).any():
            raise StateFormatError(f"media chunk at {chunk.start} overlaps text or another chunk")
        covered[chunk.start:chunk.end] = True
    if (tokens[~covered] < 0).any():
        raise StateFormatError("server tokens state holds negative ids outside media chunks")
    return MediaTokens(tokens, tuple(chunks), True)


# -- KV section ------------------------------------------------------------------------

def _parse_kv(spec, reader, layout):
    name, layers, role = layout.name, layout.layers, layout.role
    n_stream = reader.u32(f"{name} n_stream")
    if n_stream < 1 or n_stream > 256:
        raise StateFormatError(f"{name} cache has an unsupported stream count {n_stream}")
    streams = []
    for s in range(n_stream):
        where = f"{name} stream {s}"
        cells = reader.u32(f"{where} cell_count")
        if cells == 0:
            streams.append(None)
            continue
        positions = np.empty(cells, dtype=np.int32)
        ext = np.empty((cells, 3), dtype=np.int32) if layout.ext else None
        seq_ids = []
        for i in range(cells):
            positions[i] = reader.i32(f"{where} cell {i} pos")
            n_seq = reader.u32(f"{where} cell {i} n_seq_id")
            if n_seq != 1:
                raise StateFormatError(f"{where} cell {i} belongs to {n_seq} sequences; only single-sequence "
                                       "states are supported (state_read_meta requires n_seq_id == 1)")
            if ext is not None:
                ext[i] = reader.array("<i4", 3, f"{where} cell {i} ext")
            seq_ids.append(tuple(int(x) for x in reader.array("<i4", n_seq, f"{where} cell {i} seq ids")))
        if (positions < 0).any():
            raise StateFormatError(f"{where} has negative positions")
        v_trans = reader.u32(f"{where} v_trans")
        n_layer = reader.u32(f"{where} n_layer")
        if v_trans not in (0, 1):
            raise StateFormatError(f"{where} has invalid v_trans {v_trans}")
        if n_layer != len(layers):
            raise StateFormatError(f"{where} holds {n_layer} layers; the model's {name} cache has {len(layers)}")
        k, v, k_types, v_types = {}, {}, {}, {}
        for il in layers:
            shape = spec.kv_shape(il, "k", role)
            dtype, ggml_type, stored = _row_header(reader, f"{where} layer {il} K", shape,
                                                   quantized=spec.memory == "dsv4")
            k[il] = reader.array(dtype, cells * stored[0] * stored[1], f"{where} layer {il} K rows").reshape(
                (cells,) + stored)
            k_types[il] = ggml_type
        for il in layers:
            shape = spec.kv_shape(il, "v", role)
            if shape is None:
                continue
            what = f"{where} layer {il} V"
            if not v_trans:
                dtype, ggml_type, _ = _row_header(reader, what, shape)
                v[il] = reader.array(dtype, cells * shape[0] * shape[1], what + " rows").reshape((cells,) + shape)
            else:
                ggml_type = reader.i32(what + " type")
                dtype = _dtype(ggml_type, what)
                element = reader.u32(what + " element size")
                n_embd = reader.u32(what + " n_embd_v_gqa")
                if element != dtype.itemsize or n_embd != shape[0] * shape[1]:
                    raise StateFormatError(f"{what}: element size {element} / n_embd_v_gqa {n_embd} do not "
                                           f"match {dtype.itemsize} / {shape[0] * shape[1]}")
                transposed = reader.array(dtype, cells * n_embd, what + " elements").reshape(n_embd, cells)
                v[il] = transposed.T.reshape((cells,) + shape)
            v_types[il] = ggml_type
        streams.append(KVStream(positions, tuple(seq_ids), ext, bool(v_trans), k, v, k_types, v_types))
    return KVCache(name, tuple(layers), streams)


def _dtype(ggml_type, what):
    if ggml_type not in GGML_DTYPES:
        raise StateFormatError(f"{what}: ggml type {ggml_type} is not supported (F32 = 0 and F16 = 1 only)")
    return GGML_DTYPES[ggml_type]


def _storage(ggml_type, shape, what, *, quantized=False):
    """(numpy dtype, stored shape) of rows of logical shape (..., dim) in a ggml type: the shape itself
    for F32/F16, (..., dim / 32) blocks for q8_0 where `quantized` allows it (DeepSeek V4 sections)."""
    if ggml_type in GGML_DTYPES:
        return GGML_DTYPES[ggml_type], tuple(shape)
    if quantized and ggml_type in QUANT_BLOCKS:
        block, size = QUANT_BLOCKS[ggml_type]
        if shape[-1] % size:
            raise StateFormatError(f"{what}: {shape[-1]} elements do not fill whole {size}-element blocks")
        return block, tuple(shape[:-1]) + (shape[-1] // size,)
    raise StateFormatError(f"{what}: ggml type {ggml_type} is not supported (F32 = 0 and F16 = 1"
                           + (", and q8_0 = 8 in DeepSeek V4 caches" if quantized else "") + " only)")


def _type_of(array, *, quantized=False):
    """The ggml type of a row array (float32, float16, or Q8_0_BLOCK where `quantized`), or None."""
    if array.dtype == Q8_0_BLOCK:
        return GGML_TYPE_Q8_0 if quantized else None
    if array.dtype.fields is not None:
        return None
    return {np.dtype("<f4"): GGML_TYPE_F32, np.dtype("<f2"): GGML_TYPE_F16}.get(array.dtype.newbyteorder("<"))


def _row_header(reader, what, shape, *, quantized=False):
    ggml_type = reader.i32(what + " type")
    dtype, stored = _storage(ggml_type, shape, what, quantized=quantized)
    row = reader.u64(what + " row size")
    expected = dtype.itemsize * stored[0] * stored[1]
    if row != expected:
        raise StateFormatError(f"{what}: row size {row} does not match {expected} ({shape} {dtype})")
    return dtype, ggml_type, stored


def _write_kv(spec, cache, layout):
    role = layout.role
    out = [_pack("<I", len(cache.streams))]
    for s, stream in enumerate(cache.streams):
        if stream is None:
            out.append(_pack("<I", 0))
            continue
        cells = stream.n_cells
        positions = np.asarray(stream.positions)
        if cells == 0 or positions.shape != (cells,) or len(stream.seq_ids) != cells:
            raise StateFormatError(f"{cache.name} stream {s} has inconsistent cell metadata")
        if layout.ext != (stream.ext is not None) or (stream.ext is not None and np.shape(stream.ext) != (cells, 3)):
            raise StateFormatError(f"{cache.name} stream {s}: cell ext must be [cells, 3] exactly for M-RoPE and PLE "
                                   "models (ModelSpec.has_cell_ext)")
        out.append(_pack("<I", cells))
        meta = []
        for i in range(cells):
            ids = stream.seq_ids[i]
            if len(ids) != 1:
                raise StateFormatError(f"{cache.name} stream {s} cell {i} must belong to one sequence")
            meta.append(_pack("<iI", int(positions[i]), len(ids)))
            if stream.ext is not None:
                meta.append(np.asarray(stream.ext[i], dtype="<i4").tobytes())
            meta.append(np.asarray(ids, dtype="<i4").tobytes())
        out.append(b"".join(meta))
        with_v = tuple(il for il in cache.layers if spec.kv_shape(il, "v", role) is not None)
        if set(stream.k) != set(cache.layers) or set(stream.v) != set(with_v):
            if with_v == tuple(cache.layers):
                raise StateFormatError(f"{cache.name} stream {s} must hold K and V for layers {cache.layers}")
            raise StateFormatError(f"{cache.name} stream {s} must hold K for layers {cache.layers} and V for "
                                   f"{with_v} (K-only layers carry no V)")
        out.append(_pack("<II", int(bool(stream.v_trans)), len(cache.layers)))
        for il in cache.layers:
            array, ggml_type = _layer_array(stream, il, "k", (cells,) + spec.kv_shape(il, "k", role), cache.name,
                                            quantized=spec.memory == "dsv4")
            out.append(_pack("<iQ", ggml_type, array.nbytes // cells))
            out.append(array.tobytes())
        for il in with_v:
            shape = spec.kv_shape(il, "v", role)
            array, ggml_type = _layer_array(stream, il, "v", (cells,) + shape, cache.name)
            n_embd = shape[0] * shape[1]
            if stream.v_trans:
                out.append(_pack("<iII", ggml_type, array.dtype.itemsize, n_embd))
                out.append(np.ascontiguousarray(array.reshape(cells, n_embd).T).tobytes())
            else:
                out.append(_pack("<iQ", ggml_type, array.dtype.itemsize * n_embd))
                out.append(array.tobytes())
    return b"".join(out)


def _layer_array(stream, layer, key, shape, name, *, quantized=False):
    array = np.asarray((stream.k if key == "k" else stream.v)[layer])
    types = stream.k_types if key == "k" else stream.v_types
    ggml_type = _type_of(array, quantized=quantized)
    if ggml_type is None:
        raise StateFormatError(f"{name} layer {layer} {key.upper()} must be float16 or float32"
                               + (" or q8_0 blocks" if quantized else "") + f", not {array.dtype}")
    if layer in types and types[layer] != ggml_type:
        raise StateFormatError(f"{name} layer {layer} {key.upper()} dtype changed from ggml type {types[layer]}")
    _, stored = _storage(ggml_type, shape, f"{name} layer {layer} {key.upper()}", quantized=quantized)
    if array.shape != stored:
        raise StateFormatError(f"{name} layer {layer} {key.upper()} has shape {array.shape}, expected {stored}")
    if array.dtype.fields is not None:
        return np.ascontiguousarray(array), ggml_type
    return np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")), ggml_type


# -- recurrent section -------------------------------------------------------------------

def _parse_recurrent(spec, reader, name, layers):
    cells = reader.u32(f"{name} cell_count")
    positions = np.empty(cells, dtype=np.int32)
    seq_ids = []
    for i in range(cells):
        positions[i] = reader.i32(f"{name} cell {i} pos")
        n_seq = reader.u32(f"{name} cell {i} n_seq_id")
        if n_seq != 0:
            raise StateFormatError(f"{name} cell {i} lists {n_seq} sequences; single-sequence saves write 0")
        seq_ids.append(())
    s_trans = reader.u32(f"{name} s_trans")
    n_layer = reader.u32(f"{name} n_layer")
    if s_trans != 0:
        raise StateFormatError(f"{name}: transposed recurrent state is not supported")
    if n_layer != spec.n_layer_trunk:
        raise StateFormatError(f"{name} records {n_layer} model layers; the model has {spec.n_layer_trunk}")
    expected = spec.recurrent_row_bytes

    def row_size(what, index):
        row = reader.u64(what + " row size")
        known = expected[index] if expected else None
        # a family may have 0-byte rows (src/llama-hparams.cpp:208-258); unknown sizes must be nonzero
        if row % 4 or (known is None and row == 0) or (known is not None and row != known):
            raise StateFormatError(f"{what}: unexpected row size {row}"
                                   + (f", model expects {known}" if known is not None else ""))
        return int(row)

    parts = {"r": {}, "s": {}, "p": {}}
    for il in layers:
        what = f"{name} layer {il} R"
        ggml_type = reader.i32(what + " type")
        if ggml_type != GGML_TYPE_F32:
            raise StateFormatError(f"{what}: recurrent rows must be F32 (type 0), got {ggml_type}")
        row = row_size(what, 0)
        parts["r"][il] = OpaqueRows(ggml_type, row, reader.raw(cells * row, what + " rows"))
        if il in spec.ple_layers:
            # the PLE conv row follows its layer's R rows, with a row size but no type
            what = f"{name} layer {il} P"
            row = reader.u64(what + " row size")
            if row != spec.recurrent_p_bytes:
                raise StateFormatError(f"{what}: unexpected row size {row}, model expects {spec.recurrent_p_bytes}")
            parts["p"][il] = OpaqueRows(None, int(row), reader.raw(cells * row, what + " rows"))
    for il in layers:
        what = f"{name} layer {il} S"
        ggml_type = reader.i32(what + " type")
        if ggml_type != GGML_TYPE_F32:
            raise StateFormatError(f"{what}: recurrent rows must be F32 (type 0), got {ggml_type}")
        row = row_size(what, 1)
        parts["s"][il] = OpaqueRows(ggml_type, row, reader.raw(cells * row, what + " rows"))
    return RecurrentState(name, tuple(layers), positions, tuple(seq_ids), n_layer, parts["r"], parts["s"], parts["p"])


def _write_recurrent(spec, part):
    cells = part.n_cells
    if len(part.seq_ids) != cells or any(part.seq_ids):
        raise StateFormatError(f"{part.name}: single-sequence recurrent cells carry no seq ids")
    out = [_pack("<I", cells)]
    out.extend(_pack("<iI", int(p), 0) for p in np.asarray(part.positions))
    out.append(_pack("<II", 0, part.n_layer))
    p_rows = getattr(part, "p", None) or {}
    for key, rows_by_layer in (("r", part.r), ("s", part.s)):
        if set(rows_by_layer) != set(part.layers):
            raise StateFormatError(f"{part.name} must hold rows for layers {part.layers}")
        if key == "r" and set(p_rows) != set(spec.ple_layers) & set(part.layers):
            raise StateFormatError(f"{part.name} must hold PLE rows for layers {spec.ple_layers}")
        for il in part.layers:
            rows = rows_by_layer[il]
            if len(rows.data) != cells * rows.row_bytes:
                raise StateFormatError(f"{part.name} layer {il}: {len(rows.data)} bytes for {cells} rows "
                                       f"of {rows.row_bytes}")
            out.append(_pack("<iQ", rows.ggml_type, rows.row_bytes))
            out.append(bytes(rows.data))
            if key == "r" and il in p_rows:
                ple = p_rows[il]
                if len(ple.data) != cells * ple.row_bytes:
                    raise StateFormatError(f"{part.name} layer {il} PLE: {len(ple.data)} bytes for {cells} rows "
                                           f"of {ple.row_bytes}")
                out.append(_pack("<Q", ple.row_bytes))
                out.append(bytes(ple.data))
    return b"".join(out)


def recurrent_reset(part, position):
    """The recurrent section of `part`'s layout reset to zeros in one cell at `position`.

    A zero row is exactly the state a fresh sequence starts from (build_rs clears row rs_z and
    copies it to cells without a source, src/llama-graph.cpp:3427-3432), so this is the R0
    reset for every recurrent family: R, S and PLE rows alike, 0-byte rows included."""
    def zeros(rows):
        return {layer: OpaqueRows(r.ggml_type, r.row_bytes, bytes(r.row_bytes)) for layer, r in rows.items()}
    return RecurrentState(part.name, part.layers, np.asarray([position], dtype=np.int32), ((),), part.n_layer,
                          zeros(part.r), zeros(part.s), zeros(getattr(part, "p", None) or {}))


# -- DeepSeek V4 sections ------------------------------------------------------------------

def _stream_count(reader, what):
    n_stream = reader.u32(what)
    if n_stream < 1 or n_stream > 256:
        raise StateFormatError(f"{what}: unsupported stream count {n_stream}")
    return n_stream


# (Their writers are not named _write_<kind>: universal_kv's streaming state writer looks those up per
# section and hands a payload with any other section kind to SequenceState.to_bytes, which also
# writes the DSV4 header.)

def _parse_pooled_rows(spec, reader, layout):
    """A K-row section (dsv4_state_write_k_cache, src/llama-kv-cache-dsv4.cpp:332-362)."""
    pool, where = spec.pool(layout.name), f"{layout.name} rows"
    version = reader.u32(f"{where} version")
    if version != DSV4_K_ROWS_VERSION:
        raise StateFormatError(f"{where}: version {version} is not supported (only {DSV4_K_ROWS_VERSION})")
    n_rows = reader.u32(f"{where} n_rows")
    n_stream = _stream_count(reader, f"{where} stream count")
    n_layer = reader.u32(f"{where} n_layer")
    if n_layer != len(layout.layers):
        raise StateFormatError(f"{where} hold {n_layer} layers; the model has {len(layout.layers)}")
    rows, types = {}, {}
    for il in layout.layers:
        what = f"{where} layer {il}"
        if reader.u32(what + " id") != il:
            raise StateFormatError(f"{what}: layer ids differ from the model's")
        ggml_type = reader.i32(what + " type")
        dtype, stored = _storage(ggml_type, (pool.width,), what, quantized=True)
        ne0, count, row = reader.u64(what + " ne0"), reader.u64(what + " rows"), reader.u64(what + " row size")
        if ne0 != pool.width or count != n_rows or row != dtype.itemsize * stored[0]:
            raise StateFormatError(f"{what}: ne0 {ne0}, rows {count}, row size {row} do not match {pool.width}, "
                                   f"{n_rows}, {dtype.itemsize * stored[0]}")
        rows[il] = reader.array(dtype, n_stream * n_rows * stored[0], what + " data").reshape(n_stream, n_rows,
                                                                                            stored[0])
        types[il] = ggml_type
    return PooledRows(layout.name, tuple(layout.layers), n_stream, n_rows, rows, types)


def _write_pooled_rows(spec, part, layout):
    pool = spec.pool(layout.name)
    out = [_pack("<IIII", DSV4_K_ROWS_VERSION, part.n_rows, part.n_stream, len(layout.layers))]
    for il in layout.layers:
        what = f"{part.name} layer {il}"
        array = np.asarray(part.rows[il])
        ggml_type = _type_of(array, quantized=True)
        if ggml_type is None:
            raise StateFormatError(f"{what} rows must be float16, float32 or q8_0 blocks, not {array.dtype}")
        if il in part.types and part.types[il] != ggml_type:
            raise StateFormatError(f"{what} dtype changed from ggml type {part.types[il]}")
        _, stored = _storage(ggml_type, (part.n_stream, part.n_rows, pool.width), what, quantized=True)
        if array.shape != stored:
            raise StateFormatError(f"{what} rows have shape {array.shape}, expected {stored}")
        array = np.ascontiguousarray(array) if array.dtype.fields else np.ascontiguousarray(
            array, dtype=array.dtype.newbyteorder("<"))
        out.append(_pack("<IiQQQ", il, ggml_type, pool.width, part.n_rows, array.dtype.itemsize * stored[-1]))
        out.append(array.tobytes())
    return b"".join(out)


def _parse_compressor_ring(spec, reader, layout):
    """A compressor ring section (llama_dsv4_comp_state::state_write, src/llama-kv-cache-dsv4.cpp:1080-1116)."""
    pool, where = spec.pool(layout.name[:-len("_state")]), layout.name
    version, ratio, size, width = (reader.u32(f"{where} {field}")
                                   for field in ("version", "ratio", "state_size", "n_embd_state"))
    if version != DSV4_RING_VERSION or (ratio, size, width) != (pool.ratio, pool.state_size, pool.state_width):
        raise StateFormatError(f"{where}: version {version}, ratio {ratio}, state {size} x {width} differ from "
                               f"{DSV4_RING_VERSION}, {pool.ratio}, {pool.state_size} x {pool.state_width}")
    n_stream = _stream_count(reader, f"{where} stream count")
    n_layer = reader.u32(f"{where} n_layer")
    if n_layer != len(layout.layers):
        raise StateFormatError(f"{where} holds {n_layer} layers; the model has {len(layout.layers)}")
    kv, score = {}, {}
    for il in layout.layers:
        what = f"{where} layer {il}"
        if reader.u32(what + " id") != il:
            raise StateFormatError(f"{what}: layer ids differ from the model's")
        for target, name in ((kv, "kv"), (score, "score")):
            ggml_type = reader.i32(f"{what} {name} type")
            ne0, rows, row = (reader.u64(f"{what} {name} {field}") for field in ("ne0", "rows", "row size"))
            if ggml_type != GGML_TYPE_F32 or (ne0, rows, row) != (width, size, 4 * width):
                raise StateFormatError(f"{what} {name}: type {ggml_type}, ne0 {ne0}, rows {rows}, row size {row} "
                                       f"are not F32 {width} x {size}")
            target[il] = reader.array("<f4", n_stream * size * width, f"{what} {name} data").reshape(
                n_stream, size, width)
    return CompressorRing(layout.name, tuple(layout.layers), n_stream, kv, score)


def _write_compressor_ring(spec, part, layout):
    pool = spec.pool(layout.name[:-len("_state")])
    shape = (part.n_stream, pool.state_size, pool.state_width)
    out = [_pack("<IIIIII", DSV4_RING_VERSION, pool.ratio, pool.state_size, pool.state_width, part.n_stream,
                 len(layout.layers))]
    for il in layout.layers:
        out.append(_pack("<I", il))
        for name, rows in (("kv", part.kv[il]), ("score", part.score[il])):
            rows = np.asarray(rows)
            if rows.dtype != np.dtype("<f4") or rows.shape != shape:
                raise StateFormatError(f"{part.name} layer {il} {name} must be float32 {shape}, not "
                                       f"{rows.dtype} {rows.shape}")
            out.append(_pack("<iQQQ", GGML_TYPE_F32, pool.state_width, pool.state_size, 4 * pool.state_width))
            out.append(np.ascontiguousarray(rows).tobytes())
    return b"".join(out)


# -- truncation -------------------------------------------------------------------------

def media_cut(media, *, before=()):
    """Token-list index where the first media chunk's span begins (len(tokens) without media):
    the chunk's start, moved back over the `before` token ids that sit right in front of it.
    mtmd appends the projector's img_beg/aud_beg text (e.g. Qwen-VL's <|vision_start|>) to the
    text chunk before each medium (tools/mtmd/mtmd.cpp:1311-1354), so they are not part of the
    rendered conversation text; pass their ids to cut before them."""
    if isinstance(media, StateFile):
        media = media.media
    if not media.media:
        return len(media.tokens)
    cut = media.media[0].start
    before = [int(t) for t in before]
    if before and cut >= len(before) and np.asarray(media.tokens[cut - len(before):cut]).tolist() == before:
        cut -= len(before)
    return cut


def truncate_state(state_file, cut, *, recurrent=None, sliding="native"):
    """state_file cut to its first `cut` token-list entries, as a native save of that prefix.

    Every KV section drops its cells at positions >= MediaTokens.pos_next(cut); all cells of an
    M-RoPE image share one position, so `cut` must not fall inside a media chunk (an image is
    kept or dropped whole, like server_tokens::keep_first). The token words keep the entries and
    media chunks before the cut, packed as before.

    A recurrent section integrates every token up to the end and cannot be cut: with
    recurrent=None a cut before the end is refused (the state needs R0), with recurrent="reset"
    every recurrent section becomes recurrent_reset(part, position of the last kept cell). A cut
    at the end (cut == len(tokens), e.g. media_cut of a state without media) is no cut: every
    section stays as saved, the recurrent rows too, whatever `recurrent` says.

    A sliding (SWA) section of a native save holds only the last n_swa positions before its own
    end, so it rarely holds the window a save at an earlier cut would: sliding="native" keeps
    exactly that window and refuses the cut when any of it is missing (the result restores like
    a native save); sliding="keep" keeps whatever cells lie before the cut, for reading source
    rows (the result is then not checked with check_positions and may not restore)."""
    if recurrent not in (None, "reset"):
        raise ValueError("recurrent must be None or 'reset'")
    if state_file.state.spec.memory == "dsv4":
        raise StateFormatError("a DeepSeek V4 state cannot be cut: its pooled rows and compressor rings hold "
                               "every token, and only the last n_swa tokens keep their own latents")
    if sliding not in ("native", "keep"):
        raise ValueError("sliding must be 'native' or 'keep'")
    media = state_file.media
    n, cut = len(media.tokens), int(cut)
    if not 0 < cut <= n:
        raise StateFormatError(f"cut {cut} lies outside 1..{n}")
    kept = media.truncated(cut)
    dropped_from = media.pos_next(cut)
    cell_positions = media.cell_positions()[:cut]
    last = int(cell_positions[-1])
    state = state_file.state
    spec = state.spec
    parts = []
    for part, layout in zip(state.parts, spec.layouts(state.flags)):
        if layout.kind == "recurrent":
            if cut == n:
                parts.append(part)
            elif recurrent is None:
                raise StateFormatError(f"the {part.name} section integrates all {n} tokens and cannot be cut at {cut}; "
                                       "the state needs R0 (truncate_state(..., recurrent='reset'))")
            else:
                parts.append(recurrent_reset(part, last))
            continue
        streams = []
        for stream in part.streams:
            if stream is None:
                streams.append(None)
                continue
            positions = np.asarray(stream.positions, dtype=np.int64)
            keep = positions < dropped_from
            if layout.sliding and sliding == "native":
                # is_masked_swa (src/llama-hparams.h:466-478) against the new end
                keep &= last - positions < spec.n_swa
                needed = np.unique(cell_positions[last - cell_positions < spec.n_swa])
                missing = np.setdiff1d(needed, positions[keep])
                if len(missing):
                    raise StateFormatError(f"{part.name} section lacks positions {missing[:5].tolist()} a native save "
                                           "at the cut keeps (they left its sliding window); re-prefill instead")
            index = np.flatnonzero(keep)
            if not len(index):
                streams.append(None)
                continue
            # kept cells in one run (the usual cut, a prefix) are sliced: the rows stay views of the
            # saved arrays (of a memory-mapped file: nothing is copied or read) instead of copies
            if int(index[-1]) - int(index[0]) + 1 == len(index):
                index = slice(int(index[0]), int(index[-1]) + 1)
                seq_ids = tuple(stream.seq_ids[index])
            else:
                seq_ids = tuple(stream.seq_ids[i] for i in index)
            streams.append(KVStream(positions[index].astype(np.int32), seq_ids,
                                    None if stream.ext is None else np.asarray(stream.ext)[index], stream.v_trans,
                                    {il: np.asarray(rows)[index] for il, rows in stream.k.items()},
                                    {il: np.asarray(rows)[index] for il, rows in stream.v.items()},
                                    dict(stream.k_types), dict(stream.v_types)))
        parts.append(KVCache(part.name, part.layers, streams))
    out = StateFile(kept.to_words(), SequenceState(spec, state.flags, parts))
    if sliding == "native":
        out.check_positions()
    return out


# -- top level --------------------------------------------------------------------------

def _parse_payload(spec, reader, flags):
    layouts = spec.layouts(flags)
    if spec.memory == "dsv4":
        # llama_kv_cache_dsv4::state_read requires all three and the mode to match the flags
        magic, version, mode = (reader.u32(f"DSV4 {field}") for field in ("magic", "version", "mode"))
        if magic != DSV4_STATE_MAGIC or version != DSV4_STATE_VERSION:
            raise StateFormatError(f"not a DeepSeek V4 state (magic {magic:#010x}, version {version})")
        if mode != int(flags == FLAGS_PARTIAL_ONLY):
            raise StateFormatError(f"DeepSeek V4 state mode {mode} does not match flags {flags}")
    parsers = {"kv": _parse_kv, "rows": _parse_pooled_rows, "ring": _parse_compressor_ring}
    parts = []
    for layout in layouts:
        parts.append(parsers[layout.kind](spec, reader, layout) if layout.kind in parsers
                     else _parse_recurrent(spec, reader, layout.name, layout.layers))
    return SequenceState(spec, flags, parts)


def parse_state(data, spec, *, flags=FLAGS_NONE):
    """Parse a bare memory payload (what memory->state_write wrote)."""
    reader = _Reader(data)
    state = _parse_payload(spec, reader, flags)
    reader.finish("memory state")
    return state


def parse_blob(data, spec, *, flags=FLAGS_NONE):
    """Parse a llama_state_seq_get_data(_ext) blob written with `flags`."""
    reader = _Reader(data)
    magic = reader.u32("io magic")
    if magic != IO_MAGIC:
        raise StateFormatError(f"not a sequence state blob (magic {magic:#010x})")
    seq_id = reader.i32("seq id")
    state = _parse_payload(spec, reader, flags)
    reader.finish("sequence state blob")
    return StateBlob(seq_id, state)


def parse_file(data, spec):
    """Parse a llama_state_seq_save_file file (llama-server slot save)."""
    reader = _Reader(data)
    magic = reader.u32("file magic")
    version = reader.u32("file version")
    if magic != LLAMA_STATE_SEQ_MAGIC or version != LLAMA_STATE_SEQ_VERSION:
        raise StateFormatError(f"unknown (magic, version) for sequence state file: {magic:#010x}, {version}")
    count = reader.u32("token count")
    words = reader.array("<i4", count, "token words").copy()
    parse_media_tokens(words)
    state = _parse_payload(spec, reader, FLAGS_NONE)
    reader.finish("sequence state file")
    return StateFile(words, state)


def read_file(path, spec):
    return parse_file(Path(path).read_bytes(), spec)


def write_new(path, data):
    """Write bytes to a new file atomically; never overwrite."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)
    return path


def write_file(path, state_file, *, check_positions=True):
    return write_new(path, state_file.to_bytes(check_positions=check_positions))


def template_layout(state):
    """What a target template (any native save, one token is enough) fixes about the layout a
    translated state must reproduce: per section in write order its name, role, layers, whether
    its cells carry an ext and, for KV sections, the nonempty stream's index and count, cache
    dtype, v_trans and the layers that store V; for recurrent sections the row bytes (R, S, PLE)
    and the layer count."""
    if isinstance(state, (StateFile, StateBlob)):
        state = state.state
    out = []
    for part, layout in zip(state.parts, state.spec.layouts(state.flags)):
        entry = {"name": part.name, "kind": layout.kind, "role": layout.role, "layers": tuple(part.layers),
                 "sliding": layout.sliding, "ext": layout.ext}
        if layout.kind == "rows":
            types = sorted(set(part.types.values()))
            names = {code: name for name, code in CACHE_TYPES.items()}
            entry.update({"n_stream": part.n_stream, "n_rows": part.n_rows,
                          "dtype": names.get(types[0]) if len(types) == 1 else None})
        elif layout.kind == "ring":
            entry.update({"n_stream": part.n_stream})
        elif layout.kind == "kv":
            index = [i for i, s in enumerate(part.streams) if s is not None]
            stream = part.streams[index[0]] if len(index) == 1 else None
            types = sorted(set(stream.k_types.values())) if stream is not None else []
            entry.update({"n_stream": len(part.streams), "stream": index[0] if len(index) == 1 else None,
                          "dtype": {code: name for name, code in CACHE_TYPES.items()}[types[0]]
                          if len(types) == 1 else None,
                          "v_trans": stream.v_trans if stream is not None else None,
                          "v_layers": tuple(il for il in part.layers
                                            if state.spec.kv_shape(il, "v", layout.role) is not None),
                          "nextn_layers": tuple(il for il in part.layers if state.spec.layer_types[il] == "nextn")})
        else:
            first = part.layers[0] if part.layers else None
            entry.update({"n_layer": part.n_layer, "cells": part.n_cells,
                          "r_bytes": part.r[first].row_bytes if first is not None else None,
                          "s_bytes": part.s[first].row_bytes if first is not None else None,
                          "p_layers": tuple(sorted(getattr(part, "p", None) or {}))})
        out.append(entry)
    return out


# -- slot resume sidecar (<slot file>.resume) ---------------------------------------------

@dataclasses.dataclass
class Checkpoint:
    """common_prompt_checkpoint as the resume file stores it (common/common.h:1165-1181)."""

    n_tokens: int
    pos_min: int
    pos_max: int
    data_tgt: bytes          # llama_state_seq_get_data_ext(..., PARTIAL_ONLY) blob of the target
    data_dft: bytes = b""
    data_spec: bytes = b""

    def target_state(self, spec):
        return parse_blob(self.data_tgt, spec, flags=FLAGS_PARTIAL_ONLY)


@dataclasses.dataclass
class ResumeFile:
    """slot_resume_file (tools/server/server-context.cpp:41-121): draft/speculative state and checkpoints."""

    draft: bytes = b""
    speculative: bytes = b""
    checkpoints: list = dataclasses.field(default_factory=list)

    def to_bytes(self):
        def blob(data):
            return _pack("<Q", len(data)) + bytes(data)
        out = [_pack("<II", RESUME_MAGIC, RESUME_VERSION), blob(self.draft), blob(self.speculative),
               _pack("<I", len(self.checkpoints))]
        for cp in self.checkpoints:
            if cp.n_tokens < 0 or cp.pos_min < 0 or cp.pos_max < cp.pos_min:
                raise StateFormatError("invalid checkpoint positions")
            out += [_pack("<qii", cp.n_tokens, cp.pos_min, cp.pos_max), blob(cp.data_tgt), blob(cp.data_dft),
                    blob(cp.data_spec)]
        return b"".join(out)


def parse_resume(data, *, n_prompt_tokens=None):
    """Parse a slot resume file with the reader's checks (count, blob sizes, positions, no trailing data)."""
    reader = _Reader(data)
    remaining = len(reader.view)
    if reader.u32("resume magic") != RESUME_MAGIC or reader.u32("resume version") != RESUME_VERSION:
        raise StateFormatError("unsupported slot resume format")

    def blob(what):
        nonlocal remaining
        size = reader.u64(what + " size")
        if size > remaining:
            raise StateFormatError(f"invalid slot resume blob size for {what}")
        remaining -= size
        return reader.raw(size, what)

    draft, speculative = blob("draft"), blob("speculative")
    count = reader.u32("checkpoint count")
    if count > 65536 or count > remaining // 40:
        raise StateFormatError("invalid checkpoint count")
    checkpoints = []
    for i in range(count):
        n_tokens, pos_min, pos_max = reader.i64(f"checkpoint {i} n_tokens"), reader.i32(f"checkpoint {i} pos_min"), \
            reader.i32(f"checkpoint {i} pos_max")
        tgt, dft, spec_data = blob(f"checkpoint {i} target"), blob(f"checkpoint {i} draft"), blob(f"checkpoint {i} spec")
        if n_tokens < 0 or pos_min < 0 or pos_max < pos_min or (n_prompt_tokens is not None and n_tokens > n_prompt_tokens):
            raise StateFormatError("invalid checkpoint positions")
        checkpoints.append(Checkpoint(n_tokens, pos_min, pos_max, tgt, dft, spec_data))
    reader.finish("slot resume file")
    return ResumeFile(draft, speculative, checkpoints)


# -- RoPE (ggml-cpu conventions) -----------------------------------------------------------

def rope_angles(positions, n_rot, base, sections=None):
    """float32 angles [n, n_rot/2] as ggml builds its rope cache (no YaRN, no freq factors).

    With `sections` (IMROPE) each rotated pair takes theta_t/h/w/e as
    ggml_mrope_cache_init does. For text tokens the first three position components
    equal the token position and the fourth is 0 (llm_graph_input_pos::set_input,
    src/llama-graph.cpp:131-141), so pairs whose sector falls to theta_e are not rotated.
    Qwen3.8's sections [11, 11, 10, 0] over 32 pairs never reach theta_e.
    """
    if n_rot <= 0 or n_rot % 2:
        raise ValueError("n_rot must be a positive even number")
    theta_scale = rope_theta_scale(base, n_rot)
    theta = np.asarray(positions, dtype=np.int64).astype(np.float32)
    angles = np.empty((len(theta), n_rot // 2), dtype=np.float32)
    for index in range(n_rot // 2):
        angles[:, index] = theta
        theta = (theta * theta_scale).astype(np.float32)
    if sections is not None:
        sections = [int(x) for x in sections]
        if len(sections) != 4 or min(sections) < 0 or sum(sections[:3]) == 0:
            raise ValueError("IMROPE sections must be four nonnegative counts, not all of the first three zero")
        total = sum(sections)
        for index in range(n_rot // 2):
            sector = index % total
            text = ((sector % 3 == 1 and sector < 3 * sections[1]) or (sector % 3 == 2 and sector < 3 * sections[2])
                    or (sector % 3 == 0 and sector < 3 * sections[0]))
            if not text:
                angles[:, index] = 0
    return angles


_LIBM = []


def _libm():
    """(cosf, sinf, powf, logf) from the C math library through ctypes, or None when it cannot be loaded."""
    if not _LIBM:
        try:
            import ctypes
            import ctypes.util
            libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
            for name, n_args in (("cosf", 1), ("sinf", 1), ("powf", 2), ("logf", 1)):
                getattr(libm, name).restype = ctypes.c_float
                getattr(libm, name).argtypes = [ctypes.c_float] * n_args
            _LIBM.append((libm.cosf, libm.sinf, libm.powf, libm.logf))
        except (OSError, AttributeError):
            _LIBM.append(None)
    return _LIBM[0]


def _logf(x):
    """float32 logf (libm through ctypes when available, else the float64 log rounded to float32)."""
    libm = _libm()
    if libm is None:
        return np.float32(math.log(float(x)))
    return np.float32(libm[3](float(np.float32(x))))


def rope_theta_scale(base, n_rot):
    """float32 powf(freq_base, -2.0f/n_dims) as ggml_compute_forward_rope computes it
    (ggml/src/ggml-cpu/ops.cpp:6143); libm powf through ctypes when available, else the
    float64 power rounded to float32. numpy's float32 power can differ from powf by 1 ulp
    (base 1e7, n_dims 16), which changes every rotated key."""
    base, exponent = np.float32(base), np.float32(-2.0) / np.float32(n_rot)
    libm = _libm()
    if libm is None:
        return np.float32(np.power(np.float64(base), np.float64(exponent)))
    return np.float32(libm[2](float(base), float(exponent)))


def yarn_corr_dims(n_dims, n_ctx_orig, base, beta_fast=32.0, beta_slow=1.0):
    """float32 (low, high) as ggml_rope_yarn_corr_dims computes them (ggml/src/ggml.c:4418-4430):
    floor/ceil of n_dims * logf(n_ctx_orig / (beta * 2 * pi)) / (2 * logf(base)), clamped to
    [0, n_dims - 1]."""
    f32 = np.float32

    def corr_dim(beta):
        ratio = f32(f32(n_ctx_orig) / f32(f32(f32(beta) * f32(2)) * f32(math.pi)))
        return f32(f32(f32(n_dims) * _logf(ratio)) / f32(f32(2) * _logf(f32(base))))

    start, end = np.floor(corr_dim(beta_fast)), np.ceil(corr_dim(beta_slow))
    return f32(max(f32(0), start)), f32(min(f32(n_dims - 1), end))


def dsv4_rope_attn_factor(freq_scale, ext_factor):
    """float32 attn_factor of DeepSeek V4's compressed-layer RoPE calls: 1 without YaRN, else
    1 / (1 + 0.1 logf(1 / freq_scale)), which cancels the YaRN magnitude ggml applies
    (dsv4_rope_attn_factor, src/models/deepseek4.cpp:11-17; llama's sources are not FMA-contracted)."""
    f32 = np.float32
    if ext_factor == 0:
        return 1.0
    return float(f32(f32(1.0) / f32(f32(1.0) + f32(f32(0.1) * _logf(f32(f32(1.0) / f32(freq_scale)))))))


def _fma32(a, b, c):
    """float32 fused multiply-add, emulated in float64 (exact product, one rounding to float64
    then one to float32)."""
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(np.float32)


def _cos_sin(angles):
    """float32 cos/sin as ggml's rope_yarn computes them (cosf/sinf); libm through ctypes when available."""
    libm = _libm()
    flat = np.asarray(angles, dtype=np.float32).ravel()
    if libm is None:
        wide = flat.astype(np.float64)
        return np.cos(wide).astype(np.float32).reshape(angles.shape), np.sin(wide).astype(np.float32).reshape(angles.shape)
    cosf, sinf = libm[0], libm[1]
    unique, inverse = np.unique(flat, return_inverse=True)
    cos = np.fromiter((cosf(float(t)) for t in unique), dtype=np.float32, count=len(unique))
    sin = np.fromiter((sinf(float(t)) for t in unique), dtype=np.float32, count=len(unique))
    return cos[inverse].reshape(angles.shape), sin[inverse].reshape(angles.shape)


@dataclasses.dataclass(frozen=True)
class Rope:
    """One ggml_rope_ext (or text-position ggml_rope_multi) call as ggml-cpu evaluates it.

    mode: "norm" (adjacent pairs), "neox" (pairs i, i + n_dims/2) or "imrope" (NEOX pairs with
    IMROPE sectors); the pairs live in [offset, offset + n_dims) of the last axis and every
    other dim passes through. freq_scale, ext_factor, attn_factor, beta_* and n_ctx_orig are
    the runtime values (rope_runtime), so ext_factor = 0 and attn_factor = 1 give plain RoPE."""

    mode: str
    n_dims: int
    base: float
    offset: int = 0
    sections: tuple = None
    freq_scale: float = 1.0
    ext_factor: float = 0.0
    attn_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    n_ctx_orig: int = 0
    freq_factors: tuple = None    # per rotated pair: theta / factor (ggml_rope_cache_init's freq_factors)

    def __post_init__(self):
        if self.mode not in ("norm", "neox", "imrope") or self.n_dims <= 0 or self.n_dims % 2 or self.offset < 0:
            raise ValueError("RoPE needs mode norm/neox/imrope, a positive even n_dims and offset >= 0")
        if self.freq_factors is not None and len(self.freq_factors) != self.n_dims // 2:
            raise ValueError("RoPE frequency factors need one value per rotated pair")
        if (self.mode == "imrope") != (self.sections is not None):
            raise ValueError("IMROPE (and only IMROPE) takes sections")
        if self.ext_factor != 0 and self.n_ctx_orig <= 0:
            raise ValueError("YaRN needs the original context length")

    def magnitude(self):
        """float32 factor on cos/sin: attn_factor, times fmaf(logf(1 / freq_scale), 0.1f, 1.0f)
        with YaRN (rope_yarn, ops.cpp:5961-5969, as the pinned build contracts it)."""
        f32 = np.float32
        mscale = f32(self.attn_factor)
        if self.ext_factor != 0:
            log = _logf(f32(f32(1.0) / f32(self.freq_scale)))
            mscale = f32(_fma32(log, f32(0.1), f32(1.0)) * mscale)
        return mscale

    def cos_sin(self, positions):
        """float32 (cos, sin) [n, n_dims/2] with the magnitude applied, as ggml's rope cache."""
        f32 = np.float32
        extrap = rope_angles(positions, self.n_dims, self.base, self.sections)
        if self.freq_factors is not None:
            extrap = (extrap / np.asarray(self.freq_factors, dtype=f32)).astype(f32)
        interp = (extrap * f32(self.freq_scale)).astype(f32)
        theta = interp
        if self.ext_factor != 0:
            low, high = yarn_corr_dims(self.n_dims, self.n_ctx_orig, self.base, self.beta_fast, self.beta_slow)
            y = ((np.arange(self.n_dims // 2, dtype=f32) - low) / max(f32(0.001), f32(high - low))).astype(f32)
            ramp = (f32(1) - np.minimum(f32(1), np.maximum(f32(0), y))).astype(f32)
            mix = (ramp * f32(self.ext_factor)).astype(f32)
            # theta_interp * (1 - mix) + theta_extrap * mix, contracted into one fma by the pinned build
            theta = _fma32(interp, (f32(1) - mix).astype(f32), (extrap * mix).astype(f32))
        cos, sin = _cos_sin(theta)
        mscale = self.magnitude()
        if mscale != 1:
            cos, sin = (cos * mscale).astype(f32), (sin * mscale).astype(f32)
        return cos, sin

    def _pairs(self):
        o, n = self.offset, self.n_dims
        if self.mode == "norm":
            return slice(o, o + n, 2), slice(o + 1, o + n, 2)
        return slice(o, o + n // 2), slice(o + n // 2, o + n)

    def apply(self, x, positions):
        return self._rotate(x, positions, inverse=False)

    def invert(self, x, positions):
        """Undo apply (float64 math, float32 result): the transposed rotation divided by
        cos^2 + sin^2 of the float32 cache, i.e. by the squared magnitude, up to the rounding of
        cosf/sinf that the division also undoes (so forward(inverse(row)) changes fewer last bits
        of float32 rows than the plain transposed rotation)."""
        return self._rotate(x, positions, inverse=True)

    def _rotate(self, x, positions, *, inverse):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or len(positions) != len(x) or self.offset + self.n_dims > x.shape[2]:
            raise ValueError("rope needs x [n, heads, dim] with n positions and offset + n_rot <= dim")
        cos, sin = self.cos_sin(positions)
        cos, sin = cos[:, None, :], sin[:, None, :]
        a, b = self._pairs()
        out = x.copy()
        x0, x1 = x[:, :, a], x[:, :, b]
        if not inverse:
            # rotate_pairs as the pinned CPU build compiles it, with fused multiply-adds:
            # fma(x0, cos, -(x1 * sin)) and fma(x0, sin, x1 * cos), emulated in float64.
            wide = x0.astype(np.float64)
            out[:, :, a] = (wide * cos - (x1 * sin).astype(np.float64)).astype(np.float32)
            out[:, :, b] = (wide * sin + (x1 * cos).astype(np.float64)).astype(np.float32)
        else:
            c, s, y0, y1 = (v.astype(np.float64) for v in (cos, sin, x0, x1))
            norm = c * c + s * s
            out[:, :, a] = ((y0 * c + y1 * s) / norm).astype(np.float32)
            out[:, :, b] = ((y1 * c - y0 * s) / norm).astype(np.float32)
        return out


def _rotate(x, positions, n_rot, base, *, neox, inverse, sections=None):
    mode = "imrope" if sections is not None else ("neox" if neox else "norm")
    rope = Rope(mode, n_rot, base, sections=None if sections is None else tuple(int(s) for s in sections))
    return rope.invert(x, positions) if inverse else rope.apply(x, positions)


def rope_norm(x, positions, *, n_rot, base, inverse=False):
    """GGML_ROPE_TYPE_NORMAL: rotate adjacent pairs (2i, 2i+1) of the first n_rot dims."""
    return _rotate(x, positions, n_rot, base, neox=False, inverse=inverse)


def rope_neox(x, positions, *, n_rot, base, inverse=False):
    """GGML_ROPE_TYPE_NEOX: rotate pairs (i, i + n_rot/2) of the first n_rot dims."""
    return _rotate(x, positions, n_rot, base, neox=True, inverse=inverse)


def imrope_text(x, positions, *, n_rot, base, sections, inverse=False):
    """GGML_ROPE_TYPE_IMROPE for text tokens (all position components equal the token position).

    Pairs as NEOX over the first n_rot dims; the remaining dims pass through.
    inverse=True undoes the rotation (computed in float64, returned as float32).
    """
    return _rotate(x, positions, n_rot, base, neox=True, inverse=inverse, sections=sections)


def hadamard(n):
    """The orthonormal Sylvester-Walsh-Hadamard matrix of ggml_gen_hadamard (src/llama-kv-cache.cpp:
    21-56): float32 entries +-(1.0 / sqrtf(n)), symmetric, H @ H = I up to rounding."""
    if n < 1 or n & (n - 1):
        raise ValueError("the Hadamard size must be a power of two")
    h = np.ones((1, 1), dtype=np.float32) * np.float32(1.0 / float(np.sqrt(np.float32(n))))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def quantize_q8_0(x):
    """Q8_0_BLOCK [..., n / 32] of float32 rows [..., n], as the pinned x86 CPU build's quantize_row_q8_0
    (ggml/src/ggml-cpu/arch/x86/quants.c:302-375, what set_rows into a q8_0 cache runs): per block of 32,
    d = amax / 127 stored as float16, values x * (127 / amax) rounded half to even. A CUDA/HIP server
    quantizes otherwise (ggml-cuda/cpy-utils.cuh:137-153: x * (1 / d) rounded half away from zero), so
    rows synthesized for a GPU cache can differ from its own by one step in some values; other CPU
    architectures' quantizers are not emulated either."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % 32:
        raise ValueError("q8_0 rows need a multiple of 32 elements")
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    out = np.empty(len(blocks), dtype=Q8_0_BLOCK)
    out["d"] = (amax / np.float32(127)).astype(np.float32).astype(np.float16)
    with np.errstate(divide="ignore"):
        inverse = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
    out["qs"] = np.clip(np.rint((blocks * inverse[:, None]).astype(np.float32)), -128, 127).astype(np.int8)
    return out.reshape(x.shape[:-1] + (x.shape[-1] // 32,))


def dequantize_q8_0(blocks):
    """float32 rows [..., 32 * blocks] of Q8_0_BLOCK [..., blocks] (dequantize_row_q8_0: d * q)."""
    blocks = np.asarray(blocks)
    values = blocks["d"].astype(np.float32)[..., None] * blocks["qs"].astype(np.float32)
    return values.reshape(blocks.shape[:-1] + (blocks.shape[-1] * 32,))


def _hadamard_rows(x):
    """H @ row for every row of x [n, heads, dim] (float64 accumulation, float32 result)."""
    x = np.asarray(x, dtype=np.float32)
    h = hadamard(x.shape[-1]).astype(np.float64)
    return (x.astype(np.float64) @ h.T).astype(np.float32)


def cache_keys(spec, layer, k_pre_rope, positions, *, role="attention"):
    """Keys as the cache stores them, from pre-RoPE keys at `positions`.

    attention: Kcur_normed (Muse, Qwen), or the MLA row [RMSNorm(c_kv) * g | k_pe] whose k_pe
    tail gets the architecture's RoPE (YaRN included; NoPE kimi rows pass through);
    indexer: the k_norm LayerNorm output, rotated and then multiplied by H where the
    architecture stores Hadamard-rotated keys."""
    kind = spec.layer_types[layer]
    if kind in ("recurrent", "mlp"):
        raise ValueError(f"layer {layer} has no attention cache ({kind})")
    rope = spec.rope(layer, role)
    x = np.asarray(k_pre_rope, dtype=np.float32)
    out = x.copy() if rope is None else rope.apply(x, positions)
    if role == "indexer" and spec.indexer.hadamard:
        out = _hadamard_rows(out)
    return out


def pre_rope_keys(spec, layer, k_cache, positions, *, role="attention"):
    """Inverse of cache_keys: de-rotate stored keys back to pre-RoPE coordinates (float32)."""
    kind = spec.layer_types[layer]
    if kind in ("recurrent", "mlp"):
        raise ValueError(f"layer {layer} has no attention cache ({kind})")
    rope = spec.rope(layer, role)
    x = np.asarray(k_cache, dtype=np.float32)
    if role == "indexer" and spec.indexer.hadamard:
        x = _hadamard_rows(x)
    return x.copy() if rope is None else rope.invert(x, positions)


# -- MLA helpers ----------------------------------------------------------------------------

def mla_split(spec, layer, k_rows):
    """(c_kv [n, 1, kv_lora], k_pe [n, 1, dim - kv_lora]) views of MLA cache (or pre-RoPE) rows."""
    if not spec.k_only(layer):
        raise ValueError(f"layer {layer} is not an MLA layer")
    k_rows = np.asarray(k_rows)
    lora = spec.kv_lora[layer]
    return k_rows[..., :lora], k_rows[..., lora:]


def value_rows(spec, stream, layer):
    """V rows of a parsed attention stream: stream.v, or for MLA the c_kv view of K (the first
    kv_lora elements, which attention reads as V, src/llama-graph.cpp:2918-2924)."""
    if layer in stream.v:
        return stream.v[layer]
    return mla_split(spec, layer, stream.k[layer])[0]


def mla_renormalize_latent(c_kv, weight, eps=1e-6):
    """Put decoded c_kv rows back on the shell RMSNorm(x) * g occupies: divide by g, rescale to
    RMS 1 (llama.cpp's rms_norm, x / sqrt(mean(x^2) + eps)), multiply by g.

    weight is blk.N.attn_kv_a_norm.weight and eps the model's attention.layer_norm_rms_epsilon
    (build_norm passes f_norm_rms_eps, src/llama-graph.cpp:1591). Elements where g == 0 come
    back 0 (the cache holds 0 there too); their normalized values are lost, so the RMS is taken
    over the other elements, which on average hold the same share of the unit mean square."""
    c = np.asarray(c_kv, dtype=np.float64)
    g = np.asarray(weight, dtype=np.float64)
    if c.shape[-1] != g.shape[-1]:
        raise ValueError("the norm weight must match the latent width")
    nonzero = g != 0
    if not nonzero.any():
        raise ValueError("the norm weight is zero everywhere")
    y = np.where(nonzero, c / np.where(nonzero, g, 1.0), 0.0)
    y = y / np.sqrt(np.sum(y * y, axis=-1, keepdims=True) / np.count_nonzero(nonzero) + eps)
    return (y * g).astype(np.float32)


def nextn_layers(spec):
    """Layers a KV section keeps although the trunk never writes them (rows stay zero)."""
    return spec.layers_of("nextn")


# -- builders ---------------------------------------------------------------------------------

def _cast(array, dtype, what):
    array = np.asarray(array)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise StateFormatError(f"{what} must be finite floating point")
    with np.errstate(over="ignore"):
        out = array.astype(dtype)
    if not np.isfinite(out).all():
        raise StateFormatError(f"{what} overflows {dtype}")
    return out


def build_kv_stream(spec, layers, positions, k, v, *, dtype="f16", v_trans=False, seq_id=0, tokens_at=None,
                    role="attention"):
    """One stream of cache-ready K/V: k[layer], v[layer] are [len(positions), heads, dim].

    Cells are written in the given (strictly increasing) position order. For IMROPE and PLE
    models `tokens_at` (token id per cell) fills the cell ext (x = y = pos). K-only layers
    (MLA, the DSA indexer) take no V; v may then be None. A nextn layer missing from k gets
    zero rows, and so do the qwen4exp indexer's V rows (never written), as in a native save.
    """
    if dtype not in DTYPE_NAMES:
        raise StateFormatError("dtype must be f16 or f32")
    np_dtype = GGML_DTYPES[DTYPE_NAMES[dtype]]
    positions = np.asarray(positions)
    if (positions.ndim != 1 or len(positions) == 0 or positions.dtype.kind not in "iu" or (positions < 0).any()
            or (np.diff(positions) <= 0).any() or positions.max() > 2**31 - 1):
        raise StateFormatError("cell positions must be a nonempty strictly increasing int32 array")
    cells = len(positions)
    v = {} if v is None else v
    out_k, out_v = {}, {}
    for il in layers:
        for key, source, target in (("k", k, out_k), ("v", v, out_v)):
            shape = spec.kv_shape(il, key, role)
            if shape is None:
                if il in source:
                    raise StateFormatError(f"layer {il} is K-only (MLA latent / indexer keys) and takes no V")
                continue
            if il not in source:
                if (key == "k" and spec.layer_types[il] == "nextn") or (key == "v" and role == "indexer"):
                    target[il] = np.zeros((cells,) + shape, dtype=np_dtype)
                    continue
                raise StateFormatError(f"missing {key.upper()} for layer {il}")
            value = _cast(source[il], np_dtype, f"layer {il} {key.upper()}")
            if value.shape != (cells,) + shape:
                raise StateFormatError(f"layer {il} {key.upper()} has shape {value.shape}, expected "
                                       f"{(cells,) + shape}")
            target[il] = value
    ext = None
    if spec.has_cell_ext(role):
        if tokens_at is None or len(tokens_at) != cells:
            raise StateFormatError("IMROPE and PLE caches need the token id of every cell")
        ext = np.stack([positions, positions, np.asarray(tokens_at)], axis=1).astype(np.int32)
    code = DTYPE_NAMES[dtype]
    return KVStream(positions.astype(np.int32), tuple((int(seq_id),) for _ in range(cells)), ext, bool(v_trans),
                    out_k, out_v, {il: code for il in out_k}, {il: code for il in out_v})


def build_muse_state(spec, tokens, positions, k, v, *, swa_positions=None, dtype="f16", v_trans=False,
                     k_pre_rope=True, n_stream=1, stream=0, seq_id=0, server_tokens=False):
    """A Muse (iSWA) llama_state_seq_save_file from per-layer K/V at chosen positions.

    positions: token positions of the base-cache cells (every full layer gets all of them).
    k[layer], v[layer]: [len(positions), n_head_kv, head_dim] for every attention layer,
    rows aligned with `positions`; K is pre-RoPE (Kcur_normed) unless k_pre_rope=False.
    swa_positions: subset of `positions` stored for the sliding layers (default: those a
    native save keeps, pos > max(positions) - n_swa). llama-server re-processes the prompt
    unless the SWA part reaches back more than n_swa positions or a checkpoint exists, and
    aborts on zero SWA cells (tools/server/server-context.cpp:3366-3458).
    dtype/v_trans must match the receiving context (cache type; v_trans = not flash-attn).
    """
    if spec.architecture != "muse-glimmer":
        raise StateFormatError("build_muse_state needs a muse-glimmer spec")
    tokens = np.asarray(tokens)
    positions = np.asarray(positions)
    if positions.ndim != 1 or len(positions) == 0:
        raise StateFormatError("positions must be a nonempty 1-D array")
    last = int(positions.max())
    if len(tokens) != last + 1:
        raise StateFormatError(f"{len(tokens)} tokens given; the cache ends at position {last}, so the "
                               f"state must carry {last + 1} tokens (decoding continues at the token count)")
    if swa_positions is None:
        swa_positions = positions[positions > last - spec.n_swa]
    swa_positions = np.asarray(swa_positions)
    if swa_positions.ndim != 1 or len(swa_positions) == 0:
        raise StateFormatError("the SWA cache needs at least one cell (llama-server aborts on an empty one)")
    index = {int(p): i for i, p in enumerate(positions)}
    if len(index) != len(positions):
        raise StateFormatError("positions must be unique")
    missing = [int(p) for p in swa_positions if int(p) not in index]
    if missing:
        raise StateFormatError(f"SWA positions {missing[:5]} are not among the base positions")
    if int(swa_positions.max()) != last:
        raise StateFormatError("the SWA cache must end at the same position as the base cache")
    if not 0 <= stream < n_stream:
        raise StateFormatError("stream must lie in [0, n_stream)")
    for il in spec.layers_of("full", "sliding"):
        for key, source in (("K", k), ("V", v)):
            if il not in source:
                raise StateFormatError(f"missing {key} for layer {il}")
            if len(source[il]) != len(positions):
                raise StateFormatError(f"layer {il} {key} has {len(source[il])} rows for {len(positions)} positions")
    order = np.argsort(positions, kind="stable")
    swa_rows = np.asarray([index[int(p)] for p in np.sort(swa_positions)])
    full, sliding = spec.layers_of("full"), spec.layers_of("sliding")

    def keys(il, rows):
        value = np.asarray(k[il])[rows]
        return cache_keys(spec, il, value, positions[rows]) if k_pre_rope else value

    def section(name, layers, rows):
        built = build_kv_stream(spec, layers, positions[rows], {il: keys(il, rows) for il in layers},
                                {il: np.asarray(v[il])[rows] for il in layers},
                                dtype=dtype, v_trans=v_trans, seq_id=seq_id)
        streams = [None] * n_stream
        streams[stream] = built
        return KVCache(name, layers, streams)

    state = SequenceState(spec, FLAGS_NONE, [section("base", full, order), section("swa", sliding, swa_rows)])
    return StateFile(token_words(tokens, server_packed=server_tokens), state)


def build_dense_state(spec, tokens, positions, k, v=None, *, indexer_k=None, dtype="f16", v_trans=False,
                      k_pre_rope=True, n_stream=1, stream=0, seq_id=0, server_tokens=False):
    """A plain-KV (dense model, e.g. Qwen3, or an MLA model) or DSA llama_state_seq_save_file
    from per-layer K/V at chosen positions.

    positions: token positions of the cells (every layer gets all of them, written in
    position order); the state ends at max(positions), so it carries max(positions) + 1
    tokens and decoding resumes at that count. k[layer], v[layer]: [len(positions),
    n_head_kv, head_dim] for every attention layer, rows aligned with `positions`; K is
    pre-RoPE (Kcur_normed, or the MLA row [c_kv | k_pe]) unless k_pre_rope=False. MLA layers
    take no V and nextn layers get zero rows. DSA specs also need indexer_k[layer] [n, 1,
    head_size] for every indexer layer (pre-RoPE LayerNorm outputs when k_pre_rope), or
    indexer_k="zeros", which is harmless while every query sees at most indexer top_k
    unmasked cells: the scores get the KV mask before ggml_top_k (src/models/deepseek32.cpp:
    280-321), so neither the KV view padding (256 cells) nor other sequences' cells count.
    dtype/v_trans must match the receiving context (cache type; v_trans = not flash-attn).
    """
    if spec.memory not in ("kv", "dsa"):
        raise StateFormatError("build_dense_state needs a plain-KV (dense) or DSA spec")
    tokens = np.asarray(tokens)
    positions = np.asarray(positions)
    if positions.ndim != 1 or len(positions) == 0:
        raise StateFormatError("positions must be a nonempty 1-D array")
    last = int(positions.max())
    if len(tokens) != last + 1:
        raise StateFormatError(f"{len(tokens)} tokens given; the cache ends at position {last}, so the "
                               f"state must carry {last + 1} tokens (decoding continues at the token count)")
    if not 0 <= stream < n_stream:
        raise StateFormatError("stream must lie in [0, n_stream)")
    v = {} if v is None else v
    order = np.argsort(positions, kind="stable")
    cells = positions[order]
    parts = []
    for layout in spec.layouts(FLAGS_NONE):
        if layout.role == "indexer":
            if isinstance(indexer_k, str) and indexer_k == "zeros":
                source, pre_rope = {il: np.zeros((len(positions), 1, spec.indexer.head_size), np.float32)
                                    for il in layout.layers}, False
            elif indexer_k is None:
                raise StateFormatError("DSA states need indexer_k (or indexer_k='zeros')")
            else:
                source, pre_rope = indexer_k, k_pre_rope
        else:
            source, pre_rope = k, k_pre_rope
        keys, values = {}, {}
        for il in layout.layers:
            if il not in source:
                if spec.layer_types[il] == "nextn" and layout.role == "attention":
                    continue
                raise StateFormatError(f"missing K for layer {il}")
            if len(source[il]) != len(positions):
                raise StateFormatError(f"layer {il} K has {len(source[il])} rows for {len(positions)} positions")
            rows = np.asarray(source[il])[order]
            keys[il] = cache_keys(spec, il, rows, cells, role=layout.role) if pre_rope else rows
            if spec.kv_shape(il, "v", layout.role) is not None:
                if il not in v:
                    raise StateFormatError(f"missing V for layer {il}")
                if len(v[il]) != len(positions):
                    raise StateFormatError(f"layer {il} V has {len(v[il])} rows for {len(positions)} positions")
                values[il] = np.asarray(v[il])[order]
        built = build_kv_stream(spec, layout.layers, cells, keys, values, dtype=dtype, v_trans=v_trans, seq_id=seq_id,
                                role=layout.role)
        streams = [None] * n_stream
        streams[stream] = built
        parts.append(KVCache(layout.name, layout.layers, streams))
    state = SequenceState(spec, FLAGS_NONE, parts)
    return StateFile(token_words(tokens, server_packed=server_tokens), state)


def describe(state):
    """Small JSON-able summary of a SequenceState."""
    out = {"architecture": state.spec.architecture, "flags": state.flags, "sections": []}
    for part in state.parts:
        if isinstance(part, KVCache):
            streams = []
            for stream in part.streams:
                if stream is None:
                    streams.append(None)
                    continue
                streams.append({"cells": stream.n_cells, "pos_min": int(stream.positions.min()),
                                "pos_max": int(stream.positions.max()), "v_trans": stream.v_trans,
                                "dtype": sorted({str(a.dtype) for a in stream.k.values()}),
                                "k_only": sorted(set(stream.k) - set(stream.v))})
            out["sections"].append({"name": part.name, "layers": list(part.layers), "streams": streams})
        elif isinstance(part, PooledRows):
            out["sections"].append({"name": part.name, "layers": list(part.layers), "n_stream": part.n_stream,
                                    "rows": part.n_rows, "dtype": sorted({str(a.dtype) for a in part.rows.values()})})
        elif isinstance(part, CompressorRing):
            out["sections"].append({"name": part.name, "layers": list(part.layers), "n_stream": part.n_stream})
        else:
            out["sections"].append({"name": part.name, "layers": list(part.layers), "cells": part.n_cells,
                                    "positions": part.positions.tolist(), "bytes": part.nbytes})
    return out
