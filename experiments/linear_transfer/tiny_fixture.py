"""Generate untrained miniature GGUF models for CPU collector integration tests.

These files exercise the real Muse/Qwen3.5/Qwen3 graph implementations and, through
create_mla_fixture, the MLA (deepseek2, mistral4), DSA (deepseek32, glm-dsa) and
DSA-iSWA (dots3note) ones, and through create_rwkv_fixture the recurrent-only RWKV ones
(rwkv6, rwkv6qwen2, rwkv7, arwkv7), which test-llama-archs skips. Their random weights and
byte tokenizer cannot measure language quality or transfer quality. No model download or
running inference server is involved.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


FIXTURE_ARCHITECTURES = ("muse-glimmer", "qwen35", "qwen3")
MLA_FIXTURE_ARCHITECTURES = ("deepseek2", "mistral4", "deepseek32", "glm-dsa", "dots3note")
RWKV_FIXTURE_ARCHITECTURES = ("rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7")
# YaRN of the DeepSeek-V2 kind on a 64-token original context (factor 4, mscale_all_dim 0.707):
# rope.scaling.yarn_log_multiplier is stored as 0.1 * mscale_all_dim (conversion/deepseek.py)
YARN_V2 = {"factor": 4.0, "original_context_length": 64, "log_multiplier": 0.1 * 0.707}


def _gguf_module():
    gguf_path = Path(__file__).resolve().parents[2] / "runtime/source/gguf-py"
    if not gguf_path.is_dir():
        raise RuntimeError("build the pinned runtime first to provide gguf-py")
    if str(gguf_path) not in sys.path:
        sys.path.insert(0, str(gguf_path))
    import gguf
    return gguf


def _byte_vocabulary(writer, gguf):
    # Token i decodes to byte i. The writer rejects empty arrays, while the
    # tokenizer requires a merges key; this unreachable merge joins symbols
    # that cannot be produced from the byte vocabulary.
    from gguf.vocab import bytes_to_unicode
    byte_symbols = bytes_to_unicode()
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list([byte_symbols[b] for b in range(256)] + ["<bos>", "<eos>"])
    writer.add_token_types([gguf.TokenType.NORMAL] * 256 + [gguf.TokenType.CONTROL] * 2)
    writer.add_token_merges(["<unused-left> <unused-right>"])
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(257)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def create_mla_fixture(path, architecture, seed=7, *, layers=2, q_lora=24, yarn=None, nextn=0, indexer_types=None,
                       sliding_window=8, context_length=256, rope_freq_base=10000.0, rope_freq_base_swa=500.0):
    """Write a tiny all-dense absorbed-MLA model; return its tensors (numpy order) for numeric checks.

    Geometry: hidden 64, 2 query heads, kv_lora 32, k_pe 16 (rope.dimension_count), qk_nope 16,
    v_head 16, so the cache row is [c_kv (32) | k_pe (16)] = 48 elements (src/models/deepseek2.cpp:
    84-160 lists the tensors). deepseek2/mistral4 (mistral4 reuses deepseek2's loader and graph):
    q_lora 0 selects attn_q (the "lite" layout), nextn adds nextn blocks the plain cache keeps
    as zero rows. deepseek32/glm-dsa/dots3note add a lightning indexer (2 heads of 64, top_k 8;
    tensors as src/models/deepseek32.cpp:92-114) on the `indexer_types` layers (default: all for
    deepseek32/glm-dsa, the full-attention ones for dots3note); deepseek32 and dots3note need
    q_lora > 0. dots3note alternates full (even) and sliding (odd) layers with `sliding_window`,
    a sliding kv_lora of 24 (row 40), rope.freq_base_swa and the k_pe RMSNorm of
    src/models/dots3note.cpp:69-136. `yarn`: None or {"factor", "original_context_length",
    "log_multiplier" (optional)} written as rope.scaling.* with type yarn. attn_kv_a_norm weights
    are 1 + noise (the RMS-shell checks need a non-trivial g)."""
    if architecture not in MLA_FIXTURE_ARCHITECTURES:
        raise ValueError(f"MLA fixture architecture must be one of {MLA_FIXTURE_ARCHITECTURES}")
    if q_lora < 0 or (q_lora == 0 and architecture in ("deepseek32", "glm-dsa", "dots3note")):
        raise ValueError(f"{architecture} needs q_lora > 0")
    if nextn and architecture not in ("deepseek2", "mistral4"):
        raise ValueError("nextn blocks are generated for deepseek2/mistral4 only")
    if layers < 1 or nextn < 0:
        raise ValueError("layers must be positive and nextn nonnegative")
    if yarn is not None and (not isinstance(yarn, dict) or not {"factor", "original_context_length"} <= set(yarn)):
        raise ValueError("yarn must be a dict with factor and original_context_length")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gguf = _gguf_module()

    hidden, n_head, kv_lora, rope, nope, v_head, ffn = 64, 2, 32, 16, 16, 16, 128
    kv_lora_swa = 24
    idx_heads, idx_dim, top_k = 2, 64, 8
    dsa = architecture in ("deepseek32", "glm-dsa", "dots3note")
    sliding = [bool(il % 2) for il in range(layers)] if architecture == "dots3note" else [False] * layers
    if indexer_types is None:
        indexer_types = [not s for s in sliding] if dsa else None
    elif not dsa or len(indexer_types) != layers or not indexer_types[0]:
        raise ValueError("indexer_types: one flag per layer for a DSA architecture, layer 0 owning an indexer")
    n_all = layers + nextn
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_name(f"UNTRAINED MLA test fixture: {architecture}")
    writer.add_file_type(0)  # ALL_F32
    writer.add_context_length(context_length)
    writer.add_embedding_length(hidden)
    writer.add_block_count(n_all)
    if nextn:
        writer.add_nextn_predict_layers(nextn)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(1)                    # MLA: one latent "head" per cell
    writer.add_key_length(kv_lora + rope)          # cache row [c_kv | k_pe]
    writer.add_value_length(kv_lora)               # V is the first kv_lora elements of K
    writer.add_key_length_mla(nope + rope)         # per-head decompressed q/k
    writer.add_value_length_mla(v_head)
    writer.add_kv_lora_rank(kv_lora)
    writer.add_q_lora_rank(q_lora)                 # required unless the "lite" layer counts; 0 selects attn_q
    writer.add_rope_dimension_count(rope)
    writer.add_rope_freq_base(float(rope_freq_base))
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_leading_dense_block_count(n_all)    # all dense: no expert tensors
    writer.add_expert_feed_forward_length(ffn)     # required keys even without experts
    writer.add_expert_shared_count(0)
    writer.add_vocab_size(258)
    if dsa:
        writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        writer.add_indexer_head_count(idx_heads)
        writer.add_indexer_key_length(idx_dim)
        writer.add_indexer_top_k(top_k)
        if architecture != "deepseek32":
            writer.add_indexer_types([bool(t) for t in indexer_types])
    if architecture == "dots3note":
        writer.add_sliding_window(sliding_window)
        writer.add_sliding_window_pattern(sliding)
        writer.add_rope_freq_base_swa(float(rope_freq_base_swa))
        writer.add_kv_lora_rank_swa(kv_lora_swa)
        writer.add_key_length_swa(kv_lora_swa + rope)
        writer.add_value_length_swa(kv_lora_swa)
        writer.add_key_length_mla_swa(nope + rope)
        writer.add_value_length_mla_swa(v_head)
    if yarn is not None:
        writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        writer.add_rope_scaling_factor(float(yarn["factor"]))
        writer.add_rope_scaling_orig_ctx_len(int(yarn["original_context_length"]))
        if yarn.get("log_multiplier") is not None:
            writer.add_rope_scaling_yarn_log_mul(float(yarn["log_multiplier"]))
    _byte_vocabulary(writer, gguf)

    tensors = {}

    def tensor(name, shape, fill=None):
        # NumPy dimensions are reversed relative to GGML's ne array.
        value = (rng.normal(0, .05, shape).astype(np.float32) if fill is None
                 else np.full(shape, fill, dtype=np.float32))
        tensors[name] = value

    tensor("token_embd.weight", (258, hidden))
    tensor("output_norm.weight", (hidden,), 1)
    tensor("output.weight", (258, hidden))
    for il in range(n_all):
        prefix = f"blk.{il}"
        lora = kv_lora_swa if il < layers and sliding[il] else kv_lora
        tensor(f"{prefix}.attn_norm.weight", (hidden,), 1)
        if q_lora:
            tensor(f"{prefix}.attn_q_a_norm.weight", (q_lora,), 1)
            tensor(f"{prefix}.attn_q_a.weight", (q_lora, hidden))
            tensor(f"{prefix}.attn_q_b.weight", (n_head * (nope + rope), q_lora))
        else:
            tensor(f"{prefix}.attn_q.weight", (n_head * (nope + rope), hidden))
        tensors[f"{prefix}.attn_kv_a_norm.weight"] = (1 + 0.2 * rng.normal(size=lora)).astype(np.float32)
        tensor(f"{prefix}.attn_kv_a_mqa.weight", (lora + rope, hidden))
        tensor(f"{prefix}.attn_k_b.weight", (n_head, lora, nope))       # ggml {nope, kv_lora, n_head}
        tensor(f"{prefix}.attn_v_b.weight", (n_head, v_head, lora))     # ggml {kv_lora, v_head, n_head}
        tensor(f"{prefix}.attn_output.weight", (hidden, n_head * v_head))
        if architecture == "dots3note":
            tensor(f"{prefix}.attn_k_norm.weight", (rope,), 1)
            tensor(f"{prefix}.attn_gate.weight", (n_head, hidden))
        tensor(f"{prefix}.ffn_norm.weight", (hidden,), 1)
        tensor(f"{prefix}.ffn_gate.weight", (ffn, hidden))
        tensor(f"{prefix}.ffn_down.weight", (hidden, ffn))
        tensor(f"{prefix}.ffn_up.weight", (ffn, hidden))
        if dsa and indexer_types[il]:
            tensor(f"{prefix}.indexer.k_norm.weight", (idx_dim,), 1)
            tensor(f"{prefix}.indexer.k_norm.bias", (idx_dim,), 0)
            tensor(f"{prefix}.indexer.proj.weight", (idx_heads, hidden))
            tensor(f"{prefix}.indexer.attn_k.weight", (idx_dim, hidden))
            tensor(f"{prefix}.indexer.attn_q_b.weight", (idx_heads * idx_dim, q_lora))
        if il >= layers:
            tensor(f"{prefix}.nextn.eh_proj.weight", (hidden, 2 * hidden))
            tensor(f"{prefix}.nextn.enorm.weight", (hidden,), 1)
            tensor(f"{prefix}.nextn.hnorm.weight", (hidden,), 1)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def create_rwkv_fixture(path, architecture, seed=11, *, layers=2):
    """Write a tiny random RWKV model (test-llama-archs skips the RWKV architectures); return its
    tensors (numpy order).

    Geometry: hidden 64, wkv head size 16 (4 heads), FFN 128, LoRA ranks 8. rwkv6 and rwkv7 keep
    two token-shift rows per layer (attention and FFN inputs, token_shift_count 2), rwkv6qwen2 and
    arwkv7 one (their graphs assert n_embd == n_embd_r), so the recurrent rows are R = 128 or 64
    and S = 64 * 16 floats. Tensor names and shapes follow the pinned loaders
    (src/models/rwkv6.cpp, rwkv6qwen2.cpp, rwkv7.cpp, arwkv7.cpp: load_arch_tensors) and the
    metadata the converter writes (conversion/rwkv.py: head_count 0, no RoPE)."""
    if architecture not in RWKV_FIXTURE_ARCHITECTURES:
        raise ValueError(f"RWKV fixture architecture must be one of {RWKV_FIXTURE_ARCHITECTURES}")
    if layers < 1:
        raise ValueError("layers must be positive")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gguf = _gguf_module()
    hidden, head_size, ffn, rank, vocab = 64, 16, 128, 8, 258
    qwen_style = architecture in ("rwkv6qwen2", "arwkv7")      # RMS norms and a SwiGLU FFN, one shift row
    v7 = architecture in ("rwkv7", "arwkv7")
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_name(f"UNTRAINED RWKV test fixture: {architecture}")
    writer.add_file_type(0)  # ALL_F32
    writer.add_context_length(1024)
    writer.add_embedding_length(hidden)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(0)
    writer.add_wkv_head_size(head_size)
    if qwen_style:
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_token_shift_count(1)
    else:
        writer.add_layer_norm_eps(1e-5)
    if architecture == "rwkv6qwen2":
        writer.add_head_count_kv(hidden // head_size)
    if v7:
        writer.add_decay_lora_rank(rank)
        writer.add_iclr_lora_rank(rank)
        writer.add_value_residual_mix_lora_rank(rank)
        writer.add_gate_lora_rank(rank)
    else:
        writer.add_time_mix_extra_dim(rank)
        writer.add_time_decay_extra_dim(rank)
        writer.add_rescale_every_n_layers(0)
    writer.add_vocab_size(vocab)
    _byte_vocabulary(writer, gguf)

    tensors = {}

    def tensor(name, ne, fill=None, scale=.05):
        # ne in ggml order; numpy dimensions are reversed
        shape = tuple(reversed(ne))
        tensors[name] = (rng.normal(0, scale, shape).astype(np.float32) if fill is None
                         else np.full(shape, fill, dtype=np.float32))

    tensor("token_embd.weight", (hidden, vocab))
    tensor("output_norm.weight", (hidden,), 1)
    tensor("output.weight", (hidden, vocab))
    if not qwen_style:
        tensor("token_embd_norm.weight", (hidden,), 1)
        tensor("token_embd_norm.bias", (hidden,), 0)
        tensor("output_norm.bias", (hidden,), 0)
    for il in range(layers):
        p = f"blk.{il}"
        tensor(f"{p}.attn_norm.weight", (hidden,), 1)
        if not qwen_style:
            tensor(f"{p}.attn_norm.bias", (hidden,), 0)
            tensor(f"{p}.attn_norm_2.weight", (hidden,), 1)
            tensor(f"{p}.attn_norm_2.bias", (hidden,), 0)
        if v7:
            for name in ("w", "a", "v"):
                tensor(f"{p}.time_mix_{name}0.weight", (hidden,))
                tensor(f"{p}.time_mix_{name}1.weight", (hidden, rank))
                tensor(f"{p}.time_mix_{name}2.weight", (rank, hidden))
            tensor(f"{p}.time_mix_g1.weight", (hidden, rank))
            tensor(f"{p}.time_mix_g2.weight", (rank, hidden))
            tensor(f"{p}.time_mix_lerp_fused.weight", (hidden, 1, 1, 6), scale=.3)
            for name in ("k_k", "k_a", "r_k"):
                tensor(f"{p}.time_mix_{name}.weight", (hidden,), scale=.3)
        else:
            tensor(f"{p}.time_mix_w1.weight", (hidden, rank * 5))
            tensor(f"{p}.time_mix_w2.weight", (rank, hidden, 5))
            tensor(f"{p}.time_mix_lerp_x.weight", (hidden, 1, 1), scale=.3)
            tensor(f"{p}.time_mix_lerp_fused.weight", (hidden, 1, 1, 5), scale=.3)
            if architecture == "rwkv6":
                tensor(f"{p}.time_mix_first.weight", (head_size, hidden // head_size))
            tensor(f"{p}.time_mix_decay.weight", (hidden,), -1)
            tensor(f"{p}.time_mix_decay_w1.weight", (hidden, rank))
            tensor(f"{p}.time_mix_decay_w2.weight", (rank, hidden))
            tensor(f"{p}.time_mix_gate.weight", (hidden, hidden))
        for name in ("key", "value", "receptance"):
            tensor(f"{p}.time_mix_{name}.weight", (hidden, hidden))
        if not qwen_style:
            tensor(f"{p}.time_mix_ln.weight", (hidden,), 1)
            tensor(f"{p}.time_mix_ln.bias", (hidden,), 0)
        tensor(f"{p}.time_mix_output.weight", (hidden, hidden))
        if qwen_style:
            tensor(f"{p}.ffn_norm.weight", (hidden,), 1)
            tensor(f"{p}.ffn_gate.weight", (hidden, ffn))
            tensor(f"{p}.ffn_down.weight", (ffn, hidden))
            tensor(f"{p}.ffn_up.weight", (hidden, ffn))
        else:
            tensor(f"{p}.channel_mix_lerp_k.weight", (hidden, 1, 1), scale=.3)
            if architecture == "rwkv6":
                tensor(f"{p}.channel_mix_lerp_r.weight", (hidden, 1, 1), scale=.3)
                tensor(f"{p}.channel_mix_receptance.weight", (hidden, hidden))
            tensor(f"{p}.channel_mix_key.weight", (hidden, ffn))
            tensor(f"{p}.channel_mix_value.weight", (ffn, hidden))
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def copy_with_metadata(source, path, *, values=None, remove=(), drop_tensors=(), add_tensors=None):
    """Copy a GGUF with edited metadata and unchanged tensors (as gguf-py's
    gguf_new_metadata.py, without its tqdm dependency): keys in `remove` are dropped and
    `values` sets keys; an existing key keeps its GGUF type, a new one is STRING, FLOAT32 or
    UINT32 by its Python type. Empty arrays (test-llama-archs writes an empty merges list and
    classifier labels) are dropped: gguf-py cannot write them. Used to reproduce what a
    converter writes, e.g. the base head_dim in attention.value_length of kimi-linear and
    bailingmoe3 GGUFs. Tensors named in `drop_tensors` are left out and `add_tensors`
    (name -> float32 array, numpy order) appended."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gguf = _gguf_module()
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(path, reader.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    values = dict(values or {})
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name in remove:
            continue
        value_type = field.types[0]
        if value_type == gguf.GGUFValueType.ARRAY and not field.data and field.name not in values:
            continue
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, values.pop(field.name, field.contents()), value_type, sub_type=sub_type)
    for key, value in values.items():
        value_type = (gguf.GGUFValueType.STRING if isinstance(value, str)
                      else gguf.GGUFValueType.FLOAT32 if isinstance(value, float) else gguf.GGUFValueType.UINT32)
        writer.add_key_value(key, value, value_type)
    kept = [tensor for tensor in reader.tensors if tensor.name not in drop_tensors]
    if len(kept) != len(reader.tensors) - len(set(drop_tensors)):
        raise KeyError(f"{source} lacks some of {sorted(drop_tensors)}")
    added = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in (add_tensors or {}).items()}
    for tensor in kept:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    for name, value in added.items():
        writer.add_tensor_info(name, value.shape, value.dtype, value.nbytes)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in kept:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for value in added.values():
        writer.write_tensor_data(value)
    writer.close()


def create_nemotron_mtp_fixture(source, path, seed=13, *, n_expert=2, n_ff_exp=64, n_ff_shexp=32):
    """test-llama-archs' dense nemotron_h_moe (attention, Mamba-2, MLP layers) turned into what
    the converter writes for a Nemotron-3 MoE with its MTP head (conversion/nemotron.py:254-365):
    MoE experts with a shared expert on the MLP layer and one appended nextn block (attention and
    MoE sub-layers in one block index, listed as an attention and an MLP layer, so its
    feed_forward_length is the MoE width). The tensors follow load_arch_tensors
    (src/models/nemotron-h.cpp:108-171); a context without load_mtp skips the MTP ones."""
    gguf = _gguf_module()
    reader = gguf.GGUFReader(source)
    meta = {field.name: field.contents() for field in reader.fields.values()}
    if meta.get("general.architecture") != "nemotron_h_moe":
        raise ValueError("expected test-llama-archs' nemotron_h_moe fixture")
    p = "nemotron_h_moe."
    n_layer, embd = int(meta[p + "block_count"]), int(meta[p + "embedding_length"])
    n_ff = [int(x) for x in meta[p + "feed_forward_length"]]
    mlp = [il for il in range(n_layer) if n_ff[il]]
    heads, heads_kv = list(meta[p + "attention.head_count"]), list(meta[p + "attention.head_count_kv"])
    attention = next(il for il in range(n_layer) if heads_kv[il] and not n_ff[il])
    head_k = int(meta[p + "attention.key_length"])
    rng = np.random.default_rng(seed)

    def weight(*ne, scale=0.05):
        # ggml order; numpy dimensions are reversed
        return rng.normal(0, scale, tuple(reversed(ne))).astype(np.float32)

    def moe(il):
        return {f"blk.{il}.ffn_gate_inp.weight": weight(embd, n_expert),
                f"blk.{il}.exp_probs_b.bias": np.zeros(n_expert, np.float32),
                f"blk.{il}.ffn_down_exps.weight": weight(n_ff_exp, embd, n_expert),
                f"blk.{il}.ffn_up_exps.weight": weight(embd, n_ff_exp, n_expert),
                f"blk.{il}.ffn_down_shexp.weight": weight(n_ff_shexp, embd),
                f"blk.{il}.ffn_up_shexp.weight": weight(embd, n_ff_shexp)}
    added = {}
    for il in mlp:
        added.update(moe(il))
    mtp, n_q, n_kv = n_layer, head_k * heads[attention], head_k * heads_kv[attention]
    added.update({f"blk.{mtp}.nextn.enorm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.nextn.hnorm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.nextn.eh_proj.weight": weight(2 * embd, embd),
                  f"blk.{mtp}.nextn.shared_head_norm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.attn_norm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.attn_q.weight": weight(embd, n_q),
                  f"blk.{mtp}.attn_k.weight": weight(embd, n_kv),
                  f"blk.{mtp}.attn_v.weight": weight(embd, n_kv),
                  f"blk.{mtp}.attn_output.weight": weight(n_q, embd),
                  f"blk.{mtp}.post_attention_norm.weight": np.ones(embd, np.float32), **moe(mtp)})
    dense = [tensor.name for tensor in reader.tensors
             if any(tensor.name.startswith(f"blk.{il}.ffn_up") or tensor.name.startswith(f"blk.{il}.ffn_down")
                    for il in mlp)]

    def extend(key, value):
        return list(meta[p + key]) + [value]
    values = {p + "block_count": n_layer + 1, p + "nextn_predict_layers": 1,
              p + "feed_forward_length": n_ff + [n_ff_exp], p + "attention.head_count": extend("attention.head_count",
                                                                                          heads[attention]),
              p + "attention.head_count_kv": extend("attention.head_count_kv", heads_kv[attention]),
              p + "expert_count": n_expert, p + "expert_used_count": 1, p + "expert_feed_forward_length": n_ff_exp,
              p + "expert_shared_feed_forward_length": n_ff_shexp, p + "expert_shared_count": 1,
              p + "expert_gating_func": 2, p + "expert_weights_scale": 1.0}
    for key in ("attention.recurrent_layers", "swiglu_clamp_exp", "swiglu_clamp_shexp"):
        if p + key in meta and isinstance(meta[p + key], list) and len(meta[p + key]) == n_layer:
            values[p + key] = extend(key, type(meta[p + key][0])(0))
    copy_with_metadata(source, path, values=values, drop_tensors=dense, add_tensors=added)


def create_dsv4_mtp_fixture(source, path, seed=17):
    """test-llama-archs' deepseek4 model with one appended MTP (nextn) block, as a DeepSeek-V4 GGUF
    with its MTP head is laid out (src/models/deepseek4.cpp:84-171): block_count + 1,
    nextn_predict_layers 1, compress ratio 0 for the MTP layer, the per-layer arrays extended, the
    attention/FFN/hyper-connection tensors of a window-only layer (copies of layer 0's) plus the
    nextn projection and norms. A context without load_mtp skips every MTP tensor, but its raw
    cache keeps the MTP layer (src/llama-kv-cache-dsv4.cpp:1233-1260), which the trunk never writes."""
    gguf = _gguf_module()
    reader = gguf.GGUFReader(source)
    meta = {field.name: field.contents() for field in reader.fields.values()}
    if meta.get("general.architecture") != "deepseek4":
        raise ValueError("expected test-llama-archs' deepseek4 fixture")
    p = "deepseek4."
    n_layer, embd = int(meta[p + "block_count"]), int(meta[p + "embedding_length"])
    ratios = [int(x) for x in meta[p + "attention.compress_ratios"]]
    template = next(il for il in range(n_layer) if ratios[il] == 0)
    rng = np.random.default_rng(seed)
    mtp = n_layer
    added = {}
    for tensor in reader.tensors:
        prefix = f"blk.{template}."
        # the expert scale tensors are loaded for trunk layers only
        if tensor.name.startswith(prefix) and not tensor.name.endswith((".scale", ".input_scale")):
            added[f"blk.{mtp}." + tensor.name[len(prefix):]] = np.asarray(tensor.data, dtype=np.float32).copy()
    added.update({f"blk.{mtp}.nextn.eh_proj.weight": rng.normal(0, 0.05, (embd, 2 * embd)).astype(np.float32),
                  f"blk.{mtp}.nextn.enorm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.nextn.hnorm.weight": np.ones(embd, np.float32),
                  f"blk.{mtp}.nextn.shared_head_norm.weight": np.ones(embd, np.float32)})
    values = {p + "block_count": n_layer + 1, p + "nextn_predict_layers": 1,
              p + "attention.compress_ratios": ratios[:n_layer] + [0]}
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        if isinstance(meta.get(p + key), list) and len(meta[p + key]) == n_layer:
            values[p + key] = list(meta[p + key]) + [meta[p + key][template]]
    copy_with_metadata(source, path, values=values, add_tensors=added)


def create_fixture(path, architecture, seed=31, *, fused_qkv=False, context_length=1024,
                   sliding_window=16, ascii_output=False, control_tokens=None, chat_template=None,
                   bos_token="<bos>", add_bos=False, rope_dimension_count=None, rope_sections=None,
                   rope_freq_base=500000.0, hidden=64):
    """Write a four-layer random model; return its tensors for numeric checks.

    Architectures: muse-glimmer (iSWA, 2 query / 2 KV heads), qwen35 (hybrid: layer 3 is
    full attention, 2 query / 1 KV head) and qwen3 (dense: every layer is full attention,
    4 query / 2 KV heads, NEOX RoPE over the whole head; like the real Qwen3 GGUFs the file
    has no rope.dimension_count unless one is given, and it must equal the head size).

    `control_tokens` appends CONTROL tokens (ids 258, 259, ...) after the byte
    vocabulary and <bos>/<eos>; the default leaves the vocabulary unchanged.
    `chat_template` is stored as `tokenizer.chat_template`; `bos_token` is the text of
    the BOS token (id 256) and `add_bos` the tokenizer's add_bos_token flag.
    `rope_dimension_count` (default: the head size 32) and, for Qwen, `rope_sections`
    (default [8, 4, 4, 0]) allow partial rotary embeddings; `rope_freq_base` is the RoPE
    base (default 5e5). `hidden` is the embedding width (default 64; 5120 makes a Qwen3.5
    the local Qwen3.8 mmproj can feed, fixtures_media.py); head and recurrent sizes stay.
    """
    if architecture not in FIXTURE_ARCHITECTURES:
        raise ValueError("fixture architecture must be muse-glimmer, qwen35 or qwen3")
    if fused_qkv and architecture == "qwen35":
        raise ValueError("the fused QKV fixture is implemented only for Muse and Qwen3")
    if context_length < 1 or not 1 <= sliding_window <= context_length:
        raise ValueError("context length must be positive and contain the sliding window")
    control_tokens = list(control_tokens or [])
    if (any(not isinstance(token, str) or len(token.encode("utf-8")) < 2 for token in control_tokens)
            or len(set(control_tokens)) != len(control_tokens)
            or {bos_token, "<eos>"} & set(control_tokens)):
        raise ValueError("control tokens must be unique strings of at least two bytes")
    if not isinstance(bos_token, str) or len(bos_token.encode("utf-8")) < 2 or bos_token == "<eos>":
        raise ValueError("the BOS token text must be a string of at least two bytes other than <eos>")
    if chat_template is not None and (not isinstance(chat_template, str) or not chat_template):
        raise ValueError("chat_template must be a nonempty string")
    if rope_dimension_count is not None and (type(rope_dimension_count) is not int or rope_dimension_count <= 0
                                             or rope_dimension_count % 2 or rope_dimension_count > 32):
        raise ValueError("rope_dimension_count must be a positive even number up to the head size 32")
    if architecture == "qwen3" and rope_dimension_count not in (None, 32):
        raise ValueError("Qwen3 rotates the whole head (llama.cpp asserts n_rot == head size 32)")
    if rope_sections is not None and (architecture != "qwen35" or len(rope_sections) != 4
                                      or any(type(x) is not int or x < 0 for x in rope_sections)):
        raise ValueError("rope_sections must be four nonnegative integers (Qwen only)")
    if not isinstance(rope_freq_base, (int, float)) or isinstance(rope_freq_base, bool) or not rope_freq_base > 0:
        raise ValueError("rope_freq_base must be a positive number")
    if type(hidden) is not int or hidden < 32 or hidden % 32:
        raise ValueError("hidden must be a positive multiple of 32")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gguf_path = Path(__file__).resolve().parents[2] / "runtime/source/gguf-py"
    if not gguf_path.is_dir():
        raise RuntimeError("build the pinned runtime first to provide gguf-py")
    sys.path.insert(0, str(gguf_path))
    from gguf import GGUFWriter, TokenType
    from gguf.vocab import bytes_to_unicode

    head_dim, ffn, layers = 32, 128, 4
    query_heads, kv_heads = {"muse-glimmer": (2, 2), "qwen35": (2, 1), "qwen3": (4, 2)}[architecture]
    vocab_size = 258 + len(control_tokens)
    rng = np.random.default_rng(seed)
    writer = GGUFWriter(path, architecture)
    writer.add_name(f"UNTRAINED collector test fixture: {architecture}")
    writer.add_file_type(0)  # ALL_F32
    writer.add_context_length(context_length)
    writer.add_embedding_length(hidden)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(query_heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    if architecture != "qwen3" or rope_dimension_count is not None:
        writer.add_rope_dimension_count(head_dim if rope_dimension_count is None else rope_dimension_count)
    writer.add_rope_freq_base(float(rope_freq_base))
    if architecture == "muse-glimmer":
        writer.add_sliding_window(sliding_window)
        writer.add_sliding_window_pattern(4)
        writer.add_logit_scale(1.0)
    elif architecture == "qwen35":
        writer.add_full_attention_interval(4)
        writer.add_rope_dimension_sections([8, 4, 4, 0] if rope_sections is None else list(rope_sections))
        writer.add_ssm_conv_kernel(4)
        writer.add_ssm_inner_size(64)
        writer.add_ssm_state_size(32)
        writer.add_ssm_time_step_rank(2)
        writer.add_ssm_group_count(1)

    # Token i decodes to byte i. The writer rejects empty arrays, while the
    # tokenizer requires a merges key; this unreachable merge joins symbols
    # that cannot be produced from the byte vocabulary.
    byte_symbols = bytes_to_unicode()
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list([byte_symbols[b] for b in range(256)] + [bos_token, "<eos>"] + control_tokens)
    writer.add_token_types([TokenType.NORMAL] * 256 + [TokenType.CONTROL] * (2 + len(control_tokens)))
    writer.add_token_merges(["<unused-left> <unused-right>"])
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(257)
    writer.add_add_bos_token(bool(add_bos))
    writer.add_add_eos_token(False)
    if chat_template is not None:
        writer.add_chat_template(chat_template)

    tensors = {}

    def tensor(name, shape, fill=None):
        # NumPy dimensions are reversed relative to GGML's ne array.
        value = (rng.normal(0, .05, shape).astype(np.float32) if fill is None
                 else np.full(shape, fill, dtype=np.float32))
        tensors[name] = value

    tensor("token_embd.weight", (vocab_size, hidden))
    tensor("output_norm.weight", (hidden,), 1)
    tensor("output.weight", (vocab_size, hidden))
    if ascii_output:
        # Random byte outputs can be invalid UTF-8. Paired ASCII output rows
        # keep HTTP parser checks usable without changing input tokenization.
        printable = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        rows = tensors["output.weight"][printable[:len(printable) // 2]].copy()
        tensors["output.weight"].fill(0)
        tensors["output.weight"][printable] = np.concatenate((rows, -rows))
    for layer in range(layers):
        prefix = f"blk.{layer}"
        tensor(f"{prefix}.attn_norm.weight", (hidden,), 1)
        if architecture != "qwen3":
            tensor(f"{prefix}.post_attention_norm.weight", (hidden,), 1)
        if architecture != "qwen35" or layer == 3:
            query_width = head_dim * query_heads * (2 if architecture == "qwen35" else 1)
            tensor(f"{prefix}.attn_q.weight", (query_width, hidden))
            tensor(f"{prefix}.attn_k.weight", (head_dim * kv_heads, hidden))
            tensor(f"{prefix}.attn_v.weight", (head_dim * kv_heads, hidden))
            tensor(f"{prefix}.attn_output.weight", (hidden, head_dim * query_heads))
            tensor(f"{prefix}.attn_q_norm.weight", (head_dim,), 1)
            tensor(f"{prefix}.attn_k_norm.weight", (head_dim,), 1)
        else:
            key_width, value_width, conv_width = 32, 64, 128
            tensor(f"{prefix}.attn_qkv.weight", (key_width * 2 + value_width, hidden))
            tensor(f"{prefix}.attn_gate.weight", (value_width, hidden))
            tensor(f"{prefix}.ssm_conv1d.weight", (conv_width, 4))
            tensor(f"{prefix}.ssm_dt.bias", (2,), -1)
            tensor(f"{prefix}.ssm_a", (2,), -1)
            tensor(f"{prefix}.ssm_beta.weight", (2, hidden))
            tensor(f"{prefix}.ssm_alpha.weight", (2, hidden))
            tensor(f"{prefix}.ssm_norm.weight", (32,), 1)
            tensor(f"{prefix}.ssm_out.weight", (hidden, value_width))
        if architecture == "muse-glimmer":
            tensor(f"{prefix}.attn_gate.weight", (head_dim * query_heads, hidden))
            tensor(f"{prefix}.ffn_norm.weight", (hidden,), 1)
            tensor(f"{prefix}.post_ffw_norm.weight", (hidden,), 1)
        elif architecture == "qwen3":
            tensor(f"{prefix}.ffn_norm.weight", (hidden,), 1)
        tensor(f"{prefix}.ffn_gate.weight", (ffn, hidden))
        tensor(f"{prefix}.ffn_down.weight", (hidden, ffn))
        tensor(f"{prefix}.ffn_up.weight", (ffn, hidden))

    if fused_qkv:
        for layer in range(layers):
            parts = [tensors.pop(f"blk.{layer}.attn_{part}.weight") for part in "qkv"]
            tensors[f"blk.{layer}.attn_qkv.weight"] = np.concatenate(parts)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=FIXTURE_ARCHITECTURES + MLA_FIXTURE_ARCHITECTURES
                        + RWKV_FIXTURE_ARCHITECTURES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fused-qkv", action="store_true", help="Muse/Qwen3: exercise strided V views")
    parser.add_argument("--context-length", type=int, default=None, help="default 1024 (256 for MLA fixtures)")
    parser.add_argument("--sliding-window", type=int, default=None, help="Muse (default 16) and dots3note (8) window")
    parser.add_argument("--ascii-output", action="store_true", help="restrict random outputs to ASCII for HTTP tests")
    parser.add_argument("--control-token", action="append", default=[], dest="control_tokens",
                        help="append a CONTROL token (repeatable)")
    parser.add_argument("--chat-template", type=Path, help="Jinja file stored as tokenizer.chat_template")
    parser.add_argument("--bos-token", default="<bos>", help="text of the BOS token (id 256)")
    parser.add_argument("--add-bos", action="store_true", help="set tokenizer.ggml.add_bos_token")
    parser.add_argument("--q-lora", type=int, default=24, help="MLA fixtures: q_lora rank (0: attn_q)")
    parser.add_argument("--yarn", action="store_true", help="MLA fixtures: DeepSeek-V2 style YaRN (factor 4, orig 64)")
    parser.add_argument("--nextn", type=int, default=0, help="deepseek2/mistral4: nextn blocks")
    args = parser.parse_args()
    if args.architecture in RWKV_FIXTURE_ARCHITECTURES:
        create_rwkv_fixture(args.output, args.architecture)
        return
    if args.architecture in MLA_FIXTURE_ARCHITECTURES:
        create_mla_fixture(args.output, args.architecture, q_lora=args.q_lora, yarn=YARN_V2 if args.yarn else None,
                           nextn=args.nextn, context_length=args.context_length or 256,
                           sliding_window=args.sliding_window or 8)
        return
    create_fixture(args.output, args.architecture, fused_qkv=args.fused_qkv,
                   context_length=args.context_length or 1024, sliding_window=args.sliding_window or 16,
                   ascii_output=args.ascii_output, control_tokens=args.control_tokens,
                   chat_template=None if args.chat_template is None else args.chat_template.read_text(encoding="utf-8"),
                   bos_token=args.bos_token, add_bos=args.add_bos)


if __name__ == "__main__":
    main()
