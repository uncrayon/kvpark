"""Untrained miniature GGUFs for the bridge's draft-state (.resume) integration checks.

    python make_fixtures.py OUT_DIR

writes, with random weights and a byte tokenizer (no download, no inference server):

* qwen35-target.gguf   tiny Qwen3.5 hybrid (layers 0-2 gated DeltaNet, layer 3 attention,
                       IMROPE), with a <mask> control token (id 258) for DFlash drafts;
* dflash-mrope.gguf    a two-layer DFlash draft for it: every layer on an 8-token sliding
                       window, target_layers [1, 3], rope sections [16, 0, 0, 0] (an M-RoPE
                       draft, like the production Qwen3.8 DFlash2-mrope file);
* qwen3-target.gguf    tiny dense Qwen3 (NEOX), with the same <mask> token;
* dflash-neox.gguf     a DFlash draft for it without rope sections (NEOX, no cell extension);
* qwen35-mtp.gguf      the Qwen3.5 hybrid plus one nextn (MTP) block, layer 4
                       (qwen35.nextn_predict_layers = 1), for --spec-type draft-mtp;
* qwen35-vision.gguf   the Qwen3.5 hybrid 5120 wide, the width of the local Qwen3.8 mmproj
                       (qwen3vl_merger, projection_dim 5120), so that real projector can run
                       with it; with <|im_start|>/<|im_end|> control tokens and a Qwen-style
                       chat template that renders string and text-part content.

The targets' output rows are limited to printable ASCII so generated text always parses.

The tensor names and shapes follow the pinned loaders: src/models/qwen35.cpp (trunk and
load_block_mtp), src/models/qwen3.cpp and src/models/dflash.cpp (DFlash1: no selector or conv
tensors; tok_embd and output are borrowed from the target through ctx_other). The geometry
mirrors experiments/linear_transfer/tiny_fixture.py. Run it with a Python that has numpy (the
experiment venv); gguf-py comes from runtime/source.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HIDDEN, HEAD, FFN = 64, 32, 128
MASK_ID = 258
WINDOW = 8


def _gguf():
    sys.path.insert(0, str(ROOT / "runtime/source/gguf-py"))
    import gguf
    return gguf


CHAT_TOKENS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
CHAT_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\n' }}"
    "{%- if message['content'] is string %}{{- message['content'] }}"
    "{%- else %}{%- for part in message['content'] %}{%- if part['type'] == 'text' %}{{- part['text'] }}"
    "{%- endif %}{%- endfor %}{%- endif %}{{- '<|im_end|>\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}")


def _vocab(writer, gguf, extra=()):
    from gguf.vocab import bytes_to_unicode
    symbols = bytes_to_unicode()
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list([symbols[b] for b in range(256)] + ["<bos>", "<eos>", "<mask>"] + list(extra))
    writer.add_token_types([gguf.TokenType.NORMAL] * 256 + [gguf.TokenType.CONTROL] * (3 + len(extra)))
    writer.add_token_merges(["<unused-left> <unused-right>"])
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(257)
    writer.add_mask_token_id(MASK_ID)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


class _Tensors:
    def __init__(self, seed):
        self.rng, self.items = np.random.default_rng(seed), {}

    def __call__(self, name, shape, fill=None):
        # numpy shapes are ggml's ne reversed
        self.items[name] = (self.rng.normal(0, .05, shape).astype(np.float32) if fill is None
                            else np.full(shape, fill, dtype=np.float32))


def _ascii_output(t):
    # random byte outputs can be invalid UTF-8, which the server's output parser rejects; paired
    # printable rows (as in tiny_fixture.py --ascii-output) keep every argmax printable
    output = t.items["output.weight"]
    printable = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    rows = output[printable[:len(printable) // 2]].copy()
    output.fill(0)
    output[printable] = np.concatenate((rows, -rows))


def _write(path, writer, tensors):
    for name, value in tensors.items.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _common(writer, n_layer, query_heads, kv_heads, hidden=HIDDEN):
    writer.add_file_type(0)
    writer.add_context_length(1024)
    writer.add_embedding_length(hidden)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(query_heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(HEAD)
    writer.add_value_length(HEAD)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(500000.0)


def _ffn(t, prefix, hidden=HIDDEN):
    t(f"{prefix}.ffn_gate.weight", (FFN, hidden))
    t(f"{prefix}.ffn_down.weight", (hidden, FFN))
    t(f"{prefix}.ffn_up.weight", (FFN, hidden))


def qwen35(path, *, mtp=False, vision=False, seed=31):
    """Qwen3.5 hybrid: 2 query / 1 KV head of 32, IMROPE sections [8, 4, 4, 0]; +1 nextn block;
    vision: 5120 wide with chat tokens and a chat template."""
    gguf = _gguf()
    n_trunk = 4
    hidden = 5120 if vision else HIDDEN
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_name("UNTRAINED bridge fixture: qwen35" + (" + MTP" if mtp else "") + (" vision" if vision else ""))
    _common(writer, n_trunk + (1 if mtp else 0), 2, 1, hidden=hidden)
    writer.add_rope_dimension_count(HEAD)
    writer.add_full_attention_interval(4)
    writer.add_rope_dimension_sections([8, 4, 4, 0])
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_inner_size(64)
    writer.add_ssm_state_size(32)
    writer.add_ssm_time_step_rank(2)
    writer.add_ssm_group_count(1)
    if mtp:
        writer.add_nextn_predict_layers(1)
    extra = CHAT_TOKENS if vision else ()
    _vocab(writer, gguf, extra)
    if vision:
        writer.add_chat_template(CHAT_TEMPLATE)
    n_vocab = 259 + len(extra)
    t = _Tensors(seed)
    t("token_embd.weight", (n_vocab, hidden))
    t("output_norm.weight", (hidden,), 1)
    t("output.weight", (n_vocab, hidden))
    _ascii_output(t)

    def attention(prefix):
        t(f"{prefix}.attn_norm.weight", (hidden,), 1)
        t(f"{prefix}.post_attention_norm.weight", (hidden,), 1)
        t(f"{prefix}.attn_q.weight", (HEAD * 2 * 2, hidden))        # query and gate
        t(f"{prefix}.attn_k.weight", (HEAD, hidden))
        t(f"{prefix}.attn_v.weight", (HEAD, hidden))
        t(f"{prefix}.attn_output.weight", (hidden, HEAD * 2))
        t(f"{prefix}.attn_q_norm.weight", (HEAD,), 1)
        t(f"{prefix}.attn_k_norm.weight", (HEAD,), 1)
        _ffn(t, prefix, hidden)

    for layer in range(n_trunk):
        prefix = f"blk.{layer}"
        if layer == 3:
            attention(prefix)
            continue
        t(f"{prefix}.attn_norm.weight", (hidden,), 1)
        t(f"{prefix}.post_attention_norm.weight", (hidden,), 1)
        t(f"{prefix}.attn_qkv.weight", (32 * 2 + 64, hidden))
        t(f"{prefix}.attn_gate.weight", (64, hidden))
        t(f"{prefix}.ssm_conv1d.weight", (128, 4))
        t(f"{prefix}.ssm_dt.bias", (2,), -1)
        t(f"{prefix}.ssm_a", (2,), -1)
        t(f"{prefix}.ssm_beta.weight", (2, hidden))
        t(f"{prefix}.ssm_alpha.weight", (2, hidden))
        t(f"{prefix}.ssm_norm.weight", (32,), 1)
        t(f"{prefix}.ssm_out.weight", (hidden, 64))
        _ffn(t, prefix, hidden)
    if mtp:
        prefix = f"blk.{n_trunk}"
        attention(prefix)
        t(f"{prefix}.nextn.eh_proj.weight", (hidden, 2 * hidden))
        t(f"{prefix}.nextn.enorm.weight", (hidden,), 1)
        t(f"{prefix}.nextn.hnorm.weight", (hidden,), 1)
    return _write(path, writer, t)


def qwen3(path, seed=37):
    """Dense Qwen3: 4 query / 2 KV heads of 32, NEOX over the whole head."""
    gguf = _gguf()
    writer = gguf.GGUFWriter(path, "qwen3")
    writer.add_name("UNTRAINED bridge fixture: qwen3")
    _common(writer, 4, 4, 2)
    _vocab(writer, gguf)
    t = _Tensors(seed)
    t("token_embd.weight", (259, HIDDEN))
    t("output_norm.weight", (HIDDEN,), 1)
    t("output.weight", (259, HIDDEN))
    _ascii_output(t)
    for layer in range(4):
        prefix = f"blk.{layer}"
        t(f"{prefix}.attn_norm.weight", (HIDDEN,), 1)
        t(f"{prefix}.attn_q.weight", (HEAD * 4, HIDDEN))
        t(f"{prefix}.attn_k.weight", (HEAD * 2, HIDDEN))
        t(f"{prefix}.attn_v.weight", (HEAD * 2, HIDDEN))
        t(f"{prefix}.attn_output.weight", (HIDDEN, HEAD * 4))
        t(f"{prefix}.attn_q_norm.weight", (HEAD,), 1)
        t(f"{prefix}.attn_k_norm.weight", (HEAD,), 1)
        t(f"{prefix}.ffn_norm.weight", (HIDDEN,), 1)
        _ffn(t, prefix)
    return _write(path, writer, t)


def dflash(path, *, mrope, target_layers=(1, 3), seed=41):
    """DFlash1 draft: 2 layers, 2 query / 1 KV head of 32, block 4, all layers on an 8-token window."""
    gguf = _gguf()
    writer = gguf.GGUFWriter(path, "dflash")
    writer.add_name("UNTRAINED bridge fixture: dflash" + (" mrope" if mrope else ""))
    _common(writer, 2, 2, 1)
    writer.add_target_layers(list(target_layers))
    writer.add_uint32("dflash.block_size", 4)
    writer.add_sliding_window(WINDOW)
    writer.add_sliding_window_pattern([True, True])
    if mrope:
        writer.add_rope_dimension_sections([HEAD // 2, 0, 0, 0])
    _vocab(writer, gguf)
    t = _Tensors(seed)
    t("fc.weight", (HIDDEN, HIDDEN * len(target_layers)))
    t("enc.output_norm.weight", (HIDDEN,), 1)
    t("output_norm.weight", (HIDDEN,), 1)
    for layer in range(2):
        prefix = f"blk.{layer}"
        t(f"{prefix}.attn_norm.weight", (HIDDEN,), 1)
        t(f"{prefix}.attn_q.weight", (HEAD * 2, HIDDEN))
        t(f"{prefix}.attn_k.weight", (HEAD, HIDDEN))
        t(f"{prefix}.attn_v.weight", (HEAD, HIDDEN))
        t(f"{prefix}.attn_output.weight", (HIDDEN, HEAD * 2))
        t(f"{prefix}.attn_q_norm.weight", (HEAD,), 1)
        t(f"{prefix}.attn_k_norm.weight", (HEAD,), 1)
        t(f"{prefix}.ffn_norm.weight", (HIDDEN,), 1)
        _ffn(t, prefix)
    return _write(path, writer, t)


FIXTURES = {
    "qwen35-target.gguf": lambda path: qwen35(path),
    "dflash-mrope.gguf": lambda path: dflash(path, mrope=True),
    "qwen3-target.gguf": lambda path: qwen3(path),
    "dflash-neox.gguf": lambda path: dflash(path, mrope=False),
    "qwen35-mtp.gguf": lambda path: qwen35(path, mtp=True),
    "qwen35-vision.gguf": lambda path: qwen35(path, vision=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in FIXTURES.items():
        path = args.out_dir / name
        if not path.exists():
            make(path)
        print(path)


if __name__ == "__main__":
    main()
