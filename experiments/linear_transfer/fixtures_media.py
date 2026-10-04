"""Fixtures for slot files that hold media: a PNG writer, tiny vision-capable text models, a tiny
qwen3vl_merger projector and a throwaway llama-server.

* write_png: a random RGB PNG (stdlib zlib; no imaging library).
* create_vision_qwen35: the tiny Qwen3.5 hybrid of tiny_fixture.create_fixture, 5120 wide (the
  projection width of the local Qwen3.8 mmproj, qwen3vl_merger), with <|vision_start|> and
  <|vision_end|> control tokens, so the real projector can feed it (M-RoPE image positions).
* create_qwen3vl_mmproj: a random qwen3vl_merger projector (one ViT block of 32, patch 16,
  spatial merge 2) whose projection width matches a 64-wide tiny text model; paired with the
  tiny qwen3 (dense, NEOX) or muse-glimmer (iSWA, NORM) fixtures it gives images NORMAL
  (sequential) positions. Tensor names and keys follow the real Qwen3.8 mmproj and
  tools/mtmd/clip.cpp (qwen3vl_merger: patch_embd and patch_embd.1, position_embd, v.blk.N
  with a fused attn_qkv, post_ln, mm.0 and mm.2).
* Server: llama-server on 127.0.0.1 with a slot directory, killed by PID on close.

Random weights: these check layouts and bookkeeping, never quality. Nothing is downloaded.
"""

import base64
import json
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

import numpy as np

import tiny_fixture


VISION_TOKENS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
VISION_START_ID = 258 + VISION_TOKENS.index("<|vision_start|>")   # ids follow <bos>/<eos> (256, 257)


def write_png(path, width, height, seed=0):
    """A random 8-bit RGB PNG; returns its bytes (whose sha256 is the mtmd chunk id)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path.write_bytes(data)
    return data


def create_vision_qwen35(path, seed=31):
    """The tiny Qwen3.5 hybrid 5120 wide (the Qwen3.8 mmproj's projection_dim) with chat and
    vision control tokens (ids 258..261); returns its tensors."""
    return tiny_fixture.create_fixture(path, "qwen35", seed=seed, hidden=5120, control_tokens=VISION_TOKENS,
                                       ascii_output=True)


def create_vision_text_model(path, architecture, seed=31, **options):
    """A 64-wide tiny qwen3 (dense) or muse-glimmer (iSWA) with the vision control tokens; options
    go to tiny_fixture.create_fixture (e.g. sliding_window=64, so Muse's window holds an image)."""
    if architecture not in ("qwen3", "muse-glimmer"):
        raise ValueError("the tiny projector pairs with the qwen3 or muse-glimmer fixture")
    return tiny_fixture.create_fixture(path, architecture, seed=seed, control_tokens=VISION_TOKENS, ascii_output=True,
                                       **options)


def create_qwen3vl_mmproj(path, projection_dim=64, seed=5):
    """A random qwen3vl_merger mmproj: ViT width 32 (2 heads), FFN 64, one block, patch 16, image
    size 64 (a 4 x 4 position table, interpolated for other sizes), spatial merge 2 (a 2 x 2 patch
    group per token, 32 x 32 pixels), projection to `projection_dim`."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gguf = tiny_fixture._gguf_module()
    width, heads, ffn, blocks, patch, image, merge = 32, 2, 64, 1, 16, 64, 2
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(path, "clip")
    writer.add_type("mmproj")
    writer.add_name("UNTRAINED qwen3vl_merger test projector")
    writer.add_file_type(0)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.projection_dim", projection_dim)
    writer.add_uint32("clip.vision.image_size", image)
    writer.add_uint32("clip.vision.patch_size", patch)
    writer.add_uint32("clip.vision.embedding_length", width)
    writer.add_uint32("clip.vision.feed_forward_length", ffn)
    writer.add_uint32("clip.vision.block_count", blocks)
    writer.add_uint32("clip.vision.attention.head_count", heads)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_uint32("clip.vision.spatial_merge_size", merge)
    writer.add_array("clip.vision.is_deepstack_layers", [False] * blocks)

    def tensor(name, ne, fill=None):
        # ne in ggml order; numpy dimensions are reversed
        shape = tuple(reversed(ne))
        value = rng.normal(0, .05, shape).astype(np.float32) if fill is None else np.full(shape, fill, np.float32)
        writer.add_tensor(name, value)

    tensor("v.patch_embd.weight", (patch, patch, 3, width))
    tensor("v.patch_embd.weight.1", (patch, patch, 3, width))
    tensor("v.patch_embd.bias", (width,), 0)
    tensor("v.position_embd.weight", (width, (image // patch) ** 2))
    for il in range(blocks):
        p = f"v.blk.{il}"
        tensor(f"{p}.ln1.weight", (width,), 1)
        tensor(f"{p}.ln1.bias", (width,), 0)
        tensor(f"{p}.attn_qkv.weight", (width, 3 * width))
        tensor(f"{p}.attn_qkv.bias", (3 * width,), 0)
        tensor(f"{p}.attn_out.weight", (width, width))
        tensor(f"{p}.attn_out.bias", (width,), 0)
        tensor(f"{p}.ln2.weight", (width,), 1)
        tensor(f"{p}.ln2.bias", (width,), 0)
        tensor(f"{p}.ffn_up.weight", (width, ffn))
        tensor(f"{p}.ffn_up.bias", (ffn,), 0)
        tensor(f"{p}.ffn_down.weight", (ffn, width))
        tensor(f"{p}.ffn_down.bias", (width,), 0)
    tensor("v.post_ln.weight", (width,), 1)
    tensor("v.post_ln.bias", (width,), 0)
    merged = width * merge * merge
    tensor("mm.0.weight", (merged, merged))
    tensor("mm.0.bias", (merged,), 0)
    tensor("mm.2.weight", (merged, projection_dim))
    tensor("mm.2.bias", (projection_dim,), 0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def free_port(ports):
    """The first port of `ports` nothing listens on (127.0.0.1)."""
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port in {ports[0]}..{ports[-1]}")


class ServerError(RuntimeError):
    """An HTTP error answer of the server (status in .status)."""

    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class Server:
    """A throwaway llama-server (CPU, one slot, --slot-save-path) for slot save/restore checks;
    close() kills it by PID."""

    def __init__(self, binary, model, slots, log, port, *, mmproj=None, ctx=512, flash_attn="on",
                 extra=(), timeout=180):
        command = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx),
                   "-np", "1", "-fa", flash_attn, "--slot-save-path", str(slots), "-t", "4", "--no-warmup",
                   *map(str, extra)]
        if mmproj is not None:
            command += ["--mmproj", str(mmproj)]
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(log)
        with self.log.open("w") as output:
            self.process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            self._wait(timeout)
        except BaseException:
            self.close()
            raise

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.props = self.request("GET", "/props")
                return
            except (OSError, urllib.error.URLError, ServerError) as error:
                if isinstance(error, ServerError) and error.status != 503:      # 503 while loading
                    raise
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"llama-server did not start: {self.log.read_text()[-2000:]}") from None
                time.sleep(0.25)

    def request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            with error:
                body = error.read()[:600]
            raise ServerError(f"{method} {path}: HTTP {error.code} {body!r}", error.code) from None

    def complete(self, prompt, *, images=(), n_predict=1):
        """POST /completion on slot 0 (prompt: text or token ids; with images a multimodal prompt
        whose text holds the server's media marker per image)."""
        if images:
            prompt = {"prompt_string": prompt,
                      "multimodal_data": [base64.b64encode(bytes(image)).decode() for image in images]}
        return self.request("POST", "/completion", {"prompt": prompt, "n_predict": n_predict, "id_slot": 0,
                                                    "cache_prompt": True, "temperature": 0, "samplers": ["top_k"],
                                                    "top_k": 1})

    def save(self, filename):
        return self.request("POST", "/slots/0?action=save", {"filename": filename})

    def restore(self, filename):
        return self.request("POST", "/slots/0?action=restore", {"filename": filename})

    def erase(self):
        return self.request("POST", "/slots/0?action=erase")

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=30)
