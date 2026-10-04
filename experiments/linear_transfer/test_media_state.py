"""Slot files whose token words hold media: kvstate's server_tokens codec (parse_media_tokens,
MediaTokens, MediaChunk), truncate_state, and handoff's position index next to media cells.

The unit tests build token words and mtmd chunk blobs by hand, field for field as
server_tokens::serialize (tools/server/server-common.cpp:278-316, 553-573) and
mtmd_input_chunk_save (tools/mtmd/mtmd.cpp:29-136, 262-411) write them.

MediaSlotTests drive a throwaway llama-server on 127.0.0.1 (ports 18540-18549, or from
KVPARK_MEDIA_PORT) with real images and slot saves; they need KVPARK_LLAMA_SERVER=/path/to/a
kvpark-patched llama-server with mtmd (runtime/build/bin/llama-server). The tiny dense (qwen3)
and iSWA (muse-glimmer) fixtures run with a random qwen3vl_merger projector (NORMAL image
positions); KVPARK_QWEN_MMPROJ=/path/to/a 5120-wide qwen3vl_merger mmproj (the local Qwen3.8's)
adds the tiny Qwen3.5 hybrid with M-RoPE image positions. Nothing is downloaded.
"""

import hashlib
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import fixtures_media
import handoff
import kvstate
from kvstate import StateFormatError


SERVER = os.environ.get("KVPARK_LLAMA_SERVER")
QWEN_MMPROJ = os.environ.get("KVPARK_QWEN_MMPROJ")
FIRST_PORT = int(os.environ.get("KVPARK_MEDIA_PORT", "18540"))
PORTS = range(FIRST_PORT, FIRST_PORT + 10)


def chunk_blob(kind="image", *, nx=4, ny=3, pos_type=1, entries=1, merge=1, chunk_id="a" * 64, n_tokens=0,
               text=(), version=2):
    """mtmd_input_chunk_save bytes: u64 version, u32 type, u64 n_text + i32 tokens, u8 has_image
    [u32 nx, ny, pos_type, image_idx, n_temporal_merge, u64 + id, u8 is_audio, u64 n_entries,
    entries (u8 viewsep, u8 newline, i32 lead_pad, i32 nx, i32 ny)], u8 has_audio [u32 n_tokens,
    u64 + id, the entry batch]."""
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    def batch(count, is_audio):
        return struct.pack("<BQ", is_audio, count) + b"".join(struct.pack("<BBiii", 0, 0, 0, nx, ny)
                                                               for _ in range(count))
    out = struct.pack("<QI", version, {"text": 0, "image": 1, "audio": 2}[kind])
    out += struct.pack("<Q", len(text)) + b"".join(struct.pack("<i", t) for t in text)
    if kind == "image":
        out += b"\1" + struct.pack("<IIIII", nx, ny, pos_type, 0, merge) + string(chunk_id) + batch(entries, 0) + b"\0"
    elif kind == "audio":
        out += b"\0\1" + struct.pack("<I", n_tokens) + string(chunk_id) + batch(entries, 1)
    else:
        out += b"\0\0"
    return out


def server_words(tokens, chunks):
    """server_tokens::serialize words of a token list (-1 inside media) and (start, blob) chunks."""
    data = struct.pack("<iII", -1, 1, len(tokens)) + np.asarray(tokens, dtype="<i4").tobytes()
    data += struct.pack("<I", len(chunks)) + b"".join(struct.pack("<I", start) for start, _ in chunks)
    data += b"".join(struct.pack("<I", len(blob)) + blob for _, blob in chunks)
    return np.frombuffer(data + bytes(-len(data) % 4), dtype="<i4").copy()


def qwen_metadata():
    """A tiny Qwen3.5 hybrid: layers 0-2 recurrent, layer 3 attention (IMROPE, one KV head of 16)."""
    p = "qwen35."
    return {"general.architecture": "qwen35", p + "block_count": 4, p + "embedding_length": 32,
            p + "attention.head_count": 2, p + "attention.head_count_kv": 1, p + "attention.key_length": 16,
            p + "attention.value_length": 16, p + "full_attention_interval": 4, p + "rope.dimension_count": 8,
            p + "rope.dimension_sections": [2, 1, 1, 0], p + "rope.freq_base": 1e7, p + "ssm.conv_kernel": 4,
            p + "ssm.inner_size": 8, p + "ssm.state_size": 2, p + "ssm.group_count": 1}


def media_state(spec, rng, *, before=5, nx=3, ny=2, after=4):
    """A Qwen3.5 slot state of `before` text tokens, one M-RoPE image of nx x ny cells and `after`
    text tokens, laid out as llama-server saves it (image cells at the image's first position with
    ext (pos + col, pos + row, -1); text after it at pos + max(nx, ny))."""
    blob = chunk_blob(nx=nx, ny=ny, chunk_id=hashlib.sha256(b"image").hexdigest())
    tokens = list(rng.integers(0, 200, before)) + [-1] * (nx * ny) + list(rng.integers(0, 200, after))
    words = server_words(tokens, [(before, blob)])
    positions = kvstate.parse_media_tokens(words).cell_positions().astype(np.int32)
    ext = np.stack([positions, positions, np.asarray(tokens)], 1).astype(np.int32)
    for i in range(nx * ny):
        ext[before + i] = (positions[before] + i % nx, positions[before] + i // nx, -1)
    n = len(tokens)
    k = {3: rng.normal(size=(n, 1, 16)).astype(np.float16)}
    v = {3: rng.normal(size=(n, 1, 16)).astype(np.float16)}
    stream = kvstate.KVStream(positions, ((0,),) * n, ext, False, k, v, {3: 1}, {3: 1})
    r_row, s_row = spec.recurrent_row_bytes
    recurrent = kvstate.RecurrentState("recurrent", (0, 1, 2), np.asarray([positions[-1]], np.int32), ((),), 4,
                                       {il: kvstate.OpaqueRows(0, r_row, rng.bytes(r_row)) for il in (0, 1, 2)},
                                       {il: kvstate.OpaqueRows(0, s_row, rng.bytes(s_row)) for il in (0, 1, 2)})
    state = kvstate.SequenceState(spec, 0, [kvstate.KVCache("attn", (3,), [stream]), recurrent])
    return kvstate.StateFile(words, state)


class MediaTokenTests(unittest.TestCase):
    def test_chunk_blobs(self):
        image = kvstate.parse_media_chunk(chunk_blob(nx=4, ny=3, chunk_id="x" * 64), start=7)
        self.assertEqual((image.start, image.end, image.n_tokens, image.n_pos, image.modality, image.pos_type,
                          image.nx, image.ny, image.id), (7, 19, 12, 4, "image", "mrope", 4, 3, "x" * 64))
        # NORMAL images count one position per cell; HunyuanVL adds BOI, EOI and a newline per row
        normal = kvstate.parse_media_chunk(chunk_blob(nx=12, ny=1, pos_type=0))
        self.assertEqual((normal.n_tokens, normal.n_pos, normal.pos_type), (12, 12, "normal"))
        hunyuan = kvstate.parse_media_chunk(chunk_blob(nx=3, ny=2, pos_type=2))
        self.assertEqual((hunyuan.n_tokens, hunyuan.n_pos), ((3 + 1) * 2 + 2, 10))
        # video: three frames merged in pairs give two frames of cells
        video = kvstate.parse_media_chunk(chunk_blob(nx=2, ny=2, entries=3, merge=2))
        self.assertEqual((video.n_tokens, video.n_pos), (2 * 2 * 2, 2))
        audio = kvstate.parse_media_chunk(chunk_blob("audio", n_tokens=7, entries=2, chunk_id="aud"))
        self.assertEqual((audio.modality, audio.n_tokens, audio.n_pos, audio.id, audio.pos_type),
                         ("audio", 7, 7, "aud", None))
        for blob, fragment in ((chunk_blob(version=3), "version"), (chunk_blob("text"), "text chunk"),
                               (chunk_blob()[:-5], "truncated"), (chunk_blob() + b"\0", "trailing"),
                               (chunk_blob(pos_type=3), "position type"),
                               (chunk_blob()[:20] + b"\2" + chunk_blob()[21:], "flag")):
            with self.subTest(fragment=fragment), self.assertRaisesRegex(StateFormatError, fragment):
                kvstate.parse_media_chunk(blob)

    def test_token_words_with_media_round_trip(self):
        # text, two 3 x 2 M-RoPE images (6 cells, 3 positions each) and an audio chunk of 2 cells, in
        # the layout of server-common.h:147-156
        image, audio = chunk_blob(nx=3, ny=2), chunk_blob("audio", n_tokens=2)
        tokens = [10, 11, 12, 13, 14] + [-1] * 6 + [15] + [-1] * 6 + [-1] * 2 + [16]
        words = server_words(tokens, [(5, image), (12, image), (18, audio)])
        media = kvstate.parse_media_tokens(words)
        self.assertTrue(media.server_packed)
        self.assertEqual([(c.start, c.n_tokens, c.n_pos, c.modality) for c in media.media],
                         [(5, 6, 3, "image"), (12, 6, 3, "image"), (18, 2, 2, "audio")])
        np.testing.assert_array_equal(media.to_words(), words)
        # positions: the images' cells share one position, which then advances by max(nx, ny)
        self.assertEqual(media.cell_positions().tolist(),
                         [0, 1, 2, 3, 4] + [5] * 6 + [8] + [9] * 6 + [12, 13] + [14])
        self.assertEqual((media.pos_next(), media.pos_next(5), media.pos_next(11), media.pos_next(12)), (15, 5, 8, 9))
        with self.assertRaisesRegex(StateFormatError, "inside the media chunk"):
            media.pos_next(7)
        kept = media.truncated(12)
        self.assertEqual((len(kept.tokens), [c.start for c in kept.media]), (12, [5]))
        np.testing.assert_array_equal(kvstate.parse_media_tokens(kept.to_words()).tokens, tokens[:12])
        # the cut before the first medium, moved back over the img_beg tokens in front of it
        self.assertEqual(kvstate.media_cut(media), 5)
        self.assertEqual((kvstate.media_cut(media, before=[13, 14]), kvstate.media_cut(media, before=[99])), (3, 5))
        self.assertEqual(kvstate.media_cut(kvstate.parse_media_tokens([1, 2, 3])), 3)
        # text-only readers keep refusing media
        with self.assertRaisesRegex(StateFormatError, "media chunks; parse_media_tokens"):
            kvstate.decode_token_words(words)
        text = kvstate.token_words([5, 6, 7], server_packed=True)
        self.assertEqual(kvstate.parse_media_tokens(text).media, ())
        np.testing.assert_array_equal(kvstate.parse_media_tokens(text).to_words(), text)

    def test_malformed_media_tables_are_refused(self):
        image = chunk_blob(nx=3, ny=2)
        good = [1, 2] + [-1] * 6 + [3]
        cases = (([1, 2] + [-1] * 6 + [-1, 3], [(2, image)], "negative ids outside"),
                 ([1] + [-1] * 6 + [3, 4], [(2, image)], "overlaps"),
                 ([1, 2] + [-1] * 5, [(2, image)], "does not fit"),
                 (good, [(2, image), (2, image)], "increase"),
                 (good, [(2, b"")], "empty"))
        for tokens, chunks, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(StateFormatError, fragment):
                kvstate.parse_media_tokens(server_words(tokens, chunks))
        words = server_words(good, [(2, image)])
        with self.assertRaisesRegex(StateFormatError, "trailing"):
            kvstate.parse_media_tokens(np.concatenate([words, [0]]))
        padded = words.copy()
        padded[-1] |= 1 << 24                         # a nonzero padding byte
        with self.assertRaisesRegex(StateFormatError, "trailing"):
            kvstate.parse_media_tokens(padded)
        with self.assertRaisesRegex(StateFormatError, "plain token list cannot hold media"):
            kvstate.MediaTokens(np.asarray(good), kvstate.parse_media_tokens(words).media, False).to_words()


class MediaStateTests(unittest.TestCase):
    def setUp(self):
        self.spec = kvstate.ModelSpec.from_metadata(qwen_metadata())
        self.rng = np.random.default_rng(17)

    def test_state_with_an_image_round_trips_and_checks_its_end(self):
        file = media_state(self.spec, self.rng)
        data = file.to_bytes()
        parsed = kvstate.parse_file(data, self.spec)
        self.assertEqual(parsed.to_bytes(), data)
        self.assertEqual((len(parsed.media.tokens), parsed.media.pos_next(), parsed.server_packed), (15, 12, True))
        with self.assertRaisesRegex(StateFormatError, "media chunks"):
            parsed.tokens
        # the sections end at the last cell's position (11), not at the token count
        attn = parsed.state.part("attn").stream()
        attn.positions = attn.positions.copy()
        attn.positions[-1] = 14
        with self.assertRaisesRegex(StateFormatError, "attn section ends at position 14; with 15 tokens every "
                                                      "section must end at 11"):
            parsed.to_bytes()

    def test_truncation_drops_whole_images_and_needs_r0(self):
        file = media_state(self.spec, self.rng)
        # before the image: its cells (all at position 5) go with the text after it
        cut = kvstate.truncate_state(file, 5, recurrent="reset")
        stream = cut.state.part("attn").stream()
        self.assertEqual((stream.positions.tolist(), cut.media.media), ([0, 1, 2, 3, 4], ()))
        self.assertEqual(cut.state.part("recurrent").positions.tolist(), [4])
        np.testing.assert_array_equal(cut.state.part("attn").stream().k[3], file.state.part("attn").stream().k[3][:5])
        np.testing.assert_array_equal(cut.token_words, kvstate.token_words(file.media.tokens[:5], server_packed=True))
        # after the image: the whole image stays, the state ends at its shared position
        cut = kvstate.truncate_state(file, 11, recurrent="reset")
        stream = cut.state.part("attn").stream()
        self.assertEqual(stream.positions.tolist(), [0, 1, 2, 3, 4] + [5] * 6)
        np.testing.assert_array_equal(stream.ext, file.state.part("attn").stream().ext[:11])
        self.assertEqual((cut.state.part("recurrent").positions.tolist(), len(cut.media.media)), ([5], 1))
        kvstate.parse_file(cut.to_bytes(), self.spec)
        with self.assertRaisesRegex(StateFormatError, "inside the media chunk"):
            kvstate.truncate_state(file, 8, recurrent="reset")
        with self.assertRaisesRegex(StateFormatError, "needs R0"):
            kvstate.truncate_state(file, 11)
        # the whole state is no cut: the recurrent section stays as saved
        same = kvstate.truncate_state(file, 15)
        self.assertEqual(same.to_bytes(), file.to_bytes())
        with self.assertRaisesRegex(StateFormatError, "outside"):
            kvstate.truncate_state(file, 0)

    def test_truncation_keeps_a_prefix_of_cells_as_views(self):
        # the engine reads a memory-mapped source slot up to its first medium: the kept prefix of
        # every layer is sliced (a view, nothing copied), cells out of order are gathered
        data = media_state(self.spec, self.rng).to_bytes()
        file = kvstate.parse_file(np.frombuffer(data, dtype=np.uint8), self.spec)
        source = file.state.part("attn").stream()
        stream = kvstate.truncate_state(file, 5, recurrent="reset").state.part("attn").stream()
        self.assertTrue(all(np.shares_memory(stream.k[l], source.k[l]) and np.shares_memory(stream.v[l], source.v[l])
                            for l in stream.k))
        np.testing.assert_array_equal(stream.k[3], source.k[3][:5])
        self.assertEqual(stream.seq_ids, tuple(source.seq_ids[:5]))
        # cells saved out of position order (a stream that wrapped around), a dropped image cell
        # between kept ones: the kept ones are gathered
        order = np.asarray([0, 2, 9, 1, 3, 4] + [i for i in range(5, len(source.positions)) if i != 9])
        shuffled = kvstate.KVStream(source.positions[order], tuple(source.seq_ids[i] for i in order), source.ext[order],
                                    source.v_trans, {l: np.asarray(r)[order] for l, r in source.k.items()},
                                    {l: np.asarray(r)[order] for l, r in source.v.items()}, dict(source.k_types),
                                    dict(source.v_types))
        file.state.part("attn").streams[file.state.part("attn").streams.index(source)] = shuffled
        cut = kvstate.truncate_state(file, 5, recurrent="reset").state.part("attn").stream()
        self.assertEqual(cut.positions.tolist(), [0, 2, 1, 3, 4])
        self.assertEqual(cut.seq_ids, tuple(source.seq_ids[i] for i in [0, 2, 1, 3, 4]))
        np.testing.assert_array_equal(cut.k[3], np.asarray(source.k[3])[[0, 2, 1, 3, 4]])
        np.testing.assert_array_equal(cut.v[3], np.asarray(source.v[3])[[0, 2, 1, 3, 4]])
        self.assertFalse(np.shares_memory(cut.k[3], shuffled.k[3]))          # gathered, not sliced

    def test_sliding_sections_hold_only_their_own_window(self):
        meta = {"general.architecture": "muse-glimmer", "muse-glimmer.block_count": 4,
                "muse-glimmer.embedding_length": 32, "muse-glimmer.attention.head_count": 2,
                "muse-glimmer.attention.head_count_kv": 1, "muse-glimmer.attention.key_length": 8,
                "muse-glimmer.attention.value_length": 8, "muse-glimmer.attention.sliding_window": 4,
                "muse-glimmer.attention.sliding_window_pattern": 4, "muse-glimmer.rope.dimension_count": 8,
                "muse-glimmer.rope.freq_base": 5e5}
        spec = kvstate.ModelSpec.from_metadata(meta)
        layers = spec.layers_of("full", "sliding")
        k = {il: self.rng.normal(size=(12, 1, 8)) for il in layers}
        v = {il: self.rng.normal(size=(12, 1, 8)) for il in layers}
        file = kvstate.build_muse_state(spec, np.arange(12), np.arange(12), k, v, server_tokens=True)
        # a native save at 10 would hold positions 6..9 in the SWA section; this one holds 8..11
        with self.assertRaisesRegex(StateFormatError, r"swa section lacks positions \[6, 7\]"):
            kvstate.truncate_state(file, 10)
        near = kvstate.truncate_state(file, 12)
        self.assertEqual(near.to_bytes(), file.to_bytes())
        # sliding="keep" reads what is there (for source rows); the SWA section then ends early
        kept = kvstate.truncate_state(file, 10, sliding="keep")
        self.assertEqual(kept.state.part("swa").stream().positions.tolist(), [8, 9])
        self.assertEqual(kept.state.part("base").stream().positions.tolist(), list(range(10)))

    def test_handoff_index_tolerates_shared_positions(self):
        stream = media_state(self.spec, self.rng).state.part("attn").stream()
        index = handoff._stream_index(stream, "source")
        self.assertEqual((sorted(index.shared), 5 in index, 11 in index), ([5], False, True))
        np.testing.assert_array_equal(handoff._cells(index, [0, 4, 8, 11], "source"), [0, 4, 11, 14])
        with self.assertRaisesRegex(ValueError, "several cells at positions \\[5\\]"):
            handoff._cells(index, [4, 5], "source")
        with self.assertRaisesRegex(ValueError, "lacks positions \\[6\\]"):
            handoff._cells(index, [6], "source")
        # a cut leaves the image (and everything from it on) out of the index
        cut = handoff._stream_index(stream, "source", pos_cut=5)
        self.assertEqual((sorted(cut), sorted(cut.shared)), ([0, 1, 2, 3, 4], []))
        # the cut is a position: after the M-RoPE image (token entries 5..10, cells at 5, n_pos 3)
        # token index 11 is position 8, and the index then keeps what truncate_state keeps
        state = media_state(self.spec, self.rng)
        pos_cut = state.media.pos_next(11)
        truncated = kvstate.truncate_state(state, 11, recurrent="reset")
        kept = handoff._stream_index(stream, "source", pos_cut=pos_cut)
        self.assertEqual((pos_cut, sorted(kept), sorted(kept.shared)), (8, [0, 1, 2, 3, 4], [5]))
        self.assertEqual(sorted(set(truncated.state.part("attn").stream().positions.tolist())), [0, 1, 2, 3, 4, 5])


@unittest.skipUnless(SERVER, "set KVPARK_LLAMA_SERVER to run slot saves with images on a throwaway llama-server")
class MediaSlotTests(unittest.TestCase):
    """Real slot saves holding an image, from llama-server with an mtmd projector."""

    PROMPT = "Look at this: {marker} and describe it in words."

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-media-")
        cls.root = Path(cls.temp.name)
        cls.png = fixtures_media.write_png(cls.root / "image.png", 128, 96, seed=1)
        cls.mmproj = cls.root / "mmproj-tiny.gguf"
        fixtures_media.create_qwen3vl_mmproj(cls.mmproj)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def start(self, model, mmproj, **options):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        slots = directory / "slots"
        slots.mkdir()
        server = fixtures_media.Server(SERVER, model, slots, directory / "server.log", fixtures_media.free_port(PORTS),
                                       mmproj=mmproj, extra=["--image-max-tokens", "16"], **options)
        self.addCleanup(server.close)
        return server, slots

    def save_full(self, server, slots, spec, prompt=PROMPT):
        server.complete(prompt.format(marker=server.props["media_marker"]), images=[self.png])
        server.save("full.bin")
        data = (slots / "full.bin").read_bytes()
        state = kvstate.parse_file(data, spec)
        self.assertEqual(state.to_bytes(), data)
        (chunk,) = state.media.media
        self.assertEqual((chunk.modality, chunk.id), ("image", hashlib.sha256(self.png).hexdigest()))
        return state

    def write(self, slots, name, state_file):
        kvstate.write_file(slots / name, state_file)
        kvstate.write_new(slots / (name + ".resume"), kvstate.ResumeFile().to_bytes())

    def reuse(self, server, name, prompt=PROMPT):
        """Restore a slot file, then send the full image request: (restored cells, cache_n)."""
        server.erase()
        restored = server.restore(name)["n_restored"]
        answer = server.complete(prompt.format(marker=server.props["media_marker"]), images=[self.png])
        return restored, answer["timings"]["cache_n"]

    def native_prefix(self, server, slots, state, cut):
        server.erase()
        server.complete([int(t) for t in state.media.tokens[:cut]])
        server.save(f"prefix-{cut}.bin")
        return (slots / f"prefix-{cut}.bin").read_bytes()

    @unittest.skipUnless(QWEN_MMPROJ, "set KVPARK_QWEN_MMPROJ to a 5120-wide qwen3vl_merger mmproj (the Qwen3.8 one)")
    def test_mrope_image_in_a_hybrid_slot(self):
        model = self.root / "qwen35-vision.gguf"
        if not model.exists():
            fixtures_media.create_vision_qwen35(model)
        spec = kvstate.ModelSpec.from_gguf(model)
        server, slots = self.start(model, QWEN_MMPROJ, flash_attn="off")
        state = self.save_full(server, slots, spec)
        media = state.media
        (chunk,) = media.media
        self.assertEqual((chunk.pos_type, chunk.n_tokens, chunk.n_pos),
                         ("mrope", chunk.nx * chunk.ny, max(chunk.nx, chunk.ny)))
        self.assertEqual(media.tokens[chunk.start - 1], fixtures_media.VISION_START_ID)
        stream = state.state.part("attn").stream()
        # every image cell sits at the image's first position, its row and column in the cell ext;
        # text after it continues at that position + max(nx, ny)
        pos0 = int(stream.positions[chunk.start])
        np.testing.assert_array_equal(stream.positions, media.cell_positions())
        grid = np.arange(chunk.n_tokens)
        expected = np.stack([pos0 + grid % chunk.nx, pos0 + grid // chunk.nx, -np.ones_like(grid)], 1)
        np.testing.assert_array_equal(stream.ext[chunk.start:chunk.end], expected)
        self.assertEqual(int(stream.positions[chunk.end]), pos0 + chunk.n_pos)
        self.assertEqual(state.state.part("recurrent").positions.tolist(), [media.pos_next() - 1])
        # cut before <|vision_start|>: the KV section is a native save of that prefix, the recurrent
        # rows need R0
        cut = kvstate.media_cut(state, before=[fixtures_media.VISION_START_ID])
        self.assertEqual(cut, chunk.start - 1)
        with self.assertRaisesRegex(StateFormatError, "needs R0"):
            kvstate.truncate_state(state, cut)
        truncated = kvstate.truncate_state(state, cut, recurrent="reset")
        native = kvstate.parse_file(self.native_prefix(server, slots, state, cut), spec)
        np.testing.assert_array_equal(truncated.token_words, native.token_words)
        mine, theirs = truncated.state.part("attn").stream(), native.state.part("attn").stream()
        np.testing.assert_array_equal(mine.ext, theirs.ext)
        for key in ("k", "v"):
            np.testing.assert_array_equal(getattr(mine, key)[3], getattr(theirs, key)[3])
        self.write(slots, "cut.bin", truncated)
        self.assertEqual(self.reuse(server, "cut.bin"), (cut, cut))
        # cut right after the image: the image's cells come back and the same image request reuses them
        after = kvstate.truncate_state(state, chunk.end, recurrent="reset")
        self.assertEqual(len(after.media.media), 1)
        self.write(slots, "after.bin", after)
        self.assertEqual(self.reuse(server, "after.bin"), (chunk.end, chunk.end))

    def test_normal_position_images_in_dense_and_iswa_slots(self):
        cases = (("qwen3", {}), ("muse-glimmer", {"sliding_window": 64}))
        for architecture, options in cases:
            with self.subTest(architecture=architecture):
                model = self.root / f"{architecture}-vision.gguf"
                if not model.exists():
                    fixtures_media.create_vision_text_model(model, architecture, **options)
                spec = kvstate.ModelSpec.from_gguf(model)
                server, slots = self.start(model, self.mmproj)
                state = self.save_full(server, slots, spec)
                (chunk,) = state.media.media
                # NORMAL positions: one per cell (nx = n_tokens, ny = 1), no cell ext
                self.assertEqual((chunk.pos_type, chunk.n_pos, chunk.ny), ("normal", chunk.n_tokens, 1))
                positions = state.state.parts[0].stream().positions
                np.testing.assert_array_equal(positions, np.arange(len(state.media.tokens)))
                cut = kvstate.media_cut(state, before=[fixtures_media.VISION_START_ID])
                truncated = kvstate.truncate_state(state, cut)
                native = self.native_prefix(server, slots, state, cut)
                # no recurrent rows: the truncated state is the native save of the prefix, byte for byte
                self.assertEqual(truncated.to_bytes(), native)
                self.write(slots, "cut.bin", truncated)
                self.write(slots, "native.bin", kvstate.parse_file(native, spec))
                restored, cache_n = self.reuse(server, "cut.bin")
                # llama-server reprocesses an iSWA slot whose SWA cells do not reach back past n_swa
                # unless a checkpoint in the .resume covers it (server-context.cpp:3366-3445): with the
                # same empty .resume the native save behaves the same
                self.assertEqual((restored, cache_n), (cut, self.reuse(server, "native.bin")[1]))
                if architecture == "qwen3":
                    self.assertEqual(cache_n, cut)
                    after = kvstate.truncate_state(state, chunk.end)
                    self.write(slots, "after.bin", after)
                    self.assertEqual(self.reuse(server, "after.bin"), (chunk.end, chunk.end))
                else:
                    self.assertEqual(cache_n, 0)
                    # a long tail pushes the prefix out of the saved SWA window: the cut is refused, and
                    # sliding="keep" reads the rows that are left
                    long = self.save_full(server, slots, spec, self.PROMPT + " " + "and more words " * 6)
                    with self.assertRaisesRegex(StateFormatError, "swa section lacks positions"):
                        kvstate.truncate_state(long, cut)
                    kept = kvstate.truncate_state(long, cut, sliding="keep")
                    self.assertEqual(kept.state.part("base").stream().positions.tolist(), list(range(cut)))


if __name__ == "__main__":
    unittest.main()
