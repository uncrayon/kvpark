import base64
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from kvpark.bridge import resume

# Template draft blobs written by the kvpark-patched CPU llama-server (runtime/build) for a
# one-token-prompt save of five tokens, from the tiny fixtures of
# experiments/universal_kv/fixtures_bridge/make_fixtures.py. Every header byte is the server's;
# the row values (random weights) are zeroed so the samples compress.
GOLDEN = {
    # Qwen3.5 hybrid target + M-RoPE DFlash draft (2 layers, 1 KV head of 32, F16), -fa on
    "dflash-mrope": "eNq7YSOyngEIGIGYlQECGBkQ4D8QwMSQMUycCcpngmKYODNUnBmKYeIsUHEWKIaJMyDZPdDugJnlwDAKqAFGw3I0LEfDcngDAMXwKwg=",
    # the same with -fa off: V stored transposed (v_trans 1)
    "dflash-mrope-vtrans": "eNq7YSOyngEIGIGYlQECGBkQ4D8QwMSQMUycCcpngmKYODNUnBmKYeIsUHEWKEY2f7C4A2aWA8MooAYYDUvqhiUofSqMBsVoWA4iAABcICrO",
    # dense Qwen3 target + NEOX DFlash draft: no cell extension
    "dflash-neox": "eNq7YSOyngEIGIGYlQECGJFoGJsJic2MxGZBYjMg6aOGWTB1DgyjgBpgNCxHw3I0LIc3AADu1QMI",
    # Qwen3.5 hybrid with --spec-type draft-mtp: the nextn layer's own KV cache
    "mtp": "eNq7YSOyngEIGIGYlQECGBkQwANJDBmnQsWZoHwmKM6BijNDxZmhGCbOAhVngeJ8JLtgZjswjAJqgNGwpB4AAMgjBHg=",
}
# draft blob sizes the server accepted for T0 = 64 in the live check (live_resume_check.py)
LIVE_SIZES = {"dflash-mrope": 696, "dflash-neox": 648, "mtp": 16}


def golden(name):
    return zlib.decompress(base64.b64decode(GOLDEN[name]))


def kv_section(cells, layers, *, ext=False, v_trans=0, v=True, n_stream=1, home=0, positions=None):
    """A KV section of llama_kv_cache::state_write with `cells` cells and `layers` F16 layers of 8-byte rows."""
    out = [struct.pack("<I", n_stream)]
    positions = list(range(cells)) if positions is None else positions
    for s in range(n_stream):
        if s != home or not cells:
            out.append(struct.pack("<I", 0))
            continue
        out.append(struct.pack("<I", cells))
        for pos in positions:
            out.append(struct.pack("<iI", pos, 1) + (struct.pack("<iii", pos, pos, 7) if ext else b"") + struct.pack("<i", 0))
        out.append(struct.pack("<II", v_trans, layers))
        out += [struct.pack("<iQ", 1, 8) + b"\x11" * 8 * cells for _ in range(layers)]
        if v:
            if v_trans:
                out += [struct.pack("<iII", 1, 2, 4) + b"\x22" * 8 * cells for _ in range(layers)]
            else:
                out += [struct.pack("<iQ", 1, 8) + b"\x22" * 8 * cells for _ in range(layers)]
    return b"".join(out)


def recurrent_section(cells, layers):
    out = [struct.pack("<I", cells)]
    out += [struct.pack("<iI", 9, 0) for _ in range(cells)]
    out.append(struct.pack("<II", 0, layers + 1))
    out += [struct.pack("<iQ", 0, 12) + bytes(12 * cells) for _ in range(layers)]
    out += [struct.pack("<iQ", 0, 16) + bytes(16 * cells) for _ in range(layers)]
    return b"".join(out)


def blob(*sections, seq_id=0):
    return struct.pack("<Ii", resume.IO_MAGIC, seq_id) + b"".join(sections)


class ResumeFileTest(unittest.TestCase):
    def test_round_trip_and_the_servers_checks(self):
        data = resume.Resume(b"draft", b"", [resume.Checkpoint(3, 0, 2, b"t", b"d", b"")]).to_bytes()
        parsed = resume.parse_resume(data)
        self.assertEqual((parsed.draft, parsed.speculative), (b"draft", b""))
        self.assertEqual(parsed.checkpoints, [resume.Checkpoint(3, 0, 2, b"t", b"d", b"")])
        self.assertEqual(resume.Resume().to_bytes(), struct.pack("<IIQQI", 0x51534C54, 1, 0, 0, 0))
        for bad in (data + b"\0", data[:-1], b"XXXX" + data[4:], struct.pack("<IIQ", 0x51534C54, 1, 10**9)):
            with self.assertRaises(resume.StateBlobError):
                resume.parse_resume(bad)
        with self.assertRaises(ValueError):
            resume.Resume(checkpoints=[resume.Checkpoint(1, 2, 1)]).to_bytes()

    def test_slot_token_count_reads_plain_and_server_packed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            plain, packed = Path(directory) / "plain.bin", Path(directory) / "packed.bin"
            plain.write_bytes(struct.pack("<III", 0x67677371, 3, 3) + struct.pack("<3i", 5, 6, 7))
            packed.write_bytes(struct.pack("<III", 0x67677371, 3, 9) + struct.pack("<9i", -1, 1, 5, 1, 2, 3, 4, 5, 0))
            self.assertEqual(resume.slot_token_count(plain), 3)
            self.assertEqual(resume.slot_token_count(packed), 5)
            plain.write_bytes(b"GGUF" + bytes(20))
            with self.assertRaises(resume.StateBlobError):
                resume.slot_token_count(plain)

    def test_private_files_are_owner_only_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = resume.write_private(Path(directory) / "x.resume", b"abc")
            self.assertEqual(path.read_bytes(), b"abc")
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                resume.write_private(path, b"again")


class StateBlobTest(unittest.TestCase):
    def test_server_written_drafts_are_classified(self):
        kinds = {name: resume.classify_draft(golden(name), n_tokens=5) for name in GOLDEN}
        self.assertEqual({name: layout.kind for name, layout in kinds.items()},
                         {"dflash-mrope": "dflash", "dflash-mrope-vtrans": "dflash", "dflash-neox": "dflash",
                          "mtp": "mtp"})
        base, window = kinds["dflash-mrope"].blob.sections
        self.assertEqual((base.layers, base.cells, base.ext), ((), 5, True))
        self.assertEqual(window.layers, (resume.LayerRows(1, 64, 1, 64),) * 2)
        self.assertEqual(window.streams[0].positions, (0, 1, 2, 3, 4))
        self.assertEqual([x == y == p for (x, y, _), p in zip(window.streams[0].ext, range(5))], [True] * 5)
        self.assertEqual(kinds["dflash-mrope-vtrans"].blob.sections[1].layers, (resume.LayerRows(1, 64, 1, 64, 2, 32),) * 2)
        self.assertFalse(kinds["dflash-neox"].blob.sections[1].ext)
        (nextn,) = kinds["mtp"].blob.sections
        self.assertEqual((len(nextn.layers), nextn.ext, nextn.cells), (1, True, 5))
        self.assertEqual(resume.classify_draft(b"").kind, "none")

    def test_translated_drafts_keep_the_template_layout(self):
        for name in ("dflash-mrope", "dflash-mrope-vtrans", "dflash-neox"):
            layout = resume.classify_draft(golden(name), n_tokens=5)
            data = resume.translated_draft(layout, 64)
            if name in LIVE_SIZES:
                self.assertEqual(len(data), LIVE_SIZES[name])
            out = resume.parse_state_blob(data, n_tokens=64)
            for template, section in zip(layout.blob.sections, out.sections):
                self.assertEqual((section.n_stream, section.ext, section.v_trans, section.layers),
                                 (template.n_stream, template.ext, template.v_trans, template.layers))
                self.assertEqual(section.streams[0].positions, (62, 63))
                if section.ext:
                    self.assertEqual(section.streams[0].ext, ((62, 62, -1), (63, 63, -1)))
            # the rows are zero (after the cell metadata, only headers are nonzero)
            self.assertEqual(data[-64:], bytes(64))
        mtp = resume.translated_draft(resume.classify_draft(golden("mtp"), n_tokens=5), 64)
        self.assertEqual(mtp, struct.pack("<IiII", resume.IO_MAGIC, 0, 1, 0))
        self.assertEqual(len(mtp), LIVE_SIZES["mtp"])
        self.assertEqual(resume.translated_draft(resume.classify_draft(b""), 64), b"")
        with self.assertRaisesRegex(ValueError, "two restored tokens"):
            resume.translated_draft(resume.classify_draft(golden("dflash-neox"), n_tokens=5), 1)

    def test_translated_resume_has_no_speculative_state_or_checkpoints(self):
        template = resume.Resume(golden("mtp"), b"spec", [resume.Checkpoint(1, 0, 0, b"t", b"d")]).to_bytes()
        data, layout = resume.translated_resume(template, 100, n_tokens=5)
        parsed = resume.parse_resume(data)
        self.assertEqual((layout.kind, parsed.speculative, parsed.checkpoints), ("mtp", b"", []))
        self.assertEqual(len(parsed.draft), 16)
        data, layout = resume.translated_resume(resume.Resume().to_bytes(), 100)
        self.assertEqual((layout.kind, data), ("none", resume.Resume().to_bytes()))

    def test_generic_layouts(self):
        # plain KV with M-RoPE cells and transposed V; iSWA K-only (a DSV4 DSpark-style draft);
        # hybrid KV + recurrent; a second stream holding the sequence
        plain = resume.parse_state_blob(blob(kv_section(3, 2, ext=True, v_trans=1)), n_tokens=3)
        (section,) = plain.sections
        self.assertEqual((section.ext, section.v_trans, section.layers), (True, 1, (resume.LayerRows(1, 8, 1, 8, 2, 4),) * 2))
        k_only = blob(kv_section(4, 0, v=False), kv_section(4, 2, v=False))
        layout = resume.classify_draft(k_only, n_tokens=4)
        self.assertEqual(layout.kind, "dflash")
        self.assertIn("K-only", layout.note)
        data = resume.translated_draft(layout, 10)
        self.assertEqual(resume.parse_state_blob(data).sections[1].layers, (resume.LayerRows(1, 8),) * 2)
        self.assertEqual(len(data), 8 + (4 + 4 + 24 + 8) + (4 + 4 + 24 + 8 + 2 * (12 + 16)))
        hybrid = resume.parse_state_blob(blob(kv_section(3, 1), recurrent_section(1, 2)))
        self.assertEqual([type(s).__name__ for s in hybrid.sections], ["KVSection", "RecurrentSection"])
        self.assertEqual((hybrid.sections[1].r, hybrid.sections[1].s), (((0, 12),) * 2, ((0, 16),) * 2))
        other = resume.classify_draft(blob(kv_section(3, 1), recurrent_section(1, 2)), n_tokens=3)
        self.assertEqual(other.kind, "other")
        with self.assertRaisesRegex(ValueError, "no translated draft"):
            resume.translated_draft(other, 10)
        self.assertEqual(resume.classify_draft(blob(kv_section(3, 3)), n_tokens=3).kind, "other")
        streams = resume.classify_draft(blob(kv_section(2, 0, n_stream=2, home=1), kv_section(2, 1, n_stream=2, home=1)))
        self.assertEqual(streams.kind, "dflash")
        out = resume.parse_state_blob(resume.translated_draft(streams, 9))
        self.assertEqual([s is None for s in out.sections[1].streams], [True, False])
        self.assertEqual(out.sections[1].streams[1].positions, (7, 8))

    def test_unreadable_and_ambiguous_bytes(self):
        self.assertEqual(resume.classify_draft(b"\x01\x02\x03").kind, "other")
        self.assertEqual(resume.classify_draft(blob(kv_section(2, 1))[:-3]).kind, "other")
        with self.assertRaises(resume.StateBlobError):
            resume.parse_state_blob(struct.pack("<Ii", 0x12345678, 0) + kv_section(1, 1))
        empty = blob(kv_section(0, 0))        # an empty marker reads the same with or without cell extensions
        self.assertEqual(len(resume.parse_state_blob(empty).sections), 1)


if __name__ == "__main__":
    unittest.main()
