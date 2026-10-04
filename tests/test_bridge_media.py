"""Media a target model cannot take: text surrogates in switches and in the real request.

Fake llama-servers (test_bridge.FakeLlama) stand in for the source (a), the target (b) and a
describer (c); each answers a side request with the media it saw, or refuses media it has no
modality for, as llama-server does. The video tests that need real keyframes and sound use
ffmpeg and are skipped without it.
"""

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge import FAKE_ENGINE, FEATURED_ENGINE, FakeLlama, write_gguf  # noqa: E402

from kvpark.bridge import capabilities, registry, surrogates  # noqa: E402
from kvpark.bridge import cli as bridge_cli  # noqa: E402
from kvpark.bridge import switch as bridge_switch  # noqa: E402
from kvpark.bridge.client import LlamaServer, ServerError  # noqa: E402
from kvpark.bridge.messages import conversation_from_openai, flatten, media, media_bytes, media_digest  # noqa: E402

PNG, WAV, MP4 = b"\x89PNG not really an image", b"RIFF not really audio", b"not really a video"


def b64(data):
    return base64.b64encode(data).decode()


def image(data=PNG):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(data)}}


def audio(data=WAV):
    return {"type": "input_audio", "input_audio": {"data": b64(data), "format": "wav"}}


def video(data=MP4, kind="input_video"):
    return {"type": kind, kind: {"url": "data:video/mp4;base64," + b64(data)}}


def ffmpeg_video(path, *, sound):
    """A 3-second test video (ffmpeg's test pattern, with a tone when `sound`) as MP4 bytes."""
    argv = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10"]
    if sound:
        argv += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest"]
    subprocess.run(argv + ["-pix_fmt", "yuv420p", "-y", str(path)], check=True, timeout=120)
    return Path(path).read_bytes()


class FakeTranscriber:
    """An OpenAI-compatible speech-to-text endpoint (POST /v1/audio/transcriptions, multipart):
    answers {"text": text}; keeps each request's raw body."""

    def __init__(self, text=" hello there "):
        self.text, self.bodies = text, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.bodies.append((self.path, self.headers.get("Content-Type"), body))
                status, payload = (200, {"text": fake.text}) if self.path == "/v1/audio/transcriptions" else (404, {})
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/audio/transcriptions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class MediaSwitchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.config_path = root / "bridge" / "bridge.json"
        self.slots = {name: root / f"slots-{name}" for name in ("a", "b", "c")}
        for path in self.slots.values():
            path.mkdir()
        self.servers = {name: FakeLlama(self.slots[name], name) for name in ("a", "b", "c")}
        for server in self.servers.values():
            server.modalities = {}
        self.gguf = write_gguf(root / "model.gguf", "qwen3")
        hub = root / "hub"
        hub.mkdir()
        (hub / "manifest.json").write_text(json.dumps({"format_version": 1, "models": {"a": {}, "b": {}, "c": {}}}))
        engine = root / "engine.py"
        engine.write_text(FEATURED_ENGINE)
        config = registry.empty()
        for name in ("a", "b", "c"):
            registry.add_model(config, name, url=self.servers[name].url, gguf=self.gguf, slot_dir=self.slots[name])
        registry.set_engine(config, python=sys.executable, script=engine, hub=hub)
        config["tail"] = 5
        config["min_handover"] = 0          # the fake conversations are tiny
        registry.save(config, self.config_path)
        self.config = registry.load(self.config_path)
        self.bridge_dir = self.config_path.parent
        self.servers["a"].resident = 40     # the source's slot holds the conversation

    def tearDown(self):
        for server in self.servers.values():
            server.close()
        self.tmp.cleanup()

    def request(self, part, *, at="early", follow=None):
        """OpenAI messages with `part` in message 1 (early: the source has answered after it) or in
        the newest message 3; `follow`: a text part after the media part."""
        content = [{"type": "text", "text": "what is this"}, part] + ([{"type": "text", "text": follow}] if follow else [])
        return {"messages": [{"role": "system", "content": "be brief"},
                             {"role": "user", "content": content if at == "early" else "tell me about caches"},
                             {"role": "assistant", "content": "a cache"},
                             {"role": "user", "content": content if at == "newest" else "and now?"}]}

    def switch(self, raw, source="a", target="b", **kwargs):
        return bridge_switch.switch(self.config, conversation_from_openai(raw["messages"]), source, target,
                                    config_path=self.config_path, **kwargs)

    def job(self, target="b"):
        return self.servers[target].restored_payload["job"]

    def posts(self, name):
        return [path for method, path in self.servers[name].calls if method == "POST"]

    def assert_no_bridge_files(self):
        for path in self.slots.values():
            self.assertEqual(sorted(p.name for p in path.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.bridge_dir.glob("surrogates/.*")), [])

    # -- images ------------------------------------------------------------------------------

    def test_an_image_for_a_text_only_target_becomes_the_sources_description(self):
        self.servers["a"].modalities = {"vision": True}
        raw = self.request(image())
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        digest = media_digest(image())
        self.assertEqual(digest, hashlib.sha256(PNG).hexdigest())
        (part,) = report["surrogates"]["parts"]
        self.assertEqual({k: part[k] for k in ("message", "modality", "kind", "producer", "cached")},
                         {"message": 1, "modality": "vision", "kind": "vision", "producer": "a", "cached": False})
        described = f"[Image description: a saw 1: vision {digest[:8]}]"
        job = self.job()
        self.assertEqual(job["conversation"]["messages"][1]["content"], "what is this\n" + described)
        # the source's cache holds the image's own cells: T0 stops before that message, the target
        # prefills the surrogate itself; no media marker anywhere, so no marker allowance on the tail
        self.assertEqual(job["target"]["max_t0_message"], 1)
        # (the engine reads the source's cache up to the image: "media-source")
        self.assertEqual(report["media"], {"modalities": ["vision"], "first_message": 1, "source_media": True,
                                           "tokens_before": len("<system>be brief<user>")})
        self.assertEqual(job["tail"], 5)
        self.assertNotIn(self.servers["b"].marker, json.dumps(job["conversation"]))
        self.assertTrue(report["reuse_ok"])                   # no media left: the warm request runs
        # the side request: the image and the prompt, deterministic, thinking off
        (chat,) = self.servers["a"].chats
        self.assertEqual(chat["messages"][0]["content"][0], image())
        self.assertEqual(chat["messages"][0]["content"][1]["text"], surrogates.PROMPTS["vision"])
        self.assertEqual((chat["temperature"], chat["chat_template_kwargs"]), (0, {"enable_thinking": False}))
        self.assertEqual(chat["id_slot"], 0)                  # the slot the hold saves and restores
        # the source's slot was saved before it (the save the engine read) and restored after
        self.assertEqual(self.posts("a"), ["/slots/0?action=save", "/v1/chat/completions", "/slots/0?action=erase",
                                           "/slots/0?action=restore"])
        saved = [body for path, body in self.servers["a"].bodies if path.endswith("save")][0]["filename"]
        self.assertTrue(job["source"]["state"].endswith(saved))
        self.assertEqual(self.servers["a"].resident, 40)
        self.assertIn("source_restore_s", report["timings"])
        self.assert_no_bridge_files()
        # later switches take it from the cache: no side request, the same text
        again = self.switch(raw)
        self.assertEqual(len(self.servers["a"].chats), 1)
        self.assertTrue(again["surrogates"]["parts"][0]["cached"])
        self.assertEqual(self.job()["conversation"]["messages"][1]["content"], "what is this\n" + described)

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_the_surrogate_cache_is_private_and_holds_no_media(self):
        self.servers["a"].modalities = {"vision": True}
        self.switch(self.request(image()))
        directory = self.bridge_dir / surrogates.CACHE_DIR
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        (entry,) = directory.iterdir()
        self.assertEqual(entry.stat().st_mode & 0o777, 0o600)
        self.assertTrue(entry.name.startswith(f"{hashlib.sha256(PNG).hexdigest()}-vision-v{surrogates.PROMPT_VERSION}-"))
        text = entry.read_text()
        self.assertNotIn(b64(PNG), text)
        self.assertEqual(json.loads(text)["producer"], "a")

    def test_the_real_request_is_rewritten_as_the_switch_rendered_it(self):
        self.servers["a"].modalities = {"vision": True}
        raw = self.request(image(), at="newest")
        original = json.loads(json.dumps(raw))
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        # the source never saw the newest message: its cache holds no image, nothing caps T0 there
        # (T0 stops before the newest message anyway)
        self.assertNotIn("max_t0_message", self.job()["target"])
        rewritten, summary = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        self.assertEqual(raw, original)                       # the caller's request is not changed
        self.assertTrue(summary["parts"][0]["cached"])
        b = LlamaServer(self.servers["b"].url)
        self.assertEqual(b.apply_template(rewritten["messages"]),
                         b.apply_template(self.job()["conversation"]["messages"]))
        self.assertIsInstance(rewritten["messages"][3]["content"], str)
        self.assertEqual(len(self.servers["a"].chats), 1)     # from the cache
        # a target with the modality, or a request without media, needs no rewrite
        self.assertEqual(surrogates.rewrite_request(self.config, {"messages": [{"role": "user", "content": "hi"}]},
                                                    "b", bridge_dir=self.bridge_dir), (None, None))
        self.servers["b"].modalities = {"vision": True}
        self.assertEqual(surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir), (None, None))

    def test_a_text_only_source_holds_the_surrogate_so_nothing_caps_t0(self):
        # Muse -> Qwen3-4B: neither has vision; a describer (c) makes the description, and the source
        # processed the same surrogate text before (its slot holds text), so T0 is not cut there
        self.servers["c"].modalities = {"vision": True}
        self.servers["c"].resident = 25
        registry.set_media(self.config, describers={"vision": "c"})
        report = self.switch(self.request(image()))
        self.assertNotIn("skipped", report)
        self.assertEqual(report["surrogates"]["parts"][0]["producer"], "c")
        self.assertNotIn("max_t0_message", self.job()["target"])
        self.assertNotIn("first_message", report["media"])
        self.assertEqual(self.servers["a"].chats, [])
        # the describer's own conversation is saved and restored around its side request
        self.assertEqual(self.servers["c"].resident, 25)
        self.assertEqual(self.posts("c"), ["/slots/0?action=save", "/v1/chat/completions", "/slots/0?action=erase",
                                           "/slots/0?action=restore"])
        self.assertEqual(self.posts("a"), ["/slots/0?action=save"])       # step 3 only
        self.assert_no_bridge_files()

    def test_a_failing_producer_falls_back_to_the_next(self):
        self.servers["a"].modalities = {"vision": True}
        self.servers["a"].chat_error = "out of memory"
        self.servers["c"].modalities = {"vision": True}
        registry.set_media(self.config, describers={"vision": "c"})
        report = self.switch(self.request(image()))
        self.assertEqual(report["surrogates"]["parts"][0]["producer"], "c")
        self.assertIn("out of memory", report["surrogates"]["errors"][0])
        self.assertEqual(self.servers["a"].resident, 40)                 # restored all the same
        self.assertEqual(self.job()["target"]["max_t0_message"], 1)       # the source still holds the image

    def test_no_producer_or_a_skip_policy_skips_the_switch(self):
        report = self.switch(self.request(image()))
        self.assertEqual(report["skipped"], "target-no-vision")
        self.assertIn("kvpark bridge media --describer", report["note"])
        self.assertEqual(self.posts("a"), [])               # nothing asked, the source's cache untouched
        # a describer that has no vision does not help either
        registry.set_media(self.config, describers={"vision": "c"})
        self.assertEqual(self.switch(self.request(image()))["skipped"], "target-no-vision")
        self.assertEqual(self.servers["c"].chats, [])
        # the source could describe it, but the policy keeps images as they are (for b only)
        self.servers["a"].modalities = {"vision": True}
        registry.set_media(self.config, policy={"image": "skip"}, target="b")
        report = self.switch(self.request(image()))
        self.assertEqual(report["skipped"], "target-no-vision")
        self.assertIn("policy", report["note"])
        self.assertEqual(self.servers["a"].chats, [])
        registry.set_media(self.config, policy={"image": "default"}, target="b")
        self.assertNotIn("skipped", self.switch(self.request(image())))

    def test_a_surrogate_already_in_the_conversation_is_used(self):
        raw = self.request(image(), follow="[Image description: a red square]")
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        self.assertEqual(report["surrogates"]["parts"][0]["producer"], "conversation")
        self.assertEqual(self.job()["conversation"]["messages"][1]["content"],
                         "what is this\n[Image description: a red square]")
        rewritten, _ = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        self.assertEqual(rewritten["messages"][1]["content"], "what is this\n[Image description: a red square]")

    def test_an_older_engine_skips_media_the_source_holds_before_any_side_request(self):
        Path(self.config["engine"]["script"]).write_text(FAKE_ENGINE)
        self.servers["a"].modalities = {"vision": True}
        report = self.switch(self.request(image()))
        self.assertEqual(report["skipped"], "media-source")
        self.assertEqual(self.servers["a"].chats, [])

    def test_mixed_media_keep_what_the_target_reads(self):
        # an image+text target: the image stays native (marker, T0 cap), the audio becomes text
        self.servers["a"].modalities = {"vision": True, "audio": True}
        self.servers["b"].modalities = {"vision": True}
        raw = self.request(image(), at="newest")
        raw["messages"][3]["content"].append(audio())
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        self.assertEqual([p["modality"] for p in report["surrogates"]["parts"]], ["audio"])
        self.assertEqual(report["media"]["native"], ["vision"])
        marker = self.servers["b"].marker
        self.assertEqual(self.job()["conversation"]["messages"][3]["content"],
                         f"what is this{marker}[Audio transcript: a saw 1: audio {media_digest(audio())[:8]}]")
        self.assertEqual(self.job()["target"]["max_t0_message"], 3)
        self.assertNotIn("warm", report)                      # media the target reads: never warmed
        rewritten, _ = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        self.assertEqual([p["type"] for p in rewritten["messages"][3]["content"]], ["text", "image_url", "text"])
        b = LlamaServer(self.servers["b"].url)
        self.assertEqual(b.apply_template(rewritten["messages"]),
                         b.apply_template(self.job()["conversation"]["messages"]))

    def test_a_failed_source_save_fails_the_switch_before_any_producer(self):
        # step 3 needs the source's save anyway: the media go to no describer, and the switch
        # fails with the save's error (once a save failed, it is never taken as done)
        self.servers["a"].modalities = {"vision": True}
        self.servers["c"].modalities = {"vision": True}
        registry.set_media(self.config, describers={"vision": "c"})
        real, a_url = LlamaServer.slot_action, self.servers["a"].url

        def failing(server, action, filename=None):
            if server.url == a_url and action == "save":
                raise ServerError("no --slot-save-path")
            return real(server, action, filename)
        with mock.patch.object(LlamaServer, "slot_action", failing):
            with self.assertRaisesRegex(RuntimeError, "a's slot save failed: no --slot-save-path"):
                self.switch(self.request(image()))
        self.assertEqual((self.servers["a"].chats, self.servers["c"].chats), ([], []))
        self.assertEqual(self.posts("a"), [])
        self.assert_no_bridge_files()

    def test_the_same_image_twice_is_made_once_and_reported_as_repeated(self):
        from kvpark.hermes import _media_summary
        self.servers["a"].modalities = {"vision": True}
        raw = self.request(image())
        raw["messages"][3]["content"] = [{"type": "text", "text": "again"}, image()]
        report = self.switch(raw)
        first, second = report["surrogates"]["parts"]
        self.assertEqual((first["cached"], first.get("duplicate")), (False, None))
        self.assertEqual((second["cached"], second["duplicate"], second["seconds"]), (False, True, 0.0))
        self.assertEqual(len(self.servers["a"].chats), 1)
        self.assertRegex(_media_summary(report), r"^media: 2 images as text \(made by a in [0-9.]+ s; 1 repeated\)$")
        again = self.switch(raw)
        self.assertEqual([(p["cached"], p.get("duplicate")) for p in again["surrogates"]["parts"]],
                         [(True, None), (False, True)])
        self.assertEqual(_media_summary(again), "media: 2 images as text (1 from the surrogate cache or the "
                                                "conversation; 1 repeated)")

    def test_an_image_for_an_audio_only_target_becomes_a_description(self):
        self.servers["a"].modalities = {"vision": True}
        self.servers["b"].modalities = {"audio": True}
        report = self.switch(self.request(image()))
        self.assertNotIn("skipped", report)
        self.assertEqual(report["surrogates"]["parts"][0]["kind"], "vision")
        self.assertEqual(self.job()["conversation"]["messages"][1]["content"],
                         f"what is this\n[Image description: a saw 1: vision {media_digest(image())[:8]}]")
        self.assertNotIn("native", report["media"])

    def test_cli_request_out_asks_the_source_and_replaces_an_open_file(self):
        # a switch skipped before any surrogate (an engine older than "media-source"): the request
        # written still gets the source's description, and an existing world-readable file is
        # replaced by a private one
        Path(self.config["engine"]["script"]).write_text(FAKE_ENGINE)
        self.servers["a"].modalities = {"vision": True}
        import argparse
        parser = argparse.ArgumentParser()
        bridge_cli.add_parser(parser.add_subparsers(dest="command"))
        conversation, out = Path(self.tmp.name) / "conversation.json", Path(self.tmp.name) / "request.json"
        conversation.write_text(json.dumps(self.request(image())))
        out.write_text("{}")
        out.chmod(0o644)
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "switch", "--from", "a", "--to", "b", "--conversation",
                                              str(conversation), "--request-out", str(out), "--config",
                                              str(self.config_path)]))
        report = json.loads(printed.call_args[0][0])
        self.assertEqual(report["skipped"], "media-source")
        self.assertEqual(report["request_out"], {"path": str(out), "rewritten": True})
        if os.name == "posix":
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        self.assertIn("[Image description: a saw 1: vision", json.loads(out.read_text())["messages"][1]["content"])
        self.assertEqual(self.servers["a"].resident, 40)       # its slot saved and restored around it
        self.assertEqual(sorted(p.name for p in out.parent.glob(".request.json*")), [])

    # -- audio ---------------------------------------------------------------------------------

    def test_audio_for_targets_without_audio_becomes_a_transcript(self):
        self.servers["a"].modalities = {"audio": True}
        for target_modalities in ({}, {"vision": True}):
            self.servers["b"].modalities = target_modalities
            report = self.switch(self.request(audio()))
            self.assertNotIn("skipped", report)
            self.assertEqual(report["surrogates"]["parts"][0]["kind"], "audio")
            self.assertEqual(self.job()["conversation"]["messages"][1]["content"],
                             f"what is this\n[Audio transcript: a saw 1: audio {media_digest(audio())[:8]}]")
            self.assertEqual(self.job()["target"]["max_t0_message"], 1)
        self.assertEqual(len(self.servers["a"].chats), 1)                # the second from the cache
        self.assertEqual(self.servers["a"].chats[0]["messages"][0]["content"][1]["text"], surrogates.PROMPTS["audio"])
        # an audio-capable target reads it itself
        self.servers["b"].modalities = {"audio": True}
        report = self.switch(self.request(audio(), at="newest"))
        self.assertNotIn("surrogates", report)
        self.assertEqual(report["media"]["native"], ["audio"])

    def test_audio_without_a_transcriber_is_skipped(self):
        self.servers["a"].modalities = {"vision": True}
        report = self.switch(self.request(audio()))
        self.assertEqual(report["skipped"], "target-no-audio")
        registry.set_media(self.config, policy={"audio": "skip"})
        self.servers["a"].modalities = {"audio": True}
        self.assertEqual(self.switch(self.request(audio()))["skipped"], "target-no-audio")
        self.assertEqual(self.servers["a"].chats, [])

    def test_a_speech_to_text_endpoint_transcribes_audio(self):
        transcriber = FakeTranscriber()
        self.addCleanup(transcriber.close)
        registry.set_media(self.config, describers={"audio": transcriber.url}, transcription_model="whisper-v3-turbo")
        raw = self.request(audio())
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        (part,) = report["surrogates"]["parts"]
        self.assertEqual((part["producer"], part["cached"]), (transcriber.url, False))
        self.assertEqual(self.job()["conversation"]["messages"][1]["content"],
                         "what is this\n[Audio transcript: hello there]")
        ((path, content_type, body),) = transcriber.bodies
        self.assertEqual(path, "/v1/audio/transcriptions")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="model"\r\n\r\nwhisper-v3-turbo\r\n', body)
        self.assertIn(b'name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n' + WAV, body)
        # the real request gets the same text, from the cache
        rewritten, _ = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        self.assertEqual(rewritten["messages"][1]["content"], "what is this\n[Audio transcript: hello there]")
        self.assertEqual(len(transcriber.bodies), 1)
        # silence: no text
        transcriber.text = "  "
        other = audio(b"RIFF other audio")
        self.switch(self.request(other))
        self.assertTrue(self.job()["conversation"]["messages"][1]["content"].endswith("[Audio transcript: No speech.]"))
        # it takes no other media, and no audio given by URL
        producer = surrogates.Transcriber(transcriber.url)
        self.assertEqual([producer.has(m) for m in ("vision", "audio", "video")], [False, True, False])
        with self.assertRaisesRegex(ServerError, "inline audio only"):
            producer.answer([{"type": "input_audio", "input_audio": {"url": "https://example.com/a.wav"}}], "audio")

    # -- video ---------------------------------------------------------------------------------

    def test_a_video_for_a_text_only_target_is_described_by_a_video_model(self):
        self.servers["a"].modalities = {"vision": True, "video": True}
        with mock.patch.object(surrogates, "ffmpeg_tools", return_value=None):
            for kind in ("input_video", "video_url"):          # upstream's alias of input_video too
                report = self.switch(self.request(video(kind=kind)))
                self.assertNotIn("skipped", report)
                self.assertEqual(self.job()["conversation"]["messages"][1]["content"],
                                 f"what is this\n[Video description: a saw 1: video {media_digest(video())[:8]}]")
        (chat,) = self.servers["a"].chats                     # the same bytes: the alias hit the cache
        self.assertEqual(chat["messages"][0]["content"][0],
                         {"type": "input_video", "input_video": {"url": video()["input_video"]["url"]}})
        self.assertEqual(chat["messages"][0]["content"][1]["text"], surrogates.PROMPTS["video"])

    def test_a_video_for_a_vision_target_without_ffmpeg_is_described(self):
        self.servers["a"].modalities = {"vision": True, "video": True}
        self.servers["b"].modalities = {"vision": True}
        with mock.patch.object(surrogates, "ffmpeg_tools", return_value=None):
            report = self.switch(self.request(video()))
        self.assertEqual(report["surrogates"]["parts"][0]["kind"], "video")
        self.assertNotIn("native", report["media"])

    def test_a_video_nothing_can_describe_is_skipped(self):
        self.servers["a"].modalities = {"vision": True}
        with mock.patch.object(surrogates, "ffmpeg_tools", return_value=None):
            report = self.switch(self.request(video()))
        self.assertEqual(report["skipped"], "target-no-video")
        self.servers["b"].modalities = {"video": True}            # a video-capable target reads it
        report = self.switch(self.request(video(), at="newest"))
        self.assertEqual(report["media"]["native"], ["video"])

    @unittest.skipUnless(surrogates.ffmpeg_tools(), "needs ffmpeg and ffprobe")
    def test_a_video_for_a_vision_target_becomes_keyframes_and_a_transcript(self):
        data = ffmpeg_video(Path(self.tmp.name) / "clip.mp4", sound=True)
        self.servers["b"].modalities = {"vision": True}
        self.servers["a"].modalities = {"audio": True}          # transcribes the sound track
        raw = self.request(video(data))
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        (part,) = report["surrogates"]["parts"]
        self.assertEqual((part["kind"], part["producer"], part["frames"]), (surrogates.FRAMES, "ffmpeg", 2))
        # the keyframes are images the target reads itself: T0 stops at that message
        self.assertEqual(report["media"]["native"], ["vision"])
        self.assertEqual(self.job()["target"]["max_t0_message"], 1)
        marker = self.servers["b"].marker
        self.assertEqual(self.job()["tail"], 5 + 2 * len(marker))
        (chat,) = self.servers["a"].chats
        self.assertEqual(chat["messages"][0]["content"][0]["type"], "input_audio")
        content = self.job()["conversation"]["messages"][1]["content"]
        self.assertTrue(content.startswith(f"what is this\n[Video keyframes at 0.75 s, 2.25 s]{marker}{marker}"
                                           "[Video audio transcript: a saw 1: audio "), content)
        rewritten, _ = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        parts = rewritten["messages"][1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "text", "image_url", "image_url", "text"])
        self.assertTrue(parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/"))
        b = LlamaServer(self.servers["b"].url)
        self.assertEqual(b.apply_template(rewritten["messages"]),
                         b.apply_template(self.job()["conversation"]["messages"]))
        self.assert_no_bridge_files()

    @unittest.skipUnless(surrogates.ffmpeg_tools(), "needs ffmpeg and ffprobe")
    def test_a_vision_model_describes_a_videos_keyframes(self):
        data = ffmpeg_video(Path(self.tmp.name) / "clip.mp4", sound=False)
        self.servers["a"].modalities = {"vision": True}
        report = self.switch(self.request(video(data)))
        self.assertNotIn("skipped", report)
        (chat,) = self.servers["a"].chats
        content = chat["messages"][0]["content"]
        self.assertEqual([p["type"] for p in content], ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual(content[0]["text"], "[0.75 s]")
        self.assertEqual(content[-1]["text"], surrogates.PROMPTS["keyframes"])
        self.assertTrue(self.job()["conversation"]["messages"][1]["content"].startswith(
            "what is this\n[Video description: a saw 2: vision "))
        self.assertNotIn("Video audio transcript", self.job()["conversation"]["messages"][1]["content"])

    @unittest.skipUnless(surrogates.ffmpeg_tools(), "needs ffmpeg and ffprobe")
    def test_a_videos_sound_nothing_transcribes_is_noted_and_completed_later(self):
        from kvpark.hermes import _media_summary
        data = ffmpeg_video(Path(self.tmp.name) / "clip.mp4", sound=True)
        self.servers["b"].modalities = {"vision": True}
        raw = self.request(video(data))
        report = self.switch(raw)
        self.assertNotIn("skipped", report)
        (part,) = report["surrogates"]["parts"]
        self.assertEqual((part["kind"], part["sound"]), (surrogates.FRAMES, "not transcribed"))
        self.assertIn("sound track could not be transcribed", report["surrogates"]["partial"]["video"])
        self.assertIn("the sound of 1 not transcribed", _media_summary(report))
        content = self.job()["conversation"]["messages"][1]["content"]
        self.assertTrue(content.endswith(f"[Video audio transcript: {surrogates.SOUND_UNTRANSCRIBED}]"), content)
        rewritten, summary = surrogates.rewrite_request(self.config, raw, "b", bridge_dir=self.bridge_dir)
        self.assertIsNotNone(rewritten)                       # the keyframes stand; the note says what is missing
        self.assertTrue(summary["parts"][0]["cached"])
        # an audio producer now: the cached keyframes get their transcript, and the cache keeps that
        self.servers["a"].modalities = {"audio": True}
        report = self.switch(raw)
        (part,) = report["surrogates"]["parts"]
        self.assertEqual((part["cached"], part["producer"], part["audio_producer"]), (False, "ffmpeg", "a"))
        self.assertNotIn("sound", part)
        self.assertNotIn("partial", report["surrogates"])
        self.assertIn("[Video audio transcript: a saw 1: audio ", self.job()["conversation"]["messages"][1]["content"])
        (entry,) = (self.bridge_dir / surrogates.CACHE_DIR).glob("*.json")
        self.assertNotIn("sound_missing", json.loads(entry.read_text()))
        self.switch(raw)
        self.assertEqual(len(self.servers["a"].chats), 1)       # from the cache since
        self.assert_no_bridge_files()

    @unittest.skipUnless(surrogates.ffmpeg_tools(), "needs ffmpeg and ffprobe")
    def test_a_video_for_an_audio_only_target_is_described_with_its_transcript(self):
        # the target hears audio but has no vision: the video becomes text (a vision producer reads
        # its keyframes) and its sound track a transcript (the sound is not passed on as audio)
        data = ffmpeg_video(Path(self.tmp.name) / "clip.mp4", sound=True)
        self.servers["a"].modalities = {"vision": True, "audio": True}
        self.servers["b"].modalities = {"audio": True}
        report = self.switch(self.request(video(data)))
        self.assertNotIn("skipped", report)
        (part,) = report["surrogates"]["parts"]
        self.assertEqual((part["kind"], part["producer"], part["audio_producer"]), ("video", "a", "a"))
        content = self.job()["conversation"]["messages"][1]["content"]
        self.assertRegex(content, r"^what is this\n\[Video description: a saw 2: vision [0-9a-f]{8}, vision [0-9a-f]{8}\]"
                                  r"\n\[Video audio transcript: a saw 1: audio [0-9a-f]{8}\]$")
        self.assertNotIn("native", report["media"])
        self.assertEqual([chat["messages"][0]["content"][-1]["text"] for chat in self.servers["a"].chats],
                         [surrogates.PROMPTS["keyframes"], surrogates.PROMPTS["audio"]])

    def test_the_cli_writes_the_request_the_target_must_receive(self):
        self.servers["a"].modalities = {"vision": True}
        import argparse
        parser = argparse.ArgumentParser()
        bridge_cli.add_parser(parser.add_subparsers(dest="command"))
        conversation, out = Path(self.tmp.name) / "conversation.json", Path(self.tmp.name) / "request.json"
        conversation.write_text(json.dumps(self.request(image())))
        with mock.patch("builtins.print") as printed:
            bridge_cli.run(parser.parse_args(["bridge", "switch", "--from", "a", "--to", "b", "--conversation",
                                              str(conversation), "--request-out", str(out), "--config",
                                              str(self.config_path)]))
        report = json.loads(printed.call_args[0][0])
        self.assertEqual(report["request_out"], {"path": str(out), "rewritten": True})
        if os.name == "posix":
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        sent = json.loads(out.read_text())
        self.assertEqual(sent["messages"][1]["content"], self.job()["conversation"]["messages"][1]["content"])
        self.assertNotIn(b64(PNG), out.read_text())

    def test_keyframes_for_a_deepseek_v4_target_are_media_it_cannot_take_after_a_pool(self):
        self.assertEqual(capabilities.pooled_media("v4", "deepseek4", {"vision"})[0], "dsv4-media")
        self.assertIsNone(capabilities.pooled_media("v4", "deepseek4", set()))
        self.assertIsNone(capabilities.pooled_media("q", "qwen35", {"vision"}))
        # media replaced by text surrogates leave nothing for the pooled target to read
        self.assertIsNone(capabilities.media({"vision": 1}, source="a", target="v4", source_modalities={},
                                             target_modalities={}, source_saw=2, target_arch="deepseek4",
                                             replaced={"vision"}))


class SurrogateUnitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_digests_name_the_bytes_not_the_encoding(self):
        raw = {"type": "image_url", "image_url": {"url": b64(PNG)}}           # raw base64, as llama-server takes
        self.assertEqual(media_digest(raw), media_digest(image()))
        self.assertEqual(media_digest(raw), hashlib.sha256(PNG).hexdigest())
        remote = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        self.assertEqual(media_digest(remote), hashlib.sha256(b"url:https://example.com/a.png").hexdigest())
        self.assertEqual(media_digest({"type": "input_audio", "input_audio": {"url": "data:audio/wav;base64," + b64(WAV)}}),
                         hashlib.sha256(WAV).hexdigest())
        conversation = conversation_from_openai([{"role": "user", "content": [video(kind="video_url")]}])
        self.assertEqual(media(conversation), {"video": 0})
        self.assertNotIn(b64(MP4), json.dumps(flatten(conversation, marker="<M>")["messages"]))

    def test_malformed_and_unusual_media_parts(self):
        # a data URL without its comma: no bytes, a clean conversion (not an IndexError)
        bad = {"type": "image_url", "image_url": {"url": "data:image/png;base64"}}
        self.assertIsNone(media_bytes("data:image/png;base64"))
        self.assertEqual(media(conversation_from_openai([{"role": "user", "content": [bad]}])), {"vision": 0})
        # parts that name no media in a form llama-server reads never share a digest
        one = {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}
        two = {"type": "image_url", "image_url": "data:image/png;base64,BBBB"}
        self.assertNotEqual(media_digest(one), media_digest(two))
        self.assertNotEqual(media_digest({"type": "input_audio", "input_audio": {}}),
                            media_digest({"type": "input_audio", "input_audio": {"format": "mp3"}}))
        # base64 with line breaks (MIME style) names the same bytes
        wrapped = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.encodebytes(PNG * 9).decode()}}
        self.assertEqual(media_digest(wrapped), hashlib.sha256(PNG * 9).hexdigest())

    def test_the_cache_keeps_the_first_surrogate_and_trims_the_oldest(self):
        cache = surrogates.Cache(self.root / "s", max_bytes=10_000)
        self.assertIsNone(cache.get("d1", "vision"))
        first = cache.put("d1", "vision", "qwen38", {"text": "[Image description: one]"})
        cache.put("d1", "vision", "http://127.0.0.1:9", {"text": "[Image description: two]"})
        self.assertEqual(cache.get("d1", "vision")["text"], first["text"])     # the oldest, by any producer
        self.assertIsNone(cache.get("d1", "audio"))
        self.assertIsNone(cache.get("d1", surrogates.FRAMES))
        self.assertEqual(cache.stats()["entries"], 2)
        for number in range(20):
            cache.put(f"x{number}", "vision", "p", {"text": "y" * 800})
        stats = cache.stats()
        self.assertLessEqual(stats["bytes"], 10_000)
        self.assertIsNotNone(cache.get("x19", "vision"))                     # the newest stays
        self.assertIsNone(cache.get("d1", "vision"))
        self.assertEqual(cache.clear(), stats["entries"])
        self.assertEqual(cache.stats()["entries"], 0)
        with mock.patch.object(surrogates, "PROMPT_VERSION", surrogates.PROMPT_VERSION + 1):
            cache.put("d2", "vision", "p", {"text": "new prompt"})
        self.assertIsNone(cache.get("d2", "vision"))                         # another prompt version
        # a video's surrogate with its sound track comes before an older one without
        cache.put("v", "video", "p1", {"text": "older", "sound_missing": True})
        cache.put("v", "video", "p2", {"text": "complete"})
        self.assertEqual(cache.get("v", "video")["text"], "complete")

    def test_media_settings(self):
        gguf = write_gguf(self.root / "m.gguf", "qwen3")
        config = registry.empty()
        for name in ("qwen38", "muse"):
            registry.add_model(config, name, url="http://127.0.0.1:9", gguf=gguf, slot_dir=self.root)
        self.assertEqual(surrogates.policy(config), {"image": "caption", "audio": "transcribe", "video": "describe"})
        registry.set_media(config, policy={"audio": "skip"})
        registry.set_media(config, policy={"image": "skip", "video": None}, target="muse")
        self.assertEqual(surrogates.policy(config, "muse"), {"image": "skip", "audio": "skip", "video": "describe"})
        self.assertFalse(surrogates.converts(config, "muse", "vision"))
        self.assertTrue(surrogates.converts(config, "qwen38", "vision"))
        registry.set_media(config, describers={"vision": "qwen38", "audio": "http://localhost:8099"}, cache_mib=8)
        self.assertEqual(config["media"]["describers"], {"vision": "qwen38", "audio": "http://localhost:8099"})
        self.assertEqual(surrogates.cache_for(config, self.root).max_bytes, 8 << 20)
        self.assertEqual([p.label for p in surrogates.describers(config)], ["qwen38", "http://localhost:8099"])
        self.assertEqual(surrogates.describers(config, exclude={"qwen38"})[0].allowed, {"audio"})
        registry.set_media(config, describers={"audio": "http://localhost:13305/v1/audio/transcriptions/"},
                           transcription_model="whisper-v3-turbo-FLM")
        self.assertEqual(config["media"]["describers"]["audio"], "http://localhost:13305/v1/audio/transcriptions")
        (transcriber,) = surrogates.describers(config, exclude={"qwen38"})
        self.assertIsInstance(transcriber, surrogates.Transcriber)
        self.assertEqual(transcriber.model, "whisper-v3-turbo-FLM")
        registry.set_media(config, transcription_model="")
        self.assertEqual(surrogates.describers(config, exclude={"qwen38"})[0].model, surrogates.TRANSCRIPTION_MODEL)
        for bad in ({"policy": {"image": "describe"}}, {"policy": {"pictures": "skip"}},
                    {"describers": {"vision": "https://api.example.com"}}, {"describers": {"smell": "qwen38"}},
                    {"describers": {"vision": "http://127.0.0.1:13305/v1/audio/transcriptions"}},
                    {"describers": {"audio": "https://api.openai.com/v1/audio/transcriptions"}},
                    {"describers": {"audio": "http://127.0.0.1:13305/v1/audio/transcriptions?key=1"}},
                    {"cache_mib": 0}, {"policy": {"image": "skip"}, "target": "nobody"}):
            with self.assertRaises(ValueError):
                registry.set_media(config, **bad)
        registry.set_media(config, policy={"image": "default"}, target="muse", describers={"audio": None})
        self.assertNotIn("transcription_model", config["media"])
        self.assertNotIn("models", config["media"])
        self.assertEqual(config["media"]["describers"], {"vision": "qwen38"})
        registry.forget_model_media(config, "qwen38")
        self.assertNotIn("describers", config["media"])

    def test_cli_media_command(self):
        gguf = write_gguf(self.root / "m.gguf", "qwen3")
        path = self.root / "bridge" / "bridge.json"
        config = registry.empty()
        registry.add_model(config, "muse", url="http://127.0.0.1:9", gguf=gguf, slot_dir=self.root)
        registry.save(config, path)
        import argparse
        parser = argparse.ArgumentParser()
        bridge_cli.add_parser(parser.add_subparsers(dest="command"))

        def run(*argv):
            with mock.patch("builtins.print") as printed:
                bridge_cli.run(parser.parse_args(["bridge", *argv, "--config", str(path)]))
            return json.loads(printed.call_args[0][0])
        out = run("media", "--image", "skip", "--model", "muse")
        self.assertEqual(out["models"]["muse"]["image"], "skip")
        self.assertEqual(out["policy"]["image"], "caption")
        out = run("media", "--describer", "vision=muse", "--video", "skip")
        self.assertEqual((out["describers"], out["policy"]["video"]), ({"vision": "muse"}, "skip"))
        self.assertNotIn("transcription_model", out)
        out = run("media", "--describer", "audio=http://127.0.0.1:13305/v1/audio/transcriptions",
                  "--transcription-model", "whisper-v3-turbo-FLM")
        self.assertEqual(out["transcription_model"], "whisper-v3-turbo-FLM")
        out = run("media", "--describer", "audio=", "--transcription-model", "")
        self.assertEqual((out["describers"], "transcription_model" in out), ({"vision": "muse"}, False))
        surrogates.Cache(path.parent / surrogates.CACHE_DIR).put("d", "vision", "muse", {"text": "t"})
        self.assertEqual(run("media")["cache"]["entries"], 1)
        self.assertEqual(run("media", "--clear-cache")["cache"]["cleared"], 1)
        self.assertEqual(run("status")["media"]["describers"], {"vision": "muse"})
        with mock.patch("builtins.print"):
            bridge_cli.run(parser.parse_args(["bridge", "remove", "muse", "--config", str(path)]))
        self.assertEqual(registry.load(path)["media"], {"policy": {"video": "skip"}})

    def test_switch_status_summarizes_the_media(self):
        from kvpark.hermes import _media_summary
        made = {"message": 3, "modality": "vision", "producer": "qwen38", "cached": False, "seconds": 22.439}
        cached = {**made, "message": 5, "cached": True, "seconds": 0.0}
        frames = {"message": 4, "modality": "video", "producer": "ffmpeg", "cached": False, "seconds": 0.8,
                  "frames": 3}
        self.assertEqual(_media_summary({"surrogates": {"parts": [made, cached, frames]}}),
                         "media: 2 images as text (made by qwen38 in 22.4 s; 1 from the surrogate cache or the "
                         "conversation), 1 video as keyframes (made by ffmpeg in 0.8 s)")
        repeated = {**made, "message": 7, "duplicate": True, "seconds": 0.0}
        sound = {**frames, "audio_producer": "http://127.0.0.1:13305/v1/audio/transcriptions", "seconds": 2.0}
        self.assertEqual(_media_summary({"surrogates": {"parts": [made, repeated, sound]}}),
                         "media: 2 images as text (made by qwen38 in 22.4 s; 1 repeated), 1 video as keyframes "
                         "(made by ffmpeg, http://127.0.0.1:13305/v1/audio/transcriptions in 2.0 s)")
        self.assertIsNone(_media_summary({}))
        # Hermes described the image itself: the request has text, the source's cache the image
        self.assertIn("translated up to them", _media_summary({"engine": {"source_media": {"read": 900}}}))

    def test_ffmpeg_is_optional(self):
        with mock.patch.object(surrogates.shutil, "which", return_value=None):
            self.assertIsNone(surrogates.ffmpeg_tools())
        with mock.patch.object(surrogates.shutil, "which", side_effect=lambda name: None if name == "ffprobe" else name):
            self.assertIsNone(surrogates.ffmpeg_tools())


if __name__ == "__main__":
    unittest.main()
