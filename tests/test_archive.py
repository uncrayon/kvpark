import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client

from kvpark import proxy as p


class ArchiveTests(unittest.TestCase):
    def test_update_saves_resident_state_and_gates_new_requests(self):
        self.assertTrue(self.a.prepare_update()["ok"])
        self.assertIn(self.a.current_key, self.a.entries)
        self.assertTrue(self.a.status()["maintenance"])

    def test_update_save_failure_reopens_inference(self):
        self.resume = False
        with self.assertRaises(OSError):
            self.a.prepare_update()
        self.assertFalse(self.a.status()["maintenance"])
        self.assertFalse(self.a.entries)

    def test_update_wait_timeout_does_not_interrupt_generation(self):
        held, release = threading.Event(), threading.Event()
        def generation():
            with self.a.lock:
                held.set()
                release.wait(5)
        thread = threading.Thread(target=generation)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(1))
        with self.assertRaisesRegex(RuntimeError, "still active"):
            self.a.prepare_update(timeout=.01)
        self.assertFalse(self.a.status()["maintenance"])
        self.assertTrue(thread.is_alive())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.multiple(p, ARCHIVE=Path(self.tmp.name), AUTO_SAVE=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.a = p.Archive()
        self.a.identity = "weights-a"
        self.a.current_key = p.key_for("thread-a")
        self.a.current_thread = "discord:123"
        self.prompt = {"slot_archive_role": "foreground", "reasoning_effort": "xhigh",
                       "messages": [{"role": "system", "content": "stable"}]}
        self.a.current_prompt = p.prompt_signature(self.prompt)
        self.a.runtime = lambda: None
        self.tokens = 12
        self.resume = True
        self.calls = []
        self.a.request = self.upstream

    def upstream(self, method, path, body=None, timeout=600):
        self.calls.append(path)
        if path == "/slots":
            return [dict(id=0, is_processing=False, n_prompt_tokens=self.tokens)]
        if "action=save" in path:
            file = p.ARCHIVE / body["filename"]
            file.write_bytes(b"target state")
            if self.resume:
                Path(str(file) + ".resume").write_bytes(b"checkpoint and draft")
            return dict(n_saved=self.tokens, n_written=12)
        if "action=restore" in path:
            return dict(n_restored=self.tokens)
        return dict(n_erased=self.tokens)

    def test_explicit_park_bypasses_floor_and_survives_proxy_restart(self):
        key = self.a.current_key
        self.assertTrue(self.a.park(thread="discord:123")["parked"])
        reboot = p.Archive()
        reboot.identity = "weights-a"
        reboot.request = self.upstream
        reboot.restore(key)
        self.assertIn("/slots/0?action=restore", self.calls)

    def test_unparked_oneoff_never_saved(self):
        self.tokens = 100000
        self.a.save(self.a.current_key)
        self.assertFalse(self.a.entries)
        self.assertNotIn("/slots/0?action=save", self.calls)

    def test_wrong_thread_and_missing_key_cannot_park_resident(self):
        for kwargs in ({}, {"thread": "other"}, {"key": p.key_for("other")}):
            with self.assertRaises(ValueError):
                self.a.park(**kwargs)
        self.assertFalse(self.a.entries)

    def test_unpatched_backend_cannot_report_success_or_destroy_old_snapshot(self):
        key = self.a.current_key
        self.a.park(key=key)
        previous = self.a.entries[key].copy()
        self.resume = False
        with self.assertRaises(FileNotFoundError):
            self.a.park(key=key)
        self.assertEqual(self.a.entries[key], previous)
        self.assertTrue((p.ARCHIVE / previous["filename"]).exists())
        self.assertEqual(len(list(p.ARCHIVE.glob("*.bin"))), 1)

    def test_fsync_failure_after_publication_keeps_referenced_files(self):
        key = self.a.current_key
        self.a.park(key=key)
        original = p.atomic_json
        def publish_then_fail(path, data):
            original(path, data)
            raise OSError("directory fsync failed")
        with patch.object(p, "atomic_json", publish_then_fail):
            with self.assertRaises(OSError):
                self.a.park(key=key)
        reboot = p.Archive()
        self.assertTrue((p.ARCHIVE / reboot.entries[key]["filename"]).exists())
        self.assertTrue((p.ARCHIVE / (reboot.entries[key]["filename"] + ".resume")).exists())

    def test_wrong_weights_never_reach_restore(self):
        key = self.a.current_key
        self.a.park(key=key)
        self.a.identity = "weights-b"
        self.a.restore(key)
        self.assertNotIn("/slots/0?action=restore", self.calls)
        self.assertEqual(self.a.events[0]["action"], "incompatible")

    def test_expired_archive_not_restored(self):
        key = self.a.current_key
        self.a.park(key=key)
        self.a.entries[key]["saved_at"] -= 8 * 86400
        self.a.restore(key)
        self.assertNotIn("/slots/0?action=restore", self.calls)
        self.assertFalse(self.a.entries)

    def test_oversized_snapshot_preserves_previous_generation(self):
        key = self.a.current_key
        self.a.park(key=key)
        previous = self.a.entries[key].copy()
        with patch.object(p, "MAX_ARCHIVE_GB", 1 / 1024**3):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                self.a.park(key=key)
        self.assertEqual(self.a.entries[key], previous)
        self.assertTrue((p.ARCHIVE / previous["filename"]).exists())

    def test_partial_restore_and_failed_erase_is_unsafe(self):
        key = self.a.current_key
        self.a.park(key=key)
        def fail(*args, **kwargs):
            raise RuntimeError("backend failed")
        self.a.request = fail
        with self.assertRaises(p.UnsafeRestoreError):
            self.a.restore(key)

    def test_indefinite_retention(self):
        key = self.a.current_key
        self.a.park(key=key)
        self.a.entries[key]["last_used"] -= 100 * 86400
        with patch.object(p, "TTL_DAYS", 0):
            self.a.restore(key)
        self.assertIn("/slots/0?action=restore", self.calls)

    def test_erase_failure_is_not_gap_success(self):
        upstream = self.a.request
        def fail(method, path, body=None, timeout=600):
            if "erase" in path:
                raise RuntimeError("erase failed")
            return upstream(method, path, body, timeout)
        self.a.request = fail
        key = self.a.current_key
        with self.assertRaises(RuntimeError):
            self.a.park(key=key, simulate=True)
        self.assertEqual(self.a.current_key, key)
        self.assertNotIn("gap-simulated", [e["action"] for e in self.a.events])

    def test_third_day_restore_reuses_freshness_and_gc_uses_explicit_clock(self):
        key = self.a.current_key
        self.a.park(key=key)
        self.a.entries[key]["last_used"] -= 3 * 86400
        self.a.restore(key)
        self.assertLess(time.time() - self.a.entries[key]["last_used"], 2)
        self.a.gc()
        self.assertIn(key, self.a.entries)

    def test_restart_same_thread_must_restore(self):
        key = self.a.current_key
        self.a.park(key=key)
        def restarted():
            self.a.current_key = None
        self.a.runtime = restarted
        self.tokens = 0
        self.a.switch(key)
        self.assertIn("/slots/0?action=restore", self.calls)
        self.assertEqual(self.calls.count("/slots/0?action=save"), 1)

    def test_background_with_same_session_id_never_claims_foreground(self):
        for role in ("background", "unverified", None):
            request = dict(self.prompt, slot_archive_role=role, slot_archive_key="thread-a")
            self.assertIsNone(p.conversation_key(request))
        self.assertEqual(p.conversation_key(dict(self.prompt, slot_archive_key="thread-a")), self.a.current_key)

    def test_selected_foreground_is_saved_before_background_even_below_growth_threshold(self):
        key = self.a.current_key
        self.a.park(key=key)
        previous = self.a.entries[key]["filename"]
        self.tokens += 10
        self.a.switch(None, role="background")
        self.assertNotEqual(previous, self.a.entries[key]["filename"])
        self.assertIsNone(self.a.current_key)
        self.assertEqual(self.a.park(key=key)["tokens"], self.tokens)
        self.assertEqual(self.calls.count("/slots/0?action=save"), 2)
        with self.assertRaises(ValueError):
            self.a.park(key=key, simulate=True)
        self.assertNotIn("/slots/0?action=erase", self.calls)

    def test_unselected_background_cannot_be_parked_as_foreground(self):
        key = self.a.current_key
        self.a.switch(None, role="background")
        with self.assertRaises(ValueError):
            self.a.park(key=key)
        self.assertFalse(self.a.entries)

    def test_prompt_setting_change_skips_expensive_incompatible_load(self):
        key = self.a.current_key
        self.a.park(key=key)
        incoming = p.prompt_signature(dict(self.prompt, reasoning_effort="medium"))
        self.a.restore(key, incoming)
        self.assertNotIn("/slots/0?action=restore", self.calls)
        event = self.a.events[0]
        self.assertEqual(event["action"], "restore-skipped")
        self.assertEqual((event["saved_effort"], event["incoming_effort"]), ("xhigh", "medium"))
        self.a.restore(key, self.a.current_prompt)
        self.assertIn("/slots/0?action=restore", self.calls)

    def test_legacy_archive_does_not_load_unverified_background_state(self):
        key = self.a.current_key
        self.a.park(key=key)
        self.a.entries[key].pop("prompt")
        self.a.restore(key, self.a.current_prompt)
        self.assertNotIn("/slots/0?action=restore", self.calls)
        self.assertIn("foreground identity", self.a.events[0]["reason"])

    def test_status_remains_available_while_generation_owns_lock(self):
        self.a.set_activity("generating", self.a.current_key)
        held, release = threading.Event(), threading.Event()
        def generation():
            with self.a.lock:
                held.set()
                release.wait(5)
        thread = threading.Thread(target=generation)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(1))
        # This must not touch runtime, query /slots, or acquire the inference lock.
        self.a.runtime = lambda: self.fail("status mutated runtime ownership")
        started = time.monotonic()
        status = self.a.status()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(status["activity"]["phase"], "generating")
        status["activity"]["phase"] = "corrupted"
        self.assertEqual(self.a.status()["activity"]["phase"], "generating")

    def test_keys_are_not_truncated_or_derived_from_user_accounts(self):
        self.assertNotEqual(p.key_for("a/b"), p.key_for("a_b"))
        self.assertNotEqual(p.key_for("x" * 200 + "a"), p.key_for("x" * 200 + "b"))
        self.assertIsNone(p.conversation_key({"user": "same-user", "messages": []}))


class RelayTests(unittest.TestCase):
    def test_maintenance_rejects_generation_before_contacting_backend(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p, "ARCHIVE", Path(tmp)):
            state = p.Archive()
        state.maintenance_until = time.monotonic() + 60
        server = ThreadingHTTPServer(("127.0.0.1", 0), p.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with patch.object(p, "ARCHIVE_STATE", state, create=True), patch.object(p.Handler, "forward") as forward:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            self.addCleanup(conn.close)
            conn.request("POST", "/v1/chat/completions", json.dumps({"messages": []}))
            response = conn.getresponse()
            self.assertEqual(response.status, 503)
            response.read()
            forward.assert_not_called()

    def test_streaming_and_concurrent_requests_keep_slot_owned_until_eof(self):
        first_sent, release = threading.Event(), threading.Event()
        backend_order = []
        class Upstream(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                backend_order.append(body["test_id"])
                assert not any(k.startswith("slot_archive_") for k in body)
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = b'data: {"token":"first"}\n\n'
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                if body["test_id"] == "a":
                    first_sent.set()
                    release.wait(5)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), p.Handler)
        for server in (upstream, proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(p, "ARCHIVE", Path(tmp.name)):
            state = p.Archive()
        state.switch = lambda *args: None
        with patch.multiple(p, UPSTREAM_PORT=upstream.server_port), patch.object(p, "ARCHIVE_STATE", state, create=True):
            def connect(test_id):
                c = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=3)
                c.request("POST", "/v1/chat/completions", json.dumps({"test_id": test_id, "slot_archive_key": test_id, "slot_archive_role": "foreground"}))
                return c, c.getresponse()
            c, resp = connect("a")
            self.addCleanup(c.close)
            self.assertTrue(first_sent.wait(1))
            # Must receive a small SSE event while the upstream is still active.
            self.assertIn(b"first", resp.read1(8192))
            status_conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=1)
            self.addCleanup(status_conn.close)
            status_conn.request("GET", "/_kvpark/status")
            status_resp = status_conn.getresponse()
            self.assertEqual(status_resp.status, 200)
            self.assertEqual(json.loads(status_resp.read())["activity"]["phase"], "generating")
            second_done = threading.Event()
            errors = []
            def second():
                try:
                    c2, r2 = connect("b")
                    r2.read()
                    c2.close()
                except Exception as exc:
                    errors.append(exc)
                second_done.set()
            thread = threading.Thread(target=second)
            thread.start()
            time.sleep(0.1)
            self.assertEqual(backend_order, ["a"])
            release.set()
            resp.read()
            self.assertTrue(second_done.wait(3))
            thread.join()
            self.assertFalse(errors)
            self.assertEqual(backend_order, ["a", "b"])
            # Wait for the proxy's post-completion bookkeeping under its lock.
            with state.lock:
                self.assertEqual(state.current_key, p.key_for("b"))


class MetricsTests(unittest.TestCase):
    def test_fragmented_stream_records_real_completion_counters_and_first_token(self):
        m = p.ResponseMetrics(True, time.monotonic())
        m.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.assertNotIn("ttft_seconds", m.metrics)
        m.feed(b'data: {"choices":[{"delta":{"content":"Hel')
        m.feed(b'lo"}}]}\n\ndata: {"timings":{"cache_n":1000,"prompt_n":24}}\n\n')
        m.feed(b'data: [DONE]\n\n')
        result = m.finish()
        self.assertEqual(result["cached_tokens"], 1000)
        self.assertEqual(result["processed_tokens"], 24)
        self.assertIn("ttft_seconds", result)

    def test_nonstream_uses_response_not_reset_slot_counters(self):
        m = p.ResponseMetrics(False, time.monotonic())
        m.feed(b'{"usage":{"prompt_tokens":1024,"prompt_tokens_details":{"cached_tokens":1000}}}')
        self.assertEqual(m.finish(), {"cached_tokens":1000,"processed_tokens":24})


class RuntimeTests(unittest.TestCase):
    def test_manifest_detects_replaced_model_files_and_pid_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"weights")
            st = model.stat()
            from kvpark.platforms import process_identity
            data = dict(**process_identity(os.getpid()), run_id="new", identity="id",
                        files={str(model): [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]})
            manifest = Path(tmp) / "runtime.json"
            manifest.write_text(json.dumps(data))
            with patch.multiple(p, ARCHIVE=Path(tmp), RUNTIME=manifest):
                a = p.Archive()
                a.current_key = "old"
                a.runtime()
                self.assertIsNone(a.current_key)
                model.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "files changed"):
                    a.runtime()
                data["created_at"] = 0
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    a.runtime()


if __name__ == "__main__": unittest.main()
