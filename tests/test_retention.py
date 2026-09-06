import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sloth_memory import proxy


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.multiple(proxy, ARCHIVE=self.root, TTL_DAYS=7, MAX_ARCHIVE_GB=32)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.archive = proxy.Archive()

    def entry(self, session, days, *, bytes=8):
        key = proxy.key_for(session)
        filename = key + "." + "a" * 32 + ".bin"
        for suffix in ("", ".resume"):
            (self.root / (filename + suffix)).write_bytes(b"state")
        meta = dict(format=2, filename=filename, bytes=bytes, tokens=10, identity="test",
                    saved_at=time.time()-days*86400, last_used=time.time(),
                    prompt={"role": "foreground"})
        self.archive.entries[key] = meta
        proxy.atomic_json(self.root / (key + ".meta"), meta)
        self.archive.publish_status()
        return key

    def test_seven_day_default_deletes_old_even_if_recently_used(self):
        old = self.entry("old", 8)
        fresh = self.entry("fresh", 6)
        result = self.archive.gc()
        self.assertEqual(result["deleted"], 1)
        self.assertNotIn(old, self.archive.entries)
        self.assertIn(fresh, self.archive.entries)
        self.assertFalse((self.root / (old + ".meta")).exists())
        self.assertEqual(len(list(self.root.glob("*.bin*"))), 2)

    def test_disabled_cleanup_is_persistent_and_manual_cleanup_still_works(self):
        old = self.entry("old", 8)
        self.archive.update_retention({"cleanup_enabled": False})
        reboot = proxy.Archive()
        self.assertFalse(reboot.status()["cleanup_enabled"])
        reboot.gc()
        self.assertIn(old, reboot.entries)
        self.assertFalse(reboot.retention.expired(reboot.entries[old], time.time()))
        reboot.gc(manual=True)
        self.assertNotIn(old, reboot.entries)

    def test_restoring_does_not_extend_snapshot_age(self):
        key = self.entry("old", 6)
        self.archive.identity = "test"
        self.archive.request = lambda *a, **kw: {"n_restored": 10}
        saved_at = self.archive.entries[key]["saved_at"]
        self.archive.restore(key)
        self.assertEqual(self.archive.entries[key]["saved_at"], saved_at)
        with patch("time.time", return_value=saved_at + 7*86400):
            self.archive.gc()
        self.assertNotIn(key, self.archive.entries)

    def test_crash_orphans_expire_but_unrelated_files_and_symlinks_are_kept(self):
        name = proxy.key_for("orphan") + "." + "b" * 32 + ".bin"
        for suffix in ("", ".resume"):
            path = self.root / (name + suffix)
            path.write_bytes(b"orphan")
            os.utime(path, (time.time()-8*86400,)*2)
        keep = self.root / "notes.bin"
        keep.write_text("not a sloth snapshot")
        os.utime(keep, (time.time()-20*86400,)*2)
        link = self.root / (proxy.key_for("link") + "." + "c" * 32 + ".bin")
        link.symlink_to(keep)
        self.assertEqual(self.archive.gc()["orphan_files"], 2)
        self.assertTrue(keep.exists())
        self.assertTrue(link.is_symlink())

    def test_budget_remains_enforced_when_scheduled_cleanup_is_disabled(self):
        self.entry("large", 0, bytes=1024**3)
        self.archive.update_retention({"cleanup_enabled": False, "max_gib": .5})
        self.assertFalse(self.archive.entries)

    def test_invalid_settings_never_replace_saved_policy(self):
        self.archive.update_retention({"ttl_days": 14})
        for bad in ({"ttl_days": -1}, {"ttl_days": float("nan")}, {"max_gib": 0},
                    {"cleanup_enabled": "false"}, {"archive_dir": "/elsewhere"}, []):
            with self.assertRaises(ValueError):
                self.archive.update_retention(bad)
        self.assertEqual(json.loads((self.root / "retention.json").read_text()), {"ttl_days": 14})


if __name__ == "__main__":
    unittest.main()
