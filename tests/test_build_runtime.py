"""Runtime build script: pinned revision, shipped patches, and patch order."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kvpark_build_runtime", ROOT / "scripts/build_runtime.py")
build_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_runtime)


class BuildRuntimeTests(unittest.TestCase):
    def test_pin_is_a_full_commit_and_every_patch_ships(self):
        self.assertRegex(build_runtime.PIN, r"^[0-9a-f]{40}$")
        self.assertIsInstance(build_runtime.PIN_BUILD, int)
        self.assertGreater(build_runtime.PIN_BUILD, 0)
        self.assertTrue(build_runtime.PATCHES)
        self.assertEqual(len(set(build_runtime.PATCHES)), len(build_runtime.PATCHES))
        manifest = (ROOT / "MANIFEST.in").read_text()
        self.assertIn("recursive-include patches *.patch", manifest)
        for name in build_runtime.PATCHES:
            self.assertTrue((ROOT / name).is_file(), name)
            self.assertTrue(name.startswith("patches/") and name.endswith(".patch"), name)
        # upstream-relative paths, as git apply expects them inside the exported source;
        # the production set (patches/production/) is applied by hand the same way
        for path in sorted((ROOT / "patches").rglob("*.patch")):
            self.assertTrue(re.search(r"^\+\+\+ b/", path.read_text(), re.MULTILINE), path.name)

    def test_patches_apply_in_order_after_export_and_before_configure(self):
        calls = []
        envs = []
        with tempfile.TemporaryDirectory(prefix="kvpark build runtime ") as tmp:
            root = Path(tmp)
            output = root / "out"

            def run(command, **kwargs):
                calls.append((list(map(str, command)), kwargs.get("cwd")))
                if command[:2] == ["git", "apply"]:
                    envs.append(kwargs["env"])
                if command[:2] == ["cmake", "--build"]:
                    binary = output / "build/bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
                    binary.parent.mkdir(parents=True)
                    binary.write_bytes(b"server")

            archive = MagicMock()
            archive.__enter__.return_value = archive
            argv = ["build_runtime.py", "--source", str(root / "checkout"), "--output", str(output), "--jobs", "3"]
            with patch.object(build_runtime.subprocess, "run", side_effect=run), \
                    patch.object(build_runtime.tarfile, "open", return_value=archive), \
                    patch("sys.argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                build_runtime.main()
            output = output.resolve()
            source = output / "source"
        commands = [command for command, _ in calls]
        self.assertEqual(commands[0][:4], ["git", "-C", str((root / "checkout").resolve()), "archive"])
        self.assertEqual(commands[0][4], build_runtime.PIN)
        applied = []
        for command, cwd in calls[1:1 + 2 * len(build_runtime.PATCHES)]:
            self.assertEqual(cwd, source)
            applied.append(command[2:])
        expected = []
        for name in build_runtime.PATCHES:
            expected += [["--check", str(ROOT / name)], [str(ROOT / name)]]
        self.assertEqual(applied, expected)
        self.assertEqual({env["GIT_CEILING_DIRECTORIES"] for env in envs}, {str(output)})
        configure, build = commands[1 + 2 * len(build_runtime.PATCHES):]
        self.assertEqual(configure[:5], ["cmake", "-S", str(source), "-B", str(output / "build")])
        self.assertIn(f"-DLLAMA_BUILD_NUMBER={build_runtime.PIN_BUILD}", configure)
        self.assertIn(f"-DLLAMA_BUILD_COMMIT={build_runtime.PIN[:9]}", configure)
        self.assertEqual(build[:2], ["cmake", "--build"])
        self.assertIn("llama-server", build)

    @unittest.skipUnless(shutil.which("git"), "Git")
    def test_git_style_patch_applies_inside_an_enclosing_repository(self):
        # runtime/ sits inside the kvpark checkout; git apply must not resolve against it.
        with tempfile.TemporaryDirectory(prefix="kvpark nested apply ") as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            output = root / "runtime"
            target = output / "source/tools/server/server-context.cpp"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"int a = 1;\n")
            patch_file = root / "change.patch"
            patch_file.write_bytes(b"diff --git a/tools/server/server-context.cpp b/tools/server/server-context.cpp\n"
                                   b"--- a/tools/server/server-context.cpp\n"
                                   b"+++ b/tools/server/server-context.cpp\n"
                                   b"@@ -1 +1 @@\n"
                                   b"-int a = 1;\n"
                                   b"+int a = 2;\n")
            build_runtime.apply_patches(output / "source", [patch_file], output)
            self.assertEqual(target.read_bytes(), b"int a = 2;\n")


if __name__ == "__main__":
    unittest.main()
