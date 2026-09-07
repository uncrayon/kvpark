"""Installer selection, package verification, and shell integration boundaries."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("kvpark_installer", Path(__file__).resolve().parents[1] / "scripts/install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kvpark install test ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {"HOME": str(self.root), "USERPROFILE": str(self.root), "SHELL": "/bin/bash", "PATH": ""}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.info = dict(prefix=str(self.root / "hermes/venv"), base=str(self.root / "system"),
                         version=[3, 12], hermes=True, kvpark=None, legacy=None)

    def test_detects_hermes_while_ignoring_unrelated_active_environment(self):
        python = self.root / ".hermes/hermes-agent/venv/bin/python"
        with patch.dict(os.environ, VIRTUAL_ENV=str(self.root / "unrelated")), patch.object(installer, "probe", side_effect=lambda p: self.info if p == python else None):
            self.assertEqual(installer.find_hermes()[0], python)

    def test_custom_hermes_python_keeps_venv_path(self):
        python = self.root / "custom environment/bin/python"
        with patch.object(installer, "probe", return_value=self.info):
            self.assertEqual(installer.find_hermes(python)[0], python)

    def test_ambiguous_hermes_installations_require_explicit_selection(self):
        with patch.dict(os.environ, HERMES_INSTALL_DIR=str(self.root / "custom")), patch.object(installer, "probe", side_effect=lambda p: {**self.info, "prefix": str(p.parent.parent)} if 'venv' == p.parent.parent.name else None):
            with self.assertRaisesRegex(RuntimeError, "Multiple Hermes"):
                installer.find_hermes()

    def test_unidentified_launcher_never_silently_installs_standalone(self):
        with patch.object(installer.shutil, "which", return_value=str(self.root / "wrapper")), patch.object(installer, "probe", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "could not identify"):
                installer.find_hermes()

    def test_package_checksum_verified_before_install(self):
        body = b"wheel fixture"
        name = "kvpark-0.3.0a1-py3-none-any.whl"
        asset = dict(name=name, browser_download_url=f"{installer.REPOSITORY}/releases/download/v0.3.0a1/{name}",
                     digest="sha256:" + hashlib.sha256(body).hexdigest())
        for content, accepted in ((body, True), (b"damaged", False)):
            with self.subTest(accepted=accepted), patch.object(installer, "fetch", side_effect=[json.dumps({"assets": [asset]}).encode(), content]):
                if accepted:
                    self.assertEqual(installer.release_asset("0.3.0a1", name, self.root).read_bytes(), body)
                else:
                    with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                        installer.release_asset("0.3.0a1", name, self.root)

    def test_release_without_digest_is_rejected(self):
        with patch.object(installer, "fetch", return_value=b'{"assets": []}'):
            with self.assertRaisesRegex(RuntimeError, "verifiable checksum"):
                installer.release_asset("0.3.0a1", "package.whl", self.root)

    def test_launcher_conflict_is_preserved(self):
        launcher = self.root / "kvpark"
        launcher.write_text("another installation")
        with self.assertRaisesRegex(RuntimeError, "another installation"):
            installer.expose_command(Path(sys.executable), self.root, False)
        self.assertEqual(launcher.read_text(), "another installation")

    def test_path_setup_is_idempotent_and_preserves_existing_shell_config(self):
        profile = self.root / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
        profile.write_text("# user's config\n")
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                installer.expose_command(Path(sys.executable), self.root / "commands with spaces", True)
        content = profile.read_text()
        self.assertTrue(content.startswith("# user's config\n"))
        self.assertEqual(content.count("# kvpark"), 1)
        self.assertEqual(content.count("export PATH="), 1)

    def test_macos_bash_uses_login_profile(self):
        with patch.object(installer.sys, "platform", "darwin"), contextlib.redirect_stdout(io.StringIO()):
            installer.expose_command(Path(sys.executable), self.root / "bin", True)
        self.assertTrue((self.root / ".bash_profile").exists())

    def test_zsh_respects_custom_config_directory(self):
        with patch.dict(os.environ, SHELL="/bin/zsh", ZDOTDIR=str(self.root / "zsh config")), contextlib.redirect_stdout(io.StringIO()):
            installer.expose_command(Path(sys.executable), self.root / "bin", True)
        self.assertTrue((self.root / "zsh config/.zshrc").exists())
        self.assertFalse((self.root / ".zshrc").exists())

    def test_no_path_does_not_write_shell_config(self):
        installer.expose_command(Path(sys.executable), self.root / "bin", False)
        self.assertFalse((self.root / ".bashrc").exists())

    @unittest.skipIf(os.name == "nt", "POSIX launcher")
    def test_launcher_handles_shell_metacharacters_and_passes_arguments(self):
        python = self.root / "interpreter ' $ space"
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        python.chmod(0o755)
        launcher = installer.expose_command(python, self.root / "commands ' $ space", False)
        result = subprocess.run([str(launcher), "argument with spaces", "$(false)"], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), ["-I", "-m", "kvpark", "argument with spaces", "$(false)"])

    def invoke(self, info, *args):
        python = self.root / "hermes/venv/bin/python"
        argv = ["install.py", "--uv", "uv", "--version", "0.3.0a1", "--no-setup", "--no-path", "--bin-dir", str(self.root / "bin"), *args]
        with patch.object(sys, "argv", argv), patch.object(installer, "find_hermes", return_value=(python, info)), patch.object(installer, "run") as run, patch.object(installer, "release_asset", return_value=self.root / "package.whl"), contextlib.redirect_stdout(io.StringIO()):
            installer.main()
            return run.call_args_list

    def test_hermes_install_uses_target_python_and_denies_tool_overrides(self):
        calls = self.invoke(self.info)
        commands = [[str(x) for x in call.args[0]] for call in calls]
        install = next(command for command in commands if command[:3] == ["uv", "pip", "install"])
        self.assertEqual(install[install.index("--python") + 1], str(self.root / "hermes/venv/bin/python"))
        enable = next(command for command in commands if "enable" in command)
        self.assertIn("--no-allow-tool-override", enable)
        self.assertEqual(enable[0], install[install.index("--python") + 1])

    def test_repeat_install_does_not_replace_the_running_package(self):
        with patch.object(installer, "fetch", side_effect=AssertionError("repeat installs need no release download")):
            calls = self.invoke({**self.info, "kvpark": "0.3.0a1"})
        commands = [[str(x) for x in call.args[0]] for call in calls]
        install = next(command for command in commands if command[:3] == ["uv", "pip", "install"])
        self.assertEqual(install[-1], "pip>=23")
        self.assertNotIn(str(self.root / "package.whl"), install)

    def test_system_hermes_is_rejected_before_install(self):
        with self.assertRaisesRegex(RuntimeError, "system Python"):
            self.invoke({**self.info, "prefix": self.info["base"]})

    def test_legacy_hermes_install_requires_cache_preserving_migration(self):
        with self.assertRaisesRegex(RuntimeError, "rename migration"):
            self.invoke({**self.info, "legacy": "0.2.0a2"})

    def test_existing_different_version_requires_coordinated_update(self):
        with self.assertRaisesRegex(RuntimeError, "coordinated update"):
            self.invoke({**self.info, "kvpark": "0.2.0a1"})

    def test_runtime_publication_is_complete_or_absent_after_copy_failure(self):
        from types import SimpleNamespace
        source = self.root / "checkout"
        source.mkdir()
        args = SimpleNamespace(version="0.3.0a1", runtime="cpu", package=source)
        copytree = shutil.copytree
        def build(command, **kwargs):
            if "--output" in command:
                output = Path(command[command.index("--output") + 1]) / "build/bin"
                output.mkdir(parents=True)
                (output / "llama-server").write_bytes(b"server")
                (output / "libllama.so").write_bytes(b"library")
        for interrupted in (True, False):
            root = self.root / str(interrupted)
            def copy(*args, **kwargs):
                result = copytree(*args, **kwargs)
                if interrupted:
                    raise OSError("interrupted copy")
                return result
            with self.subTest(interrupted=interrupted), patch.object(installer.shutil, "which", return_value="tool"), patch.object(installer, "run", side_effect=build), patch.object(installer.shutil, "copytree", side_effect=copy), contextlib.redirect_stdout(io.StringIO()):
                if interrupted:
                    with self.assertRaisesRegex(OSError, "interrupted copy"):
                        installer.build_runtime(args, root)
                    self.assertFalse((root / "runtimes/0.3.0a1/cpu").exists())
                else:
                    binary = installer.build_runtime(args, root)
                    self.assertEqual(binary.read_bytes(), b"server")
                    self.assertEqual((binary.parent / "libllama.so").read_bytes(), b"library")
            self.assertEqual(list((root / "runtimes/0.3.0a1").glob(".kvpark-runtime-*")), [])

    def test_runtime_build_missing_compiler_is_actionable(self):
        from types import SimpleNamespace
        args = SimpleNamespace(version="0.3.0a1", runtime="cpu")
        with self.assertRaisesRegex(RuntimeError, r"C\+\+ compiler"):
            installer.build_runtime(args, self.root)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Bash bootstrap")
class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.script = Path(__file__).resolve().parents[1] / "install.sh"
        self.bash = shutil.which("bash")

    def test_help_works_over_stdin_without_python_or_uv(self):
        result = subprocess.run([self.bash, "-s", "--", "--help"], input=self.script.read_text(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--hermes-python", result.stdout)

    def test_download_truncated_before_main_does_not_install_anything(self):
        definitions = self.script.read_text().rsplit('\nmain "$@"', 1)[0]
        result = subprocess.run([self.bash], input=definitions, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_piped_bootstrap_downloads_helper_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory(prefix="kvpark pipe test ") as tmp:
            root = Path(tmp)
            curl = root / "curl"
            curl.write_text('#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\nif [ "$1" = -o ]; then shift; printf "# fixture\\n" > "$1"; exit 0; fi\nshift\ndone\nexit 1\n')
            curl.chmod(0o755)
            uv = root / "uv"
            uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENT_LOG"\n')
            uv.chmod(0o755)
            log = root / "arguments"
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "KVPARK_UV": str(uv), "ARGUMENT_LOG": str(log), "TMPDIR": tmp}
            result = subprocess.run([self.bash, "-s", "--", "--standalone", "--install-dir", "a path with spaces"],
                                    input=self.script.read_text(), env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = log.read_text().splitlines()
            self.assertEqual(args[-3:], ["--standalone", "--install-dir", "a path with spaces"])
            self.assertTrue(any(arg.endswith("/install.py") for arg in args))
            self.assertEqual(list(root.glob("kvpark-install.*")), [])

    def test_bootstrap_failure_cleans_temporary_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            uv = Path(tmp) / "uv"
            uv.write_text("#!/bin/sh\nexit 42\n")
            uv.chmod(0o755)
            result = subprocess.run([self.bash, str(self.script), "--standalone"],
                                    env={**os.environ, "KVPARK_UV": str(uv), "TMPDIR": tmp}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertNotIn("unbound variable", result.stderr)
            self.assertEqual(list(Path(tmp).glob("kvpark-install.*")), [])


if __name__ == "__main__":
    unittest.main()
