"""Exercise installation and repeat enablement in a disposable Hermes environment.

Run with Hermes' Python after installing its dependencies. Existing dependencies
are read through a .pth file; package and profile writes stay in the temporary tree.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv


def main():
    import hermes_cli  # Fail early if the caller is not a Hermes environment.
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv must be on PATH for this installer acceptance test")
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="kvpark hermes install ") as tmp:
        root = Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root / "environment")
        python = root / "environment/bin/python"
        site = Path(subprocess.check_output([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        dependencies = sysconfig.get_path("purelib")
        (site / "hermes-test-dependencies.pth").write_text(f"import site; site.addsitedir({dependencies!r})\n")
        profile = root / "profile"
        profile.mkdir()
        before = {"model": {"base_url": "http://127.0.0.1:18765/v1", "default": "existing-model"},
                  "plugins": {"enabled": ["other-plugin"], "entries": {"other-plugin": {"settings": {"keep": True}}}}}
        (profile / "config.yaml").write_text(json.dumps(before))
        env = {key: value for key, value in os.environ.items()
               if key not in {"VIRTUAL_ENV", "PYTHONPATH"} and not key.startswith(("HERMES_", "KVPARK_", "SLOTH_"))}
        env.update(HOME=str(root / "home"), HERMES_HOME=str(profile), HERMES_BUNDLED_PLUGINS=str(root / "empty-plugins"),
                   XDG_CACHE_HOME=str(root / "cache"), UV_CACHE_DIR=str(root / "cache/uv"))
        Path(env["HOME"]).mkdir()
        command = [sys.executable, str(repo / "scripts/install.py"), "--uv", uv,
                   "--hermes-python", str(python), "--package", str(repo), "--no-setup", "--no-path",
                   "--install-dir", str(root / "install"), "--bin-dir", str(root / "bin")]
        for _ in range(2):
            subprocess.run(command, cwd=root, env=env, check=True)
        subprocess.run([str(python), "-I", "-c", '''
import sys
from pathlib import Path
import kvpark
from hermes_cli.config import load_config
assert Path(kvpark.__file__).is_relative_to(sys.prefix), kvpark.__file__
cfg = load_config()
assert cfg["model"]["base_url"] == "http://127.0.0.1:18765/v1"
assert cfg["model"]["default"] == "existing-model"
assert "kvpark" in cfg["plugins"]["enabled"]
assert "other-plugin" in cfg["plugins"]["enabled"]
assert cfg["plugins"]["entries"]["other-plugin"]["settings"]["keep"] is True
assert cfg["plugins"]["entries"]["kvpark"]["allow_tool_override"] is False
'''], cwd=root, env=env, check=True)
        subprocess.run([str(python), str(repo / "tests/hermes/test_integration.py"), "-v"], cwd=root, env=env, check=True)
        print("PASS: real Hermes installation and repeat enablement preserve existing routes and unrelated plugins")


if __name__ == "__main__":
    main()
