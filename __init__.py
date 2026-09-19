"""Hermes plugin-directory entry point; the package itself lives in src/kvpark.

`hermes plugins install uncrayon/kvpark` places this checkout under
~/.hermes/plugins/kvpark and calls register(ctx) from here. Installed
packages use the hermes_agent.plugins entry point in pyproject.toml instead.
"""

import sys
from pathlib import Path

_SOURCE = str(Path(__file__).resolve().parent / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from kvpark.hermes import register  # noqa: E402

__all__ = ["register"]
