"""Persistent inference state for local open-weight models."""

import os

# Preserve existing profile credentials/settings during the distribution rename.
# Explicit new names always take precedence.
for _name, _value in list(os.environ.items()):
    if _name.startswith("SLOTH_"):
        os.environ.setdefault("KVPARK_" + _name[6:], _value)

__version__ = "0.3.0a2"
