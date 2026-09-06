"""Validated, durable cleanup preferences for one archive directory."""

import json
import math


def validate(values):
    if not isinstance(values, dict) or set(values) - {"ttl_days", "max_gib", "cleanup_enabled"}:
        raise ValueError("settings must contain only ttl_days, max_gib, and cleanup_enabled")
    result = dict(values)
    if "cleanup_enabled" in result and type(result["cleanup_enabled"]) is not bool:
        raise ValueError("cleanup_enabled must be true or false")
    for key in ("ttl_days", "max_gib"):
        if key not in result:
            continue
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == "max_gib" and value == 0):
            raise ValueError(f"{key} must be a finite {'positive' if key == 'max_gib' else 'nonnegative'} number")
    return result


class Retention:
    def __init__(self, path, defaults):
        self.path = path
        self.defaults = defaults
        self.overrides = validate(json.loads(path.read_text())) if path.exists() else {}

    def settings(self):
        return {**self.defaults(), **self.overrides}

    def update(self, values):
        from .proxy import atomic_json
        updated = {**self.overrides, **validate(values)}
        atomic_json(self.path, updated)
        self.overrides = updated
        return self.settings()

    def expired(self, meta, now, *, manual=False):
        policy = self.settings()
        return (policy["cleanup_enabled"] or manual) and policy["ttl_days"] > 0 and (
            now - meta.get("saved_at", meta["last_used"]) >= policy["ttl_days"] * 86400)
