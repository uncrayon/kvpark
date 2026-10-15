"""The copy of a format 2 hub that is published: every side that has an f16 copy keeps only that one (the GPU
engines read f16; Hub.side_flat widens it for the float32 readers), the rest (views) stay f32, and paths under
--strip-prefix in the manifest become relative to it.

    python publish_hub.py HUB-v2 OUTPUT [--strip-prefix DIR]

Files are hard-linked from HUB-v2 (same filesystem); OUTPUT must not exist. The manifest records the original
manifest's sha256 under "published".
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def relative(value, prefix):
    if isinstance(value, str):
        return value[len(prefix):] if prefix and value.startswith(prefix) else value
    if isinstance(value, list):
        return [relative(v, prefix) for v in value]
    if isinstance(value, dict):
        return {relative(k, prefix): relative(v, prefix) for k, v in value.items()}
    return value


def publish(source, output, strip_prefix=""):
    text = (source / "manifest.json").read_bytes()
    manifest = json.loads(text)
    if str(manifest.get("format_version")) != "2":
        raise ValueError(f"{source} is not a format 2 hub")
    if manifest.get("floor"):
        raise ValueError(f"{source} is a floor hub (decoders zeroed), not one to publish")
    kept = set()
    for key, side in manifest["sides"].items():
        if "f16" in side["files"]:
            side["files"] = {"f16": side["files"]["f16"]}
        kept.update(entry["file"] for entry in side["files"].values())
    for entry in manifest.get("files", {}).values():         # format 2 hubs may keep whole-file entries too
        kept.add(entry["file"])
    missing = sorted(f for f in kept if not (source / f).is_file())
    if missing:
        raise ValueError(f"{source} lacks {missing[:5]}")
    output.mkdir(parents=False)
    for name in sorted(kept):
        os.link(source / name, output / name)
    manifest = relative(manifest, strip_prefix.rstrip("/") + "/" if strip_prefix else "")
    manifest["published"] = {"of_manifest_sha256": hashlib.sha256(text).hexdigest(), "dtypes": "f16 sides, f32 views"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest, sorted(kept)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hub", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--strip-prefix", default="")
    args = parser.parse_args(argv)
    _, files = publish(args.hub, args.output, args.strip_prefix)
    size = sum((args.output / f).stat().st_size for f in files)
    print(json.dumps({"output": str(args.output), "files": len(files) + 1, "bytes": size}))


if __name__ == "__main__":
    main()
