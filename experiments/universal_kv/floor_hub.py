"""An information-free copy of a format 2 hub for live baselines: the named targets' decoders map every latent to
their train mean (decoder_<l> zeroed; z_mean and decoder_mean_<l> kept), so a switch through it restores the
decoder mean in every translated cell, pair_eval's "floor", through the live bridge's own path.

    python floor_hub.py HUB-v2 OUTPUT --target NAME [--target NAME ...]

Every other file is hard-linked from HUB-v2 (same filesystem); the targets' decoder files are written anew and
the manifest records their sha256 and size. OUTPUT must not exist.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import hub as hub_module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hub", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", action="append", required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((args.hub / "manifest.json").read_text())
    rewritten = {}
    for name in args.target:
        for kind in hub_module.KINDS:
            key = f"{name}.{kind}.decoder"
            if key not in manifest["sides"]:
                raise SystemExit(f"{args.hub} has no side {key}")
            for precision, entry in manifest["sides"][key]["files"].items():
                rewritten[entry["file"]] = (key, precision)
    args.output.mkdir(parents=False)
    for path in sorted(args.hub.iterdir()):
        if path.name == "manifest.json" or path.name in rewritten:
            continue
        os.link(path, args.output / path.name)
    for file, (key, precision) in rewritten.items():
        side = manifest["sides"][key]
        values = np.load(args.hub / file, mmap_mode="r")
        out = np.array(values)                      # flat, as converted
        for array, (offset, shape) in side["arrays"].items():
            if array.startswith("decoder_") and not array.startswith("decoder_mean"):
                out[offset:offset + int(np.prod(shape))] = 0
        np.save(args.output / file, out)
        entry = side["files"][precision]
        entry["sha256"] = hub_module.sha256_file(args.output / file)
        entry["bytes"] = (args.output / file).stat().st_size
    manifest["floor"] = {"of": str(args.hub.resolve()), "zeroed_decoders": args.target}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "rewritten": sorted(rewritten)}))


if __name__ == "__main__":
    main()
