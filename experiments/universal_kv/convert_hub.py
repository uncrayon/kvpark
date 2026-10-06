"""Convert a hub bundle from format 1 (save_hub: one npz per model and kind) to format 2.

    python convert_hub.py HUB HUB-v2 [--no-f16]

Format 2 is what the engine reads per switch, packed so that a switch reads only the two
sides it needs (Qwen3.8 -> Muse: 0.9 of the 4.9 GB of hub4-d4096). Per model and kind there
are three sides, each one flat C-contiguous float32 .npy whose arrays the manifest lists as
`name: [offset, shape]` (offsets in elements, 64-byte aligned):

  encoder  the layers a translation *from* the model reads (a sliding-window model's global
           layers): mean_<l>, basis_<l> (C-contiguous [d_l, r_l]) and the fused encoder
           weight = (basis2 * scale) @ encoder, formed in float64 ([sum r_l, D])
  decoder  z_mean, decoder_mean_<l>, decoder_<l> ([D, d_l])
  view     what only load_hub needs to rebuild the fitted HubKinds (basis2, scale, encoder;
           for a sliding-window model also the full transform and its source_* arrays)

Encoder and decoder sides also get a float16 copy (same offsets) for GPU backends unless
--no-f16. A DSA model's indexer ridge (hub.IndexerRidge, <name>.indexer.npz) becomes the side
<name>.indexer.ridge (x_mean_<l>, weight_<l>, y_mean_<l>; float32 only). A DeepSeek V4 model's
head decoders (fit_hub.py --heads: <name>.<kind>.npz per head kind, manifest["heads"]) are copied
as they are, with their "files" entries: hub.Hub reads them as format 1 npz in either format, and
a translation into the model reads them on numpy only. Every file's sha256 is in the manifest;
readers verify it once per file identity (hub.Verifier). The input bundle is only read; the
output directory must not exist.

    python convert_hub.py HUB HUB-v2 --reuse BASE-v2 [--reuse ...]

--reuse names format 2 bundles converted before (e.g. the base an onboarded HUB grew from): a
model whose format 1 files have the very sha256 such a bundle was converted from keeps that
bundle's sides as hard links (no conversion, no disk), so converting an onboarded bundle only
converts the new model.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

import hub as hub_module

ALIGN = 16                       # elements: 64-byte aligned float32 arrays
F16_MAX = 65504.0


def pack(arrays):
    """(flat float32 array, table name -> [offset, shape]) for named arrays, in the given order."""
    table, offset = {}, 0
    for name, array in arrays.items():
        table[name] = [offset, list(np.shape(array))]
        offset += -(-max(int(np.size(array)), 1) // ALIGN) * ALIGN
    flat = np.zeros(offset, dtype=np.float32)
    for name, array in arrays.items():
        start, shape = table[name]
        flat[start:start + int(np.size(array))] = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    return flat, table


def _write(path, flat):
    np.save(path, flat, allow_pickle=False)
    return {"file": path.name, "sha256": hub_module.sha256_file(path), "bytes": path.stat().st_size}


def side_arrays(arrays, layers):
    """(source layers, subset?, {role: named arrays}) of one format 1 npz (a dict of arrays)."""
    subset = "source_encoder" in arrays
    source_layers = [int(l) for l in arrays["source_layers"]] if subset else list(layers)
    prefix = "source_" if subset else ""
    encoder = {f"mean_{l}": arrays[f"{prefix}mean_{l}"] for l in source_layers}
    encoder.update({f"basis_{l}": arrays[f"{prefix}basis_{l}"] for l in source_layers})
    encoder["weight"] = hub_module.fuse_encoder(arrays[f"{prefix}basis2"], arrays[f"{prefix}scale"],
                                                arrays[f"{prefix}encoder"])
    decoder = {"z_mean": arrays["decoder_z_mean"]}
    decoder.update({f"decoder_mean_{l}": arrays[f"decoder_mean_{l}"] for l in layers})
    decoder.update({f"decoder_{l}": arrays[f"decoder_{l}"] for l in layers})
    view = {"basis2": arrays["basis2"], "scale": arrays["scale"], "encoder": arrays["encoder"]}
    if subset:
        view.update({f"mean_{l}": arrays[f"mean_{l}"] for l in layers})
        view.update({f"basis_{l}": arrays[f"basis_{l}"] for l in layers})
        view.update(source_basis2=arrays["source_basis2"], source_scale=arrays["source_scale"],
                    source_encoder=arrays["source_encoder"])
    return source_layers, subset, {"encoder": encoder, "decoder": decoder, "view": view}


def reusable(manifest, name, reuse):
    """(bundle directory, its manifest) of the first format 2 bundle in `reuse` converted from exactly
    the model's format 1 files (every file of the model, same sha256), or None."""
    keys = [key for key in manifest["files"] if hub_module._owner(key, manifest["models"]) == name]
    for directory, other in reuse:
        if other["format_version"] != hub_module.FORMAT_V2 or name not in other["models"]:
            continue
        sources = other.get("converted_from", {}).get("files", {})
        if all(sources.get(key, {}).get("sha256") == manifest["files"][key]["sha256"] for key in keys):
            return directory, other
    return None


def link_model(name, directory, other, output, new, log=print):
    """Link a model's sides (and copied head files) from a format 2 bundle into `new` (the manifest)."""
    for stem, entry in other["sides"].items():
        if hub_module._owner(stem, other["models"]) == name:
            for item in entry["files"].values():
                hub_module.link_file(directory / item["file"], output / item["file"])
            new["sides"][stem] = entry
    for key, item in (other.get("files") or {}).items():
        if hub_module._owner(key, other["models"]) == name:
            hub_module.link_file(directory / item["file"], output / item["file"])
            new.setdefault("files", {})[key] = dict(item)
    log(json.dumps({"linked": name, "from": str(directory)}))


def convert(source, output, *, f16=True, log=print, reuse=()):
    source, output = Path(source), Path(output)
    manifest = hub_module.read_manifest(source)
    if manifest["format_version"] != hub_module.FORMAT:
        raise ValueError(f"{source} is not a format {hub_module.FORMAT} hub")
    reuse = [(Path(directory), hub_module.read_manifest(directory)) for directory in reuse]
    output.mkdir(parents=True, exist_ok=False)
    verifier = hub_module.Verifier()
    new = {key: value for key, value in manifest.items() if key != "files"}
    new.update(format_version=hub_module.FORMAT_V2, sides={},
               converted_from={"directory": str(source.resolve()), "format_version": manifest["format_version"],
                               "files": manifest["files"]})
    linked = set()
    for name in manifest["models"]:
        found = reusable(manifest, name, reuse)
        if found is not None:
            link_model(name, *found, output, new, log=log)
            linked.add(name)
            continue
        for kind in hub_module.KINDS:
            started = time.perf_counter()
            entry = manifest["files"][f"{name}.{kind}"]
            path = source / entry["file"]
            verifier.check(path, entry["sha256"])
            layers = manifest["models"][name]["layers"][kind]
            with np.load(path, allow_pickle=False) as npz:
                arrays = {key: npz[key] for key in npz.files}
            source_layers, subset, roles = side_arrays(arrays, layers)
            del arrays
            for role, named in roles.items():
                flat, table = pack(named)
                stem = f"{name}.{kind}.{role}"
                files = {"f32": _write(output / f"{stem}.f32.npy", flat)}
                if f16 and role != "view":
                    big = float(np.abs(flat).max()) if len(flat) else 0.0
                    if big >= F16_MAX:
                        log(json.dumps({"skip_f16": stem, "max_abs": big}))
                    else:
                        files["f16"] = _write(output / f"{stem}.f16.npy", flat.astype(np.float16))
                new["sides"][stem] = {"layers": source_layers if role == "encoder" else list(layers),
                                      "subset": bool(subset and role == "encoder"), "files": files,
                                      "arrays": table}
            log(json.dumps({"converted": f"{name}.{kind}", "source_layers": len(source_layers),
                            "seconds": round(time.perf_counter() - started, 2)}))
    for name in manifest.get("indexer_ridges", {}):
        if name in linked:
            continue
        entry = manifest["files"][f"{name}.indexer"]
        path = source / entry["file"]
        verifier.check(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as npz:
            layers = [int(l) for l in npz["layers"]]
            named = {f"{prefix}_{l}": npz[f"{prefix}_{l}"] for l in layers
                     for prefix in ("x_mean", "weight", "y_mean")}
        flat, table = pack(named)
        stem = f"{name}.indexer.ridge"
        new["sides"][stem] = {"layers": layers, "subset": False,
                              "files": {"f32": _write(output / f"{stem}.f32.npy", flat)}, "arrays": table}
        log(json.dumps({"converted": f"{name}.indexer", "layers": len(layers)}))
    for kind, entry in sorted((manifest.get("heads") or {}).items()):
        for name in entry["models"]:
            if name in linked:
                continue
            item = manifest["files"][f"{name}.{kind}"]
            path = source / item["file"]
            verifier.check(path, item["sha256"])
            shutil.copyfile(path, output / item["file"])
            new.setdefault("files", {})[f"{name}.{kind}"] = dict(item)
            log(json.dumps({"copied": f"{name}.{kind}", "layers": len(entry["models"][name]["layers"])}))
    (output / "manifest.json").write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="format 1 hub bundle (read only)")
    parser.add_argument("output", type=Path, help="new directory for the format 2 bundle")
    parser.add_argument("--no-f16", action="store_true", help="skip the float16 copies of encoder/decoder sides")
    parser.add_argument("--reuse", type=Path, action="append", default=[], metavar="V2_BUNDLE",
                        help="link the sides of models this format 2 bundle was converted from (same files)")
    args = parser.parse_args(argv)
    convert(args.source, args.output, f16=not args.no_f16, reuse=args.reuse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
