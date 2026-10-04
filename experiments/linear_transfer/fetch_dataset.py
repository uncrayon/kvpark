"""Download the pinned SmolTalk2 calibration subsets, verifying each file's SHA-256.

Standard library only, so it runs before the research environment exists.
Files are Apache-2.0 SmolTalk2 subsets (see the dataset card's License
section); they are calibration inputs kept under evidence/, not shipped.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


REPOSITORY = "HuggingFaceTB/smoltalk2"
REVISION = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
FILES = {
    "SFT/smolagents_toolcalling_traces_think-00000-of-00001.parquet":
        (81765978, "7785c904fdaae971881bde732bd168db6d034ac60752f9396d51554a7267efe4"),
    "SFT/hermes_function_calling_v1_no_think-00000-of-00001.parquet":
        (10817864, "3b4ffeafc6fbed9075243421634f6c51aef5c714a64a0a1402df24d362afa943"),
    "SFT/smoltalk_systemchats_Qwen3_32B_think-00000-of-00001.parquet":
        (64889322, "e2411efae1369c25a28b7af6fe204dba43cda6d1e7a6d06b91c24e93a1903e5f"),
    "SFT/multi_turn_reasoning_if_think-00000-of-00001.parquet":
        (178159863, "3319a3987192cd92eb72a3b1b44b61cb653c9e2f19335baf7dd377c6f574ada4"),
    "SFT/aya_dataset_Qwen3_32B_think-00000-of-00001.parquet":
        (32597620, "e2b8f97e19856c42fffd5917b06ef64cadfa95fa9a6e3443491d4669dce64aeb"),
    "SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-00000-of-00001.parquet":
        (6326422, "baa621d326f4a884627be17aae8fab29418897bc4af110760d690fa0cd85c5e1"),
}
# Long single-document question answering (think traces), for the long-context evaluation set.
LONG_FILES = {
    "SFT/LongAlign_64k_Qwen3_32B_yarn_131k_think-00000-of-00002.parquet":
        (135166419, "2d4c8286d26c1dbe6929f3d6112383a9726dd960e9e5509ea78758baafad7e05"),
    "SFT/LongAlign_64k_Qwen3_32B_yarn_131k_think-00001-of-00002.parquet":
        (134491905, "115b28b8c9b97e23d28cb8fdef3ac539dc497b3ab21725e87f23833348144013"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, size, expected, output):
    target = output / Path(name).name
    if target.exists():
        if target.stat().st_size == size and sha256(target) == expected:
            print(f"verified existing {target.name}")
            return
        raise ValueError(f"{target} exists but does not match the pinned digest; remove it first")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
    with tempfile.NamedTemporaryFile(dir=output, prefix=".download-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                received = 0
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(block)
                    received += len(block)
                    print(f"\r{target.name}: {received / 1e6:.1f}/{size / 1e6:.1f} MB",
                          end="", flush=True)
            print()
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    if pending.stat().st_size != size or sha256(pending) != expected:
        pending.unlink()
        raise ValueError(f"{name} failed size or SHA-256 verification")
    os.link(pending, target)
    pending.unlink()
    print(f"verified {target.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory for the parquet files")
    parser.add_argument("--include-longalign", action="store_true",
                        help="also fetch the LongAlign subset (270 MB) used for long-context evaluation")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {**FILES, **(LONG_FILES if args.include_longalign else {})}
    print(f"{REPOSITORY}@{REVISION}: {len(files)} files, "
          f"{sum(size for size, _ in files.values()) / 1e6:.1f} MB")
    for name, (size, expected) in files.items():
        fetch(name, size, expected, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
