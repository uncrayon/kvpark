#!/usr/bin/env python3
"""Build the alpha's pinned llama.cpp source with checkpoint persistence."""

import argparse
from pathlib import Path
import subprocess
import tarfile
import tempfile
import os

PIN = "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e"
# Upstream's build number for PIN (`git rev-list --count`). The export has no Git
# history, so CMake would otherwise name an enclosing checkout's commit in build_info.
PIN_BUILD = 11398
# Applied in this order; a later patch may depend on an earlier one.
PATCHES = ("patches/llama-slot-resume.patch",)


def apply_patches(source, patches, ceiling):
    # Inside an enclosing repository (runtime/ in a kvpark checkout) git apply reads
    # `diff --git` paths from that repository's top level and skips them with exit 0.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(ceiling)}
    for patch in patches:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, env=env, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing llama.cpp Git checkout containing the pinned commit")
    parser.add_argument("--output", type=Path, default=Path("runtime"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake-arg", action="append", default=[], help="e.g. --cmake-arg=-DGGML_CUDA=ON")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("output already exists; choose a new directory for a clean build")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    repo = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True)
    checkout = args.source.resolve() if args.source else out / "checkout"
    if not args.source:
        subprocess.run(["git", "init", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", PIN], check=True)
    source = out / "source"
    source.mkdir()
    # Export the exact commit, never the caller's modifications or build tree.
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "-C", str(checkout), "archive", PIN], stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as files:
            # Only a fixed, trusted upstream Git commit is exported here.
            files.extractall(source, filter="data")
    apply_patches(source, [repo / patch for patch in PATCHES], out)
    build = out / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
                    f"-DLLAMA_BUILD_NUMBER={PIN_BUILD}", f"-DLLAMA_BUILD_COMMIT={PIN[:9]}",
                    "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_OPENSSL=OFF", *args.cmake_arg], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "llama-server", "-j", str(args.jobs)], check=True)
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    binaries = list((build / "bin").rglob(name))
    if len(binaries) != 1:
        raise RuntimeError("could not locate the built llama-server")
    print(f"\nBuilt {binaries[0]}\nKeep its adjacent libraries with it.")


if __name__ == "__main__":
    main()
