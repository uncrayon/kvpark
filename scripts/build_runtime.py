#!/usr/bin/env python3
"""Build the alpha's pinned llama.cpp source with checkpoint persistence."""

import argparse
from pathlib import Path
import subprocess
import tarfile
import tempfile

PIN = "85d5703a3b1b47243213a39059a6e3076c92733a"


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
    subprocess.run(["git", "apply", "--check", str(repo / "patches/llama-slot-resume.patch")], cwd=source, check=True)
    subprocess.run(["git", "apply", str(repo / "patches/llama-slot-resume.patch")], cwd=source, check=True)
    build = out / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_OPENSSL=OFF", *args.cmake_arg], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "-j", str(args.jobs)], check=True)
    print(f"\nBuilt {build / 'bin/llama-server'}\nKeep its adjacent libraries with it.")


if __name__ == "__main__":
    main()
