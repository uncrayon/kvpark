"""Build the standalone collector against existing llama.cpp libraries."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TARGETS = {"kvpark-collect-values": "collector.cpp", "kvpark-kv-probe": "kv_probe.cpp",
           # save / restore-decode / capture driver for any architecture (test_layouts.py)
           "kvpark-state-driver": "state_driver.cpp",
           # needs a runtime with libllama-common (LLAMA_BUILD_COMMON=ON), e.g. a llama-server build
           "kvpark-chat-apply": "chat_apply.cpp"}
GPU_BACKENDS = ("vulkan", "cuda", "hip", "metal", "sycl")


def runtime_backends(build):
    """ggml backends compiled into a llama.cpp build, from its CMakeCache.txt (CPU always)."""
    cache = build / "CMakeCache.txt"
    enabled = set()
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, value = line.partition("=")
            if name.startswith("GGML_") and ":" in name and value.strip().upper() in ("ON", "1", "TRUE"):
                enabled.add(name.split(":", 1)[0][5:].lower())
    return ["cpu"] + [backend for backend in GPU_BACKENDS if backend in enabled]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, default=Path("runtime/source"))
    parser.add_argument("--llama-build", type=Path, default=Path("runtime/build"))
    parser.add_argument("--runtime-revision", default="85d5703a3b1b47243213a39059a6e3076c92733a")
    parser.add_argument("--output", type=Path, required=True, help="new build directory")
    parser.add_argument("--target", choices=sorted(TARGETS), default="kvpark-collect-values",
                        help="executable to build (default: the v1 value collector)")
    parser.add_argument("--jobs", type=int, default=2, help="parallel compile jobs (default 2)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new build directory")
    source = args.llama_source.resolve()
    library_dir = args.llama_build.resolve() / "bin"
    if not (source / "include/llama.h").is_file() or not library_dir.is_dir():
        parser.error("existing llama.cpp source and matching compiled libraries are required")
    experiment = Path(__file__).resolve().parent
    output = args.output.resolve()
    subprocess.run(["cmake", "-S", str(experiment), "-B", str(output),
                    f"-DLLAMA_SOURCE={source}", f"-DLLAMA_LIBRARY_DIR={library_dir}",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(output), "--config", "Release", "-j", str(max(1, args.jobs)),
                    "--target", args.target], check=True)
    binary = output / args.target
    artifacts = {binary.resolve()}
    patterns = ("libllama.so*", "libggml*.so*", "libllama*.dylib", "libggml*.dylib")
    if args.target == "kvpark-chat-apply":
        patterns += ("libllama-common.so*",)
    for pattern in patterns:
        artifacts.update(path.resolve() for path in library_dir.glob(pattern) if path.is_file())
    manifest = {
        "format_version": 1,
        "runtime_source_revision": args.runtime_revision,
        "target": args.target,
        "collector_source": TARGETS[args.target],
        "collector_source_sha256": sha256(experiment / TARGETS[args.target]),
        "runtime_build": str(args.llama_build.resolve()),
        "backends": runtime_backends(args.llama_build.resolve()),
        "artifacts": [{"path": str(path), "sha256": sha256(path)} for path in sorted(artifacts)],
    }
    binary.with_name(binary.name + ".build.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {binary}; existing runtime files were not modified.")


if __name__ == "__main__":
    main()
