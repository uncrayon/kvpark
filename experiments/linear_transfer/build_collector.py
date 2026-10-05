"""Build the standalone collector against existing llama.cpp libraries.

The manifest (<binary>.build.json) names the runtime's llama.cpp revision, which kvstate reads as
the layout generation of the states the tool writes (kvstate.llama_revision_of). It comes from the
build's CMake build info (common/build-info.cpp, LLAMA_COMMIT) when that names a known pin
(scripts/build_runtime.py passes LLAMA_BUILD_COMMIT), and a --runtime-revision naming another one is
refused. The defaults (runtime/source, runtime/build) are the repository's runtime, a
build_runtime.py build of a7b94df2c. A build of an exported source inside another repository names
that repository's commit (the 85d5703a3 runtime kept at runtime-85d5703a3/, built before
build_runtime.py set it), which cannot be checked: --runtime-revision is then required and recorded
as given.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
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
           # llama.cpp's own vocabulary, vocab_only (experiments/universal_kv/test_engine.py KVPARK_VOCAB_TOOL)
           "kvpark-vocab": "vocab_tool.cpp",
           # needs a runtime with libllama-common (LLAMA_BUILD_COMMON=ON), e.g. a llama-server build
           "kvpark-chat-apply": "chat_apply.cpp"}
GPU_BACKENDS = ("vulkan", "cuda", "hip", "metal", "sycl")
# llama.cpp revisions kvstate reads states of (kvstate.LLAMA_REVISION_COMMITS)
PINS = ("a7b94df2c616bc1f62a73b964b4a71cb0dcc488e", "85d5703a3b1b47243213a39059a6e3076c92733a")


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


def build_info(build):
    """"b<number>-<commit>" of a llama.cpp build (its common/build-info.cpp), or None."""
    path = build / "common" / "build-info.cpp"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    number = re.search(r"LLAMA_BUILD_NUMBER\s*=\s*(\d+)", text)
    commit = re.search(r'LLAMA_COMMIT\s*=\s*"([0-9a-fA-F]+)"', text)
    return f"b{number.group(1)}-{commit.group(1).lower()}" if number and commit else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, default=Path("runtime/source"))
    parser.add_argument("--llama-build", type=Path, default=Path("runtime/build"))
    parser.add_argument("--runtime-revision",
                        help="the llama.cpp commit of --llama-source (default: the known pin the build "
                             "info names; required when it names none)")
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
    info = build_info(args.llama_build.resolve())
    named = info.rsplit("-", 1)[-1] if info else ""
    pin = next((pin for pin in PINS if named and pin.startswith(named)), None)
    given = (args.runtime_revision or "").strip().lower()
    if pin is None and not given:
        parser.error(f"the runtime build reports {info or 'no build info'}, which names no known llama.cpp "
                     f"pin ({', '.join(p[:9] for p in PINS)}); pass --runtime-revision")
    if pin is not None and given and not (given.startswith(named) or named.startswith(given)):
        parser.error(f"the runtime build reports {info}, not --runtime-revision {args.runtime_revision}")
    revision = args.runtime_revision.strip() if given else pin
    experiment = Path(__file__).resolve().parent
    output = args.output.resolve()
    subprocess.run(["cmake", "-S", str(experiment), "-B", str(output),
                    f"-DLLAMA_SOURCE={source}", f"-DLLAMA_LIBRARY_DIR={library_dir}",
                    f"-DKVPARK_LLAMA_REVISION={revision.lower()}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
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
        "runtime_source_revision": revision,
        "runtime_build_info": info,
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
