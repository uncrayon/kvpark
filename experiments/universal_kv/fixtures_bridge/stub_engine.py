"""A same-model stand-in for the bridge engine, for live checks of kvpark.bridge.switch only.

    python stub_engine.py JOB.json

Source and target are the same model here, so a perfect translation of the source slot is
the source slot itself, cut to T0 = min(P - tail, the common prefix of the source's and the
target's tokens): attention cells below T0 kept, the recurrent state reset at T0 - 1 (as the
real engine's R0 writes it). Prints the engine's report line ({"t0", "engine_seconds", "P"}).
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_resume_check import kvstate, translated_style  # noqa: E402


def main():
    started = time.perf_counter()
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    spec = kvstate.ModelSpec.from_gguf(job["target"]["gguf"])
    source = [int(t) for t in kvstate.read_file(job["source"]["state"], spec).tokens]
    target = [int(token) for token, _ in job["target"]["tokens"]]
    common = next((i for i, (a, b) in enumerate(zip(source, target)) if a != b), min(len(source), len(target)))
    t0 = min(len(target) - int(job["tail"]), common)
    translated_style(job["source"]["state"], spec, t0, job["target"]["output"])
    print(json.dumps({"t0": t0, "P": len(target), "engine_seconds": round(time.perf_counter() - started, 3)}))


if __name__ == "__main__":
    main()
