# Contributing

Start with an issue describing the user problem and a reproducible example.
Useful alpha feedback includes installation failures, agent integrations, and
model/configuration reports showing whether saved state was actually reused.

For code changes:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Keep the Python runtime dependency-free unless a dependency solves a concrete
problem. Preserve session ownership, atomic archive publication, runtime identity
checks, and honest metrics. Add regression tests for behavior changes.

Runtime changes need a fresh build and a real-model restart acceptance run.
Report the model filename, upstream commit, hardware, cache configuration,
cached/processed tokens, and whether the OS page cache was controlled. Whole
response duration is not TTFT. Remove private paths and chat contents before
sharing evidence. Never upload model weights or archive snapshots.

The project uses the MIT license. Contributions are provided under that license.
