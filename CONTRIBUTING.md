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
python scripts/check_portability.py
```

The checkout can live in any directory. Use paths relative to the checkout in
examples, user-supplied paths for models and servers, and platform data directories
for saved state. CI rejects personal home-directory paths in shared files.

After moving a checkout, recreate its virtual environment and reinstall with
`python -m pip install -e .`; activation scripts, console launchers, and editable
installs can retain the previous location. Rebuild compiled runtimes in a fresh
output directory as described in [runtime compatibility](docs/compatibility.md#moving-a-checkout).
Share the Git checkout or Python distributions, not `.venv/`, runtime build trees,
`evidence/`, or local archive/configuration files. These generated files may contain
absolute paths specific to the environment that created them.

Keep runtime dependencies small; add one only when it solves a concrete problem. Preserve session ownership, atomic archive publication, runtime identity
checks, and honest metrics. Add regression tests for behavior changes.

Runtime changes need a fresh build and a real-model restart acceptance run.
Report the model filename, upstream commit, hardware, cache configuration,
cached/processed tokens, and whether the OS page cache was controlled. Whole
response duration is not TTFT. Remove private paths and chat contents before
sharing evidence. Never upload model weights or archive snapshots.

The project uses the MIT license. Contributions are provided under that license.
