#!/usr/bin/env python3
"""Reject personal machine paths in the files shared with a checkout."""

from pathlib import Path
import re
import subprocess


PERSONAL_PATH = re.compile(
    rb"/(?:home|Users)/[\w.-]+/|[A-Za-z]:[\\/]+Users[\\/]+[\w .-]+[\\/]",
    re.IGNORECASE,
)


def main():
    repo = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo
    ).split(b"\0")
    failures = []
    for name in sorted(set(filter(None, names))):
        path = repo / name.decode("utf-8")
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            if PERSONAL_PATH.search(line):
                failures.append(f"{path.relative_to(repo)}:{number}: personal machine path")
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS: no personal machine paths in checkout files")


if __name__ == "__main__":
    main()
