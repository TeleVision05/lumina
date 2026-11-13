"""Fail when a source file is longer than the repo limit (500 lines).

Usage: python scripts/hooks/check_file_length.py [--max N] FILE...
Only files under the source folders are checked; generated data (JSON vectors, lock files,
models) is not source and is ignored by the pre-commit file filter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_PREFIXES = ("training/", "web/src/", "web/tests/", "web/e2e/", "scripts/")
SOURCE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".html", ".css", ".sh")


def is_source(path: str) -> bool:
    return path.startswith(SOURCE_PREFIXES) and path.endswith(SOURCE_SUFFIXES)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    failures = []
    for name in args.files:
        posix = Path(name).as_posix()
        if not is_source(posix) or not Path(name).is_file():
            continue
        lines = count_lines(Path(name))
        if lines > args.max:
            failures.append(f"{posix}: {lines} lines (limit {args.max})")
    for failure in failures:
        print(f"File too long, split it into modules: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
