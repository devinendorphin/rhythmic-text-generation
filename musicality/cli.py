"""Command-line interface.

    python -m musicality analyze poem.txt              # all 10 tools
    python -m musicality analyze poem.txt -t meter     # one tool
    python -m musicality compare poem.txt prose.txt    # fingerprint distance
    echo "some text" | python -m musicality analyze -  # stdin

Output is JSON, so results pipe straight into jq / pandas / notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import ALL_TOOLS
from .fingerprint import RhythmicFingerprint


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="musicality",
        description="Quantify rhythm and musicality embedded in text.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="run tools on one text")
    p_analyze.add_argument("file", help="text file, or - for stdin")
    p_analyze.add_argument(
        "-t", "--tool", choices=sorted(ALL_TOOLS), action="append",
        help="run only this tool (repeatable); default: all ten",
    )

    p_compare = sub.add_parser(
        "compare", help="rhythmic-fingerprint similarity of two texts"
    )
    p_compare.add_argument("file_a")
    p_compare.add_argument("file_b")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        text = _read(args.file)
        names = args.tool or sorted(ALL_TOOLS)
        report = {name: ALL_TOOLS[name]().analyze(text) for name in names}
        print(json.dumps(report, indent=2))
    else:
        result = RhythmicFingerprint().compare(_read(args.file_a), _read(args.file_b))
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
