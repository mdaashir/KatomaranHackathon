"""Fail a PR if any commit contains an attribution trailer."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

FORBIDDEN = (
    re.compile(r"(?im)^co-authored-by\s*:"),
    re.compile(r"(?im)^co-authored by\s*:"),
    re.compile(r"(?im)^claude-session\s*:"),
    re.compile(r"(?im)^generated with claude code"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="PR base SHA")
    parser.add_argument("--head", required=True, help="PR head SHA")
    return parser


def commit_message(sha: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%B", sha],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commits = subprocess.run(
        ["git", "rev-list", f"{args.base}..{args.head}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    violations: list[str] = []
    for sha in commits:
        message = commit_message(sha)
        if any(pattern.search(message) for pattern in FORBIDDEN):
            violations.append(f"{sha[:12]} {message.splitlines()[0]}")

    if violations:
        sys.stderr.write(
            "Attribution trailers are not allowed in PR commits:\n"
            + "\n".join(f"  - {line}" for line in violations)
            + "\n"
        )
        return 1

    sys.stdout.write(f"checked {len(commits)} commit(s); no attribution trailers\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
