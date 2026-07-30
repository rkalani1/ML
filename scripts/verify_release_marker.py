#!/usr/bin/env python3
"""Verify one fetched HTML page exposes exactly one active release marker."""
from __future__ import annotations

import argparse
from pathlib import Path

from release_marker import release_revision_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-file", type=Path, required=True)
    parser.add_argument("--expected-release-sha", required=True)
    args = parser.parse_args()
    try:
        source = args.html_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"LIVE_REVISION_ERROR unable to read fetched HTML: {exc}")
        return 1
    errors = release_revision_errors(
        source,
        args.expected_release_sha.strip(),
    )
    if errors:
        for error in errors:
            print(f"LIVE_REVISION_ERROR {error}")
        return 1
    print(f"LIVE_REVISION_OK {args.expected_release_sha.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
