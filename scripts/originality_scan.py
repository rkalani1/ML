#!/usr/bin/env python3
"""Fail CI if forbidden third-party expressive content patterns appear in docs/."""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

RULES = [
    (r"(?i)reproduced with permission", "permissioned reprint language"),
    (r"(?i)all rights reserved", "publisher rights line"),
    (r"(?i)this article is protected by copyright", "journal boilerplate"),
    (r"(?i)downloaded from\s+https?://", "publisher download residue"),
    (r"(?i)^\s*abstract\s*$", "abstract heading block"),
    (r"(?i)figure\s+\d+\s+from\s+(the\s+)?(trial|paper|study)", "lifted figure reference"),
    (r"(?i)table\s+\d+\s+reproduced", "reproduced table"),
]


def main() -> int:
    bad: list[str] = []
    for p in DOCS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for pat, name in RULES:
                if re.search(pat, line):
                    bad.append(f"{p.relative_to(DOCS)}:{i}: {name}: {line[:100]}")
    if bad:
        print("ORIGINALITY SCAN FAIL")
        for b in bad[:50]:
            print(b)
        return 1
    print("ORIGINALITY SCAN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
