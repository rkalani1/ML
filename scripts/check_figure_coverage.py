#!/usr/bin/env python3
"""Assert every teaching chapter has >=1 visual (image or substantive table)."""
from __future__ import annotations
import re, sys
from pathlib import Path

def chapter_ok(p: Path) -> tuple[bool, str]:
    t = p.read_text(encoding="utf-8", errors="replace")
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", t)
    for rel in imgs:
        target = (p.parent / rel).resolve()
        if target.exists():
            return True, f"image:{target.name}"
    # substantive table: header + separator + >=2 data rows
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\|.+\|$", line) and i + 1 < len(lines) and re.match(r"^\|[\s:-]+\|$", lines[i + 1].replace(" ", "")):
            # count following table rows
            rows = 0
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows += 1
                j += 1
            if rows >= 2:
                return True, f"table:{rows}rows"
    return False, "none"

def main(root: Path) -> int:
    curr = root / "docs" / "curriculum"
    bad = []
    ok = []
    for p in sorted(curr.glob("*.md")):
        if p.name == "index.md":
            continue
        good, why = chapter_ok(p)
        (ok if good else bad).append((p.name, why))
    print(f"ROOT {root}")
    print(f"OK {len(ok)} BAD {len(bad)}")
    for n, w in bad:
        print("MISSING", n, w)
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
