#!/usr/bin/env python3
"""Fail CI if third-party expressive-content residue appears in the repository."""
from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RULES = [
    (r"(?i)reproduced with " r"permission", "permissioned reprint language"),
    (r"(?i)all rights " r"reserved", "publisher rights line"),
    (r"(?i)this article is protected by " r"copyright", "journal boilerplate"),
    (r"(?i)downloaded from\s+https?://", "publisher download residue"),
    (r"(?i)figure\s+\d+\s+from\s+(the\s+)?(trial|paper|study)", "lifted figure reference"),
    (r"(?i)table\s+\d+\s+reproduced", "reproduced table"),
]
CROSS_LINE_RULES = [
    (r"(?i)\breproduced\s+with\s+permission\b", "permissioned reprint language"),
    (r"(?i)\ball\s+rights\s+reserved\b", "publisher rights line"),
    (
        r"(?i)\bthis\s+article\s+is\s+protected\s+by\s+copyright\b",
        "journal boilerplate",
    ),
    (r"(?i)\bdownloaded\s+from\s+https?://", "publisher download residue"),
    (
        r"(?i)\bfigure\s+\d+\s+from\s+(?:the\s+)?(?:trial|paper|study)\b",
        "lifted figure reference",
    ),
    (r"(?i)\btable\s+\d+\s+reproduced\b", "reproduced table"),
]
ALLOWED_EXACT_LINES = {
    (
        "docs/assets/fonts/OFL-1.1.txt",
        'Name "Source". All Rights ' "Reserved. Source is a trademark of Adobe in the",
    ),
    (
        "THIRD_PARTY_NOTICES.md",
        "Copyright © 2014-present, Tom Christie. All rights " "reserved.",
    ),
}
SKIP_ROOTS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "site",
}


def is_abstract_heading(line: str) -> bool:
    """Recognize standalone Abstract headings after supported Markdown rendering."""
    candidate = html.unescape(line.strip())
    while candidate.startswith(">"):
        candidate = candidate[1:].lstrip()

    raw_heading = re.fullmatch(
        r"<h([1-6])(?:\s+[^>]*)?>(.*?)</h\1>",
        candidate,
        flags=re.IGNORECASE,
    )
    if raw_heading:
        candidate = raw_heading.group(2)
    else:
        atx_heading = re.fullmatch(r"#{1,6}(?:\s+|$)(.*)", candidate)
        if atx_heading:
            candidate = atx_heading.group(1)
        candidate = re.sub(r"\s+#+\s*$", "", candidate)
        candidate = re.sub(r"\s*\{[^{}\n]*\}\s*$", "", candidate)

    candidate = re.sub(
        r"</?(?:b|code|em|i|span|strong)(?:\s+[^>]*)?>",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"[*_`~]", "", candidate)
    return candidate.strip().casefold() == "abstract"


def normalized_visible_text(relative: str, text: str) -> str:
    """Collapse visible prose so Markdown formatting and line wraps cannot evade rules."""
    retained_lines = [
        line
        for line in text.splitlines()
        if (relative, line.strip()) not in ALLOWED_EXACT_LINES
    ]
    candidate = html.unescape("\n".join(retained_lines))
    candidate = re.sub(r"<!--.*?-->", " ", candidate, flags=re.DOTALL)
    candidate = re.sub(
        r"<(https?://[^<>\s]+)>",
        r"\1",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"<[^>]+>", " ", candidate)
    candidate = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", candidate)
    candidate = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
    candidate = re.sub(r"[*_`~#>{}|]", " ", candidate)
    return re.sub(r"\s+", " ", candidate).strip()


def repository_candidates(root: Path = ROOT) -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).parts[0] not in SKIP_ROOTS
        ]
    return [
        root / Path(item.decode("utf-8", errors="surrogateescape"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def repository_text_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for path in repository_candidates(root):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            sample = path.read_bytes()[:8192]
        except OSError:
            continue
        if b"\x00" in sample:
            continue
        files.append(path)
    return files


def scan(root: Path = ROOT) -> list[str]:
    bad: list[str] = []
    for p in repository_text_files(root):
        relative = p.relative_to(root).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        found_rule_names: set[str] = set()
        for i, line in enumerate(text.splitlines(), 1):
            if (relative, line.strip()) in ALLOWED_EXACT_LINES:
                continue
            if is_abstract_heading(line):
                bad.append(
                    f"{relative}:{i}: abstract heading block: {line[:100]}"
                )
            for pat, name in RULES:
                if re.search(pat, line):
                    bad.append(f"{relative}:{i}: {name}: {line[:100]}")
                    found_rule_names.add(name)
        normalized = normalized_visible_text(relative, text)
        for pattern, name in CROSS_LINE_RULES:
            if name not in found_rule_names and re.search(pattern, normalized):
                bad.append(f"{relative}: normalized prose: {name}")
    return bad


def main() -> int:
    bad = scan()
    if bad:
        print("THIRD-PARTY RESIDUE SCAN FAIL")
        for b in bad[:50]:
            print(b)
        return 1
    print("THIRD-PARTY RESIDUE SCAN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
