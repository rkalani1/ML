#!/usr/bin/env python3
"""Deep originality / copy-paste risk scan for CRIT-APP and ML docs."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

REPOS = {
    "CRIT-APP": Path(r"C:\Users\rkala\CRIT-APP\docs"),
    "ML": Path(r"C:\Users\rkala\ML\docs"),
}
OUT = Path(r"C:\Users\rkala\OneDrive - UW\Codex\AI-Work-Sync\textbook-originality-swarm-scan.json")

PATTERNS = {
    "reproduced_permission": re.compile(
        r"(?i)reproduced with permission|with permission of|permission from the publisher"
    ),
    "copyright_line": re.compile(r"(?i)copyright\s*©|all rights reserved|this article is protected"),
    "abstract_block": re.compile(r"(?i)^\s*abstract\s*$", re.M),
    "journal_boilerplate": re.compile(
        r"(?i)downloaded from|wiley online library|springer nature|elsevier b\.?v\.?|nejm\.org|the lancet|jama network"
    ),
    "users_guides_brand": re.compile(
        r"(?i)users'? guides to the medical literature|guyatt\s+et\s+al"
    ),
    "guideline_instrument_dump": re.compile(
        r"(?i)consort 2010 checklist item|tripod checklist item\s*\d|grade working group full"
    ),
    "long_block_quote": re.compile(r"(?m)^>\s*.{120,}"),
    "figure_from_paper": re.compile(
        r"(?i)figure\s+\d+\s+(from|adapted from|reprinted from)\s+(the\s+)?(trial|paper|study|jama|lancet|nejm|stroke)"
    ),
    "table_reproduced": re.compile(r"(?i)table\s+\d+\s+(reproduced|adapted from)"),
    "isbn_or_edition": re.compile(r"(?i)\bISBN[-\s]?\d|\b\d{1,2}(st|nd|rd|th)\s+edition\b"),
    "phi_or_internal": re.compile(
        r"(?i)\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|@uw\.edu|harborview protocol|hmc stroke attending"
    ),
    "copy_paste_meta": re.compile(r"(?i)as mentioned in the textbook|see chapter \d+ of|this handbook"),
    "verbatim_doi": re.compile(r"doi:\s*10\.\d{4,}/[^\s\]\)]{5,}", re.I),
}

DISTINCTIVE = [
    r"evidence-based medicine is the conscientious",
    r"the practice of evidence-based medicine means integrating",
    r"are the results of the study valid\?",
    r"what are the results and will they help me",
    r"users' guides to the medical literature",
    r"elements of statistical learning",
    r"pattern recognition and machine learning",
    r"deep learning\s*\(goodfellow",
    r"hands-on machine learning with scikit-learn",
    r"an introduction to statistical learning",
]


def scan_repo(docs: Path) -> dict:
    flags: list[dict] = []
    files = list(docs.rglob("*.md"))
    total_chars = 0
    for p in files:
        if "_swarm_audit" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        total_chars += len(text)
        rel = str(p.relative_to(docs)).replace("\\", "/")
        for key, pat in PATTERNS.items():
            if key == "verbatim_doi":
                dois = pat.findall(text)
                if len(dois) >= 5:
                    flags.append(
                        {
                            "file": rel,
                            "line": 0,
                            "rule": "dense_doi_list",
                            "snippet": f"{len(dois)} DOIs",
                        }
                    )
                continue
            for m in pat.finditer(text):
                start = text.rfind("\n", 0, m.start()) + 1
                end = text.find("\n", m.end())
                if end < 0:
                    end = min(len(text), m.end() + 80)
                line = text[start:end].strip()[:160]
                line_no = text[: m.start()].count("\n") + 1
                flags.append(
                    {"file": rel, "line": line_no, "rule": key, "snippet": line}
                )
        for dpat in DISTINCTIVE:
            if re.search(dpat, text, re.I):
                flags.append(
                    {
                        "file": rel,
                        "line": 0,
                        "rule": "distinctive_textbook_phrase",
                        "snippet": dpat,
                    }
                )

    para_map: dict[str, list[str]] = collections.defaultdict(list)
    for p in files:
        if "_swarm_audit" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(docs)).replace("\\", "/")
        for para in re.split(r"\n\s*\n", text):
            para_n = re.sub(r"\s+", " ", para).strip()
            if len(para_n) < 280:
                continue
            if para_n.startswith("#") or para_n.startswith("|"):
                continue
            low = para_n.lower()
            if "disclaimer" in low or "web edition" in low or "educational only" in low:
                continue
            key = low[:400]
            para_map[key].append(rel)

    multi = []
    for k, v in para_map.items():
        if len(v) >= 2:
            multi.append(
                {"n": len(v), "files": sorted(set(v))[:6], "preview": k[:140]}
            )
    multi = sorted(multi, key=lambda x: -x["n"])[:25]

    return {
        "files": len([f for f in files if "_swarm_audit" not in f.parts]),
        "total_chars": total_chars,
        "flag_count": len(flags),
        "flags": flags,
        "duplicate_long_paras": multi,
        "rules": dict(collections.Counter(f["rule"] for f in flags)),
    }


def main() -> int:
    results = {}
    for name, docs in REPOS.items():
        if not docs.exists():
            results[name] = {"error": f"missing {docs}"}
            continue
        results[name] = scan_repo(docs)
        r = results[name]
        print(
            f"=== {name} files={r['files']} chars={r['total_chars']} "
            f"flags={r['flag_count']} dups={len(r['duplicate_long_paras'])}"
        )
        print(" rules:", r["rules"])
        for f in r["flags"][:15]:
            print(f"  - {f['rule']} {f['file']}:{f['line']} :: {f['snippet'][:100]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
