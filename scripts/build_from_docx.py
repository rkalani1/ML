#!/usr/bin/env python3
"""
Build ML Web Edition Markdown from Desktop DOCX curriculum.

Safety bar:
- Skip abstract-like blocks, permission/copyright/journal boilerplate
- Do not embed DOCX images (flag for code-redraw)
- Skip guideline-instrument-like full dumps when detectable
- Prepend unique Web Edition clinical frames + disclaimer banners
- Emit curriculum TOC, ORIGINALITY.md, and mkdocs.yml nav
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CURR = DOCS / "curriculum"
DOCX_PATH = Path(r"C:\Users\rkala\Desktop\ML-AI text.docx")

SKIP_H1 = {
    "table of contents",
    "part 0 — mathematical foundations",
    "part i — introduction to machine learning and artificial intelligence",
    "part ii — unsupervised learning",
    "part iii — data engineering",
    "part iv — supervised learning",
    "part v — neural networks and deep learning",
    "part vi — reinforcement learning",
    "part vii — other essential topics",
}

# Site-level pages already cover these; still convert as curriculum if useful
# Preface and How to Use become early curriculum pages.

FORBIDDEN_LINE = [
    re.compile(r"(?i)reproduced with permission"),
    re.compile(r"(?i)all rights reserved"),
    re.compile(r"(?i)this article is protected by copyright"),
    re.compile(r"(?i)downloaded from\s+https?://"),
    re.compile(r"(?i)copyright\s*©"),
    re.compile(r"(?i)^\s*keywords\s*:"),
    re.compile(r"(?i)^\s*correspondence\s*:"),
    re.compile(r"(?i)^\s*received\s*:"),
    re.compile(r"(?i)^\s*accepted\s*:"),
    re.compile(r"(?i)^\s*funding\s*:"),
    re.compile(r"(?i)^\s*conflicts?\s+of\s+interest"),
    re.compile(r"(?i)used with permission"),
    re.compile(r"(?i)permission from (the )?(publisher|journal|elsevier|springer|wiley|ieee)"),
]

ABSTRACT_HEAD = re.compile(r"(?i)^\s*abstract\s*$")
ABSTRACT_START = re.compile(r"(?i)^\s*abstract\s*[:.\-—]")
GUIDELINE_DUMP = re.compile(
    r"(?i)(full\s+checklist\s+items?\s+of\s+(consort|tripod|stard|prisma)|"
    r"reproduced\s+from\s+(aha|asa|aha/asa)\s+guideline)"
)

CLINICAL_FRAMES = {
    "preface": (
        "A telestroke consult ends. The hub radiologist mentions a new large-vessel "
        "occlusion detector with a ‘state-of-the-art AUROC.’ The spoke hospital asks "
        "whether to buy it this quarter. This book exists so that conversation starts "
        "with definitions, data, and decision impact—not vendor vocabulary."
    ),
    "how-to-use-this-textbook": (
        "You have forty-five free minutes between clinic and journal club. This chapter "
        "maps efficient paths through the curriculum so you spend time on the literacy "
        "layer that actually changes how you read a stroke-imaging model paper."
    ),
    "chapter-0": (
        "A fellow freezes at a gradient step in a methods appendix for an ICH expansion "
        "model. The clinical question is still bedside-valid, but the math barrier is "
        "blocking appraisal. Chapter 0 rebuilds the minimum calculus and linear algebra "
        "needed to read ML without surrendering clinical judgment."
    ),
    "chapter-1": (
        "An overnight resident asks whether ‘AI’ can rule out stroke on non-contrast CT. "
        "Before architecture debates begin, this chapter forces the supervised vs "
        "unsupervised vs reinforcement taxonomy and the prediction-versus-causation "
        "boundary that stroke teams violate most often."
    ),
    "chapter-2": (
        "A quality dashboard shows a green line for door-to-needle. A density plot of "
        "NIHSS looks normal until you split by transfer status. Visualization is not "
        "decoration—it is the first place site shift and selection bias become visible."
    ),
    "chapter-3": (
        "A hemorrhage-risk model quotes sensitivity 0.92 and specificity 0.88. Without "
        "prevalence, Bayes, and calibration language, those numbers are theater. This "
        "chapter rebuilds the probability spine for clinical ML consumers."
    ),
    "chapter-4": (
        "A research coordinator proposes clustering ‘phenotypes’ of cryptogenic stroke "
        "from EHR labs and imaging codes. Clustering can discover structure—or invent "
        "comforting noise. This chapter teaches how to ask which one you just saw."
    ),
    "chapter-5": (
        "A claims-analysis team wants frequent co-prescription patterns after TIA and a "
        "simple retrieval system for similar prior cases. Itemsets and sequences are "
        "powerful; they are also experts at encoding practice fashion as ‘knowledge.’"
    ),
    "chapter-6": (
        "Two sites train the same classifier for early neurologic deterioration. One "
        "encodes time-to-CT as minutes; the other as free-text. Feature engineering is "
        "where most ‘AI magic’ quietly lives—and where leakage hides."
    ),
    "chapter-7": (
        "A multiparametric MRI radiomics pipeline has 1,200 features and 180 patients. "
        "Dimensionality reduction is not optional aesthetics; it is survival against "
        "overfitting and irreproducible stroke biomarkers."
    ),
    "chapter-8": (
        "A lab wants to regress 90-day mRS from admission labs. Linear models still "
        "discipline thinking about targets, residuals, and collinearity before anyone "
        "reaches for a neural net."
    ),
    "chapter-9": (
        "A binary classifier flags LVO on CTA with impressive accuracy in the training "
        "center. Classification literacy means thresholds, class imbalance, costs of "
        "false negatives on the stroke pathway, and external validation—not leaderboard "
        "ego."
    ),
    "chapter-10": (
        "A vendor demo animates a CNN highlighting an infarct core. Deep learning can "
        "extract hierarchical image features; it can also memorize scanner fingerprints. "
        "This chapter gives neurologists the vocabulary to interrogate both."
    ),
    "chapter-11": (
        "Labeled stroke images are scarce; unlabeled scans are abundant. Self-supervised "
        "pretraining looks attractive until domain shift between scanners and protocols "
        "is measured. Read this before believing ‘we barely needed labels.’"
    ),
    "chapter-12": (
        "A multimodal model claims to fuse note text, DWI, and audio dysphagia screens. "
        "Cross-modal architectures are exciting; they also multiply failure modes. This "
        "chapter separates capability demos from clinically transportable systems."
    ),
    "chapter-13": (
        "A simulation lab proposes reinforcement learning for sequential BP targets after "
        "thrombolysis. RL requires a reward design that does not quietly optimize the "
        "wrong clinical trade-off. Stroke care is not Atari."
    ),
    "chapter-14": (
        "Your hospital’s edge device cannot run a 7-billion-parameter model during a code "
        "stroke. Compression, distillation, and efficient architectures are deployment "
        "medicine—not just engineering fashion."
    ),
    "chapter-15": (
        "A network analysis links hospitals, transfer patterns, and outcome codes. Graph "
        "methods can reveal systems structure; they can also launder confounding through "
        "edges. Literacy here protects both science and equity claims."
    ),
    "chapter-16": (
        "Missing NIHSS, duplicated MRNs, label drift after a documentation change—this is "
        "the real curriculum of clinical ML. Architecture papers rarely fail first; data "
        "pipelines do."
    ),
    "closing-synthesis": (
        "You finish a model paper and a methods appendix in one sitting. Synthesis means "
        "you can state the decision, the data generation process, the evaluation design, "
        "and the residual risks in one paragraph a charge nurse would respect."
    ),
    "selected-glossary": (
        "Journal club language collapses when people use ‘AI,’ ‘algorithm,’ and ‘model’ "
        "interchangeably. The glossary is a shared lexicon for stroke services that want "
        "precise disagreement rather than vague awe."
    ),
}

DEFAULT_FRAME = (
    "A stroke service is evaluating whether a new machine-learning tool should change "
    "pathway timing, imaging selection, or secondary-prevention counseling. Use this "
    "chapter to separate mathematical capability from clinical decision impact."
)

DISCLAIMER = (
    '<div class="disclaimer-banner" markdown="1">\n'
    "**Web Edition — original teaching text.** Educational only; not medical advice. "
    "No commercial handbook prose, paper abstracts, or publisher figures.\n"
    "</div>\n"
)


def slugify(s: str) -> str:
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[-\s]+", "-", s.strip()).lower()
    return s[:80].strip("-")


def chapter_key(title: str) -> str:
    t = title.strip()
    m = re.match(r"(?i)^chapter\s+(\d+)\b", t)
    if m:
        return f"chapter-{int(m.group(1))}"
    low = t.lower()
    if low.startswith("preface"):
        return "preface"
    if "how to use" in low:
        return "how-to-use-this-textbook"
    if low.startswith("closing"):
        return "closing-synthesis"
    if "glossary" in low:
        return "selected-glossary"
    return slugify(t)


def file_slug(order: int, title: str) -> str:
    key = chapter_key(title)
    # Prefer numeric chapter numbers when present
    m = re.match(r"(?i)^chapter\s+(\d+)\s*:\s*(.+)$", title.strip())
    if m:
        n = int(m.group(1))
        return f"{n:02d}-{slugify(m.group(2))}"
    if key == "preface":
        return "00a-preface"
    if key == "how-to-use-this-textbook":
        return "00b-how-to-use-this-textbook"
    if key == "closing-synthesis":
        return "17-closing-synthesis-senior-practice"
    if key == "selected-glossary":
        return "18-selected-glossary"
    return f"{order:02d}-{slugify(title)}"


def display_title(title: str) -> str:
    t = title.strip()
    m = re.match(r"(?i)^chapter\s+(\d+)\s*:\s*(.+)$", t)
    if m:
        return f"Chapter {int(m.group(1))}. {m.group(2).strip()}"
    return t


def is_forbidden_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    for pat in FORBIDDEN_LINE:
        if pat.search(t):
            return True
    if GUIDELINE_DUMP.search(t):
        return True
    return False


def looks_like_abstract_block(text: str) -> bool:
    t = text.strip()
    if ABSTRACT_HEAD.match(t) or ABSTRACT_START.match(t):
        return True
    # Dense abstract-ish blocks: very long para with background/methods/results/conclusions cues
    low = t.lower()
    cues = sum(
        1
        for c in (
            "background:",
            "methods:",
            "results:",
            "conclusions:",
            "objective:",
            "purpose:",
        )
        if c in low
    )
    if cues >= 3 and len(t) > 400:
        return True
    return False


def para_has_drawing(p: Paragraph) -> bool:
    el = p._element
    if el.findall(".//" + qn("w:drawing")):
        return True
    if el.findall(".//" + qn("w:pict")):
        return True
    return False


def escape_md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def table_to_md(table: Table) -> str | None:
    rows = []
    for row in table.rows:
        cells = [escape_md_cell(c.text or "") for c in row.cells]
        # Collapse duplicate merged-cell repeats common in python-docx
        cleaned = []
        prev = None
        for c in cells:
            if c == prev:
                continue
            cleaned.append(c)
            prev = c
        if not any(cleaned):
            continue
        rows.append(cleaned)
    if len(rows) < 2:
        return None
    # Skip huge tables that look like instrument dumps
    if len(rows) > 40 or max(len(r) for r in rows) > 12:
        return None
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    # If header looks empty, synthesize
    if not any(header):
        header = [f"Col{i+1}" for i in range(width)]
        body = norm
    else:
        body = norm[1:]
    # Skip if table body is mostly empty
    if not body:
        return None
    joined = " ".join(" ".join(r) for r in norm).lower()
    if GUIDELINE_DUMP.search(joined):
        return None
    if re.search(r"(?i)reproduced|permission|copyright", joined):
        return None
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for r in body:
        md += "| " + " | ".join(r) + " |\n"
    return md + "\n"


def iter_block_items(parent):
    """Yield paragraphs and tables in document order."""
    from docx.document import Document as DocClass
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import _Cell

    if isinstance(parent, DocClass):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("unsupported parent")
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def heading_level(style_name: str) -> int | None:
    if not style_name:
        return None
    m = re.match(r"(?i)^heading\s+(\d+)$", style_name)
    if m:
        return int(m.group(1))
    if style_name.lower() == "title":
        return 1
    return None


def scrub_text(text: str) -> str:
    t = text.replace("\u00a0", " ").replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def is_code_style(style_name: str) -> bool:
    s = (style_name or "").lower()
    return "code" in s or "source" in s


def convert() -> dict:
    if not DOCX_PATH.is_file():
        raise SystemExit(f"DOCX not found: {DOCX_PATH}")

    CURR.mkdir(parents=True, exist_ok=True)
    for old in CURR.glob("*.md"):
        old.unlink()

    doc = Document(str(DOCX_PATH))
    stats = {
        "chapters": 0,
        "skipped_paras": 0,
        "skipped_abstract": 0,
        "skipped_forbidden": 0,
        "skipped_images": 0,
        "skipped_tables": 0,
        "kept_tables": 0,
        "flags": [],
        "chapter_files": [],
    }

    # Split into H1 sections
    sections: list[dict] = []
    current: dict | None = None
    skip_abstract_until_heading = False

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style else "Normal"
            text = scrub_text(block.text or "")
            hl = heading_level(style)

            if hl == 1 and text:
                low = text.lower()
                if low in SKIP_H1 or low.startswith("part "):
                    current = None  # discard part dividers / TOC
                    skip_abstract_until_heading = False
                    continue
                current = {"title": text, "blocks": []}
                sections.append(current)
                skip_abstract_until_heading = False
                continue

            if current is None:
                continue

            if para_has_drawing(block):
                stats["skipped_images"] += 1
                # Keep caption text only if it is original teaching caption without permission language
                if text and not is_forbidden_line(text):
                    if not re.search(r"(?i)(reproduced|permission|copyright|from the (paper|trial|study))", text):
                        current["blocks"].append(
                            {
                                "type": "note",
                                "text": (
                                    f"*Figure omitted from Web Edition (DOCX-embedded image not "
                                    f"verified as curriculum-original). Caption text retained for "
                                    f"redraw backlog: {text[:200]}*"
                                ),
                            }
                        )
                    else:
                        stats["skipped_paras"] += 1
                        stats["skipped_forbidden"] += 1
                continue

            if not text:
                continue

            if ABSTRACT_HEAD.match(text) or ABSTRACT_START.match(text):
                stats["skipped_abstract"] += 1
                stats["skipped_paras"] += 1
                skip_abstract_until_heading = True
                continue

            if skip_abstract_until_heading:
                # stay in skip mode until next heading
                if hl and hl >= 2:
                    skip_abstract_until_heading = False
                else:
                    # end abstract skip if short heading-like line or new section cue
                    if looks_like_abstract_block(text) or len(text) > 80:
                        stats["skipped_abstract"] += 1
                        stats["skipped_paras"] += 1
                        continue
                    skip_abstract_until_heading = False

            if is_forbidden_line(text) or looks_like_abstract_block(text):
                stats["skipped_paras"] += 1
                if looks_like_abstract_block(text):
                    stats["skipped_abstract"] += 1
                else:
                    stats["skipped_forbidden"] += 1
                continue

            if hl and hl >= 2:
                current["blocks"].append({"type": "heading", "level": hl, "text": text})
            elif is_code_style(style):
                current["blocks"].append({"type": "code", "text": text})
            else:
                current["blocks"].append({"type": "para", "text": text})

        elif isinstance(block, Table):
            if current is None:
                continue
            md = table_to_md(block)
            if md is None:
                stats["skipped_tables"] += 1
            else:
                stats["kept_tables"] += 1
                current["blocks"].append({"type": "table_md", "text": md})

    # Write chapters
    nav_items = []
    toc_rows = []
    order = 0
    for sec in sections:
        title = sec["title"]
        blocks = sec["blocks"]
        if not blocks:
            continue
        order += 1
        fname = file_slug(order, title)
        # Stabilized order for non-numbered: use order index in filename already
        # Re-number preface/how-to/chapters more cleanly
        key = chapter_key(title)
        frame = CLINICAL_FRAMES.get(key, DEFAULT_FRAME)
        dtitle = display_title(title)

        # Collapse consecutive code lines into fences
        md_parts = [
            f"# {dtitle}\n\n",
            DISCLAIMER + "\n",
            "## Web Edition clinical frame\n\n",
            frame + "\n\n",
        ]

        in_code = False
        code_buf: list[str] = []

        def flush_code():
            nonlocal in_code, code_buf
            if in_code:
                md_parts.append("```\n" + "\n".join(code_buf) + "\n```\n\n")
                code_buf = []
                in_code = False

        for b in blocks:
            bt = b["type"]
            if bt == "code":
                in_code = True
                code_buf.append(b["text"])
                continue
            flush_code()
            if bt == "heading":
                level = min(max(int(b["level"]), 2), 4)
                # Avoid duplicating top-level title
                if b["text"].strip().lower() == title.strip().lower():
                    continue
                md_parts.append(f"{'#' * level} {b['text']}\n\n")
            elif bt == "para":
                md_parts.append(b["text"] + "\n\n")
            elif bt == "note":
                md_parts.append(b["text"] + "\n\n")
            elif bt == "table_md":
                md_parts.append(b["text"])
        flush_code()

        text_out = "".join(md_parts)
        # Final scrub pass against scanner patterns
        final_lines = []
        for line in text_out.splitlines():
            skip = False
            for pat, name in [
                (r"(?i)reproduced with permission", "permission"),
                (r"(?i)all rights reserved", "rights"),
                (r"(?i)this article is protected by copyright", "journal"),
                (r"(?i)downloaded from\s+https?://", "download"),
                (r"(?i)^\s*abstract\s*$", "abstract"),
                (r"(?i)figure\s+\d+\s+from\s+(the\s+)?(trial|paper|study)", "fig"),
                (r"(?i)table\s+\d+\s+reproduced", "table"),
            ]:
                if re.search(pat, line):
                    stats["flags"].append(f"{fname}: stripped residual ({name})")
                    skip = True
                    break
            if not skip:
                final_lines.append(line)
        text_out = "\n".join(final_lines).strip() + "\n"

        path = CURR / f"{fname}.md"
        path.write_text(text_out, encoding="utf-8")
        stats["chapters"] += 1
        stats["chapter_files"].append(str(path))

        # Nav label truncated like CRIT-APP
        label_core = dtitle
        if len(label_core) > 48:
            label_core = label_core[:47] + "…"
        nav_items.append((label_core, f"curriculum/{fname}.md"))
        toc_rows.append((dtitle, f"{fname}.md"))

    # curriculum index
    toc_lines = [
        "# Curriculum overview\n\n",
        f"{stats['chapters']} chapters of original Web Edition teaching material. "
        "Topic coverage spans mathematical foundations through clinical data challenges; "
        "expression is packaged for this free site with synthetic clinical frames.\n\n",
        "| Ch | Title |\n|---:|---|\n",
    ]
    for i, (dtitle, link) in enumerate(toc_rows, 1):
        toc_lines.append(f"| {i} | [{dtitle}]({link}) |\n")
    toc_lines.append(
        "\n## Sibling curriculum\n\n"
        "[Critical Appraisal for Neurologists — Web Edition]"
        "(https://rkalani1.github.io/CRIT-APP/)\n"
    )
    (CURR / "index.md").write_text("".join(toc_lines), encoding="utf-8")

    # ORIGINALITY.md
    orig = [
        "# ORIGINALITY manifest — ML Web Edition\n\n",
        "Built from Desktop DOCX curriculum (`ML-AI text.docx`) with hard filters: "
        "abstract-like blocks removed, permission/copyright/journal boilerplate skipped, "
        "DOCX-embedded images **not** published, oversized/suspect tables skipped.\n\n",
        f"- Chapters exported: **{stats['chapters']}**\n",
        f"- Paragraphs skipped (filters): **{stats['skipped_paras']}**\n",
        f"- Abstract-like skips: **{stats['skipped_abstract']}**\n",
        f"- Forbidden-line skips: **{stats['skipped_forbidden']}**\n",
        f"- DOCX images skipped (need code-redraw): **{stats['skipped_images']}**\n",
        f"- Tables kept / skipped: **{stats['kept_tables']}** / **{stats['skipped_tables']}**\n",
        f"- Residual scanner flags during scrub: **{len(stats['flags'])}**\n\n",
        "## Figure redraw backlog\n\n",
        "All figures that existed as embedded DOCX images were omitted from the Web Edition "
        "because chain-of-title as curriculum-original code-drawn assets could not be verified "
        "at export time. Future updates should add original SVG/PNG under "
        "`docs/assets/figures/` with `source: original` notes here.\n\n",
        "## Residual flags\n\n",
    ]
    if stats["flags"]:
        for f in stats["flags"][:50]:
            orig.append(f"- {f}\n")
    else:
        orig.append("- None recorded at export.\n")
    orig.append(
        "\n## Human residual\n\n"
        "Automated gates cannot prove absence of all third-party expressive content. "
        "Prefer skipping risky blocks over publishing them. Spot-check random chapters "
        "before treating the site as final. Educational disclaimer applies site-wide.\n"
    )
    (DOCS / "ORIGINALITY.md").write_text("".join(orig), encoding="utf-8")

    # mkdocs.yml
    write_mkdocs(nav_items)
    return stats


def write_mkdocs(nav_items: list[tuple[str, str]]) -> None:
    nav_lines = [
        "  - Home: index.md\n",
        "  - About & originality: about.md\n",
        "  - How to use: how-to-use.md\n",
        "  - Update protocol: update-protocol.md\n",
        "  - Curriculum:\n",
        "    - Overview: curriculum/index.md\n",
    ]
    for label, path in nav_items:
        # YAML-safe double quotes
        safe = label.replace('"', "'")
        nav_lines.append(f'    - "{safe}": {path}\n')

    content = f"""site_name: Machine Learning & AI for Neurologists — Web Edition
site_description: Free original curriculum for ML/AI literacy in neurology and stroke. Not a reprint of commercial textbooks.
site_url: https://rkalani1.github.io/ML/
repo_url: https://github.com/rkalani1/ML
repo_name: rkalani1/ML
edit_uri: edit/main/docs/
copyright: >
  Content © 2026 Rizwan Kalani / contributors · CC BY 4.0 · Site code ISC ·
  Educational only — not medical advice or institutional policy.

theme:
  name: material
  palette:
    - scheme: default
      primary: teal
      accent: amber
      toggle:
        icon: material/brightness-7
        name: Dark mode
    - scheme: slate
      primary: teal
      accent: amber
      toggle:
        icon: material/brightness-4
        name: Light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - toc.follow

plugins:
  - search

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight
  - pymdownx.arithmatex:
      generic: true
  - tables
  - toc:
      permalink: true
  - attr_list
  - md_in_html

extra_javascript:
  - javascripts/mathjax.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

extra_css:
  - stylesheets/extra.css

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/rkalani1/ML
  version:
    provider: mike
    default: web-edition

nav:
{''.join(nav_lines)}"""
    (ROOT / "mkdocs.yml").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    st = convert()
    print(f"Chapters: {st['chapters']}")
    print(f"Skipped paras: {st['skipped_paras']} (abstract={st['skipped_abstract']}, forbidden={st['skipped_forbidden']})")
    print(f"Images skipped: {st['skipped_images']}")
    print(f"Tables kept/skipped: {st['kept_tables']}/{st['skipped_tables']}")
    for p in st["chapter_files"]:
        print(" ", Path(p).name)
    sys.exit(0)
