#!/usr/bin/env python3
"""Rebuild CRIT-APP / ML as content-first open-source ebooks (nav + landing + rebrand)."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# Usage: build_ebook_site.py CRIT-APP|ML


def chapter_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            t = re.sub(r"^Chapter\s+\d+\.\s*", "", t, flags=re.I)
            return t
    return path.stem


def strip_meta_banners(md: str) -> str:
    # Remove bulky disclaimer divs; keep content clean
    md = re.sub(
        r'<div class="disclaimer-banner"[^>]*>.*?</div>\s*',
        "",
        md,
        flags=re.S | re.I,
    )
    md = md.replace("Web Edition", "open-source ebook")
    md = md.replace("web edition", "open-source ebook")
    md = md.replace("Web edition", "open-source ebook")
    # Soften meta clinical frame headings that sound like docs chrome
    md = md.replace("## Web Edition clinical frame", "## Opening")
    md = md.replace("## open-source ebook clinical frame", "## Opening")
    return md


def write_css(path: Path, accent: str) -> None:
    # accent: indigo | teal
    if accent == "teal":
        primary = "#0d9488"
        primary_deep = "#0f766e"
        glow = "rgba(13, 148, 136, 0.18)"
        hero_grad = "linear-gradient(135deg, #042f2e 0%, #0f766e 48%, #5eead4 140%)"
    else:
        primary = "#4f46e5"
        primary_deep = "#3730a3"
        glow = "rgba(79, 70, 229, 0.16)"
        hero_grad = "linear-gradient(135deg, #0b1026 0%, #312e81 45%, #818cf8 130%)"

    css = f"""/* Open-source ebook — reading-first design */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {{
  --ebook-primary: {primary};
  --ebook-deep: {primary_deep};
  --ebook-glow: {glow};
  --ebook-serif: "Cormorant Garamond", "Palatino Linotype", Palatino, Georgia, serif;
  --ebook-sans: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --ebook-measure: 42rem;
}}

/* Hide default tab chrome / reduce meta noise */
.md-header {{
  background: rgba(15, 18, 32, 0.82) !important;
  backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  box-shadow: none;
}}
.md-tabs {{ display: none !important; }}
.md-header__topic .md-ellipsis {{
  font-family: var(--ebook-serif);
  font-weight: 600;
  letter-spacing: 0.02em;
}}

.md-main {{
  background:
    radial-gradient(1200px 500px at 10% -10%, var(--ebook-glow), transparent 55%),
    radial-gradient(900px 400px at 100% 0%, rgba(201, 162, 39, 0.08), transparent 50%);
}}

.md-sidebar {{
  background: transparent;
}}
.md-nav__title {{
  font-family: var(--ebook-serif);
  font-size: 0.95rem;
  color: var(--ebook-deep);
}}
.md-nav__link {{
  font-family: var(--ebook-sans);
  font-size: 0.84rem;
  border-radius: 0.45rem;
}}
.md-nav__link--active {{
  color: var(--ebook-primary) !important;
  font-weight: 600;
}}

.md-content {{
  max-width: calc(var(--ebook-measure) + 4rem);
}}
.md-typeset {{
  font-family: var(--ebook-sans);
  font-size: 0.98rem;
  line-height: 1.72;
  color: inherit;
  text-align: left;
}}
.md-typeset h1 {{
  font-family: var(--ebook-serif);
  font-weight: 700;
  font-size: 2.35rem;
  line-height: 1.15;
  letter-spacing: -0.02em;
  margin-bottom: 0.85rem;
  border-bottom: none;
}}
.md-typeset h2 {{
  font-family: var(--ebook-serif);
  font-weight: 600;
  font-size: 1.55rem;
  margin-top: 2.2rem;
  padding-top: 0.4rem;
  border-top: 1px solid rgba(127,127,127,0.18);
}}
.md-typeset h3 {{
  font-family: var(--ebook-serif);
  font-weight: 600;
  font-size: 1.22rem;
}}
.md-typeset p,
.md-typeset li {{
  max-width: var(--ebook-measure);
}}
.md-typeset a {{
  color: var(--ebook-primary);
}}

/* Hero card on intro */
.ebook-hero {{
  margin: 0 0 2rem;
  padding: 2.1rem 2rem 1.85rem;
  border-radius: 1.15rem;
  background: {hero_grad};
  color: #f8fafc;
  box-shadow: 0 24px 50px rgba(15, 23, 42, 0.22);
  position: relative;
  overflow: hidden;
}}
.ebook-hero::after {{
  content: "";
  position: absolute;
  inset: auto -20% -40% 40%;
  height: 180px;
  background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%);
  pointer-events: none;
}}
.ebook-hero .eyebrow {{
  font-family: var(--ebook-sans);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  opacity: 0.82;
  margin: 0 0 0.65rem;
}}
.ebook-hero h1 {{
  color: #fff !important;
  margin: 0 0 0.75rem !important;
  font-size: 2.5rem !important;
  border: none !important;
}}
.ebook-hero p {{
  margin: 0;
  max-width: 38rem;
  font-size: 1.05rem;
  line-height: 1.55;
  opacity: 0.94;
}}
.ebook-hero .meta {{
  margin-top: 1.15rem;
  font-size: 0.82rem;
  opacity: 0.78;
}}

.ebook-start {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin: 1.25rem 0 2rem;
}}
.ebook-start a {{
  display: inline-block;
  padding: 0.55rem 1rem;
  border-radius: 999px;
  background: var(--ebook-primary);
  color: #fff !important;
  text-decoration: none !important;
  font-weight: 600;
  font-size: 0.9rem;
  box-shadow: 0 8px 20px var(--ebook-glow);
}}
.ebook-start a.secondary {{
  background: transparent;
  color: var(--ebook-deep) !important;
  border: 1px solid rgba(127,127,127,0.35);
  box-shadow: none;
}}

.chapter-list {{
  list-style: none;
  padding: 0;
  margin: 0 0 2rem;
}}
.chapter-list li {{
  margin: 0;
  border-bottom: 1px solid rgba(127,127,127,0.14);
}}
.chapter-list a {{
  display: flex;
  gap: 0.85rem;
  align-items: baseline;
  padding: 0.72rem 0.2rem;
  text-decoration: none !important;
  color: inherit !important;
  transition: color 0.15s ease, padding-left 0.15s ease;
}}
.chapter-list a:hover {{
  color: var(--ebook-primary) !important;
  padding-left: 0.35rem;
}}
.chapter-list .num {{
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--ebook-primary);
  min-width: 2rem;
  font-size: 0.86rem;
}}

.md-typeset img {{
  display: block;
  margin: 1.25em auto;
  max-width: 100%;
  border-radius: 0.65rem;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
}}
.md-typeset table {{
  font-size: 0.88rem;
  border-radius: 0.5rem;
  overflow: hidden;
}}
.md-typeset code {{
  font-size: 0.86em;
}}

/* Opening callouts in chapters */
.md-typeset h2#opening + p {{
  padding: 0.95rem 1.1rem;
  border-left: 3px solid var(--ebook-primary);
  background: var(--ebook-glow);
  border-radius: 0 0.65rem 0.65rem 0;
  font-size: 0.98rem;
}}

.md-footer {{
  background: transparent !important;
  border-top: 1px solid rgba(127,127,127,0.16);
}}
.md-footer-meta {{
  background: transparent !important;
}}
.md-copyright {{
  font-size: 0.75rem;
  opacity: 0.85;
}}

[data-md-color-scheme="slate"] .md-main {{
  background:
    radial-gradient(1000px 420px at 0% 0%, var(--ebook-glow), transparent 55%),
    radial-gradient(800px 360px at 100% 10%, rgba(201, 162, 39, 0.06), transparent 50%);
}}
[data-md-color-scheme="slate"] .md-typeset h2 {{
  border-top-color: rgba(255,255,255,0.08);
}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(css, encoding="utf-8")


def rebuild(root: Path, kind: str) -> None:
    docs = root / "docs"
    curr = docs / "curriculum"
    accent = "teal" if kind == "ML" else "indigo"

    # Move meta pages out of public docs (keep in repo under _meta/)
    meta_dir = root / "_meta"
    meta_dir.mkdir(exist_ok=True)
    for name in [
        "about.md",
        "how-to-use.md",
        "update-protocol.md",
        "ORIGINALITY.md",
        "ORIGINALITY_SWARM_VERDICT.md",
    ]:
        src = docs / name
        if src.exists():
            shutil.move(str(src), str(meta_dir / name))
    # hide swarm audit from public site
    swarm = docs / "_swarm_audit"
    if swarm.exists():
        dest = meta_dir / "_swarm_audit"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(swarm), str(dest))

    # Clean chapter meta chrome
    chapters = sorted(
        [
            p
            for p in curr.glob("*.md")
            if p.name != "index.md" and not p.name.startswith("00b-how-to-use")
        ],
        key=lambda p: p.name,
    )
    for p in chapters:
        text = strip_meta_banners(p.read_text(encoding="utf-8", errors="replace"))
        p.write_text(text, encoding="utf-8")

    # Optional: drop ML how-to-use chapter from tree by not listing it
    # curriculum index → simple TOC kept for internal links but not in nav
    toc_items = []
    for i, p in enumerate(chapters, 1):
        title = chapter_title(p)
        toc_items.append((i, title, p.name))

    if kind == "CRIT-APP":
        site_name = "Critical Appraisal for Neurologists"
        site_desc = "Open-source ebook on critical appraisal for neurology and stroke."
        site_url = "https://rkalani1.github.io/CRIT-APP/"
        repo = "rkalani1/CRIT-APP"
        intro_title = "Critical Appraisal for Neurologists"
        intro_blurb = (
            "An open-source ebook for stroke-aware critical appraisal—"
            "prediction versus causation, absolute effects, bias, trials, "
            "observational designs, diagnostics, and evidence-to-practice."
        )
        sibling = ("ML open-source ebook", "https://rkalani1.github.io/ML/")
        first = chapters[0].name if chapters else "index.md"
    else:
        site_name = "Machine Learning & AI for Neurologists"
        site_desc = "Open-source ebook on ML and AI literacy for neurology and stroke."
        site_url = "https://rkalani1.github.io/ML/"
        repo = "rkalani1/ML"
        intro_title = "Machine Learning & AI for Neurologists"
        intro_blurb = (
            "An open-source ebook on machine learning for neurologists—"
            "foundations, supervised and deep learning, appraisal of models, "
            "and clinical judgment with data."
        )
        sibling = ("Critical appraisal open-source ebook", "https://rkalani1.github.io/CRIT-APP/")
        first = chapters[0].name if chapters else "index.md"

    # Landing
    lines = [
        f'<div class="ebook-hero" markdown="1">\n',
        f'<p class="eyebrow">Open-source ebook</p>\n',
        f"# {intro_title}\n\n",
        f"{intro_blurb}\n\n",
        f'<p class="meta">CC BY 4.0 · Educational only — not medical advice · '
        f'<a href="https://github.com/{repo}" style="color:#fff;text-decoration:underline">'
        f"Source on GitHub</a></p>\n",
        f"</div>\n\n",
        f'<div class="ebook-start" markdown="1">\n',
        f'[Begin reading](curriculum/{first})\n',
        f'<a class="secondary" href="{sibling[1]}">{sibling[0]}</a>\n',
        f"</div>\n\n",
        f"## Contents\n\n",
        f'<ul class="chapter-list">\n',
    ]
    for n, title, fname in toc_items:
        lines.append(
            f'<li><a href="curriculum/{fname}"><span class="num">{n:02d}</span>'
            f"<span>{title}</span></a></li>\n"
        )
    lines.append("</ul>\n")
    (docs / "index.md").write_text("".join(lines), encoding="utf-8")

    # curriculum index minimal
    (curr / "index.md").write_text(
        "# Contents\n\n" + "\n".join(f"{n}. [{t}]({f})" for n, t, f in toc_items) + "\n",
        encoding="utf-8",
    )

    write_css(docs / "stylesheets" / "extra.css", accent)

    # mkdocs.yml
    nav_lines = ["nav:", "  - Introduction: index.md", "  - Book:"]
    for n, title, fname in toc_items:
        short = title if len(title) <= 52 else title[:49] + "…"
        # escape quotes in yaml
        short = short.replace('"', "'")
        nav_lines.append(f'    - "{n:02d}. {short}": curriculum/{fname}')

    yml = f"""site_name: {site_name}
site_description: {site_desc}
site_url: {site_url}
repo_url: https://github.com/{repo}
repo_name: {repo}
edit_uri: ""
copyright: >
  Open-source ebook · CC BY 4.0 content · ISC site code · Educational only — not medical advice.

theme:
  name: material
  font:
    text: Source Sans 3
    code: JetBrains Mono
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: {"teal" if accent == "teal" else "indigo"}
      accent: amber
      toggle:
        icon: material/weather-night
        name: Dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: {"teal" if accent == "teal" else "indigo"}
      accent: amber
      toggle:
        icon: material/weather-sunny
        name: Light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.expand
    - navigation.top
    - navigation.footer
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
      link: https://github.com/{repo}

{chr(10).join(nav_lines)}
"""
    (root / "mkdocs.yml").write_text(yml, encoding="utf-8")
    print(f"{kind}: {len(toc_items)} chapters, meta archived to _meta/, ebook CSS written")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: build_ebook_site.py CRIT-APP|ML|both")
        return 2
    arg = sys.argv[1]
    roots = []
    if arg in ("CRIT-APP", "both"):
        roots.append((Path(r"C:\Users\rkala\CRIT-APP"), "CRIT-APP"))
    if arg in ("ML", "both"):
        roots.append((Path(r"C:\Users\rkala\ML"), "ML"))
    for root, kind in roots:
        rebuild(root, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
