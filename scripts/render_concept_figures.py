#!/usr/bin/env python3
"""
Convert !!! note "Figure concept (text diagram) ..." admonitions into original PNG cards
and replace them with Markdown images. Deterministic, code-drawn only.
"""
from __future__ import annotations

import hashlib
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CURR = ROOT / "docs" / "curriculum"
FIG = ROOT / "docs" / "assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEAL = "#0d9488"
DEEP = "#0f766e"
INK = "#0f172a"
SOFT = "#f0fdfa"
GOLD = "#c9a227"

# Multiline title forms appear in some files
ADMON_RE = re.compile(
    r"!!! note \"Figure concept \(text diagram\)\s*([^\"]+)\"\s*\n\n((?:    .*(?:\n|$))+)",
    re.M,
)


def slug(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s.strip())
    s = s.strip("_").lower()
    return s[:60] or "fig"


def body_from_indent(block: str) -> str:
    lines = []
    for line in block.splitlines():
        if line.startswith("    "):
            lines.append(line[4:])
        elif line.strip() == "":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def render_card(fig_id: str, body: str, out: Path) -> None:
    title = f"Figure {fig_id.strip()}"
    wrapped = textwrap.fill(body, width=62)
    # height scales with lines
    nlines = max(3, wrapped.count("\n") + 1)
    h = 1.6 + 0.28 * nlines
    fig, ax = plt.subplots(figsize=(8.2, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.25, 0.4),
            9.5,
            9.1,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor=SOFT,
            edgecolor=TEAL,
            linewidth=2.2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.25, 7.9),
            9.5,
            1.6,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=DEEP,
            edgecolor="none",
        )
    )
    ax.text(5, 8.7, title, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(
        0.7,
        7.3,
        wrapped,
        ha="left",
        va="top",
        color=INK,
        fontsize=10.5,
        linespacing=1.35,
        family="DejaVu Sans",
    )
    ax.text(9.4, 0.75, "original teaching graphic", ha="right", va="center", color=GOLD, fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        fig_id = m.group(1).replace("\n", " ").strip()
        body = body_from_indent(m.group(2))
        h = hashlib.sha1(f"{path.name}:{fig_id}:{body}".encode()).hexdigest()[:8]
        fname = f"ml_concept_{slug(fig_id)}_{h}.png"
        out = FIG / fname
        if not out.exists():
            render_card(fig_id, body, out)
        count += 1
        cap = body.split("\n")[0][:120]
        return (
            f"![{title_escape(fig_id)}: {cap}](../assets/figures/{fname})\n\n"
            f"*Figure {fig_id} — original teaching graphic.*\n"
        )

    def title_escape(s: str) -> str:
        return s.replace("[", "(").replace("]", ")")

    new, n = ADMON_RE.subn(repl, text)
    # Also handle broken multiline titles like "0.\n13"
    if n == 0 and "Figure concept (text diagram)" in text:
        # looser parse
        loose = re.compile(
            r"!!! note \"Figure concept \(text diagram\)\s*([\s\S]*?)\"\s*\n\n((?:    .*(?:\n|$))+)",
            re.M,
        )

        def repl2(m: re.Match) -> str:
            nonlocal count
            fig_id = re.sub(r"\s+", " ", m.group(1)).strip()
            body = body_from_indent(m.group(2))
            h = hashlib.sha1(f"{path.name}:{fig_id}:{body}".encode()).hexdigest()[:8]
            fname = f"ml_concept_{slug(fig_id)}_{h}.png"
            out = FIG / fname
            if not out.exists():
                render_card(fig_id, body, out)
            count += 1
            cap = body.split("\n")[0][:120]
            return (
                f"![{fig_id}: {cap}](../assets/figures/{fname})\n\n"
                f"*Figure {fig_id} — original teaching graphic.*\n"
            )

        new, n = loose.subn(repl2, text)
    if n:
        path.write_text(new, encoding="utf-8")
    return n


def main() -> None:
    total = 0
    for p in sorted(CURR.glob("*.md")):
        n = process_file(p)
        if n:
            print(f"{p.name}: converted {n}")
            total += n
    # remaining?
    left = 0
    for p in CURR.glob("*.md"):
        left += p.read_text(encoding="utf-8").count("Figure concept (text diagram)")
    print(f"TOTAL_CONVERTED={total} REMAINING={left}")


if __name__ == "__main__":
    main()
