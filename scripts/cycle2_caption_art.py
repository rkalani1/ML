#!/usr/bin/env python3
"""Cycle-2: wire high-value PNGs to bare captions; convert rest to concept callouts."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURR = ROOT / "docs" / "curriculum"

# Captions that receive real original PNGs (must exist under docs/assets/figures/)
PNG_WIRE = {
    "Figure 0.2.": (
        "../assets/figures/ml_fig_core_functions.png",
        "Core functions of machine learning (original teaching catalog).",
    ),
    "Figure 1.6.": (
        "../assets/figures/ml_fig_bias_capacity.png",
        "Training vs validation error versus model capacity (original).",
    ),
    "Figure 4.5.": (
        "../assets/figures/ml_fig_elbow_wss.png",
        "Elbow plot of WSS vs k on the six-point toy set (original).",
    ),
    "Figure 10.2.": (
        "../assets/figures/ml_fig_activations.png",
        "Activation functions: sigmoid, tanh, ReLU, leaky ReLU (original).",
    ),
    "Figure 11.3.": (
        "../assets/figures/ml_fig_triplet_ssl.png",
        "Triplet loss with chapter worked numbers (original).",
    ),
    "Figure 13.3.": (
        "../assets/figures/ml_fig_value_iteration.png",
        "Value iteration on the two-state MDP (original).",
    ),
}


def convert_paragraph(text: str) -> str:
    """Turn bare 'Figure x.y. ...' paragraphs into intentional concept callouts,
    or prefix high-value ones with markdown image embeds.
    """
    # Match a full paragraph that starts with Figure N.M.
    pattern = re.compile(
        r"(?m)^(Figure\s+(\d+\.\d+)\.\s+)(.+)$"
    )

    def repl(m: re.Match) -> str:
        full_prefix = m.group(1)  # "Figure 11.3. "
        num = m.group(2)
        body = m.group(3).strip()
        key = f"Figure {num}."
        # Skip if already converted or already has image immediately above
        start = m.start()
        prior = text[max(0, start - 200) : start]
        if "![" in prior[-120:] or "Figure concept" in prior[-80:]:
            return m.group(0)
        if key in PNG_WIRE:
            rel, alt = PNG_WIRE[key]
            fname = Path(rel).name
            if fname in prior:
                # image already present nearby; keep caption italic style
                return f"![{alt}]({rel})\n\n*{full_prefix}{body}*"
            return f"![{alt}]({rel})\n\n*{full_prefix}{body}*"
        # Concept callout (Material admonition)
        return (
            f'!!! note "Figure concept (text diagram) {num}"\n\n'
            f"    {body}"
        )

    return pattern.sub(repl, text)


def main() -> None:
    n_files = 0
    n_png = 0
    n_concept = 0
    for p in sorted(CURR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        # Count bare figures before
        bare_before = len(re.findall(r"(?m)^Figure\s+\d+\.\d+\.", raw))
        new = convert_paragraph(raw)
        if new != raw:
            p.write_text(new, encoding="utf-8", newline="\n")
            n_files += 1
            n_png += sum(1 for k in PNG_WIRE if PNG_WIRE[k][0].split("/")[-1] in new and k.replace("Figure ", "").rstrip(".") in new)
            n_concept += len(re.findall(r'Figure concept \(text diagram\)', new))
            bare_after = len(re.findall(r"(?m)^Figure\s+\d+\.\d+\.", new))
            print(f"UPDATED {p.name}: bare {bare_before} → {bare_after}")
        else:
            print(f"SKIP {p.name}")
    print(f"DONE files_changed={n_files} concept_callouts_in_tree≈ scan separately")


if __name__ == "__main__":
    main()
