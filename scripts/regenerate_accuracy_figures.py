#!/usr/bin/env python3
"""Regenerate ML teaching figures whose numeric claims must not drift.

The SVG is dependency-free and deterministic. Its operating points are
asserted against the same sensitivity, specificity, and prevalence values used
in Chapters 3 and 18.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "assets" / "figures"

INK = "#142033"
MUTED = "#526277"
TEAL = "#0f8f88"
TEAL_DARK = "#08746f"
GOLD = "#c89519"
GRID = "#d8e2ea"
SURFACE = "#fbfdff"
WHITE = "#ffffff"


def ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    numerator = sensitivity * prevalence
    return numerator / (numerator + (1.0 - specificity) * (1.0 - prevalence))


def ppv_prevalence_figure() -> str:
    sensitivity, specificity = 0.85, 0.70
    lr_positive = sensitivity / (1.0 - specificity)
    ppv_05 = ppv(sensitivity, specificity, 0.05)
    ppv_20 = ppv(sensitivity, specificity, 0.20)

    assert math.isclose(lr_positive, 2.833333333333333, abs_tol=1e-12)
    assert math.isclose(ppv_05, 0.12977099236641223, abs_tol=1e-12)
    assert math.isclose(ppv_20, 0.41463414634146345, abs_tol=1e-12)

    left, top, width, height = 112.0, 142.0, 784.0, 366.0
    xmax = 0.85

    def point(prevalence: float) -> tuple[float, float]:
        value = ppv(sensitivity, specificity, prevalence)
        return left + prevalence / xmax * width, top + (1.0 - value) * height

    curve = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index in range(161)
        for prevalence in [0.80 * index / 160]
        for x, y in [point(prevalence)]
    )

    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for value in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = left + value / xmax * width
        grid_lines.append(
            f'  <path d="M {x:.1f} {top:.1f} V {top + height:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        tick_labels.append(
            f'  <text x="{x:.1f}" y="{top + height + 29:.1f}" '
            f'text-anchor="middle" class="tick">{value:.1f}</text>'
        )
    for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = top + (1.0 - value) * height
        grid_lines.append(
            f'  <path d="M {left:.1f} {y:.1f} H {left + width:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        tick_labels.append(
            f'  <text x="{left - 18:.1f}" y="{y + 6:.1f}" '
            f'text-anchor="end" class="tick">{value:.1f}</text>'
        )

    x05, y05 = point(0.05)
    x20, y20 = point(0.20)
    x55, y55 = point(0.55)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" role="img" aria-labelledby="title desc">
  <metadata>Generated deterministically by scripts/regenerate_accuracy_figures.py.</metadata>
  <title id="title">Positive predictive value across disease prevalence</title>
  <desc id="desc">For a fixed screen with sensitivity 0.85 and specificity 0.70, positive predictive value is 13.0 percent at 5 percent prevalence and 41.5 percent at 20 percent prevalence. Positive predictive value rises as prevalence rises.</desc>
  <rect width="1000" height="650" rx="24" fill="{SURFACE}"/>
  <style>
    text {{ font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .subtitle {{ font-size: 18px; fill: {MUTED}; }}
    .axis {{ font-size: 18px; font-weight: 600; }}
    .tick {{ font-size: 15px; fill: {MUTED}; }}
    .callout-title {{ font-size: 17px; font-weight: 700; }}
    .callout-value {{ font-size: 17px; fill: {TEAL_DARK}; }}
    .curve-label {{ font-size: 16px; font-weight: 600; fill: {TEAL_DARK}; }}
    .note {{ font-size: 16px; fill: {MUTED}; }}
  </style>
  <text x="112" y="54" class="title">PPV changes with prevalence</text>
  <text x="112" y="88" class="subtitle">Fixed screen: sensitivity 0.85 · specificity 0.70 · LR+ 2.83</text>
{chr(10).join(grid_lines)}
  <path d="M {left:.1f} {top:.1f} V {top + height:.1f} H {left + width:.1f}" fill="none" stroke="{INK}" stroke-width="2"/>
{chr(10).join(tick_labels)}
  <path d="{curve}" fill="none" stroke="{TEAL}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{x05:.2f}" cy="{y05:.2f}" r="10" fill="{GOLD}" stroke="{WHITE}" stroke-width="3"/>
  <path d="M {x05 + 8:.1f} {y05 - 5:.1f} L 218 438" fill="none" stroke="{TEAL_DARK}" stroke-width="2"/>
  <rect x="218" y="397" width="188" height="68" rx="12" fill="{WHITE}" stroke="{GRID}"/>
  <text x="234" y="424" class="callout-title">5% prevalence</text>
  <text x="234" y="450" class="callout-value">PPV 13.0%</text>
  <circle cx="{x20:.2f}" cy="{y20:.2f}" r="10" fill="{GOLD}" stroke="{WHITE}" stroke-width="3"/>
  <path d="M {x20 + 8:.1f} {y20 - 6:.1f} L 360 307" fill="none" stroke="{TEAL_DARK}" stroke-width="2"/>
  <rect x="360" y="266" width="204" height="68" rx="12" fill="{WHITE}" stroke="{GRID}"/>
  <text x="376" y="293" class="callout-title">20% prevalence</text>
  <text x="376" y="319" class="callout-value">PPV 41.5%</text>
  <path d="M {x55:.1f} {y55 - 7:.1f} L 654 178" fill="none" stroke="{TEAL_DARK}" stroke-width="2"/>
  <text x="665" y="176" class="curve-label">Same test, different base rate</text>
  <text x="{left + width / 2:.1f}" y="574" text-anchor="middle" class="axis">Disease prevalence P(D+)</text>
  <text x="35" y="{top + height / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 35 {top + height / 2:.1f})">Positive predictive value P(D+ | test+)</text>
  <text x="112" y="622" class="note">Predictive value changes with prevalence; sensitivity and specificity are held constant.</text>
</svg>
'''


def ppv_prevalence_mobile_figure() -> str:
    """Render a portrait companion whose smallest labels remain legible on phones."""
    sensitivity, specificity = 0.85, 0.70
    lr_positive = sensitivity / (1.0 - specificity)
    ppv_05 = ppv(sensitivity, specificity, 0.05)
    ppv_20 = ppv(sensitivity, specificity, 0.20)

    assert math.isclose(lr_positive, 2.833333333333333, abs_tol=1e-12)
    assert math.isclose(ppv_05, 0.12977099236641223, abs_tol=1e-12)
    assert math.isclose(ppv_20, 0.41463414634146345, abs_tol=1e-12)

    left, top, width, height = 82.0, 185.0, 466.0, 342.0
    xmax = 0.80

    def point(prevalence: float) -> tuple[float, float]:
        value = ppv(sensitivity, specificity, prevalence)
        return left + prevalence / xmax * width, top + (1.0 - value) * height

    curve = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index in range(161)
        for prevalence in [xmax * index / 160]
        for x, y in [point(prevalence)]
    )

    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for value in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = left + value / xmax * width
        grid_lines.append(
            f'  <path d="M {x:.1f} {top:.1f} V {top + height:.1f}" '
            f'stroke="{GRID}" stroke-width="1.5"/>'
        )
        tick_labels.append(
            f'  <text x="{x:.1f}" y="{top + height + 34:.1f}" '
            f'text-anchor="middle" class="tick">{value:.1f}</text>'
        )
    for value in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = top + (1.0 - value) * height
        grid_lines.append(
            f'  <path d="M {left:.1f} {y:.1f} H {left + width:.1f}" '
            f'stroke="{GRID}" stroke-width="1.5"/>'
        )
        tick_labels.append(
            f'  <text x="{left - 13:.1f}" y="{y + 7:.1f}" '
            f'text-anchor="end" class="tick">{value:.2g}</text>'
        )

    x05, y05 = point(0.05)
    x20, y20 = point(0.20)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 900" role="img" aria-labelledby="title-mobile desc-mobile">
  <metadata>Generated deterministically by scripts/regenerate_accuracy_figures.py.</metadata>
  <title id="title-mobile">Positive predictive value across disease prevalence</title>
  <desc id="desc-mobile">Phone layout. For a fixed screen with sensitivity 0.85 and specificity 0.70, positive predictive value is 13.0 percent at 5 percent prevalence and 41.5 percent at 20 percent prevalence.</desc>
  <rect width="600" height="900" rx="26" fill="{SURFACE}"/>
  <style>
    text {{ font-family: "Source Sans 3", "Segoe UI", Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 35px; font-weight: 750; }}
    .subtitle {{ font-size: 24px; fill: {MUTED}; }}
    .axis {{ font-size: 23px; font-weight: 650; }}
    .tick {{ font-size: 21px; fill: {MUTED}; }}
    .marker {{ font-size: 23px; font-weight: 750; fill: {TEAL_DARK}; }}
    .card-title {{ font-size: 25px; font-weight: 650; }}
    .card-value {{ font-size: 31px; font-weight: 750; fill: {TEAL_DARK}; }}
    .note-title {{ font-size: 26px; font-weight: 750; }}
    .note {{ font-size: 23px; fill: {MUTED}; }}
  </style>
  <text x="42" y="52" class="title">PPV changes with prevalence</text>
  <text x="42" y="91" class="subtitle">Sensitivity 0.85 · specificity 0.70</text>
  <text x="42" y="124" class="subtitle">LR+ 2.83 · same test, different base rate</text>
{chr(10).join(grid_lines)}
  <path d="M {left:.1f} {top:.1f} V {top + height:.1f} H {left + width:.1f}" fill="none" stroke="{INK}" stroke-width="2.5"/>
{chr(10).join(tick_labels)}
  <path d="{curve}" fill="none" stroke="{TEAL}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{x05:.2f}" cy="{y05:.2f}" r="12" fill="{GOLD}" stroke="{WHITE}" stroke-width="4"/>
  <text x="{x05 + 17:.2f}" y="{y05 - 15:.2f}" class="marker">5%</text>
  <circle cx="{x20:.2f}" cy="{y20:.2f}" r="12" fill="{GOLD}" stroke="{WHITE}" stroke-width="4"/>
  <text x="{x20 + 17:.2f}" y="{y20 - 15:.2f}" class="marker">20%</text>
  <text x="{left + width / 2:.1f}" y="582" text-anchor="middle" class="axis">Disease prevalence P(D+)</text>
  <text x="26" y="{top + height / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 26 {top + height / 2:.1f})">Positive predictive value</text>
  <rect x="34" y="615" width="252" height="104" rx="16" fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>
  <circle cx="58" cy="648" r="8" fill="{GOLD}"/>
  <text x="76" y="656" class="card-title">5% prevalence</text>
  <text x="58" y="698" class="card-value">PPV 13.0%</text>
  <rect x="314" y="615" width="252" height="104" rx="16" fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>
  <circle cx="338" cy="648" r="8" fill="{GOLD}"/>
  <text x="356" y="656" class="card-title">20% prevalence</text>
  <text x="338" y="698" class="card-value">PPV 41.5%</text>
  <text x="42" y="768" class="note-title">Why this matters</text>
  <text x="42" y="806" class="note">Predictive value changes with prevalence, even</text>
  <text x="42" y="838" class="note">when sensitivity and specificity stay constant.</text>
  <path d="M 42 865 H 558" stroke="{TEAL}" stroke-width="5" stroke-linecap="round"/>
</svg>
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    figures = {
        FIGURE_DIR / "ml_fig_ppv_prevalence.svg": ppv_prevalence_figure(),
        FIGURE_DIR / "ml_fig_ppv_prevalence_mobile.svg": ppv_prevalence_mobile_figure(),
    }
    if args.check:
        stale = [
            target
            for target, rendered in figures.items()
            if not target.is_file() or target.read_text(encoding="utf-8") != rendered
        ]
        if stale:
            for target in stale:
                print(f"ACCURACY_FIGURE_STALE {target}")
            return 1
        print(f"ACCURACY_FIGURES_OK {len(figures)}")
        return 0

    for target, rendered in figures.items():
        target.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"ACCURACY_FIGURE_WRITTEN {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
