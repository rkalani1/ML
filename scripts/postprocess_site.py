#!/usr/bin/env python3
"""Add stable image-loading and intrinsic-size attributes to built HTML."""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
INPUT_RE = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
LABEL_RE = re.compile(r"<label\b[^>]*>", re.IGNORECASE)
BUTTON_RE = re.compile(r"<button\b[^>]*>", re.IGNORECASE)
LEGACY_FIGURE_RE = re.compile(
    r"<p>\s*(<img\b[^>]*>)\s*</p>\s*<p>\s*<em>((?:(?!</?p\b).)*?)</em>\s*</p>",
    re.IGNORECASE | re.DOTALL,
)
OPENING_RE = re.compile(r'(<h2\b[^>]*\bid="opening"[^>]*>.*?</h2>)(.*?)(?=<h2\b)', re.IGNORECASE | re.DOTALL)
FIGURE_RE = re.compile(r'<figure\b[^>]*class="[^"]*teaching-figure[^"]*"[^>]*>.*?</figure>', re.IGNORECASE | re.DOTALL)
ATTR_RE = re.compile(r"([:\w-]+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)


def attributes(tag: str) -> dict[str, str]:
    return {name.lower(): value for name, _, value in ATTR_RE.findall(tag)}


def local_path(site: Path, html: Path, src: str) -> Path | None:
    parsed = urlsplit(src)
    if parsed.scheme or parsed.netloc or src.startswith("//") or not parsed.path:
        return None
    path = unquote(parsed.path)
    return (site / path.lstrip("/")) if path.startswith("/") else (html.parent / path)


def dimensions(path: Path) -> tuple[int, int] | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".png":
            data = path.read_bytes()[:24]
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", data[16:24])
        if suffix == ".svg":
            root = ElementTree.parse(path).getroot()
            viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()
            if len(viewbox) == 4:
                return round(float(viewbox[2])), round(float(viewbox[3]))
            width = re.match(r"[0-9.]+", root.attrib.get("width", ""))
            height = re.match(r"[0-9.]+", root.attrib.get("height", ""))
            if width and height:
                return round(float(width.group())), round(float(height.group()))
    except (OSError, ValueError, ElementTree.ParseError, struct.error):
        return None
    return None


def enhance(tag: str, site: Path, html: Path) -> tuple[str, int]:
    attrs = attributes(tag)
    additions: list[str] = []
    if "loading" not in attrs:
        additions.append('loading="eager"' if attrs.get("fetchpriority") == "high" else 'loading="lazy"')
    if "decoding" not in attrs:
        additions.append('decoding="async"')
    if "width" not in attrs or "height" not in attrs:
        path = local_path(site, html, attrs.get("src", ""))
        size = dimensions(path) if path and path.is_file() else None
        if size:
            if "width" not in attrs:
                additions.append(f'width="{size[0]}"')
            if "height" not in attrs:
                additions.append(f'height="{size[1]}"')
    if not additions:
        return tag, 0
    insertion = " " + " ".join(additions)
    return re.sub(r"\s*/?>$", lambda match: insertion + match.group(0), tag), 1


def label_toggle(tag: str) -> tuple[str, int]:
    attrs = attributes(tag)
    labels = {"__drawer": "Toggle navigation", "__search": "Toggle search"}
    label = labels.get(attrs.get("id", ""))
    if not label or attrs.get("aria-label") or attrs.get("aria-labelledby"):
        return tag, 0
    return re.sub(r"\s*/?>$", lambda match: f' aria-label="{label}"' + match.group(0), tag), 1


def enhance_header_toggle(tag: str) -> tuple[str, int]:
    attrs = attributes(tag)
    classes = set(attrs.get("class", "").split())
    labels = {"__drawer": "Toggle navigation", "__search": "Toggle search"}
    label = labels.get(attrs.get("for", ""))
    if "md-header__button" not in classes or not label:
        return tag, 0
    additions: list[str] = []
    for name, value in (
        ("role", "button"),
        ("tabindex", "0"),
        ("aria-label", label),
        ("aria-expanded", "false"),
    ):
        if name not in attrs:
            additions.append(f'{name}="{value}"')
    if not additions:
        return tag, 0
    insertion = " " + " ".join(additions)
    return re.sub(r"\s*>$", lambda match: insertion + match.group(0), tag), 1


def label_code_button(tag: str) -> tuple[str, int]:
    attrs = attributes(tag)
    if "md-code__button" not in attrs.get("class", "").split() or attrs.get("aria-label"):
        return tag, 0
    additions = ['aria-label="Copy code to clipboard"']
    if "type" not in attrs:
        additions.append('type="button"')
    insertion = " " + " ".join(additions)
    return re.sub(r"\s*>$", lambda match: insertion + match.group(0), tag), 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    site = args.site_dir.resolve()
    changed_tags = 0
    semantic_figures = 0
    eager_figure_images = 0
    opening_galleries = 0
    labeled_toggles = 0
    accessible_toggle_buttons = 0
    labeled_code_buttons = 0
    html_files = sorted(site.rglob("*.html"))
    for html in html_files:
        source = html.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal changed_tags
            result, count = enhance(match.group(0), site, html)
            changed_tags += count
            return result

        rendered = IMG_RE.sub(replace, source)

        def accessible_toggle(match: re.Match[str]) -> str:
            nonlocal labeled_toggles
            result, count = label_toggle(match.group(0))
            labeled_toggles += count
            return result

        rendered = INPUT_RE.sub(accessible_toggle, rendered)

        def accessible_button(match: re.Match[str]) -> str:
            nonlocal accessible_toggle_buttons
            result, count = enhance_header_toggle(match.group(0))
            accessible_toggle_buttons += count
            return result

        rendered = LABEL_RE.sub(accessible_button, rendered)

        def accessible_code_button(match: re.Match[str]) -> str:
            nonlocal labeled_code_buttons
            result, count = label_code_button(match.group(0))
            labeled_code_buttons += count
            return result

        rendered = BUTTON_RE.sub(accessible_code_button, rendered)
        def figure(match: re.Match[str]) -> str:
            nonlocal semantic_figures
            semantic_figures += 1
            image = match.group(1)
            caption = match.group(2)
            src = attributes(image).get("src", "")
            return (
                '<figure class="teaching-figure">'
                f'<a class="figure-zoom" href="{src}" title="Open full-resolution figure">{image}</a>'
                f'<figcaption>{caption}<span class="figure-zoom-hint" aria-hidden="true">Open full resolution ↗</span></figcaption>'
                '</figure>'
            )

        rendered = LEGACY_FIGURE_RE.sub(figure, rendered)

        def eager_figure(match: re.Match[str]) -> str:
            nonlocal eager_figure_images
            result, count = re.subn(
                r'\bloading=(["\'])lazy\1',
                'loading="eager"',
                match.group(0),
                flags=re.IGNORECASE,
            )
            eager_figure_images += count
            return result

        # Printing can begin before offscreen lazy images are requested. Every
        # semantic teaching figure is therefore eager in the release artifact;
        # the per-page image ceiling keeps the resulting fetch cost bounded.
        rendered = FIGURE_RE.sub(eager_figure, rendered)

        def opening(match: re.Match[str]) -> str:
            nonlocal opening_galleries
            body = match.group(2)
            figures = list(FIGURE_RE.finditer(body))
            if len(figures) <= 1:
                return match.group(0)
            opening_galleries += 1
            first = figures[0]
            extras = "".join(item.group(0) for item in figures[1:])
            extras = re.sub(
                r'\bloading=(["\'])lazy\1',
                'loading="eager"',
                extras,
                flags=re.IGNORECASE,
            )
            remainder = FIGURE_RE.sub("", body[first.end():])
            gallery = (
                '<details class="opening-figure-gallery">'
                f'<summary>More opening figures ({len(figures) - 1})</summary>{extras}</details>'
            )
            return match.group(1) + body[:first.end()] + gallery + remainder

        rendered = OPENING_RE.sub(opening, rendered)
        if rendered != source:
            html.write_text(rendered, encoding="utf-8")
    print(
        f"POSTPROCESS_OK html_files={len(html_files)} enhanced_images={changed_tags} "
        f"semantic_figures={semantic_figures} eager_figure_images={eager_figure_images} "
        f"opening_galleries={opening_galleries} "
        f"labeled_toggles={labeled_toggles} accessible_toggle_buttons={accessible_toggle_buttons} "
        f"labeled_code_buttons={labeled_code_buttons}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
