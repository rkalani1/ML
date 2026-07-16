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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    site = args.site_dir.resolve()
    changed_tags = 0
    semantic_figures = 0
    opening_galleries = 0
    html_files = sorted(site.rglob("*.html"))
    for html in html_files:
        source = html.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal changed_tags
            result, count = enhance(match.group(0), site, html)
            changed_tags += count
            return result

        rendered = IMG_RE.sub(replace, source)
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
        def opening(match: re.Match[str]) -> str:
            nonlocal opening_galleries
            body = match.group(2)
            figures = list(FIGURE_RE.finditer(body))
            if len(figures) <= 1:
                return match.group(0)
            opening_galleries += 1
            first = figures[0]
            extras = "".join(item.group(0) for item in figures[1:])
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
        f"semantic_figures={semantic_figures} opening_galleries={opening_galleries}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
