#!/usr/bin/env python3
"""Validate a built ebook as the browser receives it, without network access."""
from __future__ import annotations

import argparse
import re
import struct
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

LEGACY_CAPTION_RE = re.compile(
    r"<p>\s*<img\b[^>]*>\s*</p>\s*<p>\s*<em>(?:(?!</?p\b).)*?</em>\s*</p>",
    re.IGNORECASE | re.DOTALL,
)
FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(.*?)</figcaption>", re.IGNORECASE | re.DOTALL)


class Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str]] = []
        self.images: list[dict[str, str]] = []
        self.sources: list[dict[str, str]] = []
        self.figure_count = 0
        self.figcaption_count = 0
        self.ids: list[str] = []
        self.h1_count = 0
        self.headings: list[str] = []
        self.heading_levels: list[int] = []
        self._heading_tag: str | None = None
        self._heading_text: list[str] = []
        self.html_lang = ""
        self.main_count = 0
        self.title_text: list[str] = []
        self._in_title = False
        self.skip_targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "html":
            self.html_lang = values.get("lang", "")
        if tag == "main":
            self.main_count += 1
        if tag == "figure":
            self.figure_count += 1
        if tag == "figcaption":
            self.figcaption_count += 1
        if tag == "title":
            self._in_title = True
        if re.fullmatch(r"h[1-6]", tag):
            self._heading_tag = tag
            self._heading_text = []
        if values.get("id"):
            self.ids.append(values["id"])
        if tag == "h1":
            self.h1_count += 1
        if tag == "a" and values.get("href"):
            self.references.append(("href", values["href"]))
            if "md-skip" in values.get("class", "").split():
                self.skip_targets.append(values["href"])
        if tag in {"img", "script", "source"} and values.get("src"):
            self.references.append(("src", values["src"]))
        if tag == "source" and values.get("srcset"):
            self.sources.append(values)
            for candidate in values["srcset"].split(","):
                url = candidate.strip().split()[0]
                if url:
                    self.references.append(("srcset", url))
        if tag == "link" and values.get("href"):
            self.references.append(("href", values["href"]))
        if tag == "img":
            self.images.append(values)

    def handle_data(self, data: str) -> None:
        if self._heading_tag:
            self._heading_text.append(data)
        if self._in_title:
            self.title_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if self._heading_tag == tag:
            heading = " ".join("".join(self._heading_text).split()).rstrip("¶").strip()
            if heading:
                self.headings.append(heading)
                self.heading_levels.append(int(tag[1]))
            self._heading_tag = None
            self._heading_text = []


def resolve(site: Path, page: Path, url: str, base_path: str) -> tuple[Path | None, str]:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or url.startswith("//"):
        return None, ""
    raw = unquote(parsed.path)
    if not raw:
        target = page
    elif raw.startswith("/"):
        relative = raw.lstrip("/")
        normalized_base = base_path.strip("/")
        if normalized_base and (relative == normalized_base or relative.startswith(normalized_base + "/")):
            relative = relative[len(normalized_base):].lstrip("/")
        target = site / relative
    else:
        target = page.parent / raw
    if target.is_dir():
        target = target / "index.html"
    return target.resolve(), unquote(parsed.fragment)


def png_dimensions(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() != ".png" or not path.is_file():
        return None
    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--max-site-mb", type=float)
    parser.add_argument("--max-html-kb", type=float)
    parser.add_argument("--max-page-images", type=int, default=24)
    parser.add_argument("--max-page-image-mb", type=float, default=4.0)
    parser.add_argument("--base-path", default=Path.cwd().name)
    args = parser.parse_args()
    site = args.site_dir.resolve()
    policy = "crit" if Path.cwd().name.upper() == "CRIT-APP" else "ml"
    expected_curriculum = 28 if policy == "crit" else 20
    max_files = 500 if policy == "crit" else 400
    max_assets = 250 if policy == "crit" else 150
    max_curriculum_images = 250 if policy == "crit" else 200
    args.max_site_mb = args.max_site_mb or (40.0 if policy == "crit" else 30.0)
    args.max_html_kb = args.max_html_kb or (200.0 if policy == "crit" else 440.0)
    failures: list[str] = []
    html_files = sorted(site.rglob("*.html"))
    documents: dict[Path, Document] = {}
    total_bytes = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    all_files = [path for path in site.rglob("*") if path.is_file()]
    if len(all_files) > max_files:
        failures.append(f"site has {len(all_files)} files (limit {max_files})")
    if total_bytes > args.max_site_mb * 1024 * 1024:
        failures.append(f"site is {total_bytes / 1024 / 1024:.2f} MiB (limit {args.max_site_mb:.2f})")

    for page in html_files:
        data = page.read_text(encoding="utf-8")
        if LEGACY_CAPTION_RE.search(data):
            failures.append(f"{page.relative_to(site)} retains a nonsemantic adjacent image caption")
        if len(re.findall(r"<figcaption\b", data, re.IGNORECASE)) != len(
            re.findall(r"</figcaption>", data, re.IGNORECASE)
        ):
            failures.append(f"{page.relative_to(site)} has unbalanced figcaption tags")
        for caption in FIGCAPTION_RE.findall(data):
            if re.search(r"</?em\b", caption, re.IGNORECASE):
                failures.append(f"{page.relative_to(site)} has nested emphasis inside a generated figcaption")
            visible_caption = " ".join(re.sub(r"<[^>]+>", " ", caption).split())
            if len(visible_caption) > 3000:
                failures.append(
                    f"{page.relative_to(site)} has an implausibly long figcaption ({len(visible_caption)} chars)"
                )
        if page.stat().st_size > args.max_html_kb * 1024:
            failures.append(f"{page.relative_to(site)} HTML exceeds {args.max_html_kb:.0f} KiB")
        document = Document()
        document.feed(data)
        documents[page.resolve()] = document
        duplicates = [item for item, count in Counter(document.ids).items() if count > 1]
        if duplicates:
            failures.append(f"{page.relative_to(site)} duplicate ids: {duplicates[:5]}")
        if document.h1_count != 1:
            failures.append(f"{page.relative_to(site)} has {document.h1_count} h1 elements (expected 1)")
        duplicate_headings = [
            heading for heading, count in Counter(item.casefold() for item in document.headings).items() if count > 1
        ]
        if duplicate_headings:
            failures.append(f"{page.relative_to(site)} duplicate visible headings: {duplicate_headings[:5]}")
        for previous, current in zip(document.heading_levels, document.heading_levels[1:]):
            if current > previous + 1:
                failures.append(f"{page.relative_to(site)} heading jump h{previous} to h{current}")
                break
        if not document.html_lang.lower().startswith("en"):
            failures.append(f"{page.relative_to(site)} lacks an English lang attribute")
        if document.main_count != 1:
            failures.append(f"{page.relative_to(site)} has {document.main_count} main elements (expected 1)")
        if document.figure_count != document.figcaption_count:
            failures.append(
                f"{page.relative_to(site)} has {document.figure_count} figures but "
                f"{document.figcaption_count} figcaptions"
            )
        if not " ".join(document.title_text).strip():
            failures.append(f"{page.relative_to(site)} has an empty title")
        for skip in document.skip_targets:
            if skip.startswith("#") and skip[1:] not in document.ids:
                failures.append(f"{page.relative_to(site)} skip link target is missing: {skip}")
        if len(document.images) > args.max_page_images:
            failures.append(f"{page.relative_to(site)} has {len(document.images)} images (limit {args.max_page_images})")
        for image in document.images:
            alt = image.get("alt", "").strip()
            if not alt:
                failures.append(f"{page.relative_to(site)} image lacks descriptive alt: {image.get('src', '')}")
            elif re.search(r"\.(?:png|jpe?g|webp|svg|gif)\b", alt, re.IGNORECASE):
                failures.append(f"{page.relative_to(site)} image alt looks like a filename: {alt}")
            if image.get("loading") not in {"lazy", "eager"}:
                failures.append(f"{page.relative_to(site)} image lacks loading policy: {image.get('src', '')}")
            if image.get("decoding") not in {"async", "sync", "auto"}:
                failures.append(f"{page.relative_to(site)} image lacks decoding policy: {image.get('src', '')}")
            if not re.fullmatch(r"\d+", image.get("width", "")) or not re.fullmatch(r"\d+", image.get("height", "")):
                failures.append(f"{page.relative_to(site)} image lacks numeric dimensions: {image.get('src', '')}")
        for source in document.sources:
            if not re.fullmatch(r"\d+", source.get("width", "")) or not re.fullmatch(r"\d+", source.get("height", "")):
                failures.append(f"{page.relative_to(site)} picture source lacks numeric dimensions: {source.get('srcset', '')}")
        for element, url in [
            *((image, image.get("src", "")) for image in document.images),
            *((source, source.get("srcset", "").split(",")[0].strip().split()[0]) for source in document.sources),
        ]:
            target, _ = resolve(site, page, url, args.base_path)
            if target is None:
                continue
            size = png_dimensions(target)
            if size and (element.get("width") != str(size[0]) or element.get("height") != str(size[1])):
                failures.append(
                    f"{page.relative_to(site)} dimensions mismatch for {url}: "
                    f"declared {element.get('width')}x{element.get('height')}, actual {size[0]}x{size[1]}"
                )

    curriculum_pages = sorted((site / "curriculum").glob("*.html"))
    if len(curriculum_pages) != expected_curriculum:
        failures.append(f"curriculum has {len(curriculum_pages)} pages (expected {expected_curriculum})")
    image_assets = [
        path for path in site.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
    ]
    if len(image_assets) > max_assets:
        failures.append(f"site has {len(image_assets)} image assets (limit {max_assets})")
    oversized = [path for path in image_assets if path.stat().st_size > 512 * 1024]
    if oversized:
        failures.append(f"image assets exceed 512 KiB: {[p.relative_to(site).as_posix() for p in oversized[:5]]}")
    curriculum_image_tags = sum(len(documents[path.resolve()].images) for path in curriculum_pages)
    if curriculum_image_tags > max_curriculum_images:
        failures.append(f"curriculum has {curriculum_image_tags} image tags (limit {max_curriculum_images})")
    search_index = site / "search" / "search_index.json"
    if search_index.is_file() and search_index.stat().st_size > 2 * 1024 * 1024:
        failures.append("search/search_index.json exceeds 2 MiB")

    for page, document in documents.items():
        image_targets: set[Path] = set()
        for kind, url in document.references:
            if kind == "href" and urlsplit(url).path.lower().endswith(".md"):
                failures.append(f"{page.relative_to(site)} exposes source route: {url}")
            target, fragment = resolve(site, page, url, args.base_path)
            if target is None:
                continue
            try:
                target.relative_to(site)
            except ValueError:
                failures.append(f"{page.relative_to(site)} escapes site root: {url}")
                continue
            if not target.exists():
                failures.append(f"{page.relative_to(site)} unresolved {kind}: {url}")
                continue
            if kind in {"src", "srcset"} and target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}:
                image_targets.add(target)
            if fragment and target.suffix.lower() in {".html", ".htm"}:
                target_doc = documents.get(target)
                if target_doc and fragment not in target_doc.ids:
                    failures.append(f"{page.relative_to(site)} unresolved fragment: {url}")
        image_bytes = sum(path.stat().st_size for path in image_targets)
        if image_bytes > args.max_page_image_mb * 1024 * 1024:
            failures.append(f"{page.relative_to(site)} references {image_bytes / 1024 / 1024:.2f} MiB images (limit {args.max_page_image_mb:.2f})")

    if failures:
        print("RENDERED_VALIDATION_FAILED")
        for failure in failures[:100]:
            print(f"- {failure}")
        if len(failures) > 100:
            print(f"- ... and {len(failures) - 100} more")
        return 1
    print(
        f"RENDERED_VALIDATION_OK html_files={len(html_files)} "
        f"site_mib={total_bytes / 1024 / 1024:.2f} max_images={max(len(d.images) for d in documents.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
