#!/usr/bin/env python3
"""Validate a built ebook as the browser receives it, without network access."""
from __future__ import annotations

import argparse
import math
import re
import struct
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from book_identity import load_book_identity
from release_marker import release_revision_errors

ROOT = Path(__file__).resolve().parents[1]

LEGACY_CAPTION_RE = re.compile(
    r"<p>\s*<img\b[^>]*>\s*</p>\s*<p>\s*<em>(?:(?!</?p\b).)*?</em>\s*</p>",
    re.IGNORECASE | re.DOTALL,
)
FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(.*?)</figcaption>", re.IGNORECASE | re.DOTALL)
REVIEWED_IMAGE_SUFFIXES = {".png", ".svg"}


def parse_srcset(value: str) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    errors: list[str] = []
    descriptor_kind: str | None = None
    descriptors: set[tuple[str, float | int]] = set()
    for raw_candidate in value.split(","):
        candidate = raw_candidate.strip()
        if not candidate:
            errors.append("contains an empty candidate")
            continue
        parts = candidate.split()
        urls.append(parts[0])
        if len(parts) > 2:
            errors.append(f"candidate has extra tokens: {candidate}")
            continue
        if len(parts) == 1:
            kind = "x"
            descriptor: tuple[str, float | int] = ("x", 1.0)
        elif re.fullmatch(r"[1-9]\d*w", parts[1]):
            kind = "w"
            descriptor = ("w", int(parts[1][:-1]))
        elif re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)x", parts[1]):
            density = float(parts[1][:-1])
            if not math.isfinite(density) or density <= 0:
                errors.append(f"candidate has a nonpositive density: {candidate}")
                continue
            kind = "x"
            descriptor = ("x", density)
        else:
            errors.append(f"candidate has an invalid descriptor: {candidate}")
            continue
        if descriptor_kind is None:
            descriptor_kind = kind
        elif descriptor_kind != kind:
            errors.append("mixes width and density descriptors")
        if descriptor in descriptors:
            errors.append(f"contains a duplicate descriptor: {candidate}")
        descriptors.add(descriptor)
    if not urls:
        errors.append("contains no candidates")
    return urls, errors


class Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str]] = []
        self.image_references: list[tuple[str, str]] = []
        self.duplicate_attributes: list[tuple[str, str]] = []
        self.srcset_errors: list[tuple[str, str]] = []
        self.images: list[dict[str, str]] = []
        self.gallery_images: list[dict[str, str]] = []
        self.teaching_figure_images: list[dict[str, str]] = []
        self.toggles: list[dict[str, str]] = []
        self.header_toggle_buttons: list[dict[str, str]] = []
        self.code_buttons: list[dict[str, str]] = []
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
        self._gallery_depth = 0
        self._teaching_figure_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_names = [key.casefold() for key, _ in attrs]
        self.duplicate_attributes.extend(
            (tag.casefold(), name)
            for name, count in Counter(normalized_names).items()
            if count > 1
        )
        values = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "html":
            self.html_lang = values.get("lang", "")
        if tag == "main":
            self.main_count += 1
        if tag == "figure":
            self.figure_count += 1
            if "teaching-figure" in values.get("class", "").split():
                self._teaching_figure_depth += 1
        if tag == "details" and "opening-figure-gallery" in values.get("class", "").split():
            self._gallery_depth += 1
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
        if tag == "script" and values.get("src"):
            self.references.append(("src", values["src"]))
        if tag in {"img", "source"}:
            if values.get("src"):
                self.references.append(("src", values["src"]))
                self.image_references.append((f"{tag} src", values["src"]))
            if values.get("srcset"):
                urls, errors = parse_srcset(values["srcset"])
                self.srcset_errors.extend((tag, error) for error in errors)
                for url in urls:
                    self.references.append(("srcset", url))
                    self.image_references.append((f"{tag} srcset", url))
        if tag == "source" and (values.get("src") or values.get("srcset")):
            self.sources.append(values)
        if tag == "link" and values.get("href"):
            self.references.append(("href", values["href"]))
        if tag == "img":
            self.images.append(values)
            if self._gallery_depth:
                self.gallery_images.append(values)
            if self._teaching_figure_depth:
                self.teaching_figure_images.append(values)
        if tag == "input" and values.get("id") in {"__drawer", "__search"}:
            self.toggles.append(values)
        if (
            tag == "label"
            and values.get("for") in {"__drawer", "__search"}
            and "md-header__button" in values.get("class", "").split()
        ):
            self.header_toggle_buttons.append(values)
        if tag == "button" and "md-code__button" in values.get("class", "").split():
            self.code_buttons.append(values)

    def handle_data(self, data: str) -> None:
        if self._heading_tag:
            self._heading_text.append(data)
        if self._in_title:
            self.title_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "details" and self._gallery_depth:
            self._gallery_depth -= 1
        if tag == "figure" and self._teaching_figure_depth:
            self._teaching_figure_depth -= 1
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


def reviewed_image_target(
    site: Path,
    page: Path,
    url: str,
    base_path: str,
) -> tuple[Path | None, str | None]:
    target, _ = resolve(site, page, url, base_path)
    if target is None:
        return None, "must reference a local image file"
    try:
        target.relative_to(site)
    except ValueError:
        return None, "escapes the rendered site"
    if not target.is_file():
        return None, "does not resolve to a file"
    if target.suffix.casefold() not in REVIEWED_IMAGE_SUFFIXES:
        return None, "does not use a reviewed image-file suffix"
    return target, None


def positive_dimension(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", value) and int(value) > 0)


def png_dimensions(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    if path.suffix.lower() == ".png":
        data = path.read_bytes()[:24]
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width, height = struct.unpack(">II", data[16:24])
    elif path.suffix.lower() == ".svg":
        try:
            root = ElementTree.parse(path).getroot()
            values = root.attrib.get("viewBox", "").replace(",", " ").split()
            if len(values) != 4:
                return None
            width_value, height_value = float(values[2]), float(values[3])
            if not math.isfinite(width_value) or not math.isfinite(height_value):
                return None
            width, height = round(width_value), round(height_value)
        except (OSError, ValueError, ElementTree.ParseError):
            return None
    else:
        return None
    if (
        not 1 <= width <= 8192
        or not 1 <= height <= 8192
        or width * height > 32_000_000
    ):
        return None
    return width, height


def rendered_html_files(site: Path) -> list[Path]:
    return sorted(
        path
        for path in site.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".html"
    )


def curriculum_html_files(site: Path, html_files: list[Path]) -> list[Path]:
    curriculum = site / "curriculum"
    return [path for path in html_files if path.is_relative_to(curriculum)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--max-site-mb", type=float)
    parser.add_argument("--max-html-kb", type=float)
    parser.add_argument("--max-page-images", type=int, default=24)
    parser.add_argument("--max-page-image-mb", type=float, default=4.0)
    parser.add_argument("--base-path")
    parser.add_argument("--expected-release-sha", default="")
    args = parser.parse_args()
    site = args.site_dir.resolve()
    try:
        identity = load_book_identity(ROOT)
    except ValueError as error:
        parser.error(str(error))
    args.base_path = args.base_path or identity.slug
    policy = identity.policy
    expected_curriculum = identity.expected_curriculum
    max_files = 500 if policy == "crit" else 400
    max_assets = 250 if policy == "crit" else 150
    max_curriculum_images = 250 if policy == "crit" else 200
    args.max_site_mb = args.max_site_mb or (40.0 if policy == "crit" else 30.0)
    args.max_html_kb = args.max_html_kb or (200.0 if policy == "crit" else 440.0)
    failures: list[str] = []
    expected_release_sha = args.expected_release_sha.strip()
    if expected_release_sha and not re.fullmatch(r"[0-9a-f]{40}", expected_release_sha):
        failures.append("--expected-release-sha must be a lowercase 40-character Git SHA")
    html_files = rendered_html_files(site)
    documents: dict[Path, Document] = {}
    total_bytes = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    all_files = [path for path in site.rglob("*") if path.is_file()]
    if len(all_files) > max_files:
        failures.append(f"site has {len(all_files)} files (limit {max_files})")
    if total_bytes > args.max_site_mb * 1024 * 1024:
        failures.append(f"site is {total_bytes / 1024 / 1024:.2f} MiB (limit {args.max_site_mb:.2f})")
    controls_js = site / "javascripts" / "header-controls.js"
    if not controls_js.is_file():
        failures.append("keyboard-accessible header-control script is missing")
    else:
        script = controls_js.read_text(encoding="utf-8")
        for token in (
            "keydown",
            'event.key !== "Enter"',
            'event.key !== " "',
            "aria-expanded",
            "button.md-code__button",
            "Copy code to clipboard",
        ):
            if token not in script:
                failures.append(f"header-control script lacks required behavior: {token}")
    ebook_css = site / "stylesheets" / "extra.css"
    if not ebook_css.is_file() or '.md-header__button[role="button"]:focus-visible' not in ebook_css.read_text(
        encoding="utf-8"
    ):
        failures.append("header controls lack a visible keyboard-focus rule")

    for page in html_files:
        data = page.read_text(encoding="utf-8")
        relative = page.relative_to(site)
        if expected_release_sha:
            for error in release_revision_errors(data, expected_release_sha):
                failures.append(f"{relative} release marker: {error}")
        if relative.parts and relative.parts[0] == "curriculum" and re.search(r"<em\b", data, re.IGNORECASE):
            failures.append(
                f"{relative} contains raw emphasis markup; formulas may have been parsed as Markdown emphasis"
            )
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
        for tag, attribute in document.duplicate_attributes:
            failures.append(
                f"{page.relative_to(site)} duplicate {tag} attribute: {attribute}"
            )
        for tag, error in document.srcset_errors:
            failures.append(
                f"{page.relative_to(site)} invalid {tag} srcset: {error}"
            )
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
        for toggle in document.toggles:
            if not (toggle.get("aria-label", "").strip() or toggle.get("aria-labelledby", "").strip()):
                failures.append(
                    f"{page.relative_to(site)} toggle lacks an accessible name: {toggle.get('id', '')}"
                )
        if len(document.header_toggle_buttons) != 2:
            failures.append(
                f"{page.relative_to(site)} has {len(document.header_toggle_buttons)} accessible header toggles (expected 2)"
            )
        for button in document.header_toggle_buttons:
            expected = "Toggle navigation" if button.get("for") == "__drawer" else "Toggle search"
            if button.get("role") != "button" or button.get("tabindex") != "0":
                failures.append(
                    f"{page.relative_to(site)} header toggle is not keyboard focusable: {button.get('for', '')}"
                )
            if button.get("aria-label") != expected:
                failures.append(
                    f"{page.relative_to(site)} header toggle has the wrong accessible name: {button.get('for', '')}"
                )
            if button.get("aria-expanded") not in {"true", "false"}:
                failures.append(
                    f"{page.relative_to(site)} header toggle lacks expansion state: {button.get('for', '')}"
                )
        for button in document.code_buttons:
            if button.get("aria-label") != "Copy code to clipboard" or button.get("type") != "button":
                failures.append(f"{page.relative_to(site)} code-copy button lacks explicit semantics")
        for skip in document.skip_targets:
            if skip.startswith("#") and skip[1:] not in document.ids:
                failures.append(f"{page.relative_to(site)} skip link target is missing: {skip}")
        if len(document.images) > args.max_page_images:
            failures.append(f"{page.relative_to(site)} has {len(document.images)} images (limit {args.max_page_images})")
        for image in document.images:
            if not image.get("src") and not image.get("srcset"):
                failures.append(f"{page.relative_to(site)} image has no source")
            alt = image.get("alt", "").strip()
            if not alt:
                failures.append(f"{page.relative_to(site)} image lacks descriptive alt: {image.get('src', '')}")
            elif re.search(r"\.(?:png|jpe?g|webp|svg|gif)\b", alt, re.IGNORECASE):
                failures.append(f"{page.relative_to(site)} image alt looks like a filename: {alt}")
            if image.get("loading") not in {"lazy", "eager"}:
                failures.append(f"{page.relative_to(site)} image lacks loading policy: {image.get('src', '')}")
            if image.get("decoding") not in {"async", "sync", "auto"}:
                failures.append(f"{page.relative_to(site)} image lacks decoding policy: {image.get('src', '')}")
            if not positive_dimension(image.get("width", "")) or not positive_dimension(image.get("height", "")):
                failures.append(f"{page.relative_to(site)} image lacks numeric dimensions: {image.get('src', '')}")
        for image in document.gallery_images:
            if image.get("loading") != "eager":
                failures.append(
                    f"{page.relative_to(site)} print-expanded gallery image is not eager: {image.get('src', '')}"
                )
        for image in document.teaching_figure_images:
            if image.get("loading") != "eager":
                failures.append(
                    f"{page.relative_to(site)} print-visible teaching figure image is not eager: {image.get('src', '')}"
                )
        for source in document.sources:
            if not positive_dimension(source.get("width", "")) or not positive_dimension(source.get("height", "")):
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

    curriculum_pages = curriculum_html_files(site, html_files)
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
        for kind, url in document.image_references:
            target, error = reviewed_image_target(
                site,
                page,
                url,
                args.base_path,
            )
            if error:
                failures.append(
                    f"{page.relative_to(site)} invalid {kind}: {url} ({error})"
                )
            elif target is not None:
                image_targets.add(target)
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
