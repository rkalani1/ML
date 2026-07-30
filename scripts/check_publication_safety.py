#!/usr/bin/env python3
"""Bounded current-tree publication safety and release-integrity checks."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
from importlib import metadata as importlib_metadata
import json
import re
import site
import struct
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET
import zlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from asset_inventory import (
    figure_assets,
    sha256,
    unsupported_figure_assets,
)
from book_identity import load_book_site_url

ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".otf",
    ".png",
    ".ttf",
    ".webp",
    ".woff",
    ".woff2",
}
PUBLISHED_SOURCE_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".svg",
    ".txt",
}
PUBLISHED_SOURCE_BINARY_SUFFIXES = {".png", ".woff2"}
REPOSITORY_TEXT_SUFFIXES = PUBLISHED_SOURCE_TEXT_SUFFIXES | {
    "",
    ".cff",
    ".in",
    ".lock",
    ".mjs",
    ".py",
    ".yaml",
    ".yml",
}
RENDERED_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".map",
    ".mjs",
    ".svg",
    ".txt",
}
RENDERED_BINARY_SUFFIXES = {".png", ".woff2"}
AUDIT_BOOTSTRAP_LOCK = """pip==26.2 \\
    --hash=sha256:931c303696af6fa3417112103b1cad26890e5a07eccb5b99783700e33f2b8aad
setuptools==83.0.0 \\
    --hash=sha256:29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3
"""
REVIEWED_PNG_CHUNKS = {b"IHDR", b"IDAT", b"IEND", b"pHYs", b"tEXt"}
REVIEWED_PNG_PHYS = {
    bytes.fromhex("000017120000171201"),
    bytes.fromhex("0000189b0000189b01"),
    bytes.fromhex("00001baf00001baf01"),
    bytes.fromhex("00001ec200001ec201"),
}
REVIEWED_PNG_TEXT = {
    b"Software\x00Matplotlib version3.11.0, https://matplotlib.org/",
}
MATERIAL_FAVICON_SHA256 = (
    "023854c43fc4b25b795ee4951c8019e3de0593ffeb6f918b5d1f2a9b47a57cfe"
)
SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "site",
}
FORBIDDEN_BROWSER_DOCUMENT_SUFFIXES = {
    ".hta",
    ".htm",
    ".mht",
    ".mhtml",
    ".shtml",
    ".svgz",
    ".xht",
    ".xml",
    ".xhtml",
    ".xsl",
    ".xslt",
}
FORBIDDEN_TREE_SUFFIXES = {
    ".7z",
    ".db",
    ".dcm",
    ".doc",
    ".docx",
    ".gz",
    ".nii",
    ".parquet",
    ".pdf",
    ".pfx",
    ".p12",
    ".ppt",
    ".pptx",
    ".rar",
    ".rdata",
    ".rds",
    ".sav",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".xls",
    ".xlsx",
    ".zip",
} | FORBIDDEN_BROWSER_DOCUMENT_SUFFIXES
FORBIDDEN_GITLEAKS_CONTROL_FILES = {".gitleaks.toml", ".gitleaksignore"}
REQUIRED = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CITATION.cff",
    "requirements.lock",
    "requirements-audit-bootstrap.in",
    "requirements-audit-bootstrap.lock",
    "requirements-audit.in",
    "requirements-audit.lock",
    "sbom.cdx.json",
    "docs/publication-standards.md",
    "docs/assets/licenses/Apache-2.0.txt",
    "scripts/test_release_gates.py",
    "scripts/test_gitleaks_controls.py",
    "scripts/test_mathjax_fallback.mjs",
    "scripts/book_identity.py",
    "scripts/release_marker.py",
    "scripts/verify_release_marker.py",
    "scripts/asset_inventory.py",
    "scripts/verify_mathjax_integrity.mjs",
    "scripts/verify_sbom_schema.py",
    "_meta/ORIGINALITY.md",
    "_meta/release.json",
    "_meta/asset-rights-register.json",
    "_meta/browser-assets.json",
    ".github/workflows/pages.yml",
    ".github/dependabot.yml",
)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "GitLab token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "PyPI token": re.compile(r"\bpypi-[A-Za-z0-9_-]{20,}\b"),
    "npm token": re.compile(r"\bnpm_[A-Za-z0-9]{30,}\b"),
    "Stripe secret": re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def tracked_repository_files(root: Path) -> set[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        Path(item.decode("utf-8", errors="surrogateescape"))
        for item in process.stdout.split(b"\0")
        if item
    }


def is_skipped_root_tree(relative: Path, tracked: set[Path]) -> bool:
    """Skip untracked generated root trees, never tracked or published content."""
    return (
        bool(relative.parts)
        and relative.parts[0] in SKIP_DIRECTORIES
        and relative not in tracked
    )


def is_published_source(relative: Path) -> bool:
    return bool(relative.parts) and relative.parts[0] in {"docs", "overrides"}


def file_bytes(path: Path, label: str, errors: list[str]) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError as exc:
        errors.append(f"unable to read {label}: {exc}")
        return None


def utf8_content(data: bytes, label: str, errors: list[str]) -> str | None:
    if b"\x00" in data:
        errors.append(f"NUL byte in text publication file: {label}")
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"non-UTF-8 text publication file: {label}: {exc}")
        return None


def png_metadata_errors(data: bytes, label: str) -> list[str]:
    """Validate PNG structure and fail closed on unreviewed metadata chunks."""
    errors: list[str] = []
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [f"PNG signature mismatch: {label}"]

    chunks: list[tuple[bytes, bytes]] = []
    position = 8
    saw_iend = False
    while position < len(data):
        if position + 12 > len(data):
            errors.append(f"truncated PNG chunk framing: {label}")
            break
        length = struct.unpack(">I", data[position:position + 4])[0]
        chunk_type = data[position + 4:position + 8]
        chunk_end = position + 12 + length
        if chunk_end > len(data):
            errors.append(f"truncated PNG chunk payload: {label}")
            break
        payload = data[position + 8:position + 8 + length]
        expected_crc = struct.unpack(
            ">I", data[position + 8 + length:chunk_end]
        )[0]
        observed_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if expected_crc != observed_crc:
            errors.append(
                f"PNG chunk CRC mismatch ({chunk_type.decode('latin-1')}): {label}"
            )
        chunks.append((chunk_type, payload))
        position = chunk_end
        if chunk_type == b"IEND":
            saw_iend = True
            if payload:
                errors.append(f"nonempty PNG IEND chunk: {label}")
            if position != len(data):
                errors.append(f"trailing bytes after PNG IEND: {label}")
            break

    chunk_types = [chunk_type for chunk_type, _ in chunks]
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        errors.append(f"PNG must begin with one 13-byte IHDR chunk: {label}")
        width = height = 0
    else:
        width, height, bit_depth, color_type, compression, filtering, interlace = (
            struct.unpack(">IIBBBBB", chunks[0][1])
        )
        if (
            not 1 <= width <= 8192
            or not 1 <= height <= 8192
            or width * height > 32_000_000
        ):
            errors.append(f"PNG dimensions exceed the reviewed safety bound: {label}")
            width = height = 0
        if (bit_depth, color_type, compression, filtering, interlace) != (
            8,
            6,
            0,
            0,
            0,
        ):
            errors.append(f"unreviewed PNG pixel format: {label}")
    if chunk_types.count(b"IHDR") != 1:
        errors.append(f"PNG must contain exactly one IHDR chunk: {label}")
    if b"IDAT" not in chunk_types:
        errors.append(f"PNG is missing IDAT image data: {label}")
    if not saw_iend or chunk_types.count(b"IEND") != 1:
        errors.append(f"PNG must contain exactly one terminal IEND chunk: {label}")

    text_count = 0
    physical_count = 0
    first_idat = chunk_types.index(b"IDAT") if b"IDAT" in chunk_types else len(chunks)
    for index, (chunk_type, payload) in enumerate(chunks):
        if chunk_type not in REVIEWED_PNG_CHUNKS:
            errors.append(
                "unreviewed PNG metadata/chunk "
                f"{chunk_type.decode('latin-1')}: {label}"
            )
        if chunk_type == b"pHYs":
            physical_count += 1
            if payload not in REVIEWED_PNG_PHYS:
                errors.append(f"unreviewed PNG pHYs metadata: {label}")
            if index >= first_idat:
                errors.append(f"PNG pHYs metadata must precede image data: {label}")
        if chunk_type == b"tEXt":
            text_count += 1
            if index >= first_idat:
                errors.append(f"PNG text metadata must precede image data: {label}")
            if payload not in REVIEWED_PNG_TEXT:
                errors.append(f"unreviewed PNG text metadata: {label}")
    if text_count > 1:
        errors.append(f"multiple PNG text metadata chunks: {label}")
    if physical_count > 1:
        errors.append(f"multiple PNG pHYs metadata chunks: {label}")

    idat_indexes = [
        index for index, (chunk_type, _) in enumerate(chunks)
        if chunk_type == b"IDAT"
    ]
    if idat_indexes and idat_indexes != list(
        range(idat_indexes[0], idat_indexes[-1] + 1)
    ):
        errors.append(f"PNG IDAT chunks must be contiguous: {label}")
    if width and height and idat_indexes:
        compressed = b"".join(chunks[index][1] for index in idat_indexes)
        expected_size = height * (1 + 4 * width)
        decompressor = zlib.decompressobj()
        try:
            pixels = decompressor.decompress(compressed, expected_size + 1)
            if len(pixels) <= expected_size:
                pixels += decompressor.flush(expected_size + 1 - len(pixels))
        except zlib.error as exc:
            errors.append(f"PNG image-data decompression failed: {label}: {exc}")
        else:
            if (
                len(pixels) != expected_size
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                errors.append(f"PNG image-data length/stream is invalid: {label}")
            elif any(
                pixels[row * (1 + 4 * width)] > 4
                for row in range(height)
            ):
                errors.append(f"PNG contains an invalid row filter: {label}")
    return errors


def published_source_file_errors(path: Path, relative: Path) -> list[str]:
    errors: list[str] = []
    label = relative.as_posix()
    suffix = path.suffix.casefold()
    if suffix not in (
        PUBLISHED_SOURCE_TEXT_SUFFIXES | PUBLISHED_SOURCE_BINARY_SUFFIXES
    ):
        return [f"unreviewed published file type: {label}"]

    data = file_bytes(path, label, errors)
    if data is None:
        return errors
    if suffix in PUBLISHED_SOURCE_TEXT_SUFFIXES:
        if (
            suffix == ".css"
            and relative.parts[:2] != ("docs", "stylesheets")
        ):
            errors.append(
                f"published CSS is outside the inventoried stylesheet tree: {label}"
            )
        if (
            suffix == ".js"
            and relative.parts[:2] != ("docs", "javascripts")
        ):
            errors.append(
                f"published JavaScript is outside the inventoried script tree: {label}"
            )
        if (
            suffix == ".svg"
            and relative.parts[:3] != ("docs", "assets", "figures")
        ):
            errors.append(
                f"published SVG is outside the inventoried figure tree: {label}"
            )
        utf8_content(data, label, errors)
        return errors
    if suffix == ".png":
        if relative.parts[:3] != ("docs", "assets", "figures"):
            errors.append(f"published PNG is outside the inventoried figure tree: {label}")
        errors.extend(png_metadata_errors(data, label))
    elif suffix == ".woff2":
        if relative.parts[:3] != ("docs", "assets", "fonts"):
            errors.append(f"published WOFF2 is outside the inventoried font tree: {label}")
        if not data.startswith(b"wOF2"):
            errors.append(f"WOFF2 signature mismatch: {label}")
    return errors


def rendered_static_file_errors(
    path: Path,
    relative: str,
) -> tuple[list[str], str | None]:
    errors: list[str] = []
    suffix = path.suffix.casefold()
    if suffix in RENDERED_BINARY_SUFFIXES:
        data = file_bytes(path, f"rendered/{relative}", errors)
        if data is None:
            return errors, None
        if suffix == ".png":
            if not relative.startswith("assets/figures/"):
                if (
                    relative != "assets/images/favicon.png"
                    or hashlib.sha256(data).hexdigest() != MATERIAL_FAVICON_SHA256
                ):
                    errors.append(
                        "rendered PNG is outside the inventoried figure tree "
                        f"or reviewed theme favicon: {relative}"
                    )
            errors.extend(png_metadata_errors(data, f"rendered/{relative}"))
        else:
            if not relative.startswith("assets/fonts/"):
                errors.append(
                    f"rendered WOFF2 is outside the inventoried font tree: {relative}"
                )
            if not data.startswith(b"wOF2"):
                errors.append(f"WOFF2 signature mismatch: rendered/{relative}")
        return errors, None
    if suffix not in RENDERED_TEXT_SUFFIXES:
        return [f"unreviewed rendered file type: {relative}"], None
    if (
        suffix == ".css"
        and not relative.startswith(("stylesheets/", "assets/stylesheets/"))
    ):
        errors.append(
            f"rendered CSS is outside the reviewed stylesheet trees: {relative}"
        )
    if (
        suffix in {".js", ".mjs"}
        and not relative.startswith(("javascripts/", "assets/javascripts/"))
    ):
        errors.append(
            f"rendered JavaScript is outside the reviewed script trees: {relative}"
        )
    if (
        suffix == ".map"
        and not relative.startswith(
            ("assets/javascripts/", "assets/stylesheets/")
        )
    ):
        errors.append(
            f"rendered source map is outside the locked theme trees: {relative}"
        )
    data = file_bytes(path, f"rendered/{relative}", errors)
    if data is None:
        return errors, None
    if suffix == ".svg" and not relative.startswith("assets/figures/"):
        errors.append(
            f"rendered SVG is outside the inventoried figure tree: {relative}"
        )
    return errors, utf8_content(data, f"rendered/{relative}", errors)


def deployed_asset_tree_errors(source_root: Path, site_dir: Path) -> list[str]:
    errors: list[str] = []
    configurations = (
        (
            "figure",
            source_root / "docs" / "assets" / "figures",
            site_dir / "assets" / "figures",
            {".json", ".png", ".svg"},
        ),
        (
            "font",
            source_root / "docs" / "assets" / "fonts",
            site_dir / "assets" / "fonts",
            {".txt", ".woff2"},
        ),
        (
            "stylesheet",
            source_root / "docs" / "stylesheets",
            site_dir / "stylesheets",
            {".css"},
        ),
    )
    for label, source_directory, rendered_directory, suffixes in configurations:
        def inventory(directory: Path) -> dict[str, str]:
            if not directory.is_dir():
                return {}
            return {
                path.relative_to(directory).as_posix(): sha256(path)
                for path in directory.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.casefold() in suffixes
            }

        expected = inventory(source_directory)
        observed = inventory(rendered_directory)
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        if missing:
            errors.append(
                f"rendered {label} asset tree is missing: {', '.join(missing)}"
            )
        if extra:
            errors.append(
                f"rendered {label} asset tree has extras: {', '.join(extra)}"
            )
        for relative in sorted(set(expected) & set(observed)):
            if expected[relative] != observed[relative]:
                errors.append(
                    f"rendered {label} asset hash mismatch: {relative}"
                )
    return errors


def font_inventory_from_readme(content: str) -> tuple[dict[str, str], list[str]]:
    inventory: dict[str, str] = {}
    errors: list[str] = []
    for line in content.splitlines():
        if not line.startswith("|") or ".woff2`" not in line:
            continue
        cells = line.split("|")
        if len(cells) < 4:
            errors.append("vendored-font inventory row is malformed")
            continue
        filenames = re.findall(r"`([^`]+\.woff2)`", cells[1])
        digest_match = re.fullmatch(r"\s*`([0-9a-f]{64})`\s*", cells[2])
        if not filenames or digest_match is None:
            errors.append("vendored-font inventory row is malformed")
            continue
        digest = digest_match.group(1)
        for filename in filenames:
            if filename in inventory:
                errors.append(f"duplicate vendored-font inventory path: {filename}")
            inventory[filename] = digest
    return inventory, errors


IDENTIFIER_PATTERNS = {
    "SSN-like value": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "labeled MRN-like value": re.compile(
        r"\b(?:MRN|medical record number)\s*[:#=-]?\s*[A-Z0-9]{6,}\b",
        re.IGNORECASE,
    ),
    "labeled DOB-like value": re.compile(
        r"\b(?:DOB|date of birth)\s*[:#=-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE,
    ),
}
LOCAL_PATH_PATTERNS = {
    "macOS user path": re.compile(r"/Users/[^/\s]+/"),
    "Linux user path": re.compile(r"/home/[^/\s]+/"),
    "Windows user path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\r\n]+", re.IGNORECASE),
    "private temporary path": re.compile(r"/private/tmp/"),
}
TRACKER_MARKERS = (
    "googletagmanager",
    "google-analytics",
    "analytics.js",
    "plausible.io",
    "mixpanel",
    "segment.com/analytics",
)
FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
MATHJAX_SRI = (
    "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"
)
MATHJAX_FONT_SOURCE = (
    "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/"
)
INLINE_SCRIPT_HASHES = (
    "/8wPdzX9q0NNJXyA5lzsLojXFpkeaXVxhbfkUOQaWy8=",
    "/K9p2JtEqCycL2fSbEonMakkteWpAHv57x2wndLqMNo=",
    "DrEMJJ29sL7vIloQzly+VUGMxKcBTMII+OfW7Y8AkG4=",
    "apoQPHefCNWjxbCm+HzVDOAW4CSVWhY7VylQjgOFyfk=",
    "c7BRi6hR1MeGIZcBlBVGi8vPWbZcwNv7R3ir8hN9jP4=",
)
INLINE_SCRIPT_SOURCES = " ".join(
    f"'sha256-{digest}'" for digest in INLINE_SCRIPT_HASHES
)
REVIEWED_AUTHORED_JAVASCRIPT_SHA256 = {
    "docs/javascripts/header-controls.js": (
        "530654253a2973fd9165b94b4e2c67465d7c9835f295259047677de56cced555"
    ),
    "docs/javascripts/lazy-images.js": (
        "293b8ece9cbea2b7cd73321e18e264147ca42ff4c67d8f1e1f7dc4a8fb4a461b"
    ),
    "docs/javascripts/mathjax.js": (
        "04cd9ceed74f1acc38119b45ed1b65503b049fb3034ce754a35f61dada410b46"
    ),
}
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; base-uri 'self'; object-src 'none'; frame-src 'none'; "
    "form-action 'self'; connect-src 'self'; img-src 'self' data:; "
    f"font-src 'self' data: {MATHJAX_FONT_SOURCE}; "
    "style-src 'self' 'unsafe-inline'; "
    f"script-src 'self' {MATHJAX_URL} {INLINE_SCRIPT_SOURCES}; "
    "worker-src 'self' blob:; upgrade-insecure-requests"
)
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\\\s#]+)", re.MULTILINE)
DIRECT_PIN_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\\\s#]+)$")
LOCK_MARKER_ATOM = (
    r"[A-Za-z_][A-Za-z0-9_]*\s*"
    r"(?:===|==|!=|<=|>=|<|>|in|not\s+in)\s*"
    r"(?:'[^'\\\r\n]+'|\"[^\"\\\r\n]+\")"
)
LOCK_PIN_LINE_RE = re.compile(
    rf"^([A-Za-z0-9_.-]+)==([^\\\s#]+)"
    rf"(?:\s+;\s+{LOCK_MARKER_ATOM}"
    rf"(?:\s+(?:and|or)\s+{LOCK_MARKER_ATOM})*)?\s+\\$"
)
LOCK_HASH_LINE_RE = re.compile(
    r"^\s+--hash=sha256:([0-9a-f]{64})(?:\s+\\)?$"
)
REMOTE_CSS_LOAD_RE = re.compile(
    r"(?:@import\s+(?:url\(\s*)?|url\(\s*)[\"']?\s*"
    r"(?:(?:https?:\s*)?//|https?:)",
    re.IGNORECASE,
)
CSS_IMAGE_SET_RE = re.compile(
    r"(?:-webkit-)?image-set\s*\(",
    re.IGNORECASE,
)
REMOTE_CSS_ADDRESS_RE = re.compile(
    r"(?:(?:https?:)?/{2,}|https?:)",
    re.IGNORECASE,
)
DATA_CSS_IMPORT_RE = re.compile(
    r"@import\s+(?:url\(\s*)?[\"']?\s*data\s*:",
    re.IGNORECASE,
)
CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
CSS_DATA_URL_RE = re.compile(r"(?i)\bdata\s*:")
CSS_ESCAPE_RE = re.compile(
    r"\\(?:([0-9a-fA-F]{1,6})(?:\r\n|[ \t\r\n\f])?|(\r\n|[\n\r\f])|(.)?)",
    re.DOTALL,
)
NETWORK_PRIMITIVES = {
    "fetch call": re.compile(r"\bfetch\s*\(", re.IGNORECASE),
    "XMLHttpRequest": re.compile(r"\bXMLHttpRequest\b"),
    "WebSocket": re.compile(r"\bWebSocket\s*\("),
    "EventSource": re.compile(r"\bEventSource\s*\("),
    "sendBeacon": re.compile(r"\bsendBeacon\s*\("),
    "importScripts": re.compile(r"\bimportScripts\s*\("),
    "scripted navigation": re.compile(
        r"(?:\b(?:window\.)?location(?:\.href)?\s*=|"
        r"\bwindow\s*\[\s*[\"']location[\"']\s*\]\s*=|"
        r"\b(?:window\.)?location\.(?:assign|replace)\s*\(|"
        r"\bwindow\.open\s*\()",
        re.IGNORECASE,
    ),
    "dynamic resource creation": re.compile(
        r"(?:\bdocument\.createElement\s*\(\s*[\"']"
        r"(?:a|script|style|img|iframe|link|source|video|audio|object)[\"']\s*\)|"
        r"\bnew\s+(?:Image|Audio|Worker|SharedWorker)\s*\()",
        re.IGNORECASE,
    ),
    "dynamic resource assignment": re.compile(
        r"(?:\.(?:src|href|srcset|imagesrcset|attributionsrc|ping)\s*=|"
        r"\[\s*[\"'](?:src|href|srcset|imagesrcset|attributionsrc|ping)"
        r"[\"']\s*\]\s*=|"
        r"\bsetAttribute\s*\(\s*[\"']"
        r"(?:src|href|srcset|imagesrcset|attributionsrc|ping)[\"'])",
        re.IGNORECASE,
    ),
    "dynamic object property mutation": re.compile(
        r"(?:\bObject\.(?:assign|definePropert(?:y|ies))\s*\(|"
        r"\bReflect\.set\s*\()"
    ),
    "DOM markup injection": re.compile(
        r"(?:\.(?:innerHTML|outerHTML|srcdoc)\s*=|"
        r"\binsertAdjacentHTML\s*\(|\bdocument\.write(?:ln)?\s*\(|"
        r"\bcreateContextualFragment\s*\(|\bDOMParser\b)",
        re.IGNORECASE,
    ),
    "dynamic CSS injection": re.compile(
        r"(?:\bCSSStyleSheet\b|\badoptedStyleSheets\b|"
        r"\.(?:replace|replaceSync|insertRule)\s*\()"
    ),
    "WebRTC transport": re.compile(
        r"(?:\b(?:webkit)?RTCPeerConnection\b|\bRTCDataChannel\b)",
        re.IGNORECASE,
    ),
    "STUN or TURN transport URL": re.compile(
        r"\b(?:stuns?|turns?)\s*:",
        re.IGNORECASE,
    ),
    "dynamic module import": re.compile(r"\bimport\s*\("),
}
REMOTE_URL_RE = re.compile(r"(?:https?:)?//[^\s\"'`<>]+", re.IGNORECASE)
AUTHORED_JS_STRING_CONCAT_RE = re.compile(
    r"(?:[\"'][^\"'\r\n]*[\"']\s*\+|\+\s*[\"'])"
)
ACTIVE_SVG_PATTERNS = {
    "script element": re.compile(r"<\s*script\b", re.IGNORECASE),
    "inline event handler": re.compile(r"\son[a-z]+\s*=", re.IGNORECASE),
    "javascript URL": re.compile(r"(?:href|xlink:href)\s*=\s*[\"']\s*javascript:", re.IGNORECASE),
    "foreignObject element": re.compile(r"<\s*foreignObject\b", re.IGNORECASE),
    "embedded browsing context": re.compile(r"<\s*(?:iframe|object|embed)\b", re.IGNORECASE),
    "animation element": re.compile(
        r"<\s*(?:animate|animateMotion|animateTransform|set)\b", re.IGNORECASE
    ),
    "document type or entity declaration": re.compile(
        r"<!DOCTYPE\b|<!ENTITY\b", re.IGNORECASE
    ),
    "remote reference": re.compile(
        r"(?:href|xlink:href)\s*=\s*[\"']\s*(?:https?:)?//", re.IGNORECASE
    ),
}
XML_DECLARATION_RE = re.compile(
    r"\A(?:\ufeff)?[ \t\r\n]*<\?xml\s+version\s*=\s*(['\"])1\.[01]\1"
    r"(?:\s+encoding\s*=\s*(['\"])[A-Za-z][A-Za-z0-9._-]*\2)?"
    r"(?:\s+standalone\s*=\s*(['\"])(?:yes|no)\3)?\s*\?>",
    re.IGNORECASE,
)
INERT_SCRIPT_TYPES = {"application/json", "application/ld+json"}
SCRIPT_SUFFIXES = {".cjs", ".js", ".mjs"}
VOID_HTML_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
SELF_CLOSING_SVG_ELEMENTS = {
    "circle",
    "ellipse",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "stop",
    "use",
}
ALLOWED_SELF_CLOSING_ELEMENTS = VOID_HTML_ELEMENTS | SELF_CLOSING_SVG_ELEMENTS
ASCII_URL_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def unsafe_url_syntax(value: str) -> bool:
    return "\\" in value or bool(ASCII_URL_CONTROL_RE.search(value))


def is_remote(value: str) -> bool:
    stripped = value.strip()
    if unsafe_url_syntax(stripped):
        return True
    if re.match(r"^/{2,}", stripped):
        return True
    parsed = urlsplit(stripped)
    return parsed.scheme.casefold() in {"http", "https"} or (
        not parsed.scheme and bool(parsed.netloc)
    )


def has_remote_css_image_set(content: str) -> bool:
    return bool(
        CSS_IMAGE_SET_RE.search(content)
        and REMOTE_CSS_ADDRESS_RE.search(content)
    )


class RenderedSurfaceParser(HTMLParser):
    """Collect network-active resources and data-submission controls."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.remote_resources: list[tuple[str, str, str]] = []
        self.unsafe_forms: list[str] = []
        self.sensitive_inputs: list[str] = []
        self.dangerous_attributes: list[str] = []
        self.active_text_blocks: list[tuple[str, str, str, bool]] = []
        self.content_security_policies: list[str] = []
        self.referrer_policies: list[str] = []
        self.script_sources: list[str] = []
        self._csp_seen = False
        self._svg_depth = 0
        self._active_tag: str | None = None
        self._active_type = ""
        self._active_in_svg = False
        self._active_text: list[str] = []
        self._open_elements: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute_names = [name.casefold() for name, _ in attrs]
        for duplicate in sorted(
            {name for name in attribute_names if attribute_names.count(name) > 1}
        ):
            self.dangerous_attributes.append(
                f"{tag.casefold()} duplicate attribute [{duplicate}]"
            )
        values = {name.casefold(): (value or "") for name, value in attrs}
        lowered = tag.casefold()
        direct_head_child = bool(
            self._open_elements and self._open_elements[-1] == "head"
        )
        inside_svg = self._svg_depth > 0 or lowered == "svg"
        if lowered == "svg":
            self._svg_depth += 1
        if inside_svg:
            if lowered in {
                "script",
                "foreignobject",
                "iframe",
                "object",
                "embed",
                "animate",
                "animatemotion",
                "animatetransform",
                "set",
            }:
                self.dangerous_attributes.append(
                    f"active inline SVG element [{lowered}]"
                )
            for raw_name, raw_value in attrs:
                name = raw_name.casefold()
                value = (raw_value or "").strip()
                if (
                    name == "xmlns"
                    and value == "http://www.w3.org/2000/svg"
                ) or (
                    name == "xmlns:xlink"
                    and value == "http://www.w3.org/1999/xlink"
                ):
                    continue
                if name.startswith("on"):
                    self.dangerous_attributes.append(
                        f"active inline SVG event [{name}]"
                    )
                if name in {"base", "xml:base"}:
                    self.dangerous_attributes.append("active inline SVG xml:base")
                if name in {"href", "xlink:href"} and value and not value.startswith("#"):
                    self.dangerous_attributes.append(
                        f"active inline SVG non-fragment [{name}]"
                    )
                if active_svg_css(value):
                    self.dangerous_attributes.append(
                        f"active inline SVG attribute [{name}]"
                    )
        if lowered == "script" and values.get("src", "").strip():
            self.script_sources.append(values["src"].strip())
        if lowered == "script" and not self._csp_seen:
            self.dangerous_attributes.append("script before Content-Security-Policy")
        if lowered == "base":
            self.dangerous_attributes.append("base element")
        for name, value in values.items():
            stripped = value.strip().casefold()
            if name in {
                "href",
                "src",
                "srcset",
                "imagesrcset",
                "attributionsrc",
                "ping",
                "data",
                "poster",
                "background",
                "action",
                "formaction",
            } and unsafe_url_syntax(value):
                self.dangerous_attributes.append(
                    f"{lowered}[{name}] has unsafe URL syntax"
                )
            if name.startswith("on"):
                self.dangerous_attributes.append(f"{lowered}[{name}]")
            if name == "referrerpolicy":
                self.dangerous_attributes.append(f"{lowered}[referrerpolicy]")
            if name in {"attributionsrc", "ping"}:
                self.dangerous_attributes.append(f"{lowered}[{name}]")
            if name == "srcdoc":
                self.dangerous_attributes.append(f"{lowered}[srcdoc]")
            if name in {
                "href",
                "src",
                "srcset",
                "imagesrcset",
                "attributionsrc",
                "ping",
                "data",
                "poster",
                "action",
                "formaction",
            } and stripped.startswith("data:"):
                self.dangerous_attributes.append(f"{lowered}[{name}]={value!r}")
            if name in {"href", "src", "data", "action", "formaction"} and (
                stripped.startswith(("javascript:", "vbscript:"))
            ):
                self.dangerous_attributes.append(f"{lowered}[{name}]={value!r}")
        if values.get("formaction", "").strip():
            self.unsafe_forms.append(
                f"{lowered} formaction={values['formaction']!r}"
            )
        if lowered == "form":
            action = values.get("action", "").strip()
            method = values.get("method", "get").strip().casefold() or "get"
            if action:
                self.unsafe_forms.append(f"form action={action!r}")
            if method != "get":
                self.unsafe_forms.append(f"form method={method!r}")
        if lowered == "input":
            input_type = values.get("type", "text").strip().casefold()
            if input_type in {"file", "password"}:
                self.sensitive_inputs.append(f"input type={input_type!r}")

        resource: tuple[str, str] | None = None
        if lowered in {
            "script",
            "iframe",
            "img",
            "image",
            "source",
            "embed",
            "video",
            "audio",
            "track",
        }:
            resource = ("src", values.get("src", ""))
        elif lowered == "object":
            resource = ("data", values.get("data", ""))
        elif lowered == "input" and values.get("type", "").casefold() == "image":
            resource = ("src", values.get("src", ""))
        elif lowered == "link":
            rel = {item.casefold() for item in values.get("rel", "").split()}
            if rel & {
                "stylesheet",
                "preload",
                "modulepreload",
                "prefetch",
                "prerender",
                "preconnect",
                "dns-prefetch",
                "icon",
                "manifest",
                "apple-touch-icon",
                "mask-icon",
            }:
                resource = ("href", values.get("href", ""))
        if resource and resource[1] and is_remote(resource[1]):
            self.remote_resources.append((lowered, resource[0], resource[1]))
        if lowered == "image":
            self.dangerous_attributes.append("legacy image element aliases img")
        for attribute in (
            "poster",
            "srcset",
            "imagesrcset",
            "attributionsrc",
            "ping",
            "background",
        ):
            value = values.get(attribute, "")
            if value and (
                unsafe_url_syntax(value)
                or re.search(
                    r"(?:(?:https?:)?/{2,}|https?:)",
                    value,
                    re.IGNORECASE,
                )
            ):
                self.remote_resources.append((lowered, attribute, value))
        style = values.get("style", "")
        if style:
            decoded_style = decode_css_escapes(style)
            token_style = CSS_COMMENT_RE.sub("", decoded_style)
            if CSS_DATA_URL_RE.search(token_style):
                self.dangerous_attributes.append(
                    f"{lowered}[style] contains a data URL"
                )
            if (
                REMOTE_CSS_LOAD_RE.search(token_style)
                or has_remote_css_image_set(token_style)
                or "\\" in token_style
            ):
                self.remote_resources.append((lowered, "style", token_style))
        if lowered == "meta" and values.get("http-equiv", "").casefold() == "refresh":
            content = values.get("content", "")
            self.dangerous_attributes.append(f"meta refresh={content!r}")
        if (
            lowered == "meta"
            and values.get("http-equiv", "").casefold() == "content-security-policy"
        ):
            if direct_head_child:
                self.content_security_policies.append(values.get("content", ""))
                self._csp_seen = True
            else:
                self.dangerous_attributes.append(
                    "Content-Security-Policy meta is not a direct child of head"
                )
        if lowered == "meta" and values.get("name", "").casefold() == "referrer":
            if direct_head_child:
                self.referrer_policies.append(values.get("content", ""))
            else:
                self.dangerous_attributes.append(
                    "referrer policy meta is not a direct child of head"
                )
        if lowered == "body" and not self._csp_seen:
            self.dangerous_attributes.append(
                "body begins before a valid Content-Security-Policy"
            )
        if lowered == "style" and not self._csp_seen:
            self.dangerous_attributes.append(
                "style before Content-Security-Policy"
            )
        if resource and not self._csp_seen:
            self.dangerous_attributes.append(
                f"{lowered}[{resource[0]}] before Content-Security-Policy"
            )
        if lowered in {"script", "style"}:
            self._active_tag = lowered
            self._active_type = values.get("type", "").strip().casefold()
            self._active_in_svg = inside_svg
            self._active_text = []
        if lowered not in VOID_HTML_ELEMENTS:
            self._open_elements.append(lowered)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lowered = tag.casefold()
        if lowered not in ALLOWED_SELF_CLOSING_ELEMENTS:
            self.dangerous_attributes.append(
                f"self-closing non-void HTML element [{lowered}]"
            )
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == self._active_tag:
            content = "".join(self._active_text)
            if content.strip():
                self.active_text_blocks.append(
                    (
                        self._active_tag,
                        self._active_type,
                        content,
                        self._active_in_svg,
                    )
                )
            self._active_tag = None
            self._active_type = ""
            self._active_in_svg = False
            self._active_text = []
        if lowered == "svg" and self._svg_depth:
            self._svg_depth -= 1
        if lowered in self._open_elements:
            while self._open_elements:
                if self._open_elements.pop() == lowered:
                    break

    def handle_data(self, data: str) -> None:
        if self._active_tag:
            self._active_text.append(data)

    def handle_pi(self, data: str) -> None:
        if self._svg_depth:
            self.dangerous_attributes.append("active inline SVG processing instruction")

    def handle_decl(self, decl: str) -> None:
        if self._svg_depth:
            self.dangerous_attributes.append("active inline SVG declaration")


def text_files(
    root: Path = ROOT,
    tracked: set[Path] | None = None,
) -> list[Path]:
    tracked = tracked_repository_files(root) if tracked is None else tracked
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if is_skipped_root_tree(relative, tracked):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            if b"\x00" in path.read_bytes()[:8192]:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def release_identity_errors(
    root: Path,
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    release_path = root / "_meta" / "release.json"
    try:
        release = json.loads(read(release_path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"release metadata parse failure: {exc}"], {}
    if release.get("schema_version") != 1:
        errors.append("release metadata schema_version must be 1")
    for field in ("version", "review_date", "review_date_long", "publication_review"):
        if not isinstance(release.get(field), str) or not str(release[field]).strip():
            errors.append(f"release metadata missing field: {field}")
    if errors:
        return errors, release

    version = str(release["version"])
    review_date = str(release["review_date"])
    review_date_long = str(release["review_date_long"])
    try:
        parsed_date = dt.date.fromisoformat(review_date)
    except ValueError:
        errors.append("release metadata review_date must be an ISO calendar date")
        parsed_date = None
    if version != review_date.replace("-", "."):
        errors.append("release metadata version must match the dotted review date")
    if parsed_date is not None:
        expected_long_date = (
            f"{parsed_date.day} {parsed_date.strftime('%B')} {parsed_date.year}"
        )
        if review_date_long != expected_long_date:
            errors.append("release metadata review_date_long differs from review_date")
    review_relative = Path(str(release["publication_review"]))
    expected_review = Path("_meta") / f"PUBLICATION_REVIEW_{review_date}.md"
    if (
        review_relative.is_absolute()
        or ".." in review_relative.parts
        or review_relative != expected_review
    ):
        errors.append(
            "publication_review must be the dated release record "
            f"{expected_review.as_posix()}"
        )
    else:
        review_path = root / review_relative
        if not review_path.is_file():
            errors.append(f"publication review missing: {review_relative.as_posix()}")
        elif f"Reviewed: **{review_date_long}**" not in read(review_path):
            errors.append("publication review date differs from release metadata")

    cff_path = root / "CITATION.cff"
    if cff_path.is_file():
        cff = read(cff_path)
        if not re.search(
            rf'^version:\s*["\']?{re.escape(version)}["\']?\s*$',
            cff,
            re.MULTILINE,
        ):
            errors.append("CITATION.cff version differs from release metadata")
        if not re.search(
            rf'^date-released:\s*["\']?{re.escape(review_date)}["\']?\s*$',
            cff,
            re.MULTILINE,
        ):
            errors.append("CITATION.cff date-released differs from release metadata")
    for relative in (
        "_meta/ORIGINALITY.md",
        "docs/publication-standards.md",
    ):
        path = root / relative
        if path.is_file() and f"**{review_date_long}**" not in read(path):
            errors.append(f"review date differs from release metadata: {relative}")
    return errors, release


def pinned_pairs(content: str) -> set[tuple[str, str]]:
    return {
        (match.group(1).casefold().replace("_", "-"), match.group(2))
        for match in PIN_PATTERN.finditer(content)
    }


def lockfile_errors(
    requirements_text: str,
    lock_text: str,
    requirements_label: str,
    lock_label: str,
) -> tuple[list[str], set[tuple[str, str]]]:
    errors: list[str] = []
    direct: set[tuple[str, str]] = set()
    direct_names: set[str] = set()
    direct_invalid = False
    for line_number, line in enumerate(requirements_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = DIRECT_PIN_LINE_RE.fullmatch(stripped)
        if not match:
            direct_invalid = True
            errors.append(
                f"unexpected active line in {requirements_label}:{line_number}"
            )
            continue
        name = match.group(1).casefold().replace("_", "-")
        if name in direct_names:
            direct_invalid = True
            errors.append(f"duplicate direct pin in {requirements_label}: {name}")
        direct_names.add(name)
        direct.add((name, match.group(2)))
    if direct_invalid or not direct:
        errors.append(f"{requirements_label} must contain only exact direct pins")

    locked: set[tuple[str, str]] = set()
    locked_names: set[str] = set()
    current: tuple[str, str] | None = None
    hash_count = 0
    for line_number, line in enumerate(lock_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pin_match = LOCK_PIN_LINE_RE.fullmatch(line)
        if pin_match:
            if current is not None:
                errors.append(
                    f"unterminated dependency record in {lock_label} before "
                    f"line {line_number}"
                )
            name = pin_match.group(1).casefold().replace("_", "-")
            current = (name, pin_match.group(2))
            hash_count = 0
            continue
        hash_match = LOCK_HASH_LINE_RE.fullmatch(line)
        if hash_match:
            if current is None:
                errors.append(f"orphan hash line in {lock_label}:{line_number}")
                continue
            hash_count += 1
            if not line.rstrip().endswith("\\"):
                name, version = current
                if name in locked_names:
                    errors.append(f"duplicate locked pin in {lock_label}: {name}")
                locked_names.add(name)
                locked.add((name, version))
                current = None
                hash_count = 0
            continue
        errors.append(f"unexpected active line in {lock_label}:{line_number}")
    if current is not None:
        if hash_count == 0:
            errors.append(
                f"hash missing for {lock_label} dependency: {current[0]}"
            )
        else:
            errors.append(f"unterminated dependency record in {lock_label}")
    if not locked:
        errors.append(f"{lock_label} has no exact pins")
    elif not direct.issubset(locked):
        errors.append(f"{requirements_label} pins do not match {lock_label}")
    return errors, locked


def dependency_inventory_errors(
    requirements_text: str, lock_text: str, sbom: dict[str, object]
) -> list[str]:
    errors, locked = lockfile_errors(
        requirements_text,
        lock_text,
        "requirements.txt",
        "requirements.lock",
    )

    components = sbom.get("components", [])
    if not isinstance(components, list):
        errors.append("SBOM components must be a list")
        return errors
    inventoried: set[tuple[str, str]] = set()
    bom_refs: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"SBOM component must be an object at index {index}")
            continue
        pair = (
            str(component.get("name", "")).casefold().replace("_", "-"),
            str(component.get("version", "")),
        )
        if pair in inventoried:
            errors.append(
                "duplicate SBOM component name/version: "
                f"{pair[0]}=={pair[1]}"
            )
        inventoried.add(pair)
        bom_ref = component.get("bom-ref")
        if not isinstance(bom_ref, str) or not bom_ref.strip():
            errors.append(f"SBOM bom-ref missing for component: {pair[0]}")
        elif bom_ref in bom_refs:
            errors.append(f"duplicate SBOM bom-ref: {bom_ref}")
        else:
            bom_refs.add(bom_ref)
    if len(components) != len(inventoried) or len(inventoried) != len(locked):
        errors.append("SBOM component count differs from requirements.lock")
    if locked != inventoried:
        errors.append("SBOM component set differs from requirements.lock")
    for component in components:
        if not isinstance(component, dict):
            continue
        licenses = component.get("licenses")
        if not isinstance(licenses, list) or not licenses:
            name = str(component.get("name", "<unnamed>"))
            errors.append(f"SBOM license metadata missing for component: {name}")
    return errors


def locked_package_versions(lock_text: str, package_name: str) -> set[str]:
    normalized_name = package_name.casefold().replace("_", "-")
    return {
        version
        for name, version in pinned_pairs(lock_text)
        if name == normalized_name
    }


def browser_theme_version_errors(
    inventory_theme_package: object,
    lock_text: str,
    installed_version: str | None,
) -> list[str]:
    errors: list[str] = []
    versions = locked_package_versions(lock_text, "mkdocs-material")
    if len(versions) != 1:
        errors.append(
            "requirements.lock must contain exactly one mkdocs-material version"
        )
        return errors
    locked_version = next(iter(versions))
    if inventory_theme_package != f"mkdocs-material=={locked_version}":
        errors.append("browser-asset theme package differs from requirements.lock")
    if installed_version is None:
        errors.append("locked Material for MkDocs package is unavailable")
    elif installed_version != locked_version:
        errors.append(
            "installed Material for MkDocs version differs from requirements.lock"
        )
    return errors


def authored_javascript_baseline_errors(current: dict[str, str]) -> list[str]:
    if current != REVIEWED_AUTHORED_JAVASCRIPT_SHA256:
        return [
            "authored JavaScript differs from the manually reviewed digest baseline"
        ]
    return []


def browser_asset_inventory_errors(
    root: Path,
    review_date: str,
) -> list[str]:
    errors: list[str] = []
    inventory_path = root / "_meta" / "browser-assets.json"
    try:
        inventory = json.loads(read(inventory_path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"browser-asset inventory parse failure: {exc}"]
    if inventory.get("schema_version") != 1:
        errors.append("browser-asset inventory schema_version must be 1")
    if inventory.get("review_date") != review_date:
        errors.append("browser-asset inventory date differs from release metadata")
    records = inventory.get("authored_javascript")
    if not isinstance(records, list):
        errors.append("browser-asset inventory authored_javascript must be a list")
        records = []
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if len(by_path) != len(records):
        errors.append("duplicate or malformed browser-asset inventory path")
    current = {
        path.relative_to(root).as_posix(): path
        for path in (root / "docs" / "javascripts").rglob("*")
        if path.is_file()
    } if (root / "docs" / "javascripts").is_dir() else {}
    errors.extend(
        authored_javascript_baseline_errors(
            {relative: sha256(path) for relative, path in current.items()}
        )
    )
    if sorted(by_path) != sorted(current):
        errors.append("browser-asset inventory differs from authored JavaScript set")
    for relative in sorted(set(by_path) & set(current)):
        if current[relative].suffix.casefold() not in SCRIPT_SUFFIXES:
            errors.append(f"unsupported authored browser-script file type: {relative}")
        if by_path[relative].get("sha256") != sha256(current[relative]):
            errors.append(f"browser-asset inventory hash mismatch: {relative}")
        if not str(by_path[relative].get("purpose", "")).strip():
            errors.append(f"browser-asset purpose missing: {relative}")
    inline = inventory.get("theme_inline_script_sha256")
    if (
        not isinstance(inline, list)
        or sorted(str(item) for item in inline) != sorted(INLINE_SCRIPT_HASHES)
    ):
        errors.append("reviewed inline-script hash inventory differs from CSP")
    lock_path = root / "requirements.lock"
    try:
        lock_text = read(lock_path)
    except OSError as exc:
        errors.append(f"requirements.lock unavailable for browser assets: {exc}")
    else:
        try:
            installed_version = importlib_metadata.version("mkdocs-material")
        except importlib_metadata.PackageNotFoundError:
            installed_version = None
        errors.extend(
            browser_theme_version_errors(
                inventory.get("theme_package"),
                lock_text,
                installed_version,
            )
        )
    return errors


def scan_markers(label: str, content: str, errors: list[str]) -> None:
    folded = content.casefold()
    for tracker in TRACKER_MARKERS:
        if tracker in folded:
            errors.append(f"analytics/tracker reference in {label}: {tracker}")
    for font_host in FONT_HOSTS:
        if font_host in folded:
            errors.append(f"external font request in {label}: {font_host}")


def decode_css_escapes(content: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hexadecimal, continuation, escaped = match.groups()
        if hexadecimal:
            value = int(hexadecimal, 16)
            if value == 0 or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                return "\ufffd"
            return chr(value)
        if continuation:
            return ""
        return escaped or ""

    # CSS tokenization performs one escape-decoding pass. Re-decoding the
    # result can turn a literal backslash pair into an unrelated code point and
    # hide a browser-resolved network URL.
    return CSS_ESCAPE_RE.sub(replace, content)


def active_svg_css(content: str) -> bool:
    folded = content.casefold()
    return bool(
        REMOTE_CSS_LOAD_RE.search(content)
        or has_remote_css_image_set(content)
        or any(
            token in folded
            for token in (
                "@",
                "\\",
                "//",
                "data:",
                "http:",
                "https:",
                "image-set(",
                "javascript:",
                "url(",
            )
        )
    )


def scan_svg(label: str, content: str, errors: list[str]) -> None:
    declaration = XML_DECLARATION_RE.match(content)
    remainder = content[declaration.end():] if declaration else content
    if "<?" in remainder:
        errors.append(f"active SVG processing instruction in {label}")
        return
    for feature, pattern in ACTIVE_SVG_PATTERNS.items():
        if pattern.search(content):
            errors.append(f"active SVG {feature} in {label}")
    if ACTIVE_SVG_PATTERNS["document type or entity declaration"].search(content):
        return
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        errors.append(f"SVG XML parse failure in {label}: {exc}")
        return
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].casefold()
        if tag in {
            "script",
            "foreignobject",
            "iframe",
            "object",
            "embed",
            "animate",
            "animatemotion",
            "animatetransform",
            "set",
        }:
            errors.append(f"active SVG decoded {tag} element in {label}")
        for raw_name, raw_value in element.attrib.items():
            name = raw_name.rsplit("}", 1)[-1].casefold()
            value = raw_value.strip()
            if name.startswith("on"):
                errors.append(f"active SVG decoded event handler in {label}")
            if name == "base":
                errors.append(f"active SVG decoded xml:base attribute in {label}")
            if name in {"href", "xlink:href"} and value and not value.startswith("#"):
                errors.append(f"active SVG non-fragment reference in {label}")
            if active_svg_css(value):
                errors.append(
                    f"active SVG decoded attribute reference ({name}) in {label}"
                )
        if tag == "style":
            style = "".join(element.itertext())
            if active_svg_css(style):
                errors.append(f"active SVG CSS reference in {label}")


def scan_browser_egress(
    label: str, content: str, suffix: str, errors: list[str]
) -> None:
    if suffix == ".css":
        decoded_css = decode_css_escapes(content)
        token_css = CSS_COMMENT_RE.sub("", decoded_css)
        if (
            REMOTE_CSS_LOAD_RE.search(token_css)
            or has_remote_css_image_set(token_css)
        ):
            errors.append(f"remote CSS load in {label}")
        if "\\" in token_css:
            errors.append(f"ambiguous CSS escape in {label}")
        if DATA_CSS_IMPORT_RE.search(token_css):
            errors.append(f"nested data CSS import in {label}")
        if CSS_DATA_URL_RE.search(token_css):
            errors.append(f"data URL in authored CSS: {label}")
        scan_markers(label, decoded_css, errors)
    if suffix != ".js":
        return
    if " inline " not in label and AUTHORED_JS_STRING_CONCAT_RE.search(content):
        errors.append(f"authored JavaScript string concatenation in {label}")
    reviewed_mathjax_loader = (
        label in {
            "docs/javascripts/mathjax.js",
            "rendered/javascripts/mathjax.js",
        }
        and content.count('document.createElement("script")') == 1
        and content.count("script.src = bundle") == 1
        and f'const bundle = "{MATHJAX_URL}"' in content
        and f'script.integrity = "{MATHJAX_SRI}"' in content
        and hashlib.sha256(content.encode("utf-8")).hexdigest()
        == REVIEWED_AUTHORED_JAVASCRIPT_SHA256[
            "docs/javascripts/mathjax.js"
        ]
    )
    without_mathjax = (
        content.replace(MATHJAX_URL, "")
        if reviewed_mathjax_loader
        else content
    )
    if REMOTE_URL_RE.search(without_mathjax):
        errors.append(f"non-allowlisted remote URL in JavaScript: {label}")
    if MATHJAX_URL in content and not reviewed_mathjax_loader:
        errors.append(f"MathJax URL outside reviewed integrity loader: {label}")
    for feature, pattern in NETWORK_PRIMITIVES.items():
        if pattern.search(content):
            if reviewed_mathjax_loader and feature in {
                "dynamic resource creation",
                "dynamic resource assignment",
            }:
                continue
            errors.append(f"browser network primitive ({feature}) in {label}")


def compare_javascript_trees(
    actual_dir: Path,
    expected_dir: Path,
    label: str,
    errors: list[str],
) -> None:
    actual = {
        path.relative_to(actual_dir).as_posix(): path
        for path in actual_dir.rglob("*")
        if path.is_file()
    } if actual_dir.is_dir() else {}
    expected = {
        path.relative_to(expected_dir).as_posix(): path
        for path in expected_dir.rglob("*")
        if path.is_file()
    } if expected_dir.is_dir() else {}
    for relative in sorted(set(expected) - set(actual)):
        errors.append(f"{label} JavaScript missing from rendered site: {relative}")
    for relative in sorted(set(actual) - set(expected)):
        errors.append(f"unexpected {label} JavaScript in rendered site: {relative}")
    for relative in sorted(set(actual) & set(expected)):
        if sha256(actual[relative]) != sha256(expected[relative]):
            errors.append(f"{label} JavaScript hash mismatch: {relative}")


def compare_stylesheet_trees(
    actual_dir: Path,
    expected_dir: Path,
    errors: list[str],
) -> set[str]:
    actual = {
        path.relative_to(actual_dir).as_posix(): path
        for path in actual_dir.rglob("*")
        if path.is_file()
    } if actual_dir.is_dir() else {}
    expected = {
        path.relative_to(expected_dir).as_posix(): path
        for path in expected_dir.rglob("*")
        if path.is_file()
    } if expected_dir.is_dir() else {}
    for relative in sorted(set(expected) - set(actual)):
        errors.append(f"theme stylesheet missing from rendered site: {relative}")
    for relative in sorted(set(actual) - set(expected)):
        errors.append(f"unexpected theme stylesheet in rendered site: {relative}")
    matching: set[str] = set()
    for relative in sorted(set(actual) & set(expected)):
        if sha256(actual[relative]) != sha256(expected[relative]):
            errors.append(f"theme stylesheet hash mismatch: {relative}")
        else:
            matching.add(f"assets/stylesheets/{relative}")
    return matching


def source_dependency_shadow_errors(source_root: Path) -> list[str]:
    errors: list[str] = []
    search_roots = (source_root, source_root / "scripts")
    for search_root in search_roots:
        for name in ("material", "material.py"):
            candidate = search_root / name
            if candidate.exists():
                errors.append(
                    "repo-local dependency shadow is forbidden: "
                    f"{candidate.relative_to(source_root).as_posix()}"
                )
        if search_root.is_dir():
            for pattern in (
                "mkdocs_material-*.dist-info",
                "mkdocs-material-*.dist-info",
                "mkdocs_material-*.egg-info",
                "mkdocs-material-*.egg-info",
            ):
                for candidate in sorted(search_root.glob(pattern)):
                    errors.append(
                        "repo-local dependency metadata shadow is forbidden: "
                        f"{candidate.relative_to(source_root).as_posix()}"
                    )
    return errors


def verify_rendered_javascript_assets(
    site_dir: Path,
    source_root: Path,
    errors: list[str],
) -> tuple[dict[str, str], set[str]]:
    authored_dir = source_root / "docs" / "javascripts"
    allowlist = {
        f"javascripts/{path.relative_to(authored_dir).as_posix()}": sha256(path)
        for path in authored_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in SCRIPT_SUFFIXES
    } if authored_dir.is_dir() else {}
    compare_javascript_trees(
        site_dir / "javascripts",
        authored_dir,
        "authored",
        errors,
    )
    shadow_errors = source_dependency_shadow_errors(source_root)
    errors.extend(shadow_errors)
    if shadow_errors:
        return allowlist, set()
    try:
        lock_text = read(source_root / "requirements.lock")
    except OSError as exc:
        errors.append(f"requirements.lock unavailable for rendered assets: {exc}")
        return allowlist, set()
    locked_versions = locked_package_versions(lock_text, "mkdocs-material")
    if len(locked_versions) != 1:
        errors.append(
            "requirements.lock must contain exactly one mkdocs-material version"
        )
        return allowlist, set()
    locked_version = next(iter(locked_versions))
    try:
        distribution = importlib_metadata.distribution("mkdocs-material")
    except importlib_metadata.PackageNotFoundError:
        errors.append("locked Material for MkDocs package is unavailable")
        return allowlist, set()
    installed_version = distribution.version
    if installed_version != locked_version:
        errors.append(
            "installed Material for MkDocs version differs from requirements.lock"
        )
        return allowlist, set()
    distribution_root = Path(distribution.locate_file("")).resolve()
    trusted_roots = {
        Path(path).resolve()
        for key in ("purelib", "platlib")
        if (path := sysconfig.get_paths().get(key))
    }
    trusted_roots.update(
        Path(path).resolve()
        for path in (
            *site.getsitepackages(),
            site.getusersitepackages(),
        )
        if path
    )
    if (
        distribution_root == source_root.resolve()
        or distribution_root.is_relative_to(source_root.resolve())
        or not any(
            distribution_root == trusted
            or distribution_root.is_relative_to(trusted)
            for trusted in trusted_roots
        )
    ):
        errors.append(
            "mkdocs-material distribution is outside interpreter site-packages"
        )
        return allowlist, set()
    expected_theme = Path(
        distribution.locate_file("material/templates/assets/javascripts")
    ).resolve()
    if (
        not expected_theme.is_dir()
        or not expected_theme.is_relative_to(distribution_root)
    ):
        errors.append("locked Material for MkDocs theme path is unavailable")
        return allowlist, set()
    compare_javascript_trees(
        site_dir / "assets" / "javascripts",
        expected_theme,
        "theme",
        errors,
    )
    allowlist.update(
        {
            (
                "assets/javascripts/"
                f"{path.relative_to(expected_theme).as_posix()}"
            ): sha256(path)
            for path in expected_theme.rglob("*")
            if path.is_file() and path.suffix.casefold() in SCRIPT_SUFFIXES
        }
    )
    expected_stylesheets = Path(
        distribution.locate_file("material/templates/assets/stylesheets")
    ).resolve()
    if (
        not expected_stylesheets.is_dir()
        or not expected_stylesheets.is_relative_to(distribution_root)
    ):
        errors.append("locked Material for MkDocs stylesheet path is unavailable")
        return allowlist, set()
    locked_stylesheets = compare_stylesheet_trees(
        site_dir / "assets" / "stylesheets",
        expected_stylesheets,
        errors,
    )
    return allowlist, locked_stylesheets


def validate_local_script_source(
    site_dir: Path,
    document_path: Path,
    source: str,
    allowlist: dict[str, str] | None,
    site_path_prefix: str | None,
    errors: list[str],
) -> None:
    document = document_path.relative_to(site_dir).as_posix()
    parsed = urlsplit(source)
    if parsed.scheme or parsed.netloc:
        errors.append(
            f"non-allowlisted script scheme in rendered/{document}: {source}"
        )
        return
    if parsed.query or parsed.fragment or not parsed.path:
        errors.append(
            f"non-exact local script resource in rendered/{document}: {source}"
        )
        return
    decoded_path = unquote(parsed.path)
    if "\\" in decoded_path or "\x00" in decoded_path:
        errors.append(
            f"non-exact local script resource in rendered/{document}: {source}"
        )
        return
    if decoded_path.startswith("/"):
        if not site_path_prefix or not decoded_path.startswith(site_path_prefix):
            errors.append(
                f"non-exact local script resource in rendered/{document}: {source}"
            )
            return
        local_path = decoded_path[len(site_path_prefix):]
        if not local_path:
            errors.append(
                f"non-exact local script resource in rendered/{document}: {source}"
            )
            return
        candidate = (site_dir / local_path).resolve()
    else:
        candidate = (document_path.parent / decoded_path).resolve()
    try:
        relative = candidate.relative_to(site_dir.resolve()).as_posix()
    except ValueError:
        errors.append(
            f"local script escapes rendered site in rendered/{document}: {source}"
        )
        return
    if allowlist is None or relative not in allowlist:
        errors.append(
            f"unreviewed local script resource in rendered/{document}: {source}"
        )
        return
    if not candidate.is_file():
        errors.append(f"allowlisted local script is missing: {relative}")
        return
    if sha256(candidate) != allowlist[relative]:
        errors.append(f"allowlisted local script hash mismatch: {relative}")
        return
    if relative.startswith("javascripts/"):
        content = read(candidate)
        scan_markers(f"rendered/{relative}", content, errors)
        scan_browser_egress(f"rendered/{relative}", content, ".js", errors)


def configured_site_url(
    source_root: Path,
    errors: list[str],
) -> str | None:
    try:
        site_url = load_book_site_url(source_root)
    except ValueError as error:
        errors.append(str(error))
        return None
    return site_url


def configured_site_path_prefix(
    source_root: Path,
    errors: list[str],
) -> str | None:
    site_url = configured_site_url(source_root, errors)
    return unquote(urlsplit(site_url).path) if site_url is not None else None


def validate_rendered_sitemap(
    site_dir: Path,
    site_url: str,
    errors: list[str],
) -> None:
    sitemap_path = site_dir / "sitemap.xml"
    sitemap_gzip_path = site_dir / "sitemap.xml.gz"
    if not sitemap_path.is_file():
        errors.append("rendered passive sitemap missing: sitemap.xml")
        return
    if not sitemap_gzip_path.is_file():
        errors.append("rendered passive sitemap gzip missing: sitemap.xml.gz")
    sitemap_data = file_bytes(sitemap_path, "rendered/sitemap.xml", errors)
    if sitemap_data is None:
        return
    content = utf8_content(sitemap_data, "rendered/sitemap.xml", errors)
    if content is None:
        return
    declaration = XML_DECLARATION_RE.match(content)
    remainder = content[declaration.end():] if declaration else content
    if (
        declaration is None
        or "<?" in remainder
        or re.search(r"<!DOCTYPE\b|<!ENTITY\b", remainder, re.IGNORECASE)
    ):
        errors.append("rendered sitemap contains active XML syntax")
        return
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        errors.append(f"rendered sitemap XML parse failure: {exc}")
        return
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    url_tag = f"{{{namespace}}}url"
    loc_tag = f"{{{namespace}}}loc"
    lastmod_tag = f"{{{namespace}}}lastmod"
    if root.tag != f"{{{namespace}}}urlset" or root.attrib:
        errors.append("rendered sitemap root differs from passive sitemap schema")
        return
    locations: list[str] = []
    for entry in root:
        if entry.tag != url_tag or entry.attrib:
            errors.append("rendered sitemap contains an unsupported entry")
            continue
        children = list(entry)
        if (
            [child.tag for child in children] != [loc_tag, lastmod_tag]
            or any(child.attrib or list(child) for child in children)
        ):
            errors.append("rendered sitemap entry differs from passive schema")
            continue
        location = (children[0].text or "").strip()
        last_modified = (children[1].text or "").strip()
        if not location.startswith(site_url):
            errors.append("rendered sitemap location escapes configured site_url")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_modified):
            errors.append("rendered sitemap lastmod must be an ISO calendar date")
        locations.append(location)
    if not locations or len(locations) != len(set(locations)):
        errors.append("rendered sitemap locations must be nonempty and unique")
    if sitemap_gzip_path.is_file():
        try:
            decompressed = gzip.decompress(sitemap_gzip_path.read_bytes())
        except (OSError, EOFError) as exc:
            errors.append(f"rendered sitemap gzip parse failure: {exc}")
        else:
            if decompressed != sitemap_data:
                errors.append("rendered sitemap gzip differs from sitemap.xml")


def inline_script_inventory_errors(observed: set[str]) -> list[str]:
    errors: list[str] = []
    unused = set(INLINE_SCRIPT_HASHES) - observed
    if unused:
        errors.append(
            "CSP authorizes unused executable inline-script hashes: "
            + ", ".join(sorted(unused))
        )
    return errors


def raw_html_structure_errors(content: str) -> list[str]:
    errors: list[str] = []
    html_space = r"[\t\n\f\r ]"
    if not re.match(
        rf"{html_space}*<!doctype{html_space}+html{html_space}*>"
        rf"{html_space}*<html(?:{html_space}[^<>]*)?>",
        content,
        re.IGNORECASE,
    ):
        errors.append("document must begin with one HTML5 doctype and html element")
    matches: dict[str, list[re.Match[str]]] = {}
    closing: dict[str, list[re.Match[str]]] = {}
    for tag in ("html", "head", "body"):
        matches[tag] = list(
            re.finditer(
                rf"<{tag}(?:{html_space}[^<>]*)?>",
                content,
                re.IGNORECASE,
            )
        )
        closing[tag] = list(
            re.finditer(
                rf"</{tag}{html_space}*>",
                content,
                re.IGNORECASE,
            )
        )
        if len(matches[tag]) != 1 or len(closing[tag]) != 1:
            errors.append(
                f"document must contain exactly one explicit {tag} start/end pair"
            )
    if all(len(matches[tag]) == len(closing[tag]) == 1 for tag in matches):
        order = (
            matches["html"][0].start(),
            matches["head"][0].start(),
            closing["head"][0].start(),
            matches["body"][0].start(),
            closing["body"][0].start(),
            closing["html"][0].start(),
        )
        if list(order) != sorted(order) or len(set(order)) != len(order):
            errors.append("document html/head/body element order is malformed")
        else:
            gaps = (
                content[matches["html"][0].end():matches["head"][0].start()],
                content[closing["head"][0].end():matches["body"][0].start()],
                content[closing["body"][0].end():closing["html"][0].start()],
                content[closing["html"][0].end():],
            )
            if any(
                not re.fullmatch(r"[\t\n\f\r ]*", gap)
                for gap in gaps
            ):
                errors.append(
                    "only HTML ASCII whitespace is allowed between "
                    "document structure elements"
                )
    head = matches["head"][0] if len(matches["head"]) == 1 else None
    expected_csp = (
        '<meta http-equiv="Content-Security-Policy" '
        f'content="{CONTENT_SECURITY_POLICY}">'
    )
    head_tail = content[head.end():] if head is not None else ""
    head_space = re.match(r"[\t\n\f\r ]*", head_tail)
    if (
        head is None
        or head_space is None
        or not head_tail[head_space.end():].startswith(expected_csp)
    ):
        errors.append(
            "exact Content-Security-Policy meta must be the first head element"
        )
    for match in re.finditer(
        r"<\s*(image|plaintext|xmp|noembed|noframes|frameset|frame|"
        r"marquee|applet|bgsound|keygen)\b",
        content,
        re.IGNORECASE,
    ):
        errors.append(
            f"legacy browser-active HTML element is forbidden: {match.group(1).lower()}"
        )
    charset = re.search(r"<meta\s+charset=[\"']?utf-8[\"']?\s*>", content, re.IGNORECASE)
    if charset is None or len(content[:charset.start()].encode("utf-8")) >= 1024:
        errors.append("UTF-8 charset metadata must occur within the first 1024 bytes")
    return errors


def scan_rendered_site(
    site_dir: Path,
    errors: list[str],
    source_root: Path | None = None,
) -> int:
    if not site_dir.is_dir():
        errors.append(f"rendered site directory missing: {site_dir}")
        return 0
    if source_root is not None:
        local_script_allowlist, locked_theme_stylesheets = (
            verify_rendered_javascript_assets(site_dir, source_root, errors)
        )
    else:
        local_script_allowlist, locked_theme_stylesheets = None, set()
    if source_root is not None:
        errors.extend(deployed_asset_tree_errors(source_root, site_dir))
    site_url = configured_site_url(source_root, errors) if source_root is not None else None
    site_path_prefix = unquote(urlsplit(site_url).path) if site_url is not None else None
    if site_url is not None:
        validate_rendered_sitemap(site_dir, site_url, errors)
    html_count = 0
    observed_inline_script_hashes: set[str] = set()
    for path in sorted(site_dir.rglob("*")):
        if path.is_symlink():
            errors.append(
                "rendered site must not contain symlinks: "
                f"{path.relative_to(site_dir).as_posix()}"
            )
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(site_dir).as_posix()
        suffix = path.suffix.lower()
        if relative == "sitemap.xml" and site_url is not None:
            continue
        if relative == "sitemap.xml.gz" and site_url is not None:
            continue
        if relative.casefold().endswith(".xml.gz") and relative != "sitemap.xml.gz":
            errors.append(f"forbidden active rendered file type: {relative}")
            continue
        if suffix in FORBIDDEN_BROWSER_DOCUMENT_SUFFIXES:
            errors.append(
                f"forbidden active rendered file type: "
                f"{relative}"
            )
            continue
        static_errors, content = rendered_static_file_errors(path, relative)
        errors.extend(static_errors)
        if content is None:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} pattern in rendered/{relative}")
        for label, pattern in IDENTIFIER_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} in rendered/{relative}")
        if path.suffix.lower() != ".html":
            scan_markers(f"rendered/{relative}", content, errors)
        locked_theme_javascript = (
            source_root is not None
            and suffix in SCRIPT_SUFFIXES
            and relative.startswith("assets/javascripts/")
        )
        locked_theme_stylesheet = relative in locked_theme_stylesheets
        if (
            suffix == ".css" or suffix in SCRIPT_SUFFIXES
        ) and not locked_theme_javascript and not locked_theme_stylesheet:
            scan_browser_egress(
                f"rendered/{relative}",
                content,
                ".css" if suffix == ".css" else ".js",
                errors,
            )
        for path_label, pattern in LOCAL_PATH_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{path_label} in rendered/{relative}")
        if path.suffix.lower() == ".svg":
            scan_svg(f"rendered/{relative}", content, errors)
        if path.suffix.lower() != ".html":
            continue
        html_count += 1
        for structure_error in raw_html_structure_errors(content):
            errors.append(f"rendered/{relative} {structure_error}")
        parser = RenderedSurfaceParser()
        try:
            parser.feed(content)
            parser.close()
        except Exception as exc:  # HTMLParser can raise on malformed character references.
            errors.append(f"rendered HTML parse failure in {relative}: {exc}")
            continue
        if parser.content_security_policies != [CONTENT_SECURITY_POLICY]:
            errors.append(
                f"rendered/{relative} must contain exactly the reviewed "
                "Content-Security-Policy"
            )
        if parser.referrer_policies != ["strict-origin-when-cross-origin"]:
            errors.append(
                f"rendered/{relative} must contain exactly one reviewed "
                "strict-origin-when-cross-origin referrer policy"
            )
        for description in parser.unsafe_forms:
            errors.append(f"data-submission form in rendered/{relative}: {description}")
        for description in parser.sensitive_inputs:
            errors.append(f"sensitive input in rendered/{relative}: {description}")
        for description in parser.dangerous_attributes:
            errors.append(
                f"dangerous active markup in rendered/{relative}: {description}"
            )
        for tag, attribute, url in parser.remote_resources:
            errors.append(
                f"non-allowlisted remote resource in rendered/{relative}: "
                f"{tag}[{attribute}]={url}"
            )
        for source in parser.script_sources:
            if source == MATHJAX_URL or is_remote(source):
                continue
            validate_local_script_source(
                site_dir,
                path,
                source,
                local_script_allowlist,
                site_path_prefix,
                errors,
            )
        for active_tag, active_type, active_text, active_in_svg in (
            parser.active_text_blocks
        ):
            scan_markers(f"rendered/{relative} inline script/style", active_text, errors)
            if active_in_svg and active_tag == "style" and active_svg_css(active_text):
                errors.append(
                    f"active inline SVG style in rendered/{relative}"
                )
            if active_tag == "script" and active_type in INERT_SCRIPT_TYPES:
                continue
            if active_tag == "script":
                digest = hashlib.sha256(active_text.encode("utf-8")).digest()
                encoded = base64.b64encode(digest).decode("ascii")
                observed_inline_script_hashes.add(encoded)
                if encoded not in INLINE_SCRIPT_HASHES:
                    errors.append(
                        f"unreviewed inline script hash in rendered/{relative}: "
                        f"sha256-{encoded}"
                    )
            scan_browser_egress(
                f"rendered/{relative} inline {active_tag}",
                active_text,
                ".css" if active_tag == "style" else ".js",
                errors,
            )
    if source_root is not None:
        errors.extend(inline_script_inventory_errors(observed_inline_script_hashes))
    return html_count


def originality_provenance_errors(
    text: str,
    rights_records: list[dict[str, object]],
) -> list[str]:
    """Keep the human-readable provenance counts bound to the rights register."""
    errors: list[str] = []
    normalized = re.sub(r"[*_`]", "", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    asset_count = len(rights_records)
    status_counts: dict[str, int] = {}
    for record in rights_records:
        status = record.get("rights_status")
        if isinstance(status, str):
            status_counts[status] = status_counts.get(status, 0) + 1
    expected_asset_claim = (
        f"current figure set contains {asset_count} png/svg assets."
    )
    if expected_asset_claim not in normalized:
        errors.append(
            "ORIGINALITY.md current-figure count differs from rights register"
        )
    expected_status_claim = (
        "rights-register evidence counts: "
        f"{status_counts.get('repository-candidate-generator-link', 0)} "
        "generator-candidate records, "
        f"{status_counts.get('embedded-tool-metadata-only', 0)} "
        "embedded-tool-metadata-only records, and "
        f"{status_counts.get('repository-content-history-only', 0)} "
        "repository-content-history-only records."
    )
    if expected_status_claim not in normalized:
        errors.append(
            "ORIGINALITY.md rights-status counts differ from rights register"
        )
    return errors


def publication_errors(root: Path, site_dir: Path | None = None) -> tuple[list[str], int, int, int]:
    errors: list[str] = []
    try:
        tracked = tracked_repository_files(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"unable to inventory tracked repository files: {exc}")
        tracked = set()
    reported_symlinks: set[Path] = set()

    for relative_path in sorted(tracked, key=lambda path: path.as_posix()):
        path = root / relative_path
        if path.is_symlink():
            errors.append(
                f"tracked repository symlink is forbidden: {relative_path.as_posix()}"
            )
            reported_symlinks.add(relative_path)
            continue
        if not path.is_file() or is_published_source(relative_path):
            continue
        data = file_bytes(path, f"tracked/{relative_path.as_posix()}", errors)
        if data is None:
            continue
        if path.suffix.casefold() == ".svg":
            errors.append(
                "tracked SVG is outside the inventoried figure tree: "
                f"{relative_path.as_posix()}"
            )
            continue
        if path.suffix.casefold() not in REPOSITORY_TEXT_SUFFIXES:
            errors.append(
                "unreviewed tracked binary outside inventoried publication assets: "
                f"{relative_path.as_posix()}"
            )
            continue
        utf8_content(data, f"tracked/{relative_path.as_posix()}", errors)

    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if path.is_symlink():
            if (
                relative_path not in reported_symlinks
                and is_published_source(relative_path)
            ):
                errors.append(
                    f"published-source symlink is forbidden: {relative_path.as_posix()}"
                )
            continue
        if not path.is_file():
            continue
        if is_skipped_root_tree(relative_path, tracked):
            continue
        if is_published_source(relative_path):
            errors.extend(published_source_file_errors(path, relative_path))
        if path.suffix.lower() in FORBIDDEN_TREE_SUFFIXES:
            errors.append(f"forbidden publication-tree file type: {relative_path.as_posix()}")
        if path.name.casefold() in FORBIDDEN_GITLEAKS_CONTROL_FILES:
            errors.append(
                f"forbidden Gitleaks override file: {relative_path.as_posix()}"
            )

    for relative in REQUIRED:
        if not (root / relative).is_file():
            errors.append(f"required file missing: {relative}")
    release_errors, release = release_identity_errors(root)
    errors.extend(release_errors)

    standards_path = root / "docs" / "publication-standards.md"
    if standards_path.is_file():
        standards = read(standards_path).casefold()
        for phrase in (
            "ai tools have assisted",
            "medical advice",
            "no user accounts, uploads, or data-submission forms",
            "did not constitute a legal opinion",
            "wcag 2.2 aa",
        ):
            if phrase not in standards:
                errors.append(f"publication disclosure missing phrase: {phrase}")

    override_path = root / "overrides" / "main.html"
    if override_path.is_file() and CONTENT_SECURITY_POLICY not in read(override_path):
        errors.append("reviewed Content-Security-Policy missing from overrides/main.html")

    all_text: dict[Path, str] = {
        path: read(path) for path in text_files(root, tracked)
    }
    for path, content in all_text.items():
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        scanner_self = relative == "scripts/check_publication_safety.py"
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} pattern in {relative}")
        for label, pattern in IDENTIFIER_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} in {relative}")
        if not scanner_self:
            for path_label, pattern in LOCAL_PATH_PATTERNS.items():
                if pattern.search(content):
                    errors.append(f"{path_label} in {relative}")
            for stale_claim in (
                "Public Pages: GO",
                "Practical copyright risk: LOW",
                "full counsel opinion",
            ):
                if stale_claim.casefold() in content.casefold():
                    errors.append(
                        f"stale release/legal claim in {relative}: {stale_claim}"
                    )
        published_source = (
            bool(relative_path.parts)
            and relative_path.parts[0] in {"docs", "overrides"}
        ) or relative == "mkdocs.yml"
        if published_source:
            authored_browser_script = (
                len(relative_path.parts) >= 2
                and relative_path.parts[:2] == ("docs", "javascripts")
            )
            if path.suffix.lower() in {".md", ".html"} and re.search(
                r"<\s*script\b", content, re.IGNORECASE
            ):
                errors.append(f"raw script element in published source: {relative}")
            if path.suffix.lower() in {".md", ".html"} and re.search(
                r"<\s*(?:base|embed|iframe|object|svg)\b",
                content,
                re.IGNORECASE,
            ):
                errors.append(f"raw active element in published source: {relative}")
            if path.suffix.lower() in ({".css", ".html", ".svg"} | SCRIPT_SUFFIXES) or (
                relative == "mkdocs.yml"
            ) or authored_browser_script:
                scan_markers(relative, content, errors)
            if path.suffix.lower() == ".css":
                scan_browser_egress(relative, content, ".css", errors)
            if path.suffix.lower() in SCRIPT_SUFFIXES or authored_browser_script:
                scan_browser_egress(relative, content, ".js", errors)
            if path.suffix.lower() in {".md", ".html"} and re.search(
                r"<\s*form\b", content, re.IGNORECASE
            ):
                errors.append(f"raw form element in published source: {relative}")
            if path.suffix.lower() == ".svg":
                scan_svg(relative, content, errors)

    mathjax_path = root / "docs" / "javascripts" / "mathjax.js"
    if mathjax_path.is_file():
        mathjax = read(mathjax_path)
        if "cdn.jsdelivr.net" in mathjax:
            for requirement in (
                "@3.2.2/",
                f'script.integrity = "{MATHJAX_SRI}"',
                'script.crossOrigin = "anonymous"',
                'script.referrerPolicy = "no-referrer"',
                "mathjax-fallback-notice",
            ):
                if requirement not in mathjax:
                    errors.append(f"MathJax remote-load control missing: {requirement}")

    errors.extend(
        browser_asset_inventory_errors(
            root,
            str(release.get("review_date", "")),
        )
    )

    rights_path = root / "_meta" / "asset-rights-register.json"
    manifest_path = root / "docs" / "assets" / "figures" / "manifest.json"
    for path in unsupported_figure_assets(root):
        relative = path.relative_to(
            root / "docs" / "assets" / "figures"
        ).as_posix()
        errors.append(f"unsupported teaching-figure file type: {relative}")
    if rights_path.is_file() and manifest_path.is_file():
        try:
            rights = json.loads(read(rights_path))
            manifest = json.loads(read(manifest_path))
            review_date = str(release.get("review_date", ""))
            if rights.get("review_date") != review_date:
                errors.append("asset rights register date differs from release metadata")
            if manifest.get("audit_date") != review_date:
                errors.append("asset manifest date differs from release metadata")
            rights_records = rights.get("assets", [])
            manifest_records = manifest.get("assets", [])
            if not isinstance(rights_records, list) or not isinstance(manifest_records, list):
                raise TypeError("asset arrays must be lists")
            if rights.get("asset_count") != len(rights_records):
                errors.append("asset rights register asset_count differs from records")
            if manifest.get("asset_count") != len(manifest_records):
                errors.append("asset manifest asset_count differs from records")
            originality_path = root / "_meta" / "ORIGINALITY.md"
            if originality_path.is_file():
                errors.extend(
                    originality_provenance_errors(
                        read(originality_path),
                        [
                            record
                            for record in rights_records
                            if isinstance(record, dict)
                        ],
                    )
                )
            rights_paths = [record.get("path") for record in rights_records if isinstance(record, dict)]
            manifest_paths = [
                record.get("path") for record in manifest_records if isinstance(record, dict)
            ]
            if len(rights_paths) != len(set(rights_paths)):
                errors.append("duplicate asset path in rights register")
            if len(manifest_paths) != len(set(manifest_paths)):
                errors.append("duplicate asset path in asset manifest")
            rights_by_path = {
                record["path"]: record
                for record in rights_records
                if isinstance(record, dict) and isinstance(record.get("path"), str)
            }
            manifest_by_path = {
                record["path"]: record
                for record in manifest_records
                if isinstance(record, dict) and isinstance(record.get("path"), str)
            }
            current_paths = [
                path.relative_to(root).as_posix() for path in figure_assets(root)
            ]
            if set(rights_by_path) != set(current_paths):
                errors.append("asset rights register does not cover the current figure set")
            if set(manifest_by_path) != set(current_paths):
                errors.append("asset manifest does not cover the current figure set")
            for relative in current_paths:
                digest = sha256(root / relative)
                if rights_by_path.get(relative, {}).get("sha256") != digest:
                    errors.append(f"rights-register hash mismatch: {relative}")
                if manifest_by_path.get(relative, {}).get("sha256") != digest:
                    errors.append(f"manifest hash mismatch: {relative}")
                if not rights_by_path.get(relative, {}).get("release_basis"):
                    errors.append(f"asset release basis missing: {relative}")
                content_commit = rights_by_path.get(relative, {}).get(
                    "current_content_commit"
                )
                if not content_commit:
                    errors.append(f"current-content commit missing: {relative}")
                if manifest_by_path.get(relative, {}).get(
                    "current_content_commit"
                ) != content_commit:
                    errors.append(f"manifest current-content commit mismatch: {relative}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"asset register/manifest parse failure: {exc}")

    requirements_path = root / "requirements.txt"
    lock_path = root / "requirements.lock"
    sbom_path = root / "sbom.cdx.json"
    if lock_path.is_file() and sbom_path.is_file():
        try:
            sbom = json.loads(read(sbom_path))
            errors.extend(
                dependency_inventory_errors(
                    read(requirements_path) if requirements_path.is_file() else "",
                    read(lock_path),
                    sbom,
                )
            )
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"SBOM parse failure: {exc}")
    audit_requirements_path = root / "requirements-audit.in"
    audit_lock_path = root / "requirements-audit.lock"
    audit_bootstrap_requirements_path = root / "requirements-audit-bootstrap.in"
    audit_bootstrap_path = root / "requirements-audit-bootstrap.lock"
    bootstrap_packages: set[tuple[str, str]] = set()
    if (
        audit_bootstrap_path.is_file()
        and read(audit_bootstrap_path) != AUDIT_BOOTSTRAP_LOCK
    ):
        errors.append(
            "requirements-audit-bootstrap.lock differs from the reviewed "
            "wheel-only build-tool lock"
        )
    if (
        audit_bootstrap_requirements_path.is_file()
        and audit_bootstrap_path.is_file()
    ):
        bootstrap_errors, bootstrap_packages = lockfile_errors(
            read(audit_bootstrap_requirements_path),
            read(audit_bootstrap_path),
            "requirements-audit-bootstrap.in",
            "requirements-audit-bootstrap.lock",
        )
        errors.extend(bootstrap_errors)
        if bootstrap_packages != {
            ("pip", "26.2"),
            ("setuptools", "83.0.0"),
        }:
            errors.append(
                "audit bootstrap must contain only the reviewed pip and "
                "setuptools wheel pins"
            )
    if audit_requirements_path.is_file() and audit_lock_path.is_file():
        audit_errors, audit_packages = lockfile_errors(
            read(audit_requirements_path),
            read(audit_lock_path),
            "requirements-audit.in",
            "requirements-audit.lock",
        )
        errors.extend(audit_errors)
        if not bootstrap_packages.issubset(audit_packages):
            errors.append(
                "audit bootstrap pins must match packages in requirements-audit.lock"
            )

    font_directory = root / "docs" / "assets" / "fonts"
    font_files = (
        sorted(
            (
                path
                for path in font_directory.rglob("*")
                if path.is_file() and path.suffix.casefold() == ".woff2"
            ),
            key=lambda path: (
                path.relative_to(font_directory).as_posix().casefold(),
                path.relative_to(font_directory).as_posix(),
            ),
        )
        if font_directory.is_dir()
        else []
    )
    if font_files:
        for relative in (
            "docs/assets/fonts/OFL-1.1.txt",
            "docs/assets/fonts/README.md",
        ):
            if not (root / relative).is_file():
                errors.append(f"vendored-font notice missing: {relative}")
        font_readme = read(root / "docs/assets/fonts/README.md") if (
            root / "docs/assets/fonts/README.md"
        ).is_file() else ""
        font_inventory, font_inventory_errors = font_inventory_from_readme(font_readme)
        errors.extend(font_inventory_errors)
        current_font_paths = {
            font.relative_to(font_directory).as_posix() for font in font_files
        }
        if set(font_inventory) != current_font_paths:
            errors.append("vendored-font inventory does not cover the current font set")
        for font in font_files:
            relative = font.relative_to(font_directory).as_posix()
            digest = hashlib.sha256(font.read_bytes()).hexdigest()
            if font_inventory.get(relative) != digest:
                errors.append(f"font inventory/hash missing: {relative}")

    rendered_count = (
        scan_rendered_site(site_dir, errors, source_root=root) if site_dir else 0
    )
    figure_count = len(figure_assets(root))
    return errors, len(all_text), figure_count, rendered_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-dir",
        type=Path,
        help="Also validate the final post-processed rendered site directory.",
    )
    args = parser.parse_args()
    site_dir = args.site_dir.resolve() if args.site_dir else None
    errors, file_count, figure_count, rendered_count = publication_errors(ROOT, site_dir)
    if errors:
        for error in errors:
            print(f"PUBLICATION_SAFETY_ERROR {error}")
        return 1
    print(
        "PUBLICATION_SAFETY_OK "
        f"files={file_count} figures={figure_count} rendered_html={rendered_count} "
        "scope=current-tree-bounded"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
