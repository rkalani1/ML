#!/usr/bin/env python3
"""Derive the reviewed book identity from its canonical MkDocs URL."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode


@dataclass(frozen=True)
class BookIdentity:
    slug: str
    policy: str
    expected_curriculum: int


BOOKS = {
    "https://rkalani1.github.io/ML/": BookIdentity(
        slug="ML",
        policy="ml",
        expected_curriculum=20,
    ),
    "https://rkalani1.github.io/CRIT-APP/": BookIdentity(
        slug="CRIT-APP",
        policy="crit",
        expected_curriculum=28,
    ),
}


def load_book_site_url(root: Path) -> str:
    """Return the sole effective, reviewed top-level MkDocs site URL."""
    config = root / "mkdocs.yml"
    try:
        source = config.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read {config}: {error}") from error
    try:
        document = yaml.compose(source, Loader=yaml.SafeLoader)
        parsed = yaml.safe_load(source)
    except yaml.YAMLError as error:
        raise ValueError(f"unable to parse mkdocs.yml: {error}") from error
    if not isinstance(document, MappingNode) or not isinstance(parsed, dict):
        raise ValueError("mkdocs.yml must be a top-level mapping")
    site_url_keys = [
        key
        for key, _ in document.value
        if isinstance(key, ScalarNode) and key.value == "site_url"
    ]
    if len(site_url_keys) != 1:
        raise ValueError(
            "mkdocs.yml must declare exactly one top-level site_url key"
        )
    site_url = parsed.get("site_url")
    if not isinstance(site_url, str) or site_url not in BOOKS:
        raise ValueError(
            "mkdocs.yml site_url does not identify a reviewed book"
        )
    return site_url


def load_book_identity(root: Path) -> BookIdentity:
    """Return a fail-closed identity that is independent of checkout dirname."""
    return BOOKS[load_book_site_url(root)]
