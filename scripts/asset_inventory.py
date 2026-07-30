#!/usr/bin/env python3
"""Canonical teaching-figure inventory and hashing policy."""
from __future__ import annotations

import hashlib
from pathlib import Path

FIGURE_RELATIVE_DIR = Path("docs/assets/figures")
SUPPORTED_FIGURE_SUFFIXES = frozenset({".png", ".svg"})


def canonicalize(data: bytes, suffix: str) -> bytes:
    """Normalize text assets so hashes are stable across Git line endings."""
    return data.replace(b"\r\n", b"\n") if suffix.casefold() == ".svg" else data


def canonical_bytes(path: Path) -> bytes:
    return canonicalize(path.read_bytes(), path.suffix)


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def figure_directory(root: Path) -> Path:
    return root / FIGURE_RELATIVE_DIR


def relative_sort_key(path: Path, directory: Path) -> tuple[str, str]:
    relative = path.relative_to(directory).as_posix()
    return relative.casefold(), relative


def figure_assets(root: Path) -> list[Path]:
    directory = figure_directory(root)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_FIGURE_SUFFIXES
        ),
        key=lambda path: relative_sort_key(path, directory),
    )


def unsupported_figure_assets(root: Path) -> list[Path]:
    directory = figure_directory(root)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path != directory / "manifest.json"
            and path.suffix.casefold() not in SUPPORTED_FIGURE_SUFFIXES
        ),
        key=lambda path: relative_sort_key(path, directory),
    )
