#!/usr/bin/env python3
"""Build or verify a deterministic teaching-figure provenance manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from xml.etree import ElementTree


def dimensions(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() == ".png":
        data = path.read_bytes()[:24]
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path}: .png extension does not match PNG bytes")
        width, height = struct.unpack(">II", data[16:24])
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}: invalid PNG dimensions {width}x{height}")
        return width, height
    if path.suffix.lower() == ".svg":
        root = ElementTree.parse(path).getroot()
        viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()
        if len(viewbox) == 4:
            return round(float(viewbox[2])), round(float(viewbox[3]))
    return None, None


def build(root: Path) -> dict[str, object]:
    figure_dir = root / "docs" / "assets" / "figures"
    assets = []
    for path in sorted(figure_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".svg"}:
            continue
        data = path.read_bytes()
        width, height = dimensions(path)
        if b"Matplotlib" in data:
            software = "Matplotlib"
        elif b"Generated deterministically by scripts/regenerate_accuracy_figures.py" in data:
            software = "Python SVG generator (scripts/regenerate_accuracy_figures.py)"
        elif path.suffix.lower() == ".svg":
            software = "hand-authored SVG"
        else:
            software = "not detected"
        assets.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "width": width,
                "height": height,
                "software_metadata": software,
                "provenance_assertion": "Teaching asset present in this repository; authorship and rights have not been independently verified.",
            }
        )
    return {
        "schema_version": 1,
        "audit_date": "2026-07-16",
        "scope": "docs/assets/figures PNG and SVG assets",
        "caveat": "Hashes and software metadata support traceability but do not prove authorship or eliminate copyright risk.",
        "asset_count": len(assets),
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "docs" / "assets" / "figures" / "manifest.json"
    rendered = json.dumps(build(root), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            print("ASSET_MANIFEST_STALE")
            return 1
        print(f"ASSET_MANIFEST_OK {target}")
        return 0
    target.write_text(rendered, encoding="utf-8")
    print(f"ASSET_MANIFEST_WRITTEN {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
