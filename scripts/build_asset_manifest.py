#!/usr/bin/env python3
"""Build or verify a deterministic teaching-figure provenance manifest."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from xml.etree import ElementTree

from asset_inventory import canonical_bytes, figure_assets, sha256


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
            width_value, height_value = float(viewbox[2]), float(viewbox[3])
            width, height = round(width_value), round(height_value)
            if (
                not math.isfinite(width_value)
                or not math.isfinite(height_value)
                or width_value <= 0
                or height_value <= 0
                or width <= 0
                or height <= 0
                or width > 8192
                or height > 8192
                or width * height > 32_000_000
            ):
                raise ValueError(
                    f"{path}: invalid SVG dimensions {width_value}x{height_value}"
                )
            return width, height
        raise ValueError(f"{path}: SVG requires a four-value viewBox")
    return None, None


def build(root: Path) -> dict[str, object]:
    rights_path = root / "_meta" / "asset-rights-register.json"
    release = json.loads(
        (root / "_meta" / "release.json").read_text(encoding="utf-8")
    )
    rights_register = json.loads(rights_path.read_text(encoding="utf-8"))
    rights_by_path = {
        record["path"]: record for record in rights_register["assets"]
    }
    assets = []
    for path in figure_assets(root):
        data = canonical_bytes(path)
        width, height = dimensions(path)
        if b"Matplotlib" in data:
            software = "Matplotlib"
        elif b"Generated deterministically by scripts/regenerate_accuracy_figures.py" in data:
            software = "Python SVG generator (scripts/regenerate_accuracy_figures.py)"
        elif path.suffix.lower() == ".svg":
            software = "SVG; generator not identified in embedded metadata"
        else:
            software = "not detected"
        relative = path.relative_to(root).as_posix()
        rights = rights_by_path.get(relative)
        if rights is None:
            raise ValueError(f"{relative}: missing asset-rights-register entry")
        assets.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": len(data),
                "width": width,
                "height": height,
                "software_metadata": software,
                "rights_status": rights["rights_status"],
                "current_content_commit": rights["current_content_commit"],
                "provenance_assertion": rights["release_basis"],
            }
        )
    return {
        "schema_version": 3,
        "audit_date": str(release["review_date"]),
        "scope": "docs/assets/figures PNG and SVG assets",
        "rights_register": "_meta/asset-rights-register.json",
        "caveat": (
            "Hashes, repository history, and software metadata support "
            "traceability but do not independently prove chain of title, "
            "human authorship, copyrightability, or non-infringement."
        ),
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
