#!/usr/bin/env python3
"""Validate the committed SBOM against the official CycloneDX 1.6 JSON schema."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from jsonschema import Draft7Validator, RefResolver

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_BASE = "https://cyclonedx.org/schema/"
SCHEMA_FILES = (
    "bom-1.6.schema.json",
    "spdx.schema.json",
    "jsf-0.82.schema.json",
)


def fetch_schema(name: str) -> dict[str, object]:
    request = Request(
        SCHEMA_BASE + name,
        headers={"User-Agent": "ai-books-release-gate/2026.07.30"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    store: dict[str, dict[str, object]] = {}
    for name in SCHEMA_FILES:
        schema = fetch_schema(name)
        store[SCHEMA_BASE + name] = schema
        store["http://cyclonedx.org/schema/" + name] = schema
        store[name] = schema
    primary_url = SCHEMA_BASE + SCHEMA_FILES[0]
    primary = store[primary_url]
    document = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
    resolver = RefResolver(base_uri=primary_url, referrer=primary, store=store)
    errors = sorted(
        Draft7Validator(primary, resolver=resolver).iter_errors(document),
        key=lambda error: [str(item) for item in error.path],
    )
    if errors:
        for error in errors[:20]:
            location = "/".join(str(item) for item in error.path) or "<root>"
            print(f"SBOM_SCHEMA_ERROR {location}: {error.message}")
        return 1
    print("SBOM_SCHEMA_OK spec=1.6 source=cyclonedx.org")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
