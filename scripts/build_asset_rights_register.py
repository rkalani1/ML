#!/usr/bin/env python3
"""Build or check the current teaching-asset rights/provenance register.

Refreshing discovers provenance from repository history. CI uses --check,
which validates current asset names and hashes against the recorded commit
objects directly without repeating a full history search. Those recorded
objects must be available to Git.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import subprocess
from pathlib import Path

from asset_inventory import canonical_bytes, canonicalize, figure_assets, sha256

GENERATOR_MARKERS = (
    "matplotlib",
    "savefig(",
    ".save(",
    "write_bytes(",
    "write_text(",
    "svgwrite",
)

def run_git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout


def run_git_bytes(root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return process.stdout


@lru_cache(maxsize=None)
def file_at_commit(root: Path, commit: str, relative: str) -> bytes:
    return run_git_bytes(root, "show", f"{commit}:{relative}")


@lru_cache(maxsize=None)
def commit_metadata(root: Path, commit: str) -> tuple[str, str, str, str]:
    output = run_git(
        root,
        "show",
        "-s",
        "--format=%H%x1f%an%x1f%ad%x1f%s",
        "--date=short",
        commit,
    ).strip()
    fields = output.split("\x1f", 3)
    if len(fields) != 4:
        raise RuntimeError(f"Unable to parse provenance commit metadata: {commit}")
    return fields[0], fields[1], fields[2], fields[3]


@lru_cache(maxsize=None)
def changed_paths(root: Path, commit: str) -> tuple[str, ...]:
    return tuple(
        item
        for item in run_git(
            root,
            "show",
            "--format=",
            "--name-only",
            "--diff-filter=ACMR",
            commit,
        ).splitlines()
        if item
    )


@lru_cache(maxsize=None)
def commit_is_ancestor(root: Path, commit: str) -> bool:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if process.returncode not in {0, 1}:
        raise RuntimeError(f"Unable to test provenance ancestry: {commit}")
    return process.returncode == 0


def release_metadata(root: Path) -> dict[str, object]:
    return json.loads((root / "_meta" / "release.json").read_text(encoding="utf-8"))


def record_for_commit(
    root: Path,
    path: Path,
    review_date: str,
    commit: str,
) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    current = canonical_bytes(path)
    historical = file_at_commit(root, commit, relative)
    if canonicalize(historical, path.suffix) != current:
        raise RuntimeError(
            f"Current bytes for {relative} do not match recorded commit {commit}"
        )
    commit, author, date, subject = commit_metadata(root, commit)
    source_candidates: list[str] = []
    for item in changed_paths(root, commit):
        if not item.startswith("scripts/") or Path(item).suffix.lower() not in {
            ".py",
            ".r",
            ".ipynb",
            ".js",
        }:
            continue
        try:
            source = file_at_commit(root, commit, item).decode(
                "utf-8", errors="replace"
            )
        except subprocess.CalledProcessError:
            continue
        folded = source.casefold()
        looks_like_generator = Path(item).name.casefold().startswith(
            ("generate_", "regenerate_")
        )
        directly_names_asset = (
            path.name.casefold() in folded or path.stem.casefold() in folded
        )
        has_generator_operation = any(marker in folded for marker in GENERATOR_MARKERS)
        if looks_like_generator and directly_names_asset and has_generator_operation:
            source_candidates.append(item)
    source_candidates.sort()

    data = canonical_bytes(path)
    if b"Matplotlib" in data:
        embedded_tool = "Matplotlib metadata"
    elif b"Generated deterministically by " in data:
        embedded_tool = "embedded deterministic-generator statement"
    elif path.suffix.lower() == ".svg":
        embedded_tool = "SVG markup is directly inspectable; no generator statement detected"
    else:
        embedded_tool = "no generator metadata detected in the binary"

    if source_candidates:
        status = "repository-candidate-generator-link"
        basis = (
            "The exact current asset bytes match a repository commit that also "
            "contains a generator-named script that directly names the asset and "
            "contains a file-rendering operation. It is a candidate generator link, "
            "not proof that the script produced those bytes, independent ownership, "
            "or non-infringement clearance."
        )
    elif embedded_tool in {
        "Matplotlib metadata",
        "embedded deterministic-generator statement",
    }:
        status = "embedded-tool-metadata-only"
        basis = (
            "The exact current asset bytes match repository history and the current "
            "file contains tool metadata. No directly linked generator source was "
            "verified; this is not independent rights clearance."
        )
    else:
        status = "repository-content-history-only"
        basis = (
            "The exact current asset bytes match repository history, but no directly "
            "linked generator source or asset-specific chain-of-title attestation "
            "was verified."
        )

    return {
        "path": relative,
        "sha256": sha256(path),
        "current_content_commit": commit,
        "current_content_author": author,
        "current_content_date": date,
        "current_content_subject": subject,
        "current_content_source_candidates": source_candidates,
        "embedded_tool_evidence": embedded_tool,
        "rights_status": status,
        "release_basis": basis,
        "ai_assistance": (
            "Repository-wide AI assistance is disclosed; asset-level model "
            "contribution and human-authorship sufficiency are not fully documented."
        ),
        "review_record": (
            f"AI-assisted repository audit dated {review_date}; no asset-specific "
            "human-authorship or chain-of-title attestation is recorded."
        ),
        "release_decision": (
            "Distributed as part of the book. Any CC BY 4.0 grant applies only "
            "to rights the publisher controls; this record is not legal clearance."
        ),
    }


def current_content_record(
    root: Path, path: Path, review_date: str
) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    commits = [
        commit
        for commit in run_git(
            root,
            "log",
            "--follow",
            "--format=%H",
            "--",
            relative,
        ).splitlines()
        if commit
    ]
    current = canonical_bytes(path)
    for commit in commits:
        try:
            historical = file_at_commit(root, commit, relative)
        except subprocess.CalledProcessError:
            continue
        if canonicalize(historical, path.suffix) == current:
            return record_for_commit(root, path, review_date, commit)
    raise RuntimeError(
        f"Current bytes for {relative} do not match a committed path version"
    )


def assets(root: Path) -> list[Path]:
    return figure_assets(root)


def register_payload(
    review_date: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "review_date": review_date,
        "scope": "Current PNG and SVG teaching assets under docs/assets/figures",
        "method": (
            "Exact current hashes verified against the recorded path commit with "
            "identical bytes, plus directly linked same-commit generator candidates "
            "and embedded tool metadata. --refresh-from-history selects the newest "
            "matching path-changing commit at refresh time."
        ),
        "limitations": (
            "Repository provenance and tool metadata are not chain-of-title proof. "
            "Similarity, third-party training inputs, degree of human authorship, "
            "copyrightability, trademark, privacy, and publicity rights require "
            "fact-specific review."
        ),
        "asset_count": len(records),
        "assets": records,
    }


def build(root: Path) -> dict[str, object]:
    release = release_metadata(root)
    review_date = str(release["review_date"])
    records = [
        current_content_record(root, path, review_date) for path in assets(root)
    ]
    return register_payload(review_date, records)


def validate(root: Path, register: dict[str, object]) -> list[str]:
    errors: list[str] = []
    records = register.get("assets")
    if not isinstance(records, list):
        return ["Register assets must be a list"]
    path_values = [
        record.get("path")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    ]
    if len(path_values) != len(records):
        errors.append("Every register entry must be an object with a string path")
    if len(path_values) != len(set(path_values)):
        errors.append("Duplicate register path entries are not allowed")
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    current = {
        path.relative_to(root).as_posix(): sha256(path) for path in assets(root)
    }
    if set(by_path) != set(current):
        missing = sorted(set(current) - set(by_path))
        extra = sorted(set(by_path) - set(current))
        if missing:
            errors.append(f"Missing register entries: {', '.join(missing)}")
        if extra:
            errors.append(f"Stale register entries: {', '.join(extra)}")
    for relative, digest in current.items():
        record = by_path.get(relative, {})
        if record.get("sha256") != digest:
            errors.append(f"Hash mismatch: {relative}")
        for field in (
            "current_content_commit",
            "current_content_author",
            "current_content_date",
            "current_content_subject",
            "rights_status",
            "release_basis",
            "ai_assistance",
            "review_record",
            "release_decision",
        ):
            if not record.get(field):
                errors.append(f"Missing {field}: {relative}")
    if register.get("asset_count") != len(current):
        errors.append("asset_count does not match current asset count")
    try:
        review_date = str(release_metadata(root)["review_date"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"Unable to read release metadata: {exc}")
        return errors
    expected_records: list[dict[str, object]] = []
    for path in assets(root):
        relative = path.relative_to(root).as_posix()
        commit = by_path.get(relative, {}).get("current_content_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or commit == "0" * 40
        ):
            errors.append(f"Invalid current_content_commit: {relative}")
            continue
        try:
            if not commit_is_ancestor(root, commit):
                errors.append(
                    f"Recorded provenance commit is not an ancestor of HEAD: {relative}"
                )
                continue
            if relative not in changed_paths(root, commit):
                errors.append(
                    f"Recorded provenance commit did not change asset path: {relative}"
                )
                continue
            expected_records.append(
                record_for_commit(root, path, review_date, commit)
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"Unable to verify recorded provenance: {relative}: {exc}")
    expected = register_payload(review_date, expected_records)
    if len(expected_records) == len(current) and register != expected:
        errors.append(
            "Committed register differs from direct commit verification; "
            "run --refresh-from-history and review the resulting records"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--refresh-from-history", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "_meta" / "asset-rights-register.json"

    if args.refresh_from_history:
        rendered = json.dumps(build(root), indent=2, ensure_ascii=False) + "\n"
        target.write_text(rendered, encoding="utf-8")
        print(f"ASSET_RIGHTS_REGISTER_WRITTEN {target}")

    if not target.is_file():
        print(f"ASSET_RIGHTS_REGISTER_MISSING {target}")
        return 1
    register = json.loads(target.read_text(encoding="utf-8"))
    errors = validate(root, register)
    if errors:
        for error in errors:
            print(f"ASSET_RIGHTS_REGISTER_ERROR {error}")
        return 1
    print(f"ASSET_RIGHTS_REGISTER_OK {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
