#!/usr/bin/env python3
"""Prove that the release Gitleaks flags defeat repository suppressions."""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def scan(
    gitleaks: str,
    repository: Path,
    ignore_path: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            gitleaks,
            "git",
            "--log-opts=--all --text -m",
            "--redact",
            "--ignore-gitleaks-allow",
            "--gitleaks-ignore-path",
            str(ignore_path),
            "--no-banner",
            "--no-color",
            ".",
        ],
        cwd=repository,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gitleaks", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = root / "mutation-repository"
        repository.mkdir()
        run(["git", "init", "--quiet"], repository)
        run(["git", "config", "user.name", "Release Gate"], repository)
        run(["git", "config", "user.email", "release-gate@example.invalid"], repository)

        (repository / ".gitattributes").write_text(
            "encoded.txt -diff\n",
            encoding="utf-8",
        )
        token = "gh" + "p_" + hashlib.sha256(
            b"release-gate-suppression-mutation"
        ).hexdigest()[:36]
        suppressed = f"credential={token} # gitleaks:allow\n".encode()
        (repository / "encoded.txt").write_text(
            base64.b64encode(suppressed).decode("ascii") + "\n",
            encoding="utf-8",
        )
        run(["git", "add", ".gitattributes", "encoded.txt"], repository)
        run(["git", "commit", "--quiet", "-m", "suppression mutation"], repository)

        ignore_path = root / "empty-ignore"
        ignore_path.write_text("", encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("GITLEAKS_CONFIG", None)
        environment.pop("GITLEAKS_CONFIG_TOML", None)
        result = scan(args.gitleaks, repository, ignore_path, environment)
        if result.returncode != 1:
            print(
                "GITLEAKS_CONTROL_MUTATION_ERROR "
                f"expected leak exit 1, received {result.returncode}"
            )
            print(result.stdout)
            print(result.stderr)
            return 1

        merge_repository = root / "merge-mutation-repository"
        merge_repository.mkdir()
        run(
            ["git", "init", "--quiet", "--initial-branch", "main"],
            merge_repository,
        )
        run(["git", "config", "user.name", "Release Gate"], merge_repository)
        run(
            ["git", "config", "user.email", "release-gate@example.invalid"],
            merge_repository,
        )
        (merge_repository / "base.txt").write_text("base\n", encoding="utf-8")
        run(["git", "add", "base.txt"], merge_repository)
        run(["git", "commit", "--quiet", "-m", "base"], merge_repository)
        run(["git", "switch", "--quiet", "-c", "side"], merge_repository)
        (merge_repository / "side.txt").write_text("side\n", encoding="utf-8")
        run(["git", "add", "side.txt"], merge_repository)
        run(["git", "commit", "--quiet", "-m", "side"], merge_repository)
        run(["git", "switch", "--quiet", "main"], merge_repository)
        (merge_repository / "main.txt").write_text("main\n", encoding="utf-8")
        run(["git", "add", "main.txt"], merge_repository)
        run(["git", "commit", "--quiet", "-m", "main"], merge_repository)
        run(["git", "merge", "--quiet", "--no-ff", "--no-commit", "side"], merge_repository)
        merge_token = "gh" + "p_" + hashlib.sha256(
            b"release-gate-merge-mutation"
        ).hexdigest()[:36]
        (merge_repository / "merge-only.txt").write_text(
            f"credential={merge_token}\n",
            encoding="utf-8",
        )
        run(["git", "add", "merge-only.txt"], merge_repository)
        run(
            ["git", "commit", "--quiet", "-m", "merge resolution mutation"],
            merge_repository,
        )
        merge_result = scan(
            args.gitleaks,
            merge_repository,
            ignore_path,
            environment,
        )
        if merge_result.returncode != 1:
            print(
                "GITLEAKS_MERGE_MUTATION_ERROR "
                f"expected leak exit 1, received {merge_result.returncode}"
            )
            print(merge_result.stdout)
            print(merge_result.stderr)
            return 1

    print(
        "GITLEAKS_CONTROL_MUTATION_OK "
        "encoded=-diff inline-allow=ignored merge-patch=covered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
