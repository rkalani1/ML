#!/usr/bin/env python3
"""Reproducible verification of worked numerical examples in the ML ebook."""
from __future__ import annotations

import math
import sys


def near(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("ols_beta1", near(23 / 13, 1.76923, 1e-4), f"{23/13:.5f}"))
    checks.append(("assoc_conf", near(0.15 / 0.30, 0.5, 1e-12), "0.5"))
    checks.append(("assoc_lift", near(0.5 / 0.4, 1.25, 1e-12), "1.25"))

    scores = [1.0, 0.3, 0.8]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    w = [e / z for e in exps]
    checks.append(("attn0", near(w[0], 0.434, 0.01), f"{w[0]:.3f}"))
    checks.append(("attn1", near(w[1], 0.214, 0.01), f"{w[1]:.3f}"))
    checks.append(("attn2", near(w[2], 0.351, 0.01), f"{w[2]:.3f}"))

    acc = (82 + 49) / (82 + 8 + 11 + 49)
    checks.append(("acc", near(acc, 0.873, 0.01), f"{acc:.3f}"))

    prior, lr = 0.2, 5.0
    odds = prior / (1 - prior)
    post = (odds * lr) / (1 + odds * lr)
    checks.append(("bayes", near(post, 5 / 9, 1e-9), f"{post:.6f}"))

    failed = 0
    for name, ok, detail in checks:
        print(("PASS" if ok else "FAIL"), name, detail)
        if not ok:
            failed += 1
    if failed:
        print("FAILED", failed)
        return 1
    print("ALL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
