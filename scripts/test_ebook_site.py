#!/usr/bin/env python3
"""Structural tests for open-source ebook site (CRIT-APP or ML)."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


def load_root() -> Path:
    return Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()


ROOT = load_root()


class EbookSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ROOT
        cls.docs = cls.root / "docs"
        cls.yml = (cls.root / "mkdocs.yml").read_text(encoding="utf-8")

    def test_product_is_open_source_ebook(self) -> None:
        index = (self.docs / "index.md").read_text(encoding="utf-8")
        blob = self.yml + "\n" + index
        self.assertRegex(blob, r"(?i)open-source ebook")
        self.assertNotRegex(self.yml, r"(?i)Web Edition")
        m = re.search(r"^site_name:\s*(.+)$", self.yml, re.M)
        self.assertIsNotNone(m)
        self.assertNotIn("Web Edition", m.group(1))

    def test_no_meta_nav_entries(self) -> None:
        for pat in [
            r"How to use",
            r"Update protocol",
            r"Originality",
            r"About &",
            r"About originality",
            r"navigation\.tabs",
        ]:
            self.assertIsNone(
                re.search(pat, self.yml),
                f"forbidden nav/chrome pattern still present: {pat}",
            )

    def test_nav_is_intro_plus_book(self) -> None:
        self.assertIn("Introduction: index.md", self.yml)
        self.assertIn("Book:", self.yml)
        self.assertIn("curriculum/", self.yml)

    def test_landing_is_brief(self) -> None:
        index = (self.docs / "index.md").read_text(encoding="utf-8")
        self.assertIn("ebook-hero", index)
        self.assertIn("chapter-list", index)
        self.assertNotIn("how-to-use.md", index)
        self.assertNotIn("about.md", index)
        self.assertNotIn("update-protocol.md", index)
        self.assertLess(len(index), 12000)

    def test_chapters_exist(self) -> None:
        ch = [
            p
            for p in (self.docs / "curriculum").glob("*.md")
            if p.name != "index.md" and "how-to-use" not in p.name
        ]
        self.assertGreaterEqual(len(ch), 10)

    def test_css_ebook_system(self) -> None:
        css = (self.docs / "stylesheets" / "extra.css").read_text(encoding="utf-8")
        self.assertIn("ebook-hero", css)
        self.assertIn("chapter-list", css)
        self.assertTrue(
            "Cormorant" in css or "serif" in css.lower() or "ebook" in css
        )

    def test_meta_pages_not_in_docs_root(self) -> None:
        for name in ("about.md", "how-to-use.md", "update-protocol.md", "ORIGINALITY.md"):
            self.assertFalse(
                (self.docs / name).exists(),
                f"{name} should not be public docs root",
            )

    def test_css_responsive_and_print(self) -> None:
        css = (self.docs / "stylesheets" / "extra.css").read_text(encoding="utf-8")
        self.assertIn("@media print", css)
        self.assertTrue(
            "max-width: 768px" in css or "max-width:768px" in css,
            "tablet breakpoint missing",
        )
        self.assertTrue(
            "max-width: 480px" in css or "max-width:480px" in css,
            "phone breakpoint missing",
        )
        self.assertIn(".md-header", css)
        self.assertIn("display: none", css)

    def test_no_figure_concept_admonitions(self) -> None:
        hits = 0
        for p in (self.docs / "curriculum").glob("*.md"):
            hits += p.read_text(encoding="utf-8", errors="replace").count(
                "Figure concept (text diagram)"
            )
        self.assertEqual(hits, 0, f"leftover text-diagram callouts: {hits}")

    def test_math_verify_script_passes(self) -> None:
        import subprocess

        script = self.root / "scripts" / "verify_math_examples.py"
        self.assertTrue(script.exists(), "verify_math_examples.py missing")
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL_PASS", r.stdout)


if __name__ == "__main__":
    # Preserve path arg for setUpClass via module-level ROOT
    args = [sys.argv[0]]
    unittest.main(argv=args, verbosity=2)
