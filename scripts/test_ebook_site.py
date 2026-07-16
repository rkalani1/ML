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

    def test_raw_html_routes_target_rendered_pages(self) -> None:
        index = (self.docs / "index.md").read_text(encoding="utf-8")
        raw_hrefs = re.findall(r"<a\b[^>]*\bhref=[\"']([^\"']+)", index, re.I)
        bad = [href for href in raw_hrefs if href.split("#", 1)[0].endswith(".md")]
        self.assertEqual(bad, [], f"raw HTML links bypass MkDocs rewriting: {bad}")

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
        self.assertIn("prefers-reduced-motion", css)

    def test_navigation_labels_are_not_truncated(self) -> None:
        nav = self.yml.split("nav:", 1)[-1]
        self.assertNotIn("...", nav)
        self.assertNotIn("…", nav)

    def test_curriculum_numbering_matches_source(self) -> None:
        nav = self.yml.split("nav:", 1)[-1]
        landing = (self.docs / "index.md").read_text(encoding="utf-8")
        for chapter in sorted((self.docs / "curriculum").glob("[0-9][0-9]-*.md")):
            prefix = chapter.name[:2]
            self.assertIn(f'"{prefix}. ', nav, f"nav number does not match {chapter.name}")
            self.assertIn(f'<span class="num">{prefix}</span>', landing, f"landing number does not match {chapter.name}")
            if int(prefix) <= 16:
                title = chapter.read_text(encoding="utf-8").splitlines()[0]
                self.assertIn(f"Chapter {int(prefix)}.", title, f"H1 number does not match {chapter.name}")

    def test_figure_density_and_asset_integrity(self) -> None:
        image_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
        chapters = sorted((self.docs / "curriculum").glob("*.md"))
        for chapter in chapters:
            refs = image_re.findall(chapter.read_text(encoding="utf-8"))
            self.assertLessEqual(len(refs), 20, f"{chapter.name} has {len(refs)} raster figures")
            for ref in refs:
                if re.match(r"(?:https?:)?//", ref):
                    continue
                target = (chapter.parent / ref).resolve()
                self.assertTrue(target.is_file(), f"{chapter.name} has missing image {ref}")
        assets = [
            path
            for path in (self.docs / "assets" / "figures").rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
        ]
        self.assertLessEqual(len(assets), 300, f"figure asset ceiling exceeded: {len(assets)}")
        size = sum(path.stat().st_size for path in assets)
        self.assertLessEqual(size, 25 * 1024 * 1024, f"figure assets total {size / 1024 / 1024:.2f} MiB")

        source_blob = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for suffix in ("*.md", "*.css")
            for path in self.docs.rglob(suffix)
        )
        orphan_assets = [path.name for path in assets if path.name not in source_blob]
        self.assertEqual(
            orphan_assets,
            [],
            f"unreferenced figure assets should not ship: {orphan_assets}",
        )

    def test_no_densifier_residue(self) -> None:
        bad_names = []
        for path in (self.docs / "assets" / "figures").glob("*"):
            if re.search(r"(?i)(?:^|_)cycle\d{3,}(?:_|$)", path.name):
                bad_names.append(path.name)
        self.assertEqual(bad_names, [], f"continuous densifier assets remain: {bad_names[:5]}")

    def test_retired_mutators_are_absent(self) -> None:
        retired = {
            "build_ebook_site.py", "build_web_edition.py", "build_from_docx.py",
            "expand_chapters.py", "wire_unused_figures.py", "embed_figures.py",
            "render_concept_figures.py", "cycle2_caption_art.py", "deep_originality_scan.py",
        }
        present = sorted(path.name for path in (self.root / "scripts").glob("*.py") if path.name in retired)
        self.assertEqual(present, [], f"retired destructive mutators remain: {present}")

    def test_evidence_register_is_explicitly_bounded(self) -> None:
        register = self.docs / "evidence-register.md"
        self.assertTrue(register.is_file(), "evidence register missing")
        text = register.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("Checked 2026-07-15"), 5)
        self.assertIn("needs confirmation", text)
        self.assertIn("assets/figures/manifest.json", text)

    def test_known_content_regressions_stay_fixed(self) -> None:
        chapters = {
            path.name: path.read_text(encoding="utf-8")
            for path in (self.docs / "curriculum").glob("*.md")
        }
        blob = "\n".join(chapters.values())
        if self.root.name.upper() == "CRIT-APP":
            for false_claim in (
                "RRR) is constant across different baseline risk groups",
                "assume RR roughly equals HR",
                "OR = 11.4",
                "Absolute effects are causally identifiable",
                "## Advanced Application in Clinical Practice",
                "Precision is essentially infinite",
                "will result in a cumulative risk reduction that is much smaller than 30%",
                "which invariably exaggerate the perceived efficacy",
            ):
                self.assertNotIn(false_claim, blob)
            self.assertNotRegex(blob, r"(?m)^[ \t]*\d+\.[ \t]+\d+\.")
            self.assertIn("10/418", chapters["12-effect-sizes-absolute-benefit-nnt-and-clinical-importance.md"])
            self.assertIn("an HR is not a risk ratio", blob)
            self.assertIn("not automatically causally identified or transportable", blob)
        elif self.root.name.upper() == "ML":
            self.assertNotIn("ml_concept_", blob)
            for false_claim in (
                "positive definite (a local minimum)",
                "rolling downhill is guaranteed to find it",
                "correction matters only for the first tens of steps",
                "PSI ≳ 0.2 is a material alarm",
                "rule of thumb often near 0.2-0.4",
                "Computers cannot make truly random numbers",
                "Orthogonal vectors carry completely independent information",
                "a data matrix times its own transpose",
                "capping reliability",
                "reliability ceiling",
                "count = max(count, 1)",
                "the damping factor is the restart probability",
                "does not distort the marginal distribution",
                "its gradient cancels",
                "the sum runs away to infinity",
                "near-flat ⇒ near a minimum",
                "stacking many weight matrices, each a learned transformation, is exactly what gives",
            ):
                self.assertNotIn(false_claim, blob)
            math = chapters["00-mathematical-foundations-for-machine-learning.md"]
            self.assertIn("∇f(1, 1) = (6, 4) ≠ 𝟎", math)
            self.assertIn("therefore is not a local minimum", math)
            self.assertIn("Convergence of gradient descent is a separate result", math)
            self.assertIn("\\mathbf W^\\mathsf T", math)
            rl = chapters["13-reinforcement-learning.md"]
            self.assertIn("if count == 0:", rl)
            graph = chapters["15-graph-mining-algorithms.md"]
            self.assertRegex(graph, r"(?s)```python\s+edges = \[\(0, 1\)")

    def test_python_markdown_fences_compile(self) -> None:
        pattern = re.compile(r"(?ms)^```python[^\n]*\n(.*?)^```[ \t]*$")
        checked = 0
        for path in sorted(self.docs.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                checked += 1
                start_line = text.count("\n", 0, match.start()) + 1
                try:
                    compile(match.group(1), f"{path}:{start_line}", "exec")
                except SyntaxError as exc:
                    self.fail(f"invalid Python fence at {path}:{start_line}: {exc}")
        self.assertGreater(checked, 0, "no Python Markdown fences found")

    def test_no_generated_concept_admonitions(self) -> None:
        hits: list[str] = []
        for p in (self.docs / "curriculum").glob("*.md"):
            text = p.read_text(encoding="utf-8", errors="replace")
            if "Figure concept (text diagram)" in text or '!!! note "Concept map' in text:
                hits.append(p.name)
        self.assertEqual(hits, [], f"leftover generated concept callouts: {hits}")

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
