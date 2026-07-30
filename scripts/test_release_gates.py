#!/usr/bin/env python3
"""Negative and mutation tests for the public-release gates."""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from book_identity import load_book_identity

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safety = load_module("publication_safety_under_test", ROOT / "scripts/check_publication_safety.py")
rights = load_module("asset_rights_under_test", ROOT / "scripts/build_asset_rights_register.py")
postprocess = load_module("postprocess_under_test", ROOT / "scripts/postprocess_site.py")
originality = load_module("originality_under_test", ROOT / "scripts/originality_scan.py")
manifest = load_module("asset_manifest_under_test", ROOT / "scripts/build_asset_manifest.py")
rendered_validation = load_module(
    "rendered_validation_under_test",
    ROOT / "scripts/validate_rendered_site.py",
)


class PublicationGateTests(unittest.TestCase):
    def test_book_identity_is_independent_of_checkout_directory_name(self) -> None:
        identity = load_book_identity(ROOT)
        expected = {
            "https://rkalani1.github.io/ML/": ("ML", "ml", 20),
            "https://rkalani1.github.io/CRIT-APP/": (
                "CRIT-APP",
                "crit",
                28,
            ),
        }
        site_url = re.search(
            r"^site_url:\s*([^\s#]+)\s*$",
            (ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        self.assertIsNotNone(site_url)
        self.assertEqual(
            (identity.slug, identity.policy, identity.expected_curriculum),
            expected[site_url.group(1)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            renamed = Path(temporary) / "downloaded-source-main"
            renamed.mkdir()
            (renamed / "mkdocs.yml").write_text(
                (ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.assertEqual(load_book_identity(renamed), identity)
            original_config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
            for duplicate in (
                '"site_url": https://example.invalid/quoted/\n',
                "site_url : https://example.invalid/spaced/\n",
            ):
                (renamed / "mkdocs.yml").write_text(
                    original_config + "\n" + duplicate,
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_book_identity(renamed)
            (renamed / "mkdocs.yml").write_text(
                "site_url: https://example.invalid/unreviewed/\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_book_identity(renamed)
        for relative in (
            "scripts/test_ebook_site.py",
            "scripts/validate_rendered_site.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("root.name", source)
            self.assertNotIn("Path.cwd().name", source)

    def test_rendered_image_candidates_require_local_image_files(self) -> None:
        document = rendered_validation.Document()
        document.feed(
            '<img src="a.png" srcset="b.png 1x, c.webp 2x">'
            '<source src="d.jpg" srcset="e.svg 1x, f.gif 2x">'
            '<img SRC="bad.txt" src="valid.png">'
        )
        self.assertEqual(
            document.image_references,
            [
                ("img src", "a.png"),
                ("img srcset", "b.png"),
                ("img srcset", "c.webp"),
                ("source src", "d.jpg"),
                ("source srcset", "e.svg"),
                ("source srcset", "f.gif"),
                ("img src", "valid.png"),
            ],
        )
        self.assertEqual(document.duplicate_attributes, [("img", "src")])
        invalid_srcset = rendered_validation.Document()
        invalid_srcset.feed('<img srcset="valid.png potato">')
        self.assertEqual(
            invalid_srcset.srcset_errors,
            [("img", "candidate has an invalid descriptor: valid.png potato")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary).resolve()
            page = site / "page.html"
            page.write_text("<html></html>", encoding="utf-8")
            (site / "index.html").write_text("<html></html>", encoding="utf-8")
            (site / "note.txt").write_text("not an image", encoding="utf-8")
            directory = site / "directory"
            directory.mkdir()
            (directory / "index.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )
            (site / "valid.png").write_bytes(b"reviewed elsewhere")
            for url in ("/", "note.txt", "directory/", "https://example.invalid/x.png"):
                target, error = rendered_validation.reviewed_image_target(
                    site,
                    page,
                    url,
                    "ML",
                )
                self.assertIsNone(target)
                self.assertIsNotNone(error)
            target, error = rendered_validation.reviewed_image_target(
                site,
                page,
                "valid.png",
                "ML",
            )
            self.assertEqual(target, site / "valid.png")
            self.assertIsNone(error)

    def test_originality_scan_covers_published_subdirectories_and_text_files(self) -> None:
        self.assertEqual(originality.scan(ROOT), [])
        with tempfile.TemporaryDirectory() as temporary:
            docs = Path(temporary)
            audit = docs / "_swarm_audit"
            audit.mkdir()
            (audit / "copied.md").write_text(
                "All rights " + "reserved\n",
                encoding="utf-8",
            )
            (docs / "copied.txt").write_text(
                "Downloaded from " + "https://publisher.example/article\n",
                encoding="utf-8",
            )
            metadata = docs / "_meta"
            metadata.mkdir()
            (metadata / "copied.md").write_text(
                "## Abstract\n\nThis article is protected by " + "copyright\n",
                encoding="utf-8",
            )
            abstract_variants = {
                "attr-list.md": "## Abstract {#abstract}\n",
                "emphasis.md": "## **Abstract**\n",
                "raw-html.md": "<h2>Abstract</h2>\n",
                "blockquote.md": "> ## Abstract\n",
                "entity-abstract.md": "## Abstr&#97;ct\n",
            }
            for filename, content in abstract_variants.items():
                (metadata / filename).write_text(content, encoding="utf-8")
            boilerplate_variants = {
                "wrapped-rights.md": "All " + "rights\nreserved.\n",
                "wrapped-copyright.md": (
                    "This article is protected by\ncopyright.\n"
                ),
                "formatted-rights.md": "All " + "rights **reserved**.\n",
                "publisher-autolink.md": (
                    "Downloaded " + "from <https://publisher.example/article>\n"
                ),
                "entity-rights.md": "All rights&nbsp;" + "reserved.\n",
                "entity-copyright.md": (
                    "This article is protected by&nbsp;" + "copyright.\n"
                ),
            }
            for filename, content in boilerplate_variants.items():
                (metadata / filename).write_text(content, encoding="utf-8")
            findings = originality.scan(docs)
        joined = "\n".join(findings)
        self.assertIn("_swarm_audit/copied.md", joined)
        self.assertIn("copied.txt", joined)
        self.assertIn("_meta/copied.md", joined)
        self.assertIn("abstract heading block", joined)
        for filename in abstract_variants:
            self.assertIn(f"_meta/{filename}", joined)
        for filename in boilerplate_variants:
            self.assertIn(f"_meta/{filename}", joined)

    def test_release_revision_stamp_is_exact_and_idempotent(self) -> None:
        sha = "a" * 40
        marker = f'<meta name="release-revision" content="{sha}">'
        rendered, count = postprocess.stamp_release_revision(
            "<html><head><title>Book</title></head><body></body></html>",
            sha,
        )
        self.assertEqual(count, 1)
        self.assertIn(marker, rendered)
        self.assertEqual(postprocess.release_revision_errors(rendered, sha), [])
        repeated, count = postprocess.stamp_release_revision(rendered, sha)
        self.assertEqual(count, 0)
        self.assertEqual(repeated, rendered)
        existing_head_meta = (
            "<html><head><meta charset=\"utf-8\">"
            f"{marker}</head><body></body></html>"
        )
        unchanged, count = postprocess.stamp_release_revision(
            existing_head_meta,
            sha,
        )
        self.assertEqual(count, 0)
        self.assertEqual(unchanged, existing_head_meta)
        unusual_line_characters = (
            "<html><head><title>A\u2028B\fC</title>\n"
            "</head><body></body></html>"
        )
        stamped_unusual, count = postprocess.stamp_release_revision(
            unusual_line_characters,
            sha,
        )
        self.assertEqual(count, 1)
        self.assertIn(f"</title>\n{marker}\n</head>", stamped_unusual)
        self.assertEqual(
            postprocess.release_revision_errors(stamped_unusual, sha),
            [],
        )
        commented = (
            f"<html><head><!-- {marker} --></head><body></body></html>"
        )
        stamped_comment, count = postprocess.stamp_release_revision(
            commented,
            sha,
        )
        self.assertEqual(count, 1)
        self.assertEqual(stamped_comment.count(marker), 2)
        self.assertEqual(
            postprocess.release_revision_errors(stamped_comment, sha),
            [],
        )
        with tempfile.TemporaryDirectory() as temporary:
            fetched = Path(temporary) / "live.html"
            fetched.write_text(
                stamped_comment + (" " * 100_000),
                encoding="utf-8",
            )
            valid_live = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_release_marker.py"),
                    "--html-file",
                    str(fetched),
                    "--expected-release-sha",
                    sha,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(valid_live.returncode, 0, valid_live.stdout)
            fetched.write_text(commented, encoding="utf-8")
            commented_live = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_release_marker.py"),
                    "--html-file",
                    str(fetched),
                    "--expected-release-sha",
                    sha,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(commented_live.returncode, 0)
        self.assertIn(
            "exactly one direct-head",
            "\n".join(postprocess.release_revision_errors(commented, sha)),
        )
        for unsafe in (
            f"<html><head></head><body>{marker}</body></html>",
            f"<html><head><noscript>{marker}</noscript></head><body></body></html>",
            f"<html><head>{marker}{marker}</head><body></body></html>",
            f"<html></head><body><head>{marker}</body></html>",
            f"<html><body><head>{marker}</head></body></html>",
            f"<html><head><p></p>{marker}</head><body></body></html>",
            f"<html><head>oops{marker}</head><body></body></html>",
            f"<html><head></br>{marker}</head><body></body></html>",
            f"<html><head><script/>{marker}</head><body></body></html>",
            f"<html><head><title/>{marker}</head><body></body></html>",
            (
                "<script></script><html><head>"
                f"{marker}</head><body></body></html>"
            ),
            (
                "<html><head>"
                f"{marker}</head><body></body></html><div></div>"
            ),
            (
                "<html><head>"
                f"{marker}</head><div></div><body></body></html>"
            ),
            (
                "<html><head>"
                f"{marker}</head><body></body><div></div></html>"
            ),
            f"<html><head>{marker}</head><body></body></html>trailing",
            (
                "<html><head><meta name=\"release-revision\" "
                f"name=\"other\" content=\"{sha}\"></head><body></body></html>"
            ),
        ):
            with self.assertRaises(ValueError):
                postprocess.stamp_release_revision(unsafe, sha)
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            uppercase_html = site / "Unusual.HTML"
            uppercase_html.write_text(
                "<html><head><title>Case</title></head><body></body></html>",
                encoding="utf-8",
            )
            self.assertEqual(
                postprocess.rendered_html_files(site),
                [uppercase_html],
            )
            self.assertEqual(
                rendered_validation.rendered_html_files(site),
                [uppercase_html],
            )
            curriculum = site / "curriculum"
            curriculum.mkdir()
            uppercase_curriculum = curriculum / "Bonus.HTML"
            uppercase_curriculum.write_text(
                "<html><head><title>Bonus</title></head><body></body></html>",
                encoding="utf-8",
            )
            nested_curriculum = curriculum / "appendix" / "nested.html"
            nested_curriculum.parent.mkdir()
            nested_curriculum.write_text(
                "<html><head><title>Nested</title></head><body></body></html>",
                encoding="utf-8",
            )
            html_files = rendered_validation.rendered_html_files(site)
            self.assertEqual(
                set(
                    rendered_validation.curriculum_html_files(
                        site,
                        html_files,
                    )
                ),
                {uppercase_curriculum, nested_curriculum},
            )
            with (
                patch.dict(postprocess.os.environ, {"GITHUB_SHA": sha}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "postprocess_site.py",
                        "--site-dir",
                        str(site),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(postprocess.main(), 0)
            self.assertEqual(
                postprocess.release_revision_errors(
                    uppercase_html.read_text(encoding="utf-8"),
                    sha,
                ),
                [],
            )
        with self.assertRaises(ValueError):
            postprocess.stamp_release_revision(rendered, "not-a-sha")
        with self.assertRaises(ValueError):
            postprocess.stamp_release_revision(rendered, "b" * 40)

    def test_plain_windows_and_linux_user_paths_are_detected(self) -> None:
        windows = safety.LOCAL_PATH_PATTERNS["Windows user path"]
        sample = "C:" + r"\Users\reader\private.txt"
        self.assertIsNotNone(windows.search(sample))
        linux = safety.LOCAL_PATH_PATTERNS["Linux user path"]
        self.assertIsNotNone(linux.search("/" + "home/reader/private.txt"))

    def test_active_svg_features_are_rejected(self) -> None:
        errors: list[str] = []
        safety.scan_svg(
            "bad.svg",
            '<svg onload="run()"><script>alert(1)</script>'
            '<style>@import url(&#x2f;&#x2f;tracker.example/theme.css)</style>'
            '<image href="&#x2f;&#x2f;tracker.example/pixel"/></svg>',
            errors,
        )
        self.assertTrue(any("script element" in error for error in errors))
        self.assertTrue(any("inline event handler" in error for error in errors))
        self.assertTrue(any("CSS reference" in error for error in errors))
        self.assertTrue(any("non-fragment reference" in error for error in errors))

    def test_svg_presentation_attribute_urls_are_rejected_after_decoding(self) -> None:
        literal_errors: list[str] = []
        safety.scan_svg(
            "literal.svg",
            '<svg><rect fill="url(https://tracker.example/paint.svg#x)"/></svg>',
            literal_errors,
        )
        self.assertTrue(
            any("attribute reference (fill)" in error for error in literal_errors)
        )

        encoded_errors: list[str] = []
        safety.scan_svg(
            "encoded.svg",
            '<svg><rect filter="url(https:&#x2f;&#x2f;tracker.example/f.svg#x)"/>'
            "</svg>",
            encoded_errors,
        )
        self.assertTrue(
            any("attribute reference (filter)" in error for error in encoded_errors)
        )
        processing_instruction_errors: list[str] = []
        safety.scan_svg(
            "processing-instruction.svg",
            '<?xml-stylesheet href="data:text/css,@import '
            'url(//tracker.example/x.css)"?><svg/>',
            processing_instruction_errors,
        )
        self.assertTrue(
            any(
                "processing instruction" in error
                for error in processing_instruction_errors
            )
        )
        xml_base_errors: list[str] = []
        safety.scan_svg(
            "xml-base.svg",
            '<svg xml:base="https://tracker.example/"><rect/></svg>',
            xml_base_errors,
        )
        self.assertTrue(any("xml:base" in error for error in xml_base_errors))

    def test_rendered_site_allows_local_controls_and_rejects_remote_scripts(self) -> None:
        safe_html = f"""<!doctype html><html><head>
        <meta http-equiv="Content-Security-Policy" content="{safety.CONTENT_SECURITY_POLICY}">
        <meta charset="utf-8">
        <meta name="referrer" content="strict-origin-when-cross-origin"></head><body>
        <form><input type="text" name="q"></form></body></html>
        """
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "index.html").write_text(safe_html, encoding="utf-8")
            errors: list[str] = []
            count = safety.scan_rendered_site(site, errors)
        self.assertEqual(count, 1)
        self.assertEqual(errors, [])
        remote_html = safe_html + f'<script src="{safety.MATHJAX_URL}"></script>'
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "index.html").write_text(remote_html, encoding="utf-8")
            remote_errors: list[str] = []
            safety.scan_rendered_site(site, remote_errors)
        self.assertTrue(
            any("non-allowlisted remote resource" in error for error in remote_errors)
        )

    def test_rendered_site_rejects_submission_and_remote_resources(self) -> None:
        unsafe_html = """<!doctype html>
        <meta http-equiv="refresh" content="0; url=https://tracker.example/redirect">
        <meta name="referrer" content="unsafe-url">
        <a href="https://example.test/" referrerpolicy="unsafe-url">unsafe</a>
        <div style="background:url(data:image/png;base64,AA==)">embedded</div>
        <form action="/collect" method="post"><input type="file"></form>
        <style>@import url("https://tracker.example/style.css");</style>
        <img src="/safe.png" srcset="/safe.png 1x, https://tracker.example/two.png 2x">
        <script src="https://tracker.example/app.js"></script>
        <script src="//cdn.jsdelivr.net/npm/unreviewed-package/evil.js"></script>
        <script>navigator.sendBeacon("https://www.google-analytics.com/collect")</script>
        """
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "index.html").write_text(unsafe_html, encoding="utf-8")
            (site / "payload.htm").write_text(
                "<script>location.href='https://tracker.example'</script>",
                encoding="utf-8",
            )
            (site / "payload.xml").write_text(
                '<?xml version="1.0"?><html '
                'xmlns="http://www.w3.org/1999/xhtml"><body>'
                '<script src="https://tracker.example/attack.js"></script>'
                "</body></html>",
                encoding="utf-8",
            )
            (site / "payload.xht").write_text(
                "<html xmlns=\"http://www.w3.org/1999/xhtml\"><script/></html>",
                encoding="utf-8",
            )
            errors: list[str] = []
            safety.scan_rendered_site(site, errors)
        joined = "\n".join(errors)
        self.assertIn("data-submission form", joined)
        self.assertIn("sensitive input", joined)
        self.assertIn("non-allowlisted remote resource", joined)
        self.assertIn("analytics/tracker reference", joined)
        self.assertIn("remote CSS load", joined)
        self.assertIn("browser network primitive", joined)
        self.assertIn("dangerous active markup", joined)
        self.assertIn("Content-Security-Policy", joined)
        self.assertIn("referrer policy", joined)
        self.assertIn("referrerpolicy", joined)
        self.assertIn("contains a data URL", joined)
        self.assertIn("forbidden active rendered file type: payload.htm", joined)
        self.assertIn("forbidden active rendered file type: payload.xml", joined)
        self.assertIn("forbidden active rendered file type: payload.xht", joined)
        self.assertTrue(safety.is_remote("//cdn.jsdelivr.net/npm/unreviewed.js"))
        self.assertTrue(safety.is_remote("///tracker.example/unreviewed.js"))
        self.assertTrue(safety.is_remote("////tracker.example/unreviewed.js"))

    def test_csp_must_be_an_active_direct_head_child(self) -> None:
        csp = (
            '<meta http-equiv="Content-Security-Policy" '
            f'content="{safety.CONTENT_SECURITY_POLICY}">'
        )
        samples = {
            "body": f"<html><head></head><body>{csp}</body></html>",
            "template": (
                f"<html><head><template>{csp}</template></head><body></body></html>"
            ),
            "late": (
                "<html><head><link rel=\"icon\" href=\"favicon.png\">"
                f"{csp}</head><body></body></html>"
            ),
            "duplicate_head": (
                "<!doctype html><html><head><title>x</title></head>"
                f"<p>implied body</p><head>{csp}</head></html>"
            ),
            "head_breaker": (
                f"<!doctype html><html><head><p>x</p>{csp}</head>"
                "<body></body></html>"
            ),
            "pre_head_breaker": (
                f"<!doctype html><html><br><head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "leading_nbsp": (
                f"\u00a0<!doctype html><html><head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "doctype_nbsp": (
                f"<!doctype html>\u00a0<html><head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "html_bang": (
                f"<!doctype html><html!><head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "html_head_nbsp": (
                f"<!doctype html><html>\u00a0<head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "head_csp_nbsp": (
                f"<!doctype html><html><head>\u00a0{csp}"
                '<meta charset="utf-8"></head><body></body></html>'
            ),
            "head_body_nbsp": (
                f"<!doctype html><html><head>{csp}"
                '<meta charset="utf-8"></head>\u00a0<body></body></html>'
            ),
            "body_html_nbsp": (
                f"<!doctype html><html><head>{csp}"
                '<meta charset="utf-8"></head><body></body>\u00a0</html>'
            ),
            "trailing_nbsp": (
                f"<!doctype html><html><head>{csp}"
                '<meta charset="utf-8"></head><body></body></html>\u00a0'
            ),
            "self_closing_style": (
                f"<!doctype html><html><head>{csp}<meta charset=\"utf-8\">"
                "</head><body><style/>body{color:red}</style></body></html>"
            ),
            "legacy_image": (
                f"<!doctype html><html><head>{csp}<meta charset=\"utf-8\">"
                "</head><body><image src=\"https://tracker.example/pixel\">"
                "</body></html>"
            ),
        }
        for label, content in samples.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                site = Path(temporary)
                (site / "index.html").write_text(content, encoding="utf-8")
                errors: list[str] = []
                safety.scan_rendered_site(site, errors)
            joined = "\n".join(errors)
            if label in {"body", "template"}:
                self.assertIn("not a direct child of head", joined)
            elif label == "late":
                self.assertIn("before Content-Security-Policy", joined)
            elif label in {"duplicate_head", "head_breaker"}:
                self.assertIn("Content-Security-Policy meta must be the first", joined)
            elif label in {"leading_nbsp", "doctype_nbsp", "html_bang"}:
                self.assertIn("document must begin with one HTML5 doctype", joined)
            elif label in {
                "pre_head_breaker",
                "html_head_nbsp",
                "head_body_nbsp",
                "body_html_nbsp",
                "trailing_nbsp",
            }:
                self.assertIn("only HTML ASCII whitespace is allowed", joined)
            elif label == "head_csp_nbsp":
                self.assertIn("Content-Security-Policy meta must be the first", joined)
            elif label == "self_closing_style":
                self.assertIn("self-closing non-void HTML element", joined)
            else:
                self.assertIn("legacy image element aliases img", joined)
        for non_html_space in ("\u0085", "\u00a0", "\u2003", "\u2028"):
            with self.subTest(codepoint=f"U+{ord(non_html_space):04X}"):
                structure_errors = safety.raw_html_structure_errors(
                    f"<!doctype html><html>{non_html_space}<head>{csp}"
                    '<meta charset="utf-8"></head><body></body></html>'
                )
                self.assertTrue(
                    any(
                        "only HTML ASCII whitespace is allowed" in error
                        for error in structure_errors
                    )
                )
                first_csp_errors = safety.raw_html_structure_errors(
                    f"<!doctype html><html><head>{non_html_space}{csp}"
                    '<meta charset="utf-8"></head><body></body></html>'
                )
                self.assertTrue(
                    any(
                        "Content-Security-Policy meta must be the first" in error
                        for error in first_csp_errors
                    )
                )

    def test_rendered_sitemap_is_narrowly_passive(self) -> None:
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://example.test/book/index.html</loc>"
            "<lastmod>2026-07-30</lastmod></url></urlset>"
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "sitemap.xml").write_text(content, encoding="utf-8")
            (site / "sitemap.xml.gz").write_bytes(
                safety.gzip.compress(content.encode("utf-8"))
            )
            errors: list[str] = []
            safety.validate_rendered_sitemap(
                site,
                "https://example.test/book/",
                errors,
            )
            self.assertEqual(errors, [])
            (site / "sitemap.xml").write_text(
                '<?xml version="1.0"?>'
                '<?xml-stylesheet href="https://tracker.example/x.xsl"?>'
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>',
                encoding="utf-8",
            )
            active_errors: list[str] = []
            safety.validate_rendered_sitemap(
                site,
                "https://example.test/book/",
                active_errors,
            )
        self.assertTrue(
            any("active XML syntax" in error for error in active_errors)
        )

    def test_every_csp_inline_hash_must_be_observed(self) -> None:
        observed = set(safety.INLINE_SCRIPT_HASHES)
        self.assertEqual(safety.inline_script_inventory_errors(observed), [])
        observed.remove(next(iter(observed)))
        self.assertTrue(
            any(
                "unused executable inline-script hashes" in error
                for error in safety.inline_script_inventory_errors(observed)
            )
        )

    def test_authored_browser_egress_is_rejected(self) -> None:
        errors: list[str] = []
        safety.scan_browser_egress(
            "bad.css",
            '@import url("//tracker.example/theme.css");',
            ".css",
            errors,
        )
        escaped_css_errors: list[str] = []
        safety.scan_browser_egress(
            "escaped.css",
            r"@font-face{font-family:x;src:u\72 l(https\3a "
            r"//cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/"
            r"fonts/woff-v2/MathJax_Zero.woff?leak=x)}",
            ".css",
            escaped_css_errors,
        )
        self.assertTrue(
            any("remote CSS load" in error for error in escaped_css_errors)
        )
        scheme_without_slashes_errors: list[str] = []
        safety.scan_browser_egress(
            "scheme-without-slashes.css",
            "@font-face{font-family:x;src:url("
            "http:cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/"
            "fonts/woff-v2/MathJax_Zero.woff?bypass=1)}",
            ".css",
            scheme_without_slashes_errors,
        )
        self.assertTrue(
            any(
                "remote CSS load" in error
                for error in scheme_without_slashes_errors
            )
        )
        nested_css_errors: list[str] = []
        safety.scan_browser_egress(
            "nested.css",
            '@import url("data:text/css,%40import%20url%28https%3A%2F%2F'
            'tracker.example%2Fx.css%29%3B");',
            ".css",
            nested_css_errors,
        )
        self.assertTrue(
            any("nested data CSS import" in error for error in nested_css_errors)
        )
        data_image_errors: list[str] = []
        safety.scan_browser_egress(
            "data-image.css",
            "body{background:url(data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=)}",
            ".css",
            data_image_errors,
        )
        self.assertTrue(
            any("data URL in authored CSS" in error for error in data_image_errors)
        )
        image_set_errors: list[str] = []
        safety.scan_browser_egress(
            "image-set.css",
            'body{background-image:image-set("https://tracker.example/x.png" 1x)}',
            ".css",
            image_set_errors,
        )
        self.assertTrue(
            any("remote CSS load" in error for error in image_set_errors)
        )
        nested_image_set_errors: list[str] = []
        safety.scan_browser_egress(
            "nested-image-set.css",
            'body{background-image:image-set('
            '"data:image/svg+xml;utf8,<svg/>" .5x, '
            '"https://tracker.example/nested.png" 1x)}',
            ".css",
            nested_image_set_errors,
        )
        self.assertTrue(
            any("remote CSS load" in error for error in nested_image_set_errors)
        )
        escaped_backslash_errors: list[str] = []
        safety.scan_browser_egress(
            "four-backslash.css",
            r'@font-face{font-family:x;src:url("\\\\cdn.jsdelivr.net/'
            r'npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/'
            r'MathJax_Zero.woff")}',
            ".css",
            escaped_backslash_errors,
        )
        self.assertTrue(
            any(
                "ambiguous CSS escape" in error
                for error in escaped_backslash_errors
            )
        )
        safety.scan_browser_egress(
            "bad.js",
            'window.location = "//tracker.example/redirect";'
            'const image = document.createElement("img");'
            'image.src = "//tracker.example/collect";'
            'window["location"] = "/" + "/" + "tracker.example/redirect";'
            'const pixel = new Image();'
            'pixel["src"] = "/" + "/" + "tracker.example/collect";'
            'const hop=document.createElement("a");'
            'Object.assign(hop,{href:"/"+"/"+"tracker.example/collect"});'
            "document.body.append(hop);hop.click();"
            'const peer=new RTCPeerConnection({iceServers:[{'
            'urls:"stun:127.0.0.1:59720"}]});'
            'peer.createDataChannel("x");'
            'import("https:"+"/"+"/"+"cdn.jsdelivr.net/npm/mathjax@3.2.2/'
            'es5/tex-mml-chtml.js?egress=probe").catch(()=>{});',
            ".js",
            errors,
        )
        joined = "\n".join(errors)
        self.assertIn("remote CSS load", joined)
        self.assertIn("non-allowlisted remote URL", joined)
        self.assertIn("scripted navigation", joined)
        self.assertIn("dynamic resource creation", joined)
        self.assertIn("dynamic resource assignment", joined)
        self.assertIn("dynamic object property mutation", joined)
        self.assertIn("WebRTC transport", joined)
        self.assertIn("STUN or TURN transport URL", joined)
        self.assertIn("dynamic module import", joined)
        self.assertIn("authored JavaScript string concatenation", joined)
        nonloader_errors: list[str] = []
        safety.scan_browser_egress(
            "docs/javascripts/not-mathjax.js",
            f'const bundle = "{safety.MATHJAX_URL}";',
            ".js",
            nonloader_errors,
        )
        self.assertTrue(
            any(
                "outside reviewed integrity loader" in error
                for error in nonloader_errors
            )
        )
        reviewed: list[str] = []
        safety.scan_browser_egress(
            "docs/javascripts/mathjax.js",
            (ROOT / "docs/javascripts/mathjax.js").read_text(encoding="utf-8"),
            ".js",
            reviewed,
        )
        self.assertEqual(reviewed, [])

    def test_rendered_generated_javascript_is_scanned(self) -> None:
        safe_html = f"""<!doctype html>
        <meta http-equiv="Content-Security-Policy"
              content="{safety.CONTENT_SECURITY_POLICY}">
        <script src="javascripts/generated.js"></script>
        """
        generated = (
            'window["location"] = "/" + "/" + "tracker.example/redirect";'
            "const pixel = new Image();"
            'pixel["src"] = "/" + "/" + "tracker.example/collect";'
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "javascripts").mkdir()
            (site / "index.html").write_text(safe_html, encoding="utf-8")
            (site / "javascripts" / "generated.js").write_text(
                generated, encoding="utf-8"
            )
            errors: list[str] = []
            safety.scan_rendered_site(site, errors)
        joined = "\n".join(errors)
        self.assertIn("scripted navigation", joined)
        self.assertIn("dynamic resource creation", joined)
        self.assertIn("dynamic resource assignment", joined)

    def test_javascript_tree_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "expected"
            actual = root / "actual"
            expected.mkdir()
            actual.mkdir()
            (expected / "bundle.js").write_text("reviewed();", encoding="utf-8")
            (actual / "bundle.js").write_text("modified();", encoding="utf-8")
            (actual / "extra.js").write_text("extra();", encoding="utf-8")
            (actual / "payload.mjs").write_text(
                'window["location"] = "//tracker.example/";',
                encoding="utf-8",
            )
            errors: list[str] = []
            safety.compare_javascript_trees(
                actual,
                expected,
                "theme",
                errors,
            )
        joined = "\n".join(errors)
        self.assertIn("hash mismatch", joined)
        self.assertIn("unexpected theme JavaScript", joined)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "expected-styles"
            actual = root / "actual-styles"
            expected.mkdir()
            actual.mkdir()
            (expected / "theme.css").write_text(
                "i{background:url(data:image/svg+xml;base64,AA==)}",
                encoding="utf-8",
            )
            (actual / "theme.css").write_text(
                "i{background:url(data:image/svg+xml;base64,EVIL)}",
                encoding="utf-8",
            )
            style_errors: list[str] = []
            matching = safety.compare_stylesheet_trees(
                actual,
                expected,
                style_errors,
            )
        self.assertEqual(matching, set())
        self.assertIn("theme stylesheet hash mismatch", "\n".join(style_errors))

    def test_uninventoried_local_module_script_is_rejected(self) -> None:
        csp = (
            '<meta http-equiv="Content-Security-Policy" '
            f'content="{safety.CONTENT_SECURITY_POLICY}">'
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "index.html").write_text(
                csp + '<script src="payload.mjs"></script>',
                encoding="utf-8",
            )
            (site / "payload.mjs").write_text(
                'window["location"] = "//tracker.example/";',
                encoding="utf-8",
            )
            errors: list[str] = []
            safety.scan_rendered_site(site, errors, source_root=ROOT)
        joined = "\n".join(errors)
        self.assertIn("unreviewed local script resource", joined)
        self.assertIn("scripted navigation", joined)
        prefix_errors: list[str] = []
        prefix = safety.configured_site_path_prefix(ROOT, prefix_errors)
        self.assertEqual(prefix_errors, [])
        self.assertIsNotNone(prefix)
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "javascripts").mkdir()
            source = ROOT / "docs" / "javascripts" / "header-controls.js"
            rendered = site / "javascripts" / "header-controls.js"
            rendered.write_bytes(source.read_bytes())
            root_relative_errors: list[str] = []
            safety.validate_local_script_source(
                site,
                site / "404.html",
                f"{prefix}javascripts/header-controls.js",
                {"javascripts/header-controls.js": safety.sha256(source)},
                prefix,
                root_relative_errors,
            )
        self.assertEqual(root_relative_errors, [])

    def test_legacy_and_data_scripts_before_csp_are_rejected(self) -> None:
        csp = (
            '<meta http-equiv="Content-Security-Policy" '
            f'content="{safety.CONTENT_SECURITY_POLICY}">'
        )
        duplicate_csp = csp.replace(
            'content="',
            'content="script-src *" content="',
            1,
        )
        unsafe_html = (
            '<script type="application/x-javascript">'
            'location.href="https://tracker.example/collect"</script>'
            '<script src="data:text/javascript,location.href%3D'
            'https%3A%2F%2Ftracker.example%2Fcollect"></script>'
            r'<div style="background-image:u\72 l(https\3a '
            r'//tracker.example/pixel.png)"></div>'
            '<style>@import url("data:text/css,%40import%20url%28https%3A%2F%2F'
            'tracker.example%2Fnested.css%29%3B");</style>'
            r'<link rel="preconnect" href="\\tracker.example/">'
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<image href="https://tracker.example/pixel.svg"></image></svg>'
            + duplicate_csp
            + '<link rel="modulepreload" href="'
            + safety.MATHJAX_URL
            + '?leak=abc">'
            + '<link rel="preload" as="image" href="/safe.png" '
            'imagesrcset="http:tracker.example/attack.png 1x" '
            'attributionsrc="http://tracker.example/register">'
            + '<a href="/" ping="http://tracker.example/ping">home</a>'
            + '<script src="'
            + safety.MATHJAX_URL
            + '?leak=first" src="javascripts/header-controls.js"></script>'
        )
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / "index.html").write_text(unsafe_html, encoding="utf-8")
            errors: list[str] = []
            safety.scan_rendered_site(site, errors)
        joined = "\n".join(errors)
        self.assertIn("script before Content-Security-Policy", joined)
        self.assertIn("unreviewed inline script hash", joined)
        self.assertIn("scripted navigation", joined)
        self.assertIn("data:text/javascript", joined)
        self.assertIn("modulepreload", unsafe_html)
        self.assertIn("non-allowlisted remote resource", joined)
        self.assertIn("imagesrcset", joined)
        self.assertIn("attributionsrc", joined)
        self.assertIn("ping", joined)
        self.assertIn("duplicate attribute [src]", joined)
        self.assertIn("duplicate attribute [content]", joined)
        self.assertIn("background-image:url(https://tracker.example/pixel.png)", joined)
        self.assertIn("active inline SVG non-fragment [href]", joined)
        self.assertIn("nested data CSS import", joined)
        self.assertIn("unsafe URL syntax", joined)
        self.assertTrue(safety.is_remote(r"\\tracker.example/path"))

    def test_secret_scan_covers_unusual_text_files_and_rejects_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            figures = root / "docs" / "assets" / "figures"
            figures.mkdir(parents=True)
            samples = {
                ".env": "TOKEN=" + "ghp_" + ("A" * 30),
                "github-fine-grained.env": (
                    "TOKEN=" + "github_" + "pat_" + ("A" * 40)
                ),
                "gitlab.env": "TOKEN=" + "glpat-" + ("A" * 30),
                "slack.env": "TOKEN=" + "xox" + "b-" + ("1-" * 6) + ("A" * 24),
                "pypi.env": "TOKEN=" + "pypi-" + ("A" * 30),
                "npm.env": "TOKEN=" + "npm_" + ("A" * 36),
                "stripe.env": "TOKEN=" + "sk_" + "live_" + ("A" * 30),
                "google.env": "TOKEN=" + "AIza" + ("A" * 35),
                "private.pem": "-----BEGIN " + "PRIVATE KEY-----",
                "script.sh": "/Users/" + "reader/private.txt",
                "notebook.ipynb": '{"cells":[{"source":["MRN: ' + "ABC123456" + '"]}]}',
                "NOTICE": "DOB: " + "1/2/2000",
            }
            for name, content in samples.items():
                (root / name).write_text(content, encoding="utf-8")
            (root / "old-release.zip").write_bytes(b"PK")
            (root / "payload.htm").write_text(
                "<script>location.href='https://tracker.example'</script>",
                encoding="utf-8",
            )
            (root / "payload.xml").write_text(
                '<?xml version="1.0"?><html '
                'xmlns="http://www.w3.org/1999/xhtml"><script/></html>',
                encoding="utf-8",
            )
            (root / ".gitleaks.toml").write_text(
                "[allowlist]\npaths = ['.*']\n", encoding="utf-8"
            )
            (root / ".gitleaksignore").write_text(
                "f" * 64 + "\n", encoding="utf-8"
            )
            scanner = root / "scripts" / "check_publication_safety.py"
            scanner.parent.mkdir()
            scanner.write_text("MR" + "N: ABC123456\n", encoding="utf-8")
            (figures / "unsupported.jpg").write_bytes(b"not an image")
            (root / "docs" / "extra.css").write_text(
                "body{color:black}\n",
                encoding="utf-8",
            )
            (root / "docs" / "extra.js").write_text(
                '"use strict";\n',
                encoding="utf-8",
            )
            published_shadow = root / "docs" / "site"
            published_shadow.mkdir()
            shadow_token = "gh" + "p_" + ("B" * 36)
            (published_shadow / "leak.txt").write_text(
                f"TOKEN={shadow_token}\n",
                encoding="utf-8",
            )
            (published_shadow / "case.pdf").write_bytes(b"%PDF")
            tracked_site = root / "site"
            tracked_site.mkdir()
            (tracked_site / "tracked-leak.txt").write_text(
                "DO" + "B: 1/2/2000\n",
                encoding="utf-8",
            )
            (tracked_site / "tracked-case.pdf").write_bytes(b"%PDF")
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "site/tracked-leak.txt", "site/tracked-case.pdf"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            (root / "docs" / "injected.md").write_text(
                '<script>window["location"] = "/" + "/" + "tracker.example"</script>'
                '<iframe srcdoc="<p>unreviewed</p>"></iframe>',
                encoding="utf-8",
            )
            errors, _, _, _ = safety.publication_errors(root)
        joined = "\n".join(errors)
        for name in samples:
            self.assertIn(name, joined)
        self.assertIn("forbidden publication-tree file type: old-release.zip", joined)
        self.assertIn("forbidden publication-tree file type: payload.htm", joined)
        self.assertIn("forbidden publication-tree file type: payload.xml", joined)
        self.assertIn(
            "forbidden publication-tree file type: docs/site/case.pdf",
            joined,
        )
        self.assertIn(
            "forbidden publication-tree file type: site/tracked-case.pdf",
            joined,
        )
        self.assertIn("unsupported teaching-figure file type: unsupported.jpg", joined)
        self.assertIn("forbidden Gitleaks override file: .gitleaks.toml", joined)
        self.assertIn("forbidden Gitleaks override file: .gitleaksignore", joined)
        self.assertIn(
            "published CSS is outside the inventoried stylesheet tree",
            joined,
        )
        self.assertIn(
            "published JavaScript is outside the inventoried script tree",
            joined,
        )
        self.assertIn("raw script element in published source", joined)
        self.assertIn("raw active element in published source", joined)
        self.assertIn("GitHub token pattern in docs/site/leak.txt", joined)
        self.assertIn(
            "labeled MRN-like value in scripts/check_publication_safety.py",
            joined,
        )
        self.assertIn("labeled DOB-like value in site/tracked-leak.txt", joined)
        with tempfile.TemporaryDirectory() as temporary:
            rendered_root = Path(temporary)
            for name in ("extra.css", "extra.js", "extra.map"):
                path = rendered_root / name
                path.write_text("/* passive */\n", encoding="utf-8")
                scope_errors, _ = safety.rendered_static_file_errors(path, name)
                self.assertTrue(
                    any(
                        "outside the reviewed" in error
                        or "outside the locked" in error
                        for error in scope_errors
                    )
                )

    def test_published_binary_nul_symlink_and_png_metadata_are_rejected(self) -> None:
        def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + chunk_type
                + payload
                + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
            )

        unsafe_png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + png_chunk(b"tEXt", b"Comment\x00DOB: 1/2/2000")
            + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
            + png_chunk(b"IEND", b"")
        )
        unsafe_physical_metadata_png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + png_chunk(b"pHYs", b"MRN" + b"123456")
            + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
            + png_chunk(b"IEND", b"")
        )
        covert_version_metadata_png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + png_chunk(
                b"tEXt",
                b"Software\x00Matplotlib version123456789012345.987654321.111223333, "
                b"https://matplotlib.org/",
            )
            + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
            + png_chunk(b"IEND", b"")
        )
        self.assertIn(
            "unreviewed PNG pHYs metadata",
            "\n".join(
                safety.png_metadata_errors(
                    unsafe_physical_metadata_png,
                    "identifier.png",
                )
            ),
        )
        self.assertIn(
            "unreviewed PNG text metadata",
            "\n".join(
                safety.png_metadata_errors(
                    covert_version_metadata_png,
                    "covert-version.png",
                )
            ),
        )

        def clean_png(red: int) -> bytes:
            return (
                b"\x89PNG\r\n\x1a\n"
                + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
                + png_chunk(
                    b"IDAT",
                    zlib.compress(bytes((0, red, 0, 0, 255))),
                )
                + png_chunk(b"IEND", b"")
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_figures = root / "source/docs/assets/figures"
            rendered_figures = root / "site/assets/figures"
            source_fonts = root / "source/docs/assets/fonts"
            rendered_fonts = root / "site/assets/fonts"
            source_styles = root / "source/docs/stylesheets"
            rendered_styles = root / "site/stylesheets"
            for directory in (
                source_figures,
                rendered_figures,
                source_fonts,
                rendered_fonts,
                source_styles,
                rendered_styles,
            ):
                directory.mkdir(parents=True)
            (source_figures / "same.png").write_bytes(clean_png(0))
            (rendered_figures / "same.png").write_bytes(clean_png(255))
            (source_figures / "missing.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 1 1"></svg>',
                encoding="utf-8",
            )
            (rendered_figures / "extra.png").write_bytes(clean_png(127))
            (source_fonts / "same.woff2").write_bytes(b"wOF2source")
            (rendered_fonts / "same.woff2").write_bytes(b"wOF2rendered")
            (source_fonts / "OFL-1.1.txt").write_text("OFL\n", encoding="utf-8")
            (rendered_fonts / "extra.woff2").write_bytes(b"wOF2extra")
            (source_styles / "authored.css").write_text(
                "body{color:black}\n", encoding="utf-8"
            )
            (rendered_styles / "authored.css").write_text(
                "body{background:url(data:image/png;base64,AA==)}\n",
                encoding="utf-8",
            )
            binding_errors = safety.deployed_asset_tree_errors(
                root / "source",
                root / "site",
            )
        binding_joined = "\n".join(binding_errors)
        self.assertIn("rendered figure asset hash mismatch: same.png", binding_joined)
        self.assertIn("rendered figure asset tree is missing: missing.svg", binding_joined)
        self.assertIn("rendered figure asset tree has extras: extra.png", binding_joined)
        self.assertIn("rendered font asset hash mismatch: same.woff2", binding_joined)
        self.assertIn("rendered font asset tree is missing: OFL-1.1.txt", binding_joined)
        self.assertIn("rendered font asset tree has extras: extra.woff2", binding_joined)
        self.assertIn(
            "rendered stylesheet asset hash mismatch: authored.css",
            binding_joined,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs"
            figures = docs / "assets" / "figures"
            figures.mkdir(parents=True)
            (docs / "nul-case.txt").write_bytes(b"\x00DOB: 1/2/2000\n")
            (docs / "payload.bin").write_bytes(b"arbitrary static payload")
            images = docs / "assets" / "images"
            images.mkdir()
            (images / "unreviewed.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 1 1"></svg>',
                encoding="utf-8",
            )
            outside = root / "outside.txt"
            outside.write_text("HOST_ONLY\n", encoding="utf-8")
            (docs / "linked.txt").symlink_to(Path("../outside.txt"))
            (figures / "metadata.png").write_bytes(unsafe_png)
            zero_svg = figures / "zero.svg"
            zero_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 0 100"></svg>',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                manifest.dimensions(zero_svg)
            self.assertIsNone(postprocess.dimensions(zero_svg))
            self.assertFalse(rendered_validation.positive_dimension("0"))
            metadata = root / "_meta"
            metadata.mkdir()
            (metadata / "tracked-case.txt").write_bytes(b"\x00DOB: 1/2/2000\n")
            (root / "notes.bin").write_bytes(b"unreviewed repository payload")
            (root / "unreviewed.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 1 1"></svg>',
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "add",
                    "_meta/tracked-case.txt",
                    "notes.bin",
                    "unreviewed.svg",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            source_errors, _, _, _ = safety.publication_errors(root)

            site = root / "rendered"
            site.mkdir()
            (site / "nul-case.txt").write_bytes(b"\x00DOB: 1/2/2000\n")
            (site / "payload.dat").write_bytes(b"arbitrary static payload")
            (site / "metadata.png").write_bytes(unsafe_png)
            (site / "unreviewed.png").write_bytes(clean_png(32))
            (site / "unreviewed.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 1 1"></svg>',
                encoding="utf-8",
            )
            (site / "unreviewed.woff2").write_bytes(b"wOF2unreviewed")
            rendered_errors: list[str] = []
            safety.scan_rendered_site(site, rendered_errors)
        with tempfile.TemporaryDirectory() as temporary:
            unavailable_root = Path(temporary)
            with patch.object(
                safety.subprocess,
                "run",
                side_effect=OSError("git unavailable"),
            ):
                inventory_errors, _, _, _ = safety.publication_errors(
                    unavailable_root
                )

        source_joined = "\n".join(source_errors)
        self.assertIn("NUL byte in text publication file: docs/nul-case.txt", source_joined)
        self.assertIn("unreviewed published file type: docs/payload.bin", source_joined)
        self.assertIn(
            "published SVG is outside the inventoried figure tree: "
            "docs/assets/images/unreviewed.svg",
            source_joined,
        )
        self.assertIn("published-source symlink is forbidden: docs/linked.txt", source_joined)
        self.assertIn(
            "NUL byte in text publication file: tracked/_meta/tracked-case.txt",
            source_joined,
        )
        self.assertIn(
            "unreviewed tracked binary outside inventoried publication assets: notes.bin",
            source_joined,
        )
        self.assertIn(
            "tracked SVG is outside the inventoried figure tree: unreviewed.svg",
            source_joined,
        )
        self.assertIn(
            "unreviewed PNG text metadata: docs/assets/figures/metadata.png",
            source_joined,
        )
        rendered_joined = "\n".join(rendered_errors)
        self.assertIn(
            "NUL byte in text publication file: rendered/nul-case.txt",
            rendered_joined,
        )
        self.assertIn("unreviewed rendered file type: payload.dat", rendered_joined)
        self.assertIn("unreviewed PNG text metadata: rendered/metadata.png", rendered_joined)
        self.assertIn(
            "rendered PNG is outside the inventoried figure tree",
            rendered_joined,
        )
        self.assertIn(
            "rendered SVG is outside the inventoried figure tree",
            rendered_joined,
        )
        self.assertIn(
            "rendered WOFF2 is outside the inventoried font tree",
            rendered_joined,
        )
        self.assertIn(
            "unable to inventory tracked repository files",
            "\n".join(inventory_errors),
        )

    def test_nested_figure_and_font_assets_are_recursively_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            figures = root / "docs" / "assets" / "figures"
            nested_figures = figures / "chapter"
            nested_figures.mkdir(parents=True)
            nested_svg = nested_figures / "reviewed.svg"
            nested_svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
                encoding="utf-8",
            )
            (nested_figures / "unsupported.jpg").write_bytes(b"not an image")
            (nested_figures / "manifest.json").write_text("{}\n", encoding="utf-8")

            metadata = root / "_meta"
            metadata.mkdir()
            (metadata / "release.json").write_text(
                '{"review_date":"2026-07-30"}\n', encoding="utf-8"
            )
            (metadata / "asset-rights-register.json").write_text(
                '{"review_date":"2026-07-30","assets":[]}\n',
                encoding="utf-8",
            )
            (figures / "manifest.json").write_text(
                '{"audit_date":"2026-07-30","assets":[]}\n',
                encoding="utf-8",
            )

            fonts = root / "docs" / "assets" / "fonts"
            nested_fonts = fonts / "subset"
            nested_fonts.mkdir(parents=True)
            (nested_fonts / "unrecorded.woff2").write_bytes(b"font")
            (fonts / "OFL-1.1.txt").write_text("OFL\n", encoding="utf-8")
            (fonts / "README.md").write_text(
                "# Vendored webfont provenance\n", encoding="utf-8"
            )

            figure_paths = [
                path.relative_to(root).as_posix()
                for path in safety.figure_assets(root)
            ]
            unsupported_paths = [
                path.relative_to(figures).as_posix()
                for path in safety.unsupported_figure_assets(root)
            ]
            errors, _, figure_count, _ = safety.publication_errors(root)

        self.assertEqual(
            figure_paths,
            ["docs/assets/figures/chapter/reviewed.svg"],
        )
        self.assertEqual(
            unsupported_paths,
            ["chapter/manifest.json", "chapter/unsupported.jpg"],
        )
        self.assertEqual(figure_count, 1)
        joined = "\n".join(errors)
        self.assertIn(
            "unsupported teaching-figure file type: chapter/manifest.json",
            joined,
        )
        self.assertIn(
            "unsupported teaching-figure file type: chapter/unsupported.jpg",
            joined,
        )
        self.assertIn(
            "asset rights register does not cover the current figure set",
            joined,
        )
        self.assertIn(
            "asset manifest does not cover the current figure set",
            joined,
        )
        self.assertIn("font inventory/hash missing: subset/unrecorded.woff2", joined)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fonts = root / "docs" / "assets" / "fonts"
            fonts.mkdir(parents=True)
            first = b"wOF2first-reviewed-font"
            second = b"wOF2second-reviewed-font"
            (fonts / "first.woff2").write_bytes(second)
            (fonts / "second.woff2").write_bytes(first)
            (fonts / "OFL-1.1.txt").write_text("OFL\n", encoding="utf-8")
            (fonts / "README.md").write_text(
                "# Vendored webfont provenance\n\n"
                "| Files | SHA-256 | Embedded metadata |\n"
                "| --- | --- | --- |\n"
                f"| `first.woff2` | `{safety.hashlib.sha256(first).hexdigest()}` | first |\n"
                f"| `second.woff2` | `{safety.hashlib.sha256(second).hexdigest()}` | second |\n",
                encoding="utf-8",
            )
            swap_errors, _, _, _ = safety.publication_errors(root)
        swap_joined = "\n".join(swap_errors)
        self.assertIn("font inventory/hash missing: first.woff2", swap_joined)
        self.assertIn("font inventory/hash missing: second.woff2", swap_joined)

    def test_dependency_mutations_are_rejected(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
        self.assertEqual(
            safety.dependency_inventory_errors(requirements, lock, sbom), []
        )
        errors = safety.dependency_inventory_errors(
            requirements + "\nunpinned-package>=1\n",
            lock + "\nrogue-package==1.0\n",
            sbom,
        )
        joined = "\n".join(errors)
        self.assertIn("unexpected active line in requirements.txt", joined)
        self.assertIn("unexpected active line in requirements.lock", joined)
        for injection in (
            "\n-r rogue.lock\n",
            "\n--requirement rogue.lock\n",
            "\nrogue @ https://example.invalid/rogue.whl\n",
        ):
            injection_errors = safety.dependency_inventory_errors(
                requirements,
                lock + injection,
                sbom,
            )
            self.assertTrue(
                any(
                    "unexpected active line in requirements.lock" in error
                    for error in injection_errors
                )
            )
        extra_component = copy.deepcopy(sbom)
        extra_component["components"].append(
            {
                "name": "rogue-package",
                "version": "1.0",
                "licenses": [{"license": {"id": "MIT"}}],
            }
        )
        self.assertTrue(
            any(
                "SBOM component set differs" in error
                for error in safety.dependency_inventory_errors(
                    requirements,
                    lock,
                    extra_component,
                )
            )
        )
        duplicate_component = copy.deepcopy(sbom)
        duplicate = copy.deepcopy(duplicate_component["components"][0])
        duplicate["bom-ref"] = "duplicate-name-version-with-new-reference"
        duplicate_component["components"].append(duplicate)
        duplicate_errors = safety.dependency_inventory_errors(
            requirements,
            lock,
            duplicate_component,
        )
        self.assertTrue(
            any(
                "duplicate SBOM component name/version" in error
                for error in duplicate_errors
            )
        )
        self.assertTrue(
            any(
                "SBOM component count differs" in error
                for error in duplicate_errors
            )
        )
        duplicate_reference = copy.deepcopy(sbom)
        duplicate_reference["components"][1]["bom-ref"] = (
            duplicate_reference["components"][0]["bom-ref"]
        )
        self.assertTrue(
            any(
                "duplicate SBOM bom-ref" in error
                for error in safety.dependency_inventory_errors(
                    requirements,
                    lock,
                    duplicate_reference,
                )
            )
        )
        missing_license = copy.deepcopy(sbom)
        missing_license["components"][0].pop("licenses")
        license_errors = safety.dependency_inventory_errors(
            requirements, lock, missing_license
        )
        self.assertTrue(
            any("SBOM license metadata missing" in error for error in license_errors)
        )
        audit_in = (ROOT / "requirements-audit.in").read_text(encoding="utf-8")
        audit_lock = (ROOT / "requirements-audit.lock").read_text(encoding="utf-8")
        audit_errors, audit_packages = safety.lockfile_errors(
            audit_in,
            audit_lock,
            "requirements-audit.in",
            "requirements-audit.lock",
        )
        self.assertEqual(audit_errors, [])
        self.assertGreaterEqual(len(audit_packages), 40)
        audit_errors, _ = safety.lockfile_errors(
            audit_in,
            audit_lock + "\nrogue-audit-package==1.0\n",
            "requirements-audit.in",
            "requirements-audit.lock",
        )
        self.assertTrue(
            any("unexpected active line" in error for error in audit_errors)
        )
        bootstrap_in = (
            ROOT / "requirements-audit-bootstrap.in"
        ).read_text(encoding="utf-8")
        bootstrap_lock = (
            ROOT / "requirements-audit-bootstrap.lock"
        ).read_text(encoding="utf-8")
        bootstrap_errors, bootstrap_packages = safety.lockfile_errors(
            bootstrap_in,
            bootstrap_lock,
            "requirements-audit-bootstrap.in",
            "requirements-audit-bootstrap.lock",
        )
        self.assertEqual(bootstrap_errors, [])
        self.assertEqual(
            bootstrap_packages,
            {("pip", "26.2"), ("setuptools", "83.0.0")},
        )
        self.assertEqual(bootstrap_lock, safety.AUDIT_BOOTSTRAP_LOCK)
        self.assertTrue(bootstrap_packages.issubset(audit_packages))
        self.assertNotEqual(
            bootstrap_lock.replace("setuptools==83.0.0", "setuptools==82.0.0"),
            safety.AUDIT_BOOTSTRAP_LOCK,
        )

    def test_browser_theme_inventory_follows_lock_and_environment(self) -> None:
        inventory = json.loads(
            (ROOT / "_meta" / "browser-assets.json").read_text(encoding="utf-8")
        )
        lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        installed = safety.importlib_metadata.version("mkdocs-material")
        self.assertEqual(
            safety.browser_theme_version_errors(
                inventory["theme_package"],
                lock,
                installed,
            ),
            [],
        )
        mutated_lock = lock.replace(
            f"mkdocs-material=={installed}",
            "mkdocs-material==0.0.0",
            1,
        )
        errors = safety.browser_theme_version_errors(
            inventory["theme_package"],
            mutated_lock,
            installed,
        )
        joined = "\n".join(errors)
        self.assertIn("theme package differs from requirements.lock", joined)
        self.assertIn(
            "installed Material for MkDocs version differs from requirements.lock",
            joined,
        )
        current_javascript = {
            relative: safety.sha256(ROOT / relative)
            for relative in safety.REVIEWED_AUTHORED_JAVASCRIPT_SHA256
        }
        self.assertEqual(
            safety.authored_javascript_baseline_errors(current_javascript),
            [],
        )
        mutated_javascript = dict(current_javascript)
        first_path = next(iter(mutated_javascript))
        mutated_javascript[first_path] = "0" * 64
        self.assertTrue(
            any(
                "manually reviewed digest baseline" in error
                for error in safety.authored_javascript_baseline_errors(
                    mutated_javascript
                )
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            shadow_root = Path(temporary)
            (shadow_root / "material").mkdir()
            (shadow_root / "scripts" / "material").mkdir(parents=True)
            (shadow_root / "mkdocs_material-9.7.6.dist-info").mkdir()
            shadow_errors = safety.source_dependency_shadow_errors(shadow_root)
        joined = "\n".join(shadow_errors)
        self.assertIn("material", joined)
        self.assertIn("scripts/material", joined)
        self.assertIn("mkdocs_material-9.7.6.dist-info", joined)

    def test_release_identity_drift_is_rejected(self) -> None:
        current_errors, _ = safety.release_identity_errors(ROOT)
        self.assertEqual(current_errors, [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_meta").mkdir()
            (root / "docs").mkdir()
            release = {
                "schema_version": 1,
                "version": "2099.01.01",
                "review_date": "2099-01-01",
                "review_date_long": "1 January 2099",
                "publication_review": "_meta/PUBLICATION_REVIEW_2099-01-01.md",
            }
            (root / "_meta" / "release.json").write_text(
                json.dumps(release), encoding="utf-8"
            )
            (root / "_meta" / "PUBLICATION_REVIEW_2099-01-01.md").write_text(
                "Reviewed: **31 December 2098**", encoding="utf-8"
            )
            (root / "_meta" / "ORIGINALITY.md").write_text(
                "Reviewed: **31 December 2098**", encoding="utf-8"
            )
            (root / "docs" / "publication-standards.md").write_text(
                "Last reviewed: **31 December 2098**", encoding="utf-8"
            )
            (root / "CITATION.cff").write_text(
                'version: "2026.07.30"\ndate-released: "2026-07-30"\n',
                encoding="utf-8",
            )
            errors, _ = safety.release_identity_errors(root)
        joined = "\n".join(errors)
        self.assertIn("publication review date differs", joined)
        self.assertIn("CITATION.cff version differs", joined)
        self.assertIn("CITATION.cff date-released differs", joined)
        self.assertIn("review date differs", joined)

    def test_originality_provenance_counts_follow_rights_register(self) -> None:
        register = json.loads(
            (ROOT / "_meta/asset-rights-register.json").read_text(
                encoding="utf-8"
            )
        )
        text = (ROOT / "_meta/ORIGINALITY.md").read_text(encoding="utf-8")
        self.assertEqual(
            safety.originality_provenance_errors(text, register["assets"]),
            [],
        )
        stale_asset_count = re.sub(
            r"(current figure set contains\s+)\d+",
            r"\g<1>999",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        stale_status_count = re.sub(
            r"(Rights-register evidence counts:\s+\*\*)\d+",
            r"\g<1>999",
            text,
            count=1,
        )
        self.assertIn(
            "current-figure count differs",
            "\n".join(
                safety.originality_provenance_errors(
                    stale_asset_count,
                    register["assets"],
                )
            ),
        )
        self.assertIn(
            "rights-status counts differ",
            "\n".join(
                safety.originality_provenance_errors(
                    stale_status_count,
                    register["assets"],
                )
            ),
        )

    def test_rights_register_rejects_duplicate_and_fake_provenance(self) -> None:
        register = json.loads(
            (ROOT / "_meta/asset-rights-register.json").read_text(encoding="utf-8")
        )
        for asset in register["assets"]:
            for candidate in asset["current_content_source_candidates"]:
                self.assertTrue(
                    Path(candidate).name.casefold().startswith(
                        ("generate_", "regenerate_")
                    )
                )
        self.assertEqual(rights.validate(ROOT, register), [])
        duplicated = copy.deepcopy(register)
        duplicated["assets"].append(copy.deepcopy(duplicated["assets"][0]))
        self.assertIn(
            "Duplicate register path",
            "\n".join(rights.validate(ROOT, duplicated)),
        )
        fake = copy.deepcopy(register)
        fake["assets"][0]["current_content_commit"] = "0" * 40
        self.assertIn(
            "Invalid current_content_commit",
            "\n".join(rights.validate(ROOT, fake)),
        )
        with patch.object(rights, "commit_is_ancestor", return_value=False):
            self.assertIn(
                "Recorded provenance commit is not an ancestor of HEAD",
                "\n".join(rights.validate(ROOT, register)),
            )
        with patch.object(rights, "changed_paths", return_value=()):
            self.assertIn(
                "Recorded provenance commit did not change asset path",
                "\n".join(rights.validate(ROOT, register)),
            )

    def test_rights_register_cli_requires_an_explicit_nonmutating_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with patch.object(sys, "argv", ["build_asset_rights_register.py"]):
                with self.assertRaises(SystemExit):
                    rights.main()
            with patch.object(
                sys,
                "argv",
                [
                    "build_asset_rights_register.py",
                    "--check",
                    "--refresh-from-history",
                ],
            ):
                with self.assertRaises(SystemExit):
                    rights.main()
        with (
            patch.object(
                sys,
                "argv",
                ["build_asset_rights_register.py", "--root", str(ROOT), "--check"],
            ),
            patch.object(rights, "validate", return_value=[]),
            patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("--check attempted to write"),
            ),
        ):
            self.assertEqual(rights.main(), 0)

    def test_workflow_fetches_history_tests_rendered_output_and_deploys_main_only(self) -> None:
        override = (ROOT / "overrides/main.html").read_text(encoding="utf-8")
        self.assertIn(
            '<link rel="license" href="{{ config.site_url }}publication-standards.html">',
            override,
        )
        self.assertNotIn(
            '<link rel="license" href="https://creativecommons.org/',
            override,
        )
        workflow = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("Fetch current pull-request head", workflow)
        self.assertIn("if: github.event_name == 'pull_request'", workflow)
        self.assertIn("PR_NUMBER: ${{ github.event.pull_request.number }}", workflow)
        self.assertIn(
            'if ! remote_ref_line="$(git ls-remote --refs origin "${remote_ref}")"',
            workflow,
        )
        self.assertIn(
            "Unable to inspect current pull-request head",
            workflow,
        )
        self.assertNotIn("git ls-remote --exit-code", workflow)
        self.assertIn(
            "+${remote_ref}:refs/remotes/pull/${PR_NUMBER}/head",
            workflow,
        )
        self.assertNotIn("refs/pull/*", workflow)
        self.assertNotIn("for namespace in head merge", workflow)
        self.assertIn(
            "Scan release-reachable and current pull-request history for secrets",
            workflow,
        )
        self.assertNotIn("filter: blob:none", workflow)
        self.assertIn("timeout-minutes: 45", workflow)
        self.assertIn("${RUNNER_TEMP}/ai-books-audit-venv", workflow)
        self.assertIn('GITLEAKS_VERSION: "8.30.1"', workflow)
        self.assertIn(
            "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
            workflow,
        )
        self.assertIn("gitleaks\" git --log-opts='--all --text -m' --redact", workflow)
        self.assertIn("--ignore-gitleaks-allow", workflow)
        self.assertIn("--gitleaks-ignore-path", workflow)
        self.assertIn("python scripts/test_gitleaks_controls.py", workflow)
        self.assertNotIn("python -m venv .audit-venv", workflow)
        self.assertIn("check_publication_safety.py --site-dir site", workflow)
        self.assertIn("python scripts/test_release_gates.py", workflow)
        self.assertIn("node scripts/test_mathjax_fallback.mjs", workflow)
        self.assertIn("node scripts/verify_mathjax_integrity.mjs", workflow)
        self.assertGreaterEqual(workflow.count("-m pip_audit"), 3)
        self.assertIn("requirements-audit-bootstrap.lock", workflow)
        self.assertIn("--no-deps --only-binary=:all:", workflow)
        self.assertIn("--no-build-isolation", workflow)
        self.assertIn('"${AUDIT_VENV}/bin/cffconvert" --validate --infile CITATION.cff', workflow)
        self.assertIn("scripts/verify_sbom_schema.py", workflow)
        self.assertIn('--expected-release-sha "${GITHUB_SHA}"', workflow)
        self.assertIn("python scripts/verify_release_marker.py", workflow)
        self.assertIn('--output "${response}"', workflow)
        self.assertNotIn("grep --fixed-strings", workflow)
        deploy_block = workflow.split("\n  deploy:", 1)[1]
        self.assertIn("contents: read", deploy_block)
        self.assertIn("steps.deployment.outputs.page_url", workflow)
        self.assertGreaterEqual(
            workflow.count("github.ref == 'refs/heads/main'"),
            3,
        )
        for relative in ("_meta/update-protocol.md", "CONTRIBUTING.md"):
            documented = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("--refresh-from-history", documented)
            self.assertRegex(
                " ".join(documented.casefold().split()),
                r"do not squash or rebase",
            )
            self.assertIn("build_asset_manifest.py", documented)
            for required_gate in (
                "verify_mathjax_integrity.mjs",
                "verify_sbom_schema.py",
                "test_gitleaks_controls.py",
                "pip-audit",
                "cffconvert",
                "Gitleaks",
            ):
                self.assertIn(required_gate, documented, f"{relative}: {required_gate}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
