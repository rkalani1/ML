#!/usr/bin/env python3
"""Structurally inspect and stamp the exact deployed-revision HTML marker."""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

RELEASE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REVIEWED_HEAD_ELEMENTS = {"link", "meta", "script", "title"}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass(frozen=True)
class ReleaseMarkerInspection:
    html_count: int
    html_end_count: int
    head_count: int
    head_end_offsets: tuple[int, ...]
    body_count: int
    body_end_count: int
    structure_error_count: int
    direct_head_contents: tuple[str, ...]
    outside_direct_head_count: int
    ambiguous_attribute_count: int


class ReleaseMarkerParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.line_starts = [0]
        self.line_starts.extend(
            match.end() for match in re.finditer("\n", source)
        )
        self.stack: list[str] = []
        self.html_count = 0
        self.html_end_count = 0
        self.head_count = 0
        self.head_end_offsets: list[int] = []
        self.body_count = 0
        self.body_end_count = 0
        self.structure_error_count = 0
        self.valid_head_open = False
        self.valid_head_closed = False
        self.valid_body_open = False
        self.valid_body_closed = False
        self.direct_head_contents: list[str] = []
        self.outside_direct_head_count = 0
        self.ambiguous_attribute_count = 0

    def absolute_offset(self) -> int:
        line_number, column = self.getpos()
        return self.line_starts[line_number - 1] + column

    def inspect_meta(self, attrs: list[tuple[str, str | None]]) -> None:
        names = [
            value or ""
            for key, value in attrs
            if key.casefold() == "name"
        ]
        if not any(value.casefold() == "release-revision" for value in names):
            return
        contents = [
            value or ""
            for key, value in attrs
            if key.casefold() == "content"
        ]
        if len(names) != 1 or len(contents) != 1:
            self.ambiguous_attribute_count += 1
        content = contents[0] if len(contents) == 1 else ""
        if self.valid_head_open and self.stack == ["html", "head"]:
            self.direct_head_contents.append(content)
        else:
            self.outside_direct_head_count += 1

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if (
            (tag != "html" and not self.stack)
            or (
                self.stack == ["html"]
                and tag not in {"head", "body"}
            )
        ):
            self.structure_error_count += 1
        if self.valid_head_open and tag not in REVIEWED_HEAD_ELEMENTS:
            self.structure_error_count += 1
        if tag == "html":
            self.html_count += 1
            if self.stack or self.html_count != 1:
                self.structure_error_count += 1
        elif tag == "head":
            self.head_count += 1
            if (
                self.stack == ["html"]
                and not self.valid_head_open
                and not self.valid_head_closed
                and self.body_count == 0
            ):
                self.valid_head_open = True
            else:
                self.structure_error_count += 1
        elif tag == "body":
            self.body_count += 1
            if (
                self.stack == ["html"]
                and self.valid_head_closed
                and not self.valid_body_open
                and not self.valid_body_closed
                and self.body_count == 1
            ):
                self.valid_body_open = True
            else:
                self.structure_error_count += 1
        if tag == "meta":
            self.inspect_meta(attrs)
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if (
            (tag != "html" and not self.stack)
            or (
                self.stack == ["html"]
                and tag not in {"head", "body"}
            )
        ):
            self.structure_error_count += 1
        if self.valid_head_open and tag not in REVIEWED_HEAD_ELEMENTS:
            self.structure_error_count += 1
        if self.valid_head_open and tag in {"script", "title"}:
            self.structure_error_count += 1
        if tag == "html":
            self.html_count += 1
            self.structure_error_count += 1
        elif tag == "head":
            self.head_count += 1
            self.structure_error_count += 1
        elif tag == "body":
            self.body_count += 1
            self.structure_error_count += 1
        if tag == "meta":
            self.inspect_meta(attrs)

    def handle_data(self, data: str) -> None:
        visible = bool(data.strip(" \t\r\n\f"))
        if visible and (not self.stack or self.stack == ["html"]):
            self.structure_error_count += 1
        if (
            self.valid_head_open
            and self.stack == ["html", "head"]
            and visible
        ):
            self.structure_error_count += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if not self.stack:
            self.structure_error_count += 1
        if self.stack == ["html"] and tag != "html":
            self.structure_error_count += 1
        if (
            self.valid_head_open
            and tag not in {"head", "script", "title"}
        ):
            self.structure_error_count += 1
        if tag == "head":
            if self.valid_head_open and self.stack == ["html", "head"]:
                self.head_end_offsets.append(self.absolute_offset())
                self.valid_head_open = False
                self.valid_head_closed = True
            else:
                self.structure_error_count += 1
        elif tag == "body":
            if self.valid_body_open and self.stack == ["html", "body"]:
                self.body_end_count += 1
                self.valid_body_open = False
                self.valid_body_closed = True
            else:
                self.structure_error_count += 1
        elif tag == "html":
            if self.stack == ["html"] and self.valid_body_closed:
                self.html_end_count += 1
            else:
                self.structure_error_count += 1
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == tag:
                del self.stack[index:]
                break

    def inspection(self) -> ReleaseMarkerInspection:
        return ReleaseMarkerInspection(
            html_count=self.html_count,
            html_end_count=self.html_end_count,
            head_count=self.head_count,
            head_end_offsets=tuple(self.head_end_offsets),
            body_count=self.body_count,
            body_end_count=self.body_end_count,
            structure_error_count=self.structure_error_count,
            direct_head_contents=tuple(self.direct_head_contents),
            outside_direct_head_count=self.outside_direct_head_count,
            ambiguous_attribute_count=self.ambiguous_attribute_count,
        )


def inspect_release_revision(source: str) -> ReleaseMarkerInspection:
    parser = ReleaseMarkerParser(source)
    parser.feed(source)
    parser.close()
    return parser.inspection()


def release_revision_errors(source: str, expected_sha: str) -> list[str]:
    if not RELEASE_SHA_RE.fullmatch(expected_sha):
        return ["expected release revision must be a lowercase 40-character Git SHA"]
    inspection = inspect_release_revision(source)
    errors: list[str] = []
    if (
        inspection.html_count != 1
        or inspection.html_end_count != 1
        or inspection.head_count != 1
        or len(inspection.head_end_offsets) != 1
        or inspection.body_count != 1
        or inspection.body_end_count != 1
        or inspection.structure_error_count
    ):
        errors.append(
            "HTML must contain one ordered html root with direct head then body"
        )
    if inspection.ambiguous_attribute_count:
        errors.append("release-revision meta contains ambiguous attributes")
    if inspection.outside_direct_head_count:
        errors.append("release-revision meta must be a direct child of head")
    if len(inspection.direct_head_contents) != 1:
        errors.append("HTML must contain exactly one direct-head release-revision meta")
    elif inspection.direct_head_contents[0] != expected_sha:
        errors.append(
            "direct-head release-revision content differs from expected revision"
        )
    return errors


def stamp_release_revision(source: str, release_sha: str) -> tuple[str, int]:
    if not release_sha:
        return source, 0
    if not RELEASE_SHA_RE.fullmatch(release_sha):
        raise ValueError("GITHUB_SHA must be a lowercase 40-character Git SHA")
    inspection = inspect_release_revision(source)
    if (
        inspection.html_count != 1
        or inspection.html_end_count != 1
        or inspection.head_count != 1
        or len(inspection.head_end_offsets) != 1
        or inspection.body_count != 1
        or inspection.body_end_count != 1
        or inspection.structure_error_count
    ):
        raise ValueError(
            "rendered HTML must contain one ordered html root with direct "
            "head then body"
        )
    if inspection.ambiguous_attribute_count:
        raise ValueError("release-revision meta contains ambiguous attributes")
    if inspection.outside_direct_head_count:
        raise ValueError("release-revision meta must be a direct child of head")
    if len(inspection.direct_head_contents) > 1:
        raise ValueError("rendered HTML contains duplicate release revisions")
    if inspection.direct_head_contents:
        if inspection.direct_head_contents[0] != release_sha:
            raise ValueError("rendered HTML contains a different release revision")
        return source, 0
    marker = f'<meta name="release-revision" content="{release_sha}">'
    offset = inspection.head_end_offsets[0]
    return source[:offset] + marker + "\n" + source[offset:], 1
