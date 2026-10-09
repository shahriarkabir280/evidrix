import pytest

from app.rag.extract import (
    PAGE_SEPARATOR,
    DocumentKind,
    ExtractionError,
    extract,
    normalize_pdf_text,
)
from tests.factories import encrypt_pdf, make_image_only_pdf, make_pdf

PAGE_ONE = "Dependency injection lets a path operation declare what it needs."
PAGE_TWO = "OAuth2 with Password and Bearer tokens secures the API endpoints."


# --- PDF ---------------------------------------------------------------------------------


def test_pdf_keeps_one_entry_per_page_with_page_numbers():
    document = extract(make_pdf([PAGE_ONE, PAGE_TWO]), "guide.pdf", DocumentKind.PDF)

    assert [page.number for page in document.pages] == [1, 2]
    assert PAGE_ONE in document.pages[0].text
    assert PAGE_TWO in document.pages[1].text


def test_pdf_page_spans_slice_back_to_each_page():
    document = extract(make_pdf([PAGE_ONE, PAGE_TWO]), "guide.pdf", DocumentKind.PDF)

    spans = document.page_spans()

    assert [span.number for span in spans] == [1, 2]
    for span, page in zip(spans, document.pages, strict=True):
        assert document.text[span.char_start : span.char_end] == page.text
    assert document.text.count(PAGE_SEPARATOR) == 1


def test_pdf_title_comes_from_metadata():
    document = extract(make_pdf([PAGE_ONE], title="FastAPI Guide"), "x.pdf", DocumentKind.PDF)

    assert document.title == "FastAPI Guide"


def test_pdf_title_falls_back_to_filename():
    document = extract(make_pdf([PAGE_ONE]), "security-intro.pdf", DocumentKind.PDF)

    assert document.title == "security-intro"


def test_pdf_blank_pages_keep_their_numbers():
    document = extract(make_pdf([PAGE_ONE, "", PAGE_TWO]), "gap.pdf", DocumentKind.PDF)

    assert [page.number for page in document.pages] == [1, 2, 3]
    assert document.pages[1].text == ""


def test_image_only_pdf_has_no_extractable_text():
    with pytest.raises(ExtractionError) as error:
        extract(make_image_only_pdf(), "scan.pdf", DocumentKind.PDF)

    assert error.value.code == "no_extractable_text"


def test_password_protected_pdf_is_rejected():
    locked = encrypt_pdf(make_pdf([PAGE_ONE]), user_password="secret")

    with pytest.raises(ExtractionError) as error:
        extract(locked, "locked.pdf", DocumentKind.PDF)

    assert error.value.code == "encrypted_pdf"


def test_pdf_encrypted_only_against_editing_is_still_read():
    restricted = encrypt_pdf(make_pdf([PAGE_ONE]), user_password="")

    document = extract(restricted, "restricted.pdf", DocumentKind.PDF)

    assert PAGE_ONE in document.text


def test_corrupt_pdf_is_reported_as_invalid():
    with pytest.raises(ExtractionError) as error:
        extract(b"%PDF-1.7\nthis is not really a pdf", "broken.pdf", DocumentKind.PDF)

    assert error.value.code == "invalid_pdf"


# --- Markdown and plain text -------------------------------------------------------------

MARKDOWN = """---
title: ignored front matter
tags: [fastapi]
---
```bash
# this is a shell comment, not a heading
pip install fastapi
```

# Dependencies with yield

FastAPI supports dependencies that do extra steps after finishing.
"""


def test_markdown_strips_front_matter():
    document = extract(MARKDOWN.encode(), "deps.md", DocumentKind.MARKDOWN)

    assert "tags:" not in document.text
    assert document.text.startswith("```bash")


def test_markdown_title_skips_comments_inside_code_fences():
    document = extract(MARKDOWN.encode(), "deps.md", DocumentKind.MARKDOWN)

    assert document.title == "Dependencies with yield"


def test_markdown_is_a_single_unnumbered_page():
    document = extract(MARKDOWN.encode(), "deps.md", DocumentKind.MARKDOWN)

    assert len(document.pages) == 1
    assert document.pages[0].number is None
    assert document.page_spans() == []


def test_markdown_without_heading_uses_filename():
    body = b"Just a paragraph of notes about request bodies and validation."

    document = extract(body, "notes.md", DocumentKind.MARKDOWN)

    assert document.title == "notes"


def test_text_normalizes_line_endings_and_byte_order_mark():
    body = "﻿First line of the file\r\nSecond line of the file\r\n".encode()

    document = extract(body, "readme.txt", DocumentKind.TEXT)

    assert document.text == "First line of the file\nSecond line of the file"
    assert document.title == "readme"


def test_invalid_utf8_is_rejected():
    with pytest.raises(ExtractionError) as error:
        extract(b"caf\xe9 au lait is not utf-8 encoded", "latin1.txt", DocumentKind.TEXT)

    assert error.value.code == "invalid_encoding"


def test_binary_content_with_nul_bytes_is_rejected():
    with pytest.raises(ExtractionError) as error:
        extract(b"MZ\x00\x00binary executable payload", "tool.txt", DocumentKind.TEXT)

    assert error.value.code == "invalid_encoding"


def test_whitespace_only_text_has_no_extractable_text():
    with pytest.raises(ExtractionError) as error:
        extract(b"   \n\n\t  \n", "empty.md", DocumentKind.MARKDOWN)

    assert error.value.code == "no_extractable_text"


# --- PDF text normalization --------------------------------------------------------------


def test_normalize_rejoins_words_hyphenated_across_lines():
    assert normalize_pdf_text("semantic retri-\neval works") == "semantic retrieval works"


def test_normalize_collapses_spaces_and_extra_blank_lines():
    raw = "Title   with\t\tgaps \n\n\n\n Next paragraph"

    assert normalize_pdf_text(raw) == "Title with gaps\n\nNext paragraph"


def test_normalize_removes_page_separator_characters():
    assert PAGE_SEPARATOR not in normalize_pdf_text(f"before{PAGE_SEPARATOR}after")
