"""Text extraction from PDF, Markdown, and TXT into ExtractedDocument.

Component 1C · Plan: Part 1, step 6.
"""

import io
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath

from pypdf import PdfReader

# Pages are joined with a form feed, so page boundaries survive in Document.extracted_text.
PAGE_SEPARATOR = "\f"
# Below this many non-whitespace characters we treat the document as having no text
# (typically a scanned PDF, which would need OCR).
MIN_TEXT_CHARS = 20

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_H1 = re.compile(r"#[ \t]+(.+?)[ \t#]*")
_FENCE = re.compile(r"[ \t]{0,3}(```|~~~)")


class DocumentKind(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class ExtractionError(Exception):
    """A permanent failure: retrying the same bytes will fail the same way."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PageText:
    number: int | None  # 1-based PDF page; None for formats without pages
    text: str


@dataclass(frozen=True)
class PageSpan:
    number: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(page.text for page in self.pages)

    def page_spans(self) -> list[PageSpan]:
        """Character range of each numbered page within `text`."""
        spans = []
        offset = 0
        for page in self.pages:
            end = offset + len(page.text)
            if page.number is not None:
                spans.append(PageSpan(page.number, offset, end))
            offset = end + len(PAGE_SEPARATOR)
        return spans


def extract(data: bytes, filename: str, kind: DocumentKind) -> ExtractedDocument:
    match kind:
        case DocumentKind.PDF:
            document = _extract_pdf(data, filename)
        case DocumentKind.MARKDOWN:
            document = _extract_markdown(data, filename)
        case DocumentKind.TEXT:
            document = _extract_plain_text(data, filename)

    if sum(not char.isspace() for char in document.text) < MIN_TEXT_CHARS:
        raise ExtractionError(
            "no_extractable_text",
            "No text could be extracted. Scanned PDFs need OCR, which is not supported.",
        )
    return document


def normalize_pdf_text(text: str) -> str:
    text = _clean(text)
    # Re-join words that the PDF hyphenated across a line break ("retri-\neval").
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    # Keep paragraph breaks (one blank line) but no larger gaps.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        # Many PDFs are encrypted only to restrict editing and open with an empty password.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("encrypted_pdf", "The PDF is password-protected.")
        pages = tuple(
            PageText(number=i, text=normalize_pdf_text(page.extract_text() or ""))
            for i, page in enumerate(reader.pages, start=1)
        )
        metadata_title = reader.metadata.title if reader.metadata else None
    except ExtractionError:
        raise
    except Exception as exc:
        # Uploaded PDFs are untrusted input; any parser failure means "unreadable PDF".
        raise ExtractionError("invalid_pdf", f"The PDF could not be read: {exc}") from exc

    title = (metadata_title or "").strip() or _stem(filename)
    return ExtractedDocument(title=title, pages=pages)


def _extract_markdown(data: bytes, filename: str) -> ExtractedDocument:
    text = _clean(_decode(data))
    text = _FRONT_MATTER.sub("", text, count=1).strip()
    title = _markdown_title(text) or _stem(filename)
    return ExtractedDocument(title=title, pages=(PageText(number=None, text=text),))


def _extract_plain_text(data: bytes, filename: str) -> ExtractedDocument:
    text = _clean(_decode(data)).strip()
    return ExtractedDocument(title=_stem(filename), pages=(PageText(number=None, text=text),))


def _markdown_title(text: str) -> str | None:
    """First `# Heading` outside fenced code blocks (where `#` is often a comment)."""
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (match := _H1.fullmatch(line)):
            return match.group(1)
    return None


def _decode(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")  # -sig drops a leading byte-order mark if present
    except UnicodeDecodeError as exc:
        raise ExtractionError("invalid_encoding", "The file is not valid UTF-8 text.") from exc
    if "\x00" in text:
        raise ExtractionError("invalid_encoding", "The file contains NUL bytes; it is not text.")
    return text


def _clean(text: str) -> str:
    """Normalize line endings and remove characters that would break page offsets."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace(PAGE_SEPARATOR, "\n").replace("\x00", "")


def _stem(filename: str) -> str:
    return PurePath(filename).stem or filename
