"""Text extraction from PDF, Markdown, and TXT into ExtractedDocument.

Component 1C · Plan: Part 1, step 6.
"""

import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTPage, LTTextContainer, LTTextLine
from pypdf import PdfReader

# Pages are joined with a form feed, so page boundaries survive in Document.extracted_text.
PAGE_SEPARATOR = "\f"
# Below this many non-whitespace characters we treat the document as having no text
# (typically a scanned PDF, which would need OCR).
MIN_TEXT_CHARS = 20
# A title read from the first page: at most this many lines, and set at least this much larger
# than the body text (otherwise a page in one font size would yield its first paragraph).
MAX_TITLE_LINES = 3
MAX_TITLE_CHARS = 200
MIN_TITLE_SIZE_RATIO = 1.15

# Typographic ligatures that some PDFs (often LaTeX) store as single characters: "ﬁne-tune" would
# not match a search for "fine-tune". Only these are mapped; full NFKC normalization would also
# rewrite superscripts and other symbols whose meaning matters.
_LIGATURES = str.maketrans(
    {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
     "\ufb05": "st", "\ufb06": "st"}
)  # fmt: skip
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_H1 = re.compile(r"#[ \t]+(.+?)[ \t#]*")
_FENCE = re.compile(r"[ \t]{0,3}(```|~~~)")
# Metadata titles that name the file or the tool instead of the document.
_PLACEHOLDER_TITLE = re.compile(
    r"untitled|microsoft word\b.*|.*\.(?:docx?|pdf|tex|dvi)", re.IGNORECASE
)


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
    title: str  # indexed: part of every chunk's embedding header and section path
    pages: tuple[PageText, ...]
    display_title: str | None = None  # shown to users instead of `title` when set

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
    text = _clean(text).translate(_LIGATURES)
    # Re-join words that the PDF hyphenated across a line break ("retri-\neval" -> "retrieval"),
    # but keep the hyphen of a compound that continues with more hyphens ("state-\nof-the-art").
    # A compound broken at its only hyphen ("cosine-\nsimilarity") cannot be told apart.
    text = re.sub(
        r"(\w)-\n([a-z]\S*)",
        lambda m: f"{m[1]}{'-' if '-' in m[2] else ''}{m[2]}",
        text,
    )
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    # Keep paragraph breaks (one blank line) but no larger gaps.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    """pypdf checks encryption and reads metadata; pdfminer.six extracts the text.

    pypdf's text extraction drops the spaces between words for some fonts (Type0 /
    Identity-H, common in LaTeX/Sphinx PDFs such as the Python docs). pdfminer places words by
    their measured positions, keeps the spacing, and separates paragraphs with blank lines.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        # Many PDFs are encrypted only to restrict editing and open with an empty password.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("encrypted_pdf", "The PDF is password-protected.")
        metadata_title = (reader.metadata.title if reader.metadata else None) or ""
        heading = None
        pages = []
        for i, layout in enumerate(extract_pages(io.BytesIO(data)), start=1):
            if i == 1:
                heading = _first_page_title(layout)
            pages.append(PageText(number=i, text=normalize_pdf_text(_page_text(layout))))
    except ExtractionError:
        raise
    except Exception as exc:
        # Uploaded PDFs are untrusted input; any parser failure means "unreadable PDF".
        raise ExtractionError("invalid_pdf", f"The PDF could not be read: {exc}") from exc

    # The indexed title stays as measured (metadata, else the file name); the first page's title
    # is for display only. In every chunk's header, long paper titles lowered paper retrieval
    # (hybrid_rerank MRR@5: dev 0.712 -> 0.659, test 0.522 -> 0.500).
    metadata_title = metadata_title.strip()
    title = metadata_title or _stem(filename)
    display_title = None
    if not metadata_title or _PLACEHOLDER_TITLE.fullmatch(metadata_title):
        display_title = heading
    return ExtractedDocument(title=title, pages=tuple(pages), display_title=display_title)


def _page_text(layout: LTPage) -> str:
    # One text box per paragraph-like block; a blank line between boxes keeps paragraphs apart.
    return "\n".join(box.get_text() for box in layout if isinstance(box, LTTextContainer))


def _first_page_title(page: LTPage) -> str | None:
    """The lines set in the largest type on the first page: the title, in most papers.

    Many papers (arXiv preprints especially) have no title in their metadata. A line's size is
    the median of its characters, so a large chapter number at the start doesn't count, and only
    upright characters count, which leaves out arXiv's sideways identifier stamp.
    """
    lines = []  # (top, size, text)
    char_sizes: Counter[float] = Counter()
    for box in page:
        if not isinstance(box, LTTextContainer):
            continue
        for line in [box] if isinstance(box, LTTextLine) else box:
            if not isinstance(line, LTTextLine):
                continue
            sizes = [
                round(char.size, 1)
                for char in line
                if isinstance(char, LTChar) and char.upright and not char.get_text().isspace()
            ]
            if sizes:
                lines.append((line.y1, statistics.median(sizes), line.get_text()))
                char_sizes.update(sizes)
    if not lines:
        return None

    body_size = char_sizes.most_common(1)[0][0]
    largest = max(size for _, size, _ in lines)
    if largest < body_size * MIN_TITLE_SIZE_RATIO:
        return None
    lines.sort(key=lambda line: -line[0])  # top to bottom
    first = next(i for i, (_, size, _) in enumerate(lines) if size == largest)
    title_lines = []
    for _, size, text in lines[first : first + MAX_TITLE_LINES]:
        if size != largest:
            break
        title_lines.append(text.strip())
    title = " ".join(normalize_pdf_text("\n".join(title_lines)).split())
    if len(title) > MAX_TITLE_CHARS or sum(char.isalpha() for char in title) < 3:
        return None
    return title


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
