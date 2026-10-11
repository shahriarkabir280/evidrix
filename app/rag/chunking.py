"""Structure-aware, token-limited chunker with offsets and section paths.

Component 2B · Plan: Part 2, step 3.

A chunk is always an exact slice of ExtractedDocument.text:
    document.text[chunk.char_start:chunk.char_end] == chunk.text
so a citation can point back to the precise place in the source.
"""

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

from app.rag.extract import ExtractedDocument

TokenCounter = Callable[[str], int]

_FENCE = re.compile(r"[ \t]{0,3}(```|~~~)")
_HEADING = re.compile(r"(#{1,6})[ \t]+(.+?)[ \t#]*")
# Places where a span may be cut, from most to least preferred.
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+")
_LINE_END = re.compile(r"\n+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = 400
    # Hard ceiling. bge-small reads 512 tokens; leave room for special tokens and a
    # short title/section header that is prepended at embedding time.
    max_tokens: int = 480
    overlap_tokens: int = 64
    # A chunk smaller than this may grow past chunk_size_tokens (never past max_tokens)
    # rather than be left as a stub, e.g. a heading plus "Here are the options:".
    min_chunk_tokens: int = 100
    # Never pack text from different sections at this heading level or above.
    section_level: int = 2


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    page_start: int | None
    page_end: int | None
    section_path: str | None


@dataclass(frozen=True)
class _Unit:
    """The smallest piece the packer moves around: a paragraph, heading, code block or sentence."""

    start: int
    end: int
    section_path: str | None
    starts_section: bool = False
    is_heading: bool = False


def chunk_document(
    document: ExtractedDocument, config: ChunkingConfig, count_tokens: TokenCounter
) -> list[Chunk]:
    text = document.text
    units = [
        piece
        for unit in _segment(document, config)
        for piece in _split_oversized(
            text, unit, config.max_tokens, config.chunk_size_tokens, count_tokens
        )
    ]
    spans = _pack(text, units, config, count_tokens)
    return [
        _make_chunk(index, document, start, end, section, count_tokens)
        for index, (start, end, section) in enumerate(spans)
    ]


# --- 1. Segment the document into structural units ---------------------------------------


def _segment(document: ExtractedDocument, config: ChunkingConfig) -> list[_Unit]:
    if any(page.number is not None for page in document.pages):
        return _segment_pages(document, config.section_level)
    return _segment_markdown(document.text, config.section_level)


def _segment_markdown(text: str, section_level: int) -> list[_Unit]:
    """Paragraphs, headings and fenced code blocks, each tagged with its heading path."""
    units: list[_Unit] = []
    headings: list[tuple[int, str]] = []
    paragraph_start: int | None = None
    fence_start: int | None = None
    offset = 0

    def path() -> str | None:
        return " > ".join(title for _, title in headings) or None

    def close_paragraph(end: int) -> None:
        nonlocal paragraph_start
        if paragraph_start is not None:
            units.append(_Unit(paragraph_start, end, path()))
            paragraph_start = None

    for line in text.splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        content = line.rstrip("\n")
        line_end = line_start + len(content)

        if fence_start is not None:
            if _FENCE.match(content):
                units.append(_Unit(fence_start, line_end, path()))
                fence_start = None
        elif _FENCE.match(content):
            close_paragraph(line_start)
            fence_start = line_start
        elif heading := _HEADING.fullmatch(content):
            close_paragraph(line_start)
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
            units.append(
                _Unit(
                    line_start,
                    line_end,
                    path(),
                    starts_section=level <= section_level,
                    is_heading=True,
                )
            )
        elif not content.strip():
            close_paragraph(line_start)
        elif paragraph_start is None:
            paragraph_start = line_start

    if fence_start is not None:  # unclosed fence: keep the rest as one code block
        units.append(_Unit(fence_start, len(text), path()))
    close_paragraph(len(text))
    return [_trim(text, unit) for unit in units if text[unit.start : unit.end].strip()]


def _segment_pages(document: ExtractedDocument, section_level: int) -> list[_Unit]:
    """PDF pages, split into paragraphs at blank lines (extraction separates text boxes).

    Paragraphs that look like numbered section headings ("3 Model", "3.1 Training Details")
    become heading units, so chunks get a section path such as "Title > 3 Model > 3.1 ...".
    A page without blank lines arrives as one unit and is cut at sentence boundaries by
    _split_oversized.
    """
    text = document.text
    paragraphs = []
    for span in document.page_spans():
        page_text = text[span.char_start : span.char_end]
        cursor = 0
        for gap in re.finditer(r"\n\s*\n", page_text):
            paragraphs.append(_Unit(span.char_start + cursor, span.char_start + gap.start(), None))
            cursor = gap.end()
        paragraphs.append(_Unit(span.char_start + cursor, span.char_end, None))
    paragraphs = [_trim(text, u) for u in paragraphs if text[u.start : u.end].strip()]

    detector = _PdfHeadingDetector()
    headings: list[tuple[int, str]] = [(1, document.title)]
    units = []
    for paragraph in paragraphs:
        content = text[paragraph.start : paragraph.end]
        heading_end = paragraph.end
        heading = detector.detect(content)
        if heading is None and "\n" in content:
            # Some layouts (e.g. ACM) put "1 INTRODUCTION" in the same text box as the
            # paragraph below it; a short first line may be the heading on its own.
            first_line = content[: content.index("\n")]
            heading = detector.detect(first_line, max_words=MAX_RUN_IN_HEADING_WORDS)
            heading_end = paragraph.start + len(first_line)
        if heading is None:
            units.append(_Unit(paragraph.start, paragraph.end, _join_path(headings)))
            continue
        level, title = heading
        # The document title is level 1, so a top-level section ("3 Model") is level 2.
        headings = [h for h in headings if h[0] < level + 1] + [(level + 1, title)]
        units.append(
            _Unit(
                paragraph.start,
                heading_end,
                _join_path(headings),
                starts_section=level + 1 <= section_level,
                is_heading=True,
            )
        )
        if heading_end < paragraph.end:
            rest = _trim(text, _Unit(heading_end, paragraph.end, None))
            units.append(_Unit(rest.start, rest.end, _join_path(headings)))
    return units


_PDF_HEADING = re.compile(r"(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z].*)")
_APPENDIX_HEADING = re.compile(r"([A-H](?:\.\d{1,2}){0,3})\s+([A-Z].*)")
_BACK_MATTER = re.compile(r"(references|bibliography|acknowledge?ments?)", re.IGNORECASE)
_DOT_LEADERS = re.compile(r"\.\s?\.\s?\.")
MAX_HEADING_WORDS = 12
MAX_RUN_IN_HEADING_WORDS = 8  # stricter for a first line that shares a box with body text


class _PdfHeadingDetector:
    """Recognizes numbered section headings in PDF text, which has no heading markup.

    A candidate must be one or two short lines, not end with a full stop, and continue the
    section numbering plausibly (2 may follow 1 or 2.3, but 9 may not follow 2). "3. We
    evaluate..." is a list item, not a heading: section numbers have no trailing dot. A missed
    heading does not block its subsections: "2.1" may follow section 1, and "1 ..." restarts
    the numbering (a table of contents may already have counted to 8). Lines with dot leaders
    are table-of-contents entries. After References, lettered appendix headings
    ("A Implementation Details") are accepted too.
    """

    def __init__(self) -> None:
        self.top = 0  # number of the current top-level section
        self.in_back_matter = False

    def detect(self, paragraph: str, max_words: int = MAX_HEADING_WORDS) -> tuple[int, str] | None:
        lines = paragraph.split("\n")
        candidate = " ".join(line.strip() for line in lines)
        if len(lines) > 2 or len(candidate.split()) > max_words:
            return None
        if candidate.endswith(".") or _DOT_LEADERS.search(candidate):
            return None
        if _BACK_MATTER.fullmatch(candidate):
            self.in_back_matter = True
            return 1, candidate
        numbered = _PDF_HEADING.fullmatch(candidate)
        if numbered and not self.in_back_matter:
            parts = [int(n) for n in numbered.group(1).split(".")]
            restart = parts == [1]  # body sections restart at 1 after a table of contents
            if not (self.top <= parts[0] <= self.top + 2 or restart):
                return None
            self.top = parts[0]
            return len(parts), candidate
        lettered = _APPENDIX_HEADING.fullmatch(candidate)
        if lettered and self.in_back_matter:
            return lettered.group(1).count(".") + 1, candidate
        return None


def _join_path(headings: list[tuple[int, str]]) -> str | None:
    return " > ".join(title for _, title in headings if title) or None


# --- 2. Split units that are too big on their own ----------------------------------------


def _split_oversized(
    text: str, unit: _Unit, split_above: int, merge_up_to: int, count_tokens: TokenCounter
) -> list[_Unit]:
    """Cut a unit bigger than `split_above` tokens into pieces of about `merge_up_to` tokens.

    Pieces are merged only up to the target chunk size, not the hard maximum, so the packer
    still has room to put an overlap in front of them.
    """
    if count_tokens(text[unit.start : unit.end]) <= split_above:
        return [unit]
    # Code blocks split on lines (a "sentence" means nothing in code); prose on sentences.
    is_code = _FENCE.match(text, unit.start) is not None
    for boundary in ([_LINE_END] if is_code else [_SENTENCE_END, _LINE_END]) + [_WHITESPACE]:
        pieces = _cut(text, unit, boundary)
        if len(pieces) > 1:
            merged = _greedy_merge(text, pieces, min(merge_up_to, split_above), count_tokens)
            return [
                part
                for piece in merged
                for part in _split_oversized(text, piece, split_above, merge_up_to, count_tokens)
            ]
    return [unit]  # a single unbreakable token run; nothing more we can do


def _cut(text: str, unit: _Unit, boundary: re.Pattern[str]) -> list[_Unit]:
    pieces, cursor = [], unit.start
    for match in boundary.finditer(text, unit.start, unit.end):
        if match.start() > cursor:
            pieces.append(_Unit(cursor, match.start(), unit.section_path))
        cursor = match.end()
    if cursor < unit.end:
        pieces.append(_Unit(cursor, unit.end, unit.section_path))
    if pieces:  # keep the flags of the unit we split on its first piece
        pieces[0] = _Unit(
            pieces[0].start, pieces[0].end, unit.section_path, unit.starts_section, unit.is_heading
        )
    return pieces


def _greedy_merge(
    text: str, pieces: list[_Unit], max_tokens: int, count_tokens: TokenCounter
) -> list[_Unit]:
    merged = [pieces[0]]
    for piece in pieces[1:]:
        last = merged[-1]
        if count_tokens(text[last.start : piece.end]) <= max_tokens:
            merged[-1] = _Unit(
                last.start, piece.end, last.section_path, last.starts_section, last.is_heading
            )
        else:
            merged.append(piece)
    return merged


# --- 3. Pack units into chunks -----------------------------------------------------------


def _pack(
    text: str, units: list[_Unit], config: ChunkingConfig, count_tokens: TokenCounter
) -> list[tuple[int, int, str | None]]:
    spans: list[tuple[int, int, str | None]] = []
    current: list[_Unit] = []
    start = 0  # may sit before current[0].start when the chunk begins with overlap

    def flush(final: bool = False) -> None:
        nonlocal current
        content = [u for u in current if not u.is_heading]
        if not content and not final:
            return  # only headings so far: keep them with the text that follows
        # A heading belongs with the text under it, never alone at the end of a chunk.
        carried = []
        while content and current[-1].is_heading and not final:
            carried.insert(0, current.pop())
        section = content[0].section_path if content else current[0].section_path
        spans.append((start, current[-1].end, section))
        current = carried

    queue = list(units)
    while queue:
        unit = queue.pop(0)
        if current and unit.starts_section:
            flush()
            if current:
                start = current[0].start
        if not current:
            current, start = [unit], unit.start
            continue
        with_unit = count_tokens(text[start : unit.end])
        is_stub = count_tokens(text[start : current[-1].end]) < config.min_chunk_tokens
        if with_unit <= config.chunk_size_tokens or (is_stub and with_unit <= config.max_tokens):
            current.append(unit)
            continue

        previous_end = current[-1].end
        flush()
        if current:  # carried headings start the next chunk; no overlap across them
            start = current[0].start
            budget = config.max_tokens - count_tokens(text[start : unit.start])
            if count_tokens(text[start : unit.end]) > config.max_tokens and budget > 0:
                pieces = _split_oversized(
                    text, unit, budget, config.chunk_size_tokens, count_tokens
                )
                if len(pieces) > 1:
                    queue[0:0] = pieces[1:]
                    unit = pieces[0]
            current.append(unit)
        else:
            start = _overlap_start(text, spans[-1][0], previous_end, unit, config, count_tokens)
            current = [unit]

    if current:
        flush(final=True)
    return spans


def _overlap_start(
    text: str,
    chunk_start: int,
    chunk_end: int,
    next_unit: _Unit,
    config: ChunkingConfig,
    count_tokens: TokenCounter,
) -> int:
    """Begin the next chunk with the previous chunk's last whole sentences (or lines).

    Takes the earliest sentence/line start whose tail fits in overlap_tokens and still lets
    the new chunk stay under max_tokens. Returns next_unit.start when nothing fits.
    """
    candidates = [m.end() for m in _SENTENCE_END.finditer(text, chunk_start, chunk_end)]
    candidates += [m.end() for m in _LINE_END.finditer(text, chunk_start, chunk_end)]
    for position in sorted(set(candidates)):
        if position >= chunk_end:
            continue
        fits_overlap = count_tokens(text[position:chunk_end]) <= config.overlap_tokens
        if fits_overlap and count_tokens(text[position : next_unit.end]) <= config.max_tokens:
            return position
    return next_unit.start


# --- Helpers -----------------------------------------------------------------------------


def _make_chunk(
    index: int,
    document: ExtractedDocument,
    start: int,
    end: int,
    section_path: str | None,
    count_tokens: TokenCounter,
) -> Chunk:
    chunk_text = document.text[start:end]
    page_start, page_end = _pages_for(document, start, end)
    return Chunk(
        index=index,
        text=chunk_text,
        char_start=start,
        char_end=end,
        token_count=count_tokens(chunk_text),
        page_start=page_start,
        page_end=page_end,
        section_path=section_path,
    )


def _pages_for(document: ExtractedDocument, start: int, end: int) -> tuple[int | None, int | None]:
    spans = document.page_spans()
    if not spans:
        return None, None
    starts = [span.char_start for span in spans]
    first = spans[max(bisect.bisect_right(starts, start) - 1, 0)]
    last = spans[max(bisect.bisect_right(starts, end - 1) - 1, 0)]
    return first.number, last.number


def _trim(text: str, unit: _Unit) -> _Unit:
    """Shrink a unit so it neither starts nor ends with whitespace."""
    start, end = unit.start, unit.end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Unit(start, end, unit.section_path, unit.starts_section, unit.is_heading)
