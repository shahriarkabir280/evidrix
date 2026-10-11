import re

import pytest

from app.rag.chunking import Chunk, ChunkingConfig, chunk_document
from app.rag.extract import DocumentKind, ExtractedDocument, PageText, extract
from tests.factories import make_pdf

# Words stand in for tokens, so expected sizes are easy to reason about.
CONFIG = ChunkingConfig(chunk_size_tokens=40, max_tokens=50, overlap_tokens=10)


def count_words(text: str) -> int:
    return len(text.split())


def sentences(label: str, count: int, words_per_sentence: int = 8) -> str:
    """`count` sentences of distinct words, e.g. "alpha1 alpha2 ... alpha8."."""
    out, n = [], 0
    for _ in range(count):
        words = [f"{label}{n + i}" for i in range(words_per_sentence)]
        n += words_per_sentence
        out.append(" ".join(words) + ".")
    return " ".join(out)


def markdown(text: str) -> ExtractedDocument:
    return ExtractedDocument(title="t", pages=(PageText(number=None, text=text),))


def chunk(document: ExtractedDocument, config: ChunkingConfig = CONFIG) -> list[Chunk]:
    return chunk_document(document, config, count_words)


LONG_GUIDE = markdown(
    "# Guide\n\n"
    "## Install\n\n"
    f"{sentences('install', 6)}\n\n"
    "```bash\npip install fastapi\nuvicorn main:app --reload\n```\n\n"
    f"{sentences('more', 5)}\n\n"
    "### Linux\n\n"
    f"{sentences('linux', 4)}\n\n"
    "## Usage\n\n"
    f"{sentences('usage', 9)}\n"
)


# --- Invariants that must hold for every document ----------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        LONG_GUIDE,
        markdown(sentences("solo", 30)),
        extract(
            make_pdf([sentences("p1x", 3), sentences("p2x", 3), sentences("p3x", 3)]),
            "a.pdf",
            DocumentKind.PDF,
        ),
    ],
    ids=["markdown", "single-paragraph", "pdf"],
)
def test_chunks_are_exact_slices_within_the_token_limit(document):
    chunks = chunk(document)

    assert chunks
    for c in chunks:
        assert document.text[c.char_start : c.char_end] == c.text
        assert c.token_count == count_words(c.text) <= CONFIG.max_tokens
        assert c.text == c.text.strip()
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_every_word_of_the_document_is_in_some_chunk():
    chunks = chunk(LONG_GUIDE)

    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(LONG_GUIDE.text.split())


# --- Sections and headings ---------------------------------------------------------------


def test_title_and_first_heading_stay_with_their_content():
    document = markdown("# Title\n\n## First\n\nShort body text under the first section.")

    chunks = chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("# Title")
    assert chunks[0].section_path == "Title > First"


def test_new_top_level_section_starts_a_new_chunk_without_overlap():
    document = markdown(f"## Alpha\n\n{sentences('a', 2)}\n\n## Beta\n\n{sentences('b', 2)}")

    first, second = chunk(document)

    assert second.text.startswith("## Beta")
    assert second.char_start > first.char_end
    assert (first.section_path, second.section_path) == ("Alpha", "Beta")


def test_short_intro_is_not_left_as_a_stub_before_a_long_list():
    config = ChunkingConfig(
        chunk_size_tokens=40, max_tokens=50, overlap_tokens=10, min_chunk_tokens=10
    )
    long_list = "\n".join(f"* item{i} explained here" for i in range(11))  # 44 words
    document = markdown(f"## Options\n\nHere are the options:\n\n{long_list}")

    chunks = chunk(document, config)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("## Options\n\nHere are the options:")


def test_section_path_follows_heading_nesting():
    config = ChunkingConfig(chunk_size_tokens=40, max_tokens=50, overlap_tokens=10, section_level=3)

    paths = {c.section_path for c in chunk(LONG_GUIDE, config)}

    assert "Guide > Install" in paths
    assert "Guide > Install > Linux" in paths
    assert "Guide > Usage" in paths


def test_a_chunk_never_ends_with_a_heading():
    for c in chunk(LONG_GUIDE):
        assert not re.search(r"(^|\n)#{1,6} [^\n]*$", c.text), c.text


# --- Overlap -----------------------------------------------------------------------------


def test_consecutive_chunks_in_one_section_share_whole_sentences():
    chunks = chunk(markdown(sentences("w", 20)))

    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = previous.text[current.char_start - previous.char_start :]
        assert current.char_start < previous.char_end
        assert 0 < count_words(shared) <= CONFIG.overlap_tokens
        assert shared.endswith(".")


# --- Code blocks and oversized units -----------------------------------------------------


def test_code_block_that_fits_is_never_split():
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(10)) + "\n```"
    document = markdown(f"{sentences('before', 3)}\n\n{code}\n\n{sentences('after', 3)}")

    assert any(code in c.text for c in chunk(document))


def test_oversized_code_block_is_cut_at_line_boundaries():
    code = (
        "```python\n" + "\n".join(f"value_{i} = compute({i}, {i + 1})" for i in range(40)) + "\n```"
    )
    document = markdown(code)

    chunks = chunk(document)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= CONFIG.max_tokens
        assert c.char_start == 0 or document.text[c.char_start - 1] == "\n"


def test_paragraph_without_any_punctuation_is_cut_on_whitespace():
    document = markdown(" ".join(f"word{i}" for i in range(200)))

    chunks = chunk(document)

    assert len(chunks) >= 4
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)


# --- Pages -------------------------------------------------------------------------------


def test_pdf_chunks_report_the_pages_they_span():
    pages = [sentences(f"page{n}x", 3) for n in range(1, 5)]
    document = extract(make_pdf(pages), "guide.pdf", DocumentKind.PDF)

    chunks = chunk(document)

    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 4
    assert any(c.page_start < c.page_end for c in chunks)
    for c in chunks:
        assert f"page{c.page_start}x" in c.text


def test_markdown_chunks_have_no_pages():
    assert all(c.page_start is None and c.page_end is None for c in chunk(LONG_GUIDE))
