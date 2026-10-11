import pytest

from scripts.fetch_corpus import CorpusError, is_safe_member_path, normalize_fastapi_markdown

SOURCE_FILES = {
    "docs_src/deps/tutorial001.py": "line one\nline two\nline three\nline four\n",
}


def read_file(path: str) -> str:
    if path not in SOURCE_FILES:
        raise CorpusError(f"missing {path}")
    return SOURCE_FILES[path]


def normalize(text: str) -> str:
    return normalize_fastapi_markdown(text, read_file)


def test_include_becomes_a_fenced_code_block():
    result = normalize("Intro\n\n{* ../../docs_src/deps/tutorial001.py hl[2] *}\n")

    assert "```python\nline one\nline two\nline three\nline four\n```" in result


def test_include_line_ranges_select_only_those_lines():
    result = normalize("{* ../../docs_src/deps/tutorial001.py ln[1:2,4] hl[1] *}")

    assert "```python\nline one\nline two\nline four\n```" in result
    assert "line three" not in result


def test_missing_include_fails_loudly():
    with pytest.raises(CorpusError):
        normalize("{* ../../docs_src/deps/does_not_exist.py *}")


def test_heading_anchors_are_removed():
    result = normalize("# First Steps { #first-steps }\n\n## Recap { #recap }\n")

    assert result == "# First Steps\n\n## Recap\n"


def test_admonitions_become_labelled_paragraphs():
    text = "/// tip\n\nUse yield once.\n\n///\n\n/// note | Technical Details\n\nX\n///"

    result = normalize(text)

    assert "**Tip:**" in result
    assert "**Note: Technical Details**" in result
    assert "///" not in result


def test_nested_tab_blocks_with_four_slashes_are_handled():
    result = normalize("//// tab | Python 3.10+\n\nUse Annotated.\n\n////")

    assert result == "**Tab: Python 3.10+**\n\nUse Annotated.\n"


def test_html_tags_are_dropped_but_their_text_is_kept():
    result = normalize("Do some <dfn title='cleanup code'>extra steps</dfn> after.<br>")

    assert result == "Do some extra steps after.\n"


def test_code_fences_are_left_untouched():
    text = '```html\n<div class="x">kept</div>\n/// not an admonition\n```\n'

    assert normalize(text) == text


@pytest.mark.parametrize(
    ("path", "safe"),
    [
        ("docs/en/docs/tutorial/body.md", True),
        ("docs/en/docs/tutorial/../../../../etc/passwd.md", False),
        ("/etc/passwd", False),
        ("docs/en/./docs/tutorial/body.md", False),
    ],
)
def test_archive_paths_cannot_escape_the_output_folder(path, safe):
    assert is_safe_member_path(path) is safe
