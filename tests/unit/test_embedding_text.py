from app.rag.embeddings import text_for_embedding

CHUNK = "Use Form to declare form data input parameters."


def build(section_path, *, title="Form Data", contextual_header=True, max_tokens=512):
    return text_for_embedding(
        CHUNK,
        title=title,
        section_path=section_path,
        contextual_header=contextual_header,
        count_tokens=lambda text: len(text.split()),
        max_tokens=max_tokens,
    )


def test_markdown_section_path_already_starts_with_the_title():
    assert build("Form Data > Recap") == f"Form Data > Recap\n\n{CHUNK}"


def test_pdf_chunk_without_section_gets_the_document_title():
    assert build(None, title="Logging HOWTO") == f"Logging HOWTO\n\n{CHUNK}"


def test_section_path_not_starting_with_title_is_prefixed_with_it():
    assert build("Recap") == f"Form Data > Recap\n\n{CHUNK}"


def test_header_can_be_switched_off():
    assert build("Form Data > Recap", contextual_header=False) == CHUNK


def test_header_is_dropped_when_it_would_push_the_chunk_past_the_model_limit():
    # The chunk is 8 "tokens"; with the 4-token header and 2 special tokens it would need 14.
    assert build("Form Data > Recap", max_tokens=13) == CHUNK
