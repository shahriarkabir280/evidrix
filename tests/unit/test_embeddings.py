import numpy as np
import pytest

from app.rag import embeddings
from app.rag.embeddings import SentenceTransformerEmbedder, resolve_device
from tests.fakes import FakeEmbedder

INSTRUCTION = "Represent this sentence for searching relevant passages: "
LOGGING = "To log to a file, pass a filename to logging.basicConfig and set the level."
SORTING = "Use the key parameter of sorted() to sort strings case-insensitively."

# --- Device selection --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert resolve_device("auto") == expected


def test_explicit_device_is_used_as_is():
    assert resolve_device("cpu") == "cpu"


# --- FakeEmbedder (used by every retrieval test) -----------------------------------------


def test_fake_embedder_is_deterministic_and_normalized():
    fake = FakeEmbedder()

    first, second = fake.embed_documents([LOGGING, LOGGING])

    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_fake_embedder_ranks_shared_words_higher():
    fake = FakeEmbedder()
    query = fake.embed_query("how to log to a file with logging")

    logging_vec, sorting_vec = fake.embed_documents([LOGGING, SORTING])

    assert query @ logging_vec > query @ sorting_vec


# --- The real model ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def bge():
    return SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5", query_instruction=INSTRUCTION, device="cpu"
    )


@pytest.mark.model
def test_documents_become_normalized_float32_vectors(bge):
    vectors = bge.embed_documents([LOGGING, SORTING])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


@pytest.mark.model
def test_no_documents_gives_an_empty_matrix(bge):
    assert bge.embed_documents([]).shape == (0, 384)


@pytest.mark.model
def test_query_gets_the_instruction_but_documents_do_not(bge):
    question = "How do I log to a file?"

    query = bge.embed_query(question)
    with_instruction, without = bge.embed_documents([INSTRUCTION + question, question])

    assert query @ with_instruction == pytest.approx(1.0, abs=1e-5)
    assert query @ without < 0.99


@pytest.mark.model
def test_query_is_closer_to_the_passage_that_answers_it(bge):
    query = bge.embed_query("How can I write log messages into a file?")

    logging_vec, sorting_vec = bge.embed_documents([LOGGING, SORTING])

    assert query @ logging_vec > query @ sorting_vec


@pytest.mark.model
def test_count_tokens_uses_the_model_tokenizer_without_special_tokens(bge):
    assert bge.count_tokens("hello world") == 2
    assert bge.max_tokens == 512


def test_get_embedder_is_built_once(monkeypatch):
    built = []
    monkeypatch.setattr(
        embeddings, "SentenceTransformerEmbedder", lambda *a, **kw: built.append(1) or object()
    )
    embeddings.get_embedder.cache_clear()

    first, second = embeddings.get_embedder(), embeddings.get_embedder()

    assert first is second
    assert len(built) == 1
    embeddings.get_embedder.cache_clear()
