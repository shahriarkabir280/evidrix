"""The real models under concurrent requests (component 6D regression test).

FastAPI runs sync endpoints in a thread pool, so concurrent requests call the embedder and the
reranker from several threads at once. On Apple's MPS backend that crashed the process
(segfault) or deadlocked it until INFERENCE_LOCK serialized the calls.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.rag.embeddings import get_embedder
from app.rag.rerank import get_reranker

pytestmark = pytest.mark.model


def test_concurrent_embedding_and_reranking_from_threads():
    embedder, reranker = get_embedder(), get_reranker()
    passages = [f"passage {k} about cookies and headers" for k in range(20)]

    def work(worker: int) -> int:
        for step in range(10):
            vector = embedder.embed_query(f"question {worker} {step} about cookies")
            scores = reranker.score(f"question {worker} {step}", passages)
            assert vector.shape == (embedder.dim,) and len(scores) == len(passages)
        return worker

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sorted(pool.map(work, range(6))) == list(range(6))
