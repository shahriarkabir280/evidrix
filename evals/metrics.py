"""Label resolver plus retrieval, answer, and abstention metrics.

Component 4B (resolver) · 4C (retrieval) · 4D (answer).

Retrieval metrics are computed per question over the top-k retrieved chunks, for answerable
questions only (an unanswerable question has no evidence to find).
"""

import math
from dataclasses import dataclass

# A chunk counts as containing an evidence quote when it holds at least half of the quote's
# characters. Overlapping chunks can split a quote; half is enough to carry the answer.
MIN_QUOTE_COVERAGE = 0.5


@dataclass(frozen=True)
class SourceSpan:
    source: str  # path relative to evals/corpus/raw
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


# One evidence item = every place its quote occurs (usually exactly one).
EvidenceItem = list[SourceSpan]


def covers(chunk: SourceSpan, quote: SourceSpan) -> bool:
    if chunk.source != quote.source:
        return False
    overlap = min(chunk.char_end, quote.char_end) - max(chunk.char_start, quote.char_start)
    return overlap >= MIN_QUOTE_COVERAGE * quote.length


def covered_items(chunk: SourceSpan, items: list[EvidenceItem]) -> set[int]:
    """Indexes of the evidence items this chunk contains (at any of their occurrences)."""
    return {i for i, item in enumerate(items) if any(covers(chunk, span) for span in item)}


@dataclass(frozen=True)
class RetrievalScores:
    recall: float  # share of evidence items found in the top k
    all_evidence: float  # 1 if every evidence item is in the top k (matters for multi-hop)
    hit: float  # 1 if any relevant chunk is in the top k
    mrr: float  # 1 / rank of the first relevant chunk, 0 if none
    ndcg: float  # rank-discounted gain, normalized by the best possible ranking


def retrieval_scores(
    ranked: list[SourceSpan], items: list[EvidenceItem], k: int
) -> RetrievalScores:
    """Score one question's ranked chunks against its evidence.

    nDCG uses binary gain per *new* evidence item: a chunk earns 1 only if it covers an item that
    no higher-ranked chunk covered, so two overlapping chunks holding the same quote count once.
    The ideal ranking puts one chunk per evidence item at the top.
    """
    if not items:
        raise ValueError("retrieval metrics need at least one evidence item")
    found: set[int] = set()
    first_relevant: int | None = None
    dcg = 0.0
    for rank, chunk in enumerate(ranked[:k], start=1):
        covered = covered_items(chunk, items)
        if covered and first_relevant is None:
            first_relevant = rank
        new = covered - found
        if new:
            dcg += 1.0 / math.log2(rank + 1)
        found |= covered
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(items)) + 1))
    return RetrievalScores(
        recall=len(found) / len(items),
        all_evidence=float(len(found) == len(items)),
        hit=float(first_relevant is not None),
        mrr=1.0 / first_relevant if first_relevant else 0.0,
        ndcg=dcg / ideal,
    )


def mean_scores(scores: list[RetrievalScores]) -> RetrievalScores:
    if not scores:
        raise ValueError("no scores to average")
    n = len(scores)
    return RetrievalScores(
        recall=sum(s.recall for s in scores) / n,
        all_evidence=sum(s.all_evidence for s in scores) / n,
        hit=sum(s.hit for s in scores) / n,
        mrr=sum(s.mrr for s in scores) / n,
        ndcg=sum(s.ndcg for s in scores) / n,
    )
