"""The labeled benchmark: loading, validation, and resolving evidence quotes to text positions.

Component 4A/4B.

    uv run python -m evals.benchmark   # check that every evidence quote is found in its source

Labels are quotes, not chunk IDs: a quote keeps pointing at the same text however the corpus is
chunked, so the same benchmark can compare chunk sizes (plan §0, item 3).

Provenance: 60 questions written from the source documents only (never from retrieval output) by a
single annotator who also built the system; no second-annotator review was done. Splits were
assigned at creation, stratified by type with seed 2026 (60% dev, 40% test), before any experiment.
"""

import bisect
import json
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.rag.extract import ExtractedDocument, extract
from app.services.ingestion import detect_kind

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
CORPUS_DIR = Path(__file__).parent / "corpus" / "raw"

QuestionType = Literal["factual", "identifier", "multi_hop", "distractor", "unanswerable"]


class Evidence(BaseModel):
    source: str  # path relative to evals/corpus/raw, e.g. "fastapi/tutorial/cors.md"
    quote: str = Field(min_length=10)


class Question(BaseModel):
    id: str
    split: Literal["dev", "test"]
    type: QuestionType
    question: str
    answerable: bool
    evidence: list[Evidence]
    expected_answer: str | None
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "Question":
        if self.answerable != (self.type != "unanswerable"):
            raise ValueError("only unanswerable questions may have answerable=false")
        if self.answerable and not (self.evidence and self.expected_answer):
            raise ValueError("answerable questions need evidence and an expected answer")
        if not self.answerable and (self.evidence or self.expected_answer):
            raise ValueError("unanswerable questions must not have evidence or an answer")
        return self


def load_questions(path: Path = QUESTIONS_PATH, split: str | None = None) -> list[Question]:
    questions = [
        Question.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return [q for q in questions if split in (None, q.split)]


@dataclass(frozen=True)
class Span:
    char_start: int
    char_end: int


@cache
def source_document(source: str) -> ExtractedDocument:
    """The source as the system sees it: extracted by the same code that ingests uploads."""
    path = CORPUS_DIR / source
    data = path.read_bytes()
    return extract(data, path.name, detect_kind(path.name, data))


def find_quote(text: str, quote: str) -> list[Span]:
    """Every place `quote` occurs in `text`, ignoring differences in whitespace.

    PDF extraction breaks lines where the page did, so "level is WARNING, which" can come out
    as "level is WARNING,\\nwhich". Both sides are compared with whitespace runs collapsed, and
    matches are mapped back to offsets in the original text.
    """
    collapsed, origin = _collapse_whitespace(text)
    needle = " ".join(quote.split())
    spans = []
    start = collapsed.find(needle)
    while start != -1:
        end = start + len(needle)
        spans.append(Span(origin[start], origin[end - 1] + 1))
        start = collapsed.find(needle, start + 1)
    return spans


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """`text` with each whitespace run turned into one space, plus each output char's origin."""
    out: list[str] = []
    origin: list[int] = []
    for match in re.finditer(r"\s+|\S+", text):
        if match.group().isspace():
            out.append(" ")
            origin.append(match.start())
        else:
            out.append(match.group())
            origin.extend(range(match.start(), match.end()))
    return "".join(out), origin


def resolve(evidence: Evidence) -> list[Span]:
    return find_quote(source_document(evidence.source).text, evidence.quote)


def page_of(source: str, char_offset: int) -> int | None:
    spans = source_document(source).page_spans()
    if not spans:
        return None
    starts = [s.char_start for s in spans]
    return spans[max(bisect.bisect_right(starts, char_offset) - 1, 0)].number


def check(questions: list[Question]) -> list[str]:
    problems = []
    for question in questions:
        for evidence in question.evidence:
            if not (CORPUS_DIR / evidence.source).is_file():
                problems.append(f"{question.id}: missing source {evidence.source}")
            elif not resolve(evidence):
                problems.append(
                    f"{question.id}: quote not found in {evidence.source}: {evidence.quote[:60]!r}"
                )
    return problems


def main() -> None:
    questions = load_questions()
    problems = check(questions)
    for problem in problems:
        print(problem)
    quotes = sum(len(q.evidence) for q in questions)
    print(f"{len(questions)} questions, {quotes} evidence quotes, {len(problems)} problems")
    by_split = {s: sum(q.split == s for q in questions) for s in ("dev", "test")}
    print(f"splits: {json.dumps(by_split)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
