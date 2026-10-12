"""Runs a retrieval config over a split and writes results with provenance.

Component 4E.

    uv run python -m evals.run --preset hybrid_rerank --split dev --mode retrieval
    uv run python -m evals.run --preset hybrid_rerank --split dev --mode full
    uv run python -m evals.run --benchmark papers --preset hybrid_rerank --split dev

`retrieval` scores only what was retrieved (seconds). `full` also generates answers with the
production answer path and grades them with the judge (minutes; uses the local LLM).
The test split is held out for the final report and needs --final.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentStatus
from app.db.repositories import index_versions as index_versions_repo
from app.db.repositories.users import DEV_USER_ID
from app.db.session import get_session_factory
from app.rag.embeddings import get_embedder
from app.rag.generation import BadModelOutputError
from app.rag.llm import LLMError, get_llm
from app.rag.rerank import get_reranker
from app.rag.retrieval import load_preset, retrieve
from app.services.answering import answer_question
from evals.benchmark import (
    PAPERS_QUESTIONS_PATH,
    QUESTIONS_PATH,
    Question,
    evidence_items,
    load_questions,
)
from evals.judge import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, JudgeError, get_judge
from evals.metrics import (
    SourceSpan,
    abstention_scores,
    citation_precision,
    covered_items,
    retrieval_scores,
)

RESULTS_DIR = Path(__file__).parent / "results"
# Answers saved between the generation and judge passes (not committed).
WORK_DIR = RESULTS_DIR / "work"
MANIFEST_PATH = Path(__file__).parent / "corpus" / "manifest.json"
# Each benchmark searches only its own collection of the corpus.
BENCHMARKS = {
    "docs": {"questions": QUESTIONS_PATH, "collection": "docs"},
    "papers": {"questions": PAPERS_QUESTIONS_PATH, "collection": "papers"},
}
RETRIEVAL_METRICS = ("recall", "all_evidence", "hit", "mrr", "ndcg")


class CorpusNotIndexedError(Exception):
    pass


def corpus_documents(session: Session, collection: str = "docs") -> dict[Any, str]:
    """Map document id -> corpus source path, for one collection of the evaluation corpus.

    Retrieval is restricted to these documents, so anything else uploaded to the same database
    (manual tests, demos, the other collection) cannot leak into the results.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    source_by_hash = {
        d["sha256"]: d["path"]
        for d in manifest["documents"]
        if d.get("collection", "docs") == collection
    }
    rows = session.execute(
        select(Document.id, Document.content_hash).where(
            Document.owner_id == DEV_USER_ID,
            Document.status == DocumentStatus.COMPLETED,
            Document.content_hash.in_(source_by_hash),
        )
    ).all()
    found = {doc_id: source_by_hash[content_hash] for doc_id, content_hash in rows}
    missing = set(source_by_hash.values()) - set(found.values())
    if missing:
        raise CorpusNotIndexedError(
            f"{len(missing)} corpus documents are not ingested (e.g. {sorted(missing)[0]}). "
            "Run `make corpus` and `uv run python -m app.cli ingest-dir evals/corpus/raw`."
        )
    return found


def evaluate_question(
    question: Question,
    *,
    session: Session,
    config,
    documents: dict[Any, str],
    mode: str,
    k: int,
) -> tuple[dict[str, Any], list[str]]:
    """Retrieve (and in full mode, answer) one question. Returns the row and the passage texts
    the answer was generated from, which the judge pass needs but the results file doesn't keep."""
    settings = get_settings()
    embedder, reranker = get_embedder(), get_reranker()
    row: dict[str, Any] = {
        "id": question.id,
        "type": question.type,
        "answerable": question.answerable,
        "errors": [],
    }
    index_version = index_versions_repo.get_active(session)
    document_ids = list(documents)

    if mode == "retrieval":
        found = retrieve(
            session,
            embedder,
            reranker,
            config,
            owner_id=DEV_USER_ID,
            index_version_id=index_version.id,
            question=question.question,
            document_ids=document_ids,
        )
        hits, timings = found.hits, found.timings_ms
        generated = None
    else:
        try:
            result = answer_question(
                session,
                embedder,
                reranker,
                get_llm(),
                owner_id=DEV_USER_ID,
                question=question.question,
                document_ids=document_ids,
                retrieval=config,
                max_tokens=settings.llm_max_output_tokens,
                prompt_version=settings.answer_prompt_version,
            )
        except (LLMError, BadModelOutputError) as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
            return row, []
        hits, timings, generated = result.hits, result.timings_ms, result.generated

    spans = [SourceSpan(documents[h.document_id], h.char_start, h.char_end) for h in hits]
    items = evidence_items(question) if question.answerable else []
    row["timings_ms"] = timings
    row["retrieved"] = [
        {"rank": h.rank, "score": round(h.score, 4), "source": s.source, "start": s.char_start,
         "end": s.char_end, "relevant": bool(items and covered_items(s, items))}
        for h, s in zip(hits, spans, strict=True)
    ]  # fmt: skip
    if question.answerable:
        row["retrieval"] = vars(retrieval_scores(spans, items, k))

    if generated is None:
        return row, []

    row["answer"] = generated.answer
    row["abstained"] = generated.abstained
    row["abstain_reason"] = generated.abstain_reason
    row["usage"] = {
        "prompt_tokens": sum(c.prompt_tokens for c in generated.llm_calls),
        "completion_tokens": sum(c.completion_tokens for c in generated.llm_calls),
    }
    cited = [spans[c.marker - 1] for c in generated.citations]
    row["citations"] = [c.marker for c in generated.citations]
    if question.answerable:
        row["citation_precision"] = citation_precision(cited, items)

    return row, [h.text for h in hits]


def judge_answer(row: dict[str, Any], question: Question, sources: list[str], judge) -> None:
    """Add the judge's correctness and faithfulness verdicts to a row that has an answer."""
    if "answer" not in row:
        return
    if question.answerable and row["abstained"]:
        # Refusing an answerable question is an incorrect answer; no need to ask the judge.
        row["correctness"] = {"verdict": "incorrect", "score": 0.0, "reason": "abstained"}
    elif question.answerable:
        try:
            correctness = judge.correctness(
                question.question, question.expected_answer, row["answer"]
            )
            row["correctness"] = vars(correctness)
        except JudgeError as exc:
            row["errors"].append(f"correctness: {exc}")
    if not row["abstained"]:
        try:
            faithfulness = judge.faithfulness(row["answer"], sources)
            row["faithfulness"] = {
                "score": faithfulness.score,
                "claims": [vars(c) for c in faithfulness.claims],
            }
        except JudgeError as exc:
            row["errors"].append(f"faithfulness: {exc}")


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Aggregate per-question rows. Each metric averages only the questions it applies to."""
    scored = [r for r in rows if "retrieval" in r]
    summary: dict[str, Any] = {
        "questions": len(rows),
        "errors": sum(bool(r["errors"]) for r in rows),
        "retrieval": {m: _mean(r["retrieval"][m] for r in scored) for m in RETRIEVAL_METRICS},
        "retrieval_by_type": {
            question_type: {
                m: _mean(r["retrieval"][m] for r in scored if r["type"] == question_type)
                for m in RETRIEVAL_METRICS
            }
            for question_type in sorted({r["type"] for r in scored})
        },
        "latency_ms": _latency(rows),
    }
    if mode == "full":
        answered = [r for r in rows if "abstained" in r]
        summary["answers"] = {
            "correctness": _mean(r["correctness"]["score"] for r in rows if "correctness" in r),
            "faithfulness": _mean(
                r["faithfulness"]["score"]
                for r in rows
                if r.get("faithfulness") and r["faithfulness"]["score"] is not None
            ),
            "citation_precision": _mean(
                r["citation_precision"] for r in rows if r.get("citation_precision") is not None
            ),
            "abstention": vars(
                abstention_scores([(r["answerable"], r["abstained"]) for r in answered])
            ),
            "prompt_tokens": _mean(r["usage"]["prompt_tokens"] for r in answered),
            "completion_tokens": _mean(r["usage"]["completion_tokens"] for r in answered),
        }
    return summary


def _mean(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _latency(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stages = sorted({stage for r in rows for stage in r.get("timings_ms", {})})
    out = {}
    for stage in stages:
        values = sorted(r["timings_ms"][stage] for r in rows if stage in r.get("timings_ms", {}))
        out[stage] = {
            "p50": round(statistics.median(values), 1),
            "p95": round(values[min(len(values) - 1, round(0.95 * (len(values) - 1)))], 1),
        }
    return out


def provenance(
    benchmark: str, preset: str, config, split: str, mode: str, k: int, index_version: str
) -> dict:
    settings = get_settings()
    info = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "benchmark": benchmark,
        "preset": preset,
        "retrieval": config.model_dump(),
        "split": split,
        "mode": mode,
        "k": k,
        "index_version": index_version,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if config.rerank else None,
        "benchmark_sha256": hashlib.sha256(
            BENCHMARKS[benchmark]["questions"].read_bytes()
        ).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    }
    if mode == "full":
        info |= {
            "llm_model": settings.llm_model,
            "prompt_version": settings.answer_prompt_version,
            "judge_model": settings.judge_model,
            "judge_prompts": [FAITHFULNESS_PROMPT, CORRECTNESS_PROMPT],
        }
    return info


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(benchmark: str, preset: str, split: str, mode: str, k: int, limit: int | None) -> Path:
    """Retrieve (and answer) every question. Full mode then hands judging to a fresh process."""
    config = load_preset(preset)
    spec = BENCHMARKS[benchmark]
    questions = load_questions(spec["questions"], split=split)[:limit]
    rows: list[dict[str, Any]] = []
    sources: list[list[str]] = []
    started = time.perf_counter()
    with get_session_factory()() as session:
        documents = corpus_documents(session, spec["collection"])
        index_version = index_versions_repo.get_active(session).name
        for n, question in enumerate(questions, start=1):
            row, passages = evaluate_question(
                question, session=session, config=config, documents=documents, mode=mode, k=k
            )
            rows.append(row)
            sources.append(passages)
            session.rollback()  # end the read transaction (and its SET LOCALs) between questions
            status = "error" if row["errors"] else ("abstained" if row.get("abstained") else "ok")
            print(
                f"  [{n:2}/{len(questions)}] {question.id} {question.type:12} {status}", flush=True
            )

    info = provenance(benchmark, preset, config, split, mode, k, index_version)
    info["duration_s"] = round(time.perf_counter() - started, 1)
    if mode == "retrieval":
        return write_results(info, rows)

    # Save the answers before judging: if judging is interrupted it can be resumed with
    # --judge-from, without generating the answers again.
    checkpoint = WORK_DIR / f"{_stamp()}_{_name(info)}.json"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"provenance": info, "rows": rows, "sources": sources}))
    print(f"Saved answers to {checkpoint}", flush=True)
    # Replace this process with a fresh judge process. This one holds the embedder and reranker
    # in GPU memory; next to a 12B judge model that exhausts 16 GB and the machine starts
    # swapping. (A child process is not enough: the waiting parent would keep that memory.)
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, "-m", "evals.run", "--judge-from", str(checkpoint)])


def judge_checkpoint(checkpoint: Path) -> Path:
    """Grade the answers saved by a full run, then write the results file."""
    saved = json.loads(checkpoint.read_text())
    info, rows, sources = saved["provenance"], saved["rows"], saved["sources"]
    questions_path = BENCHMARKS[info.get("benchmark", "docs")]["questions"]
    by_id = {q.id: q for q in load_questions(questions_path)}
    judge = get_judge()
    started = time.perf_counter()
    print(f"Judging {len(rows)} answers with {judge.model}...", flush=True)
    for n, (row, passages) in enumerate(zip(rows, sources, strict=True), start=1):
        if "correctness" in row or "faithfulness" in row:
            continue  # already judged (resumed run)
        judge_answer(row, by_id[row["id"]], passages, judge)
        print(f"  [{n:2}/{len(rows)}] judged {row['id']}", flush=True)
    info["judge_duration_s"] = round(time.perf_counter() - started, 1)
    return write_results(info, rows)


def write_results(info: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    result = {"provenance": info, "summary": summarize(rows, info["mode"]), "questions": rows}
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{_stamp()}_{_name(info)}_{info['mode']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return path


def _name(info: dict[str, Any]) -> str:
    """e.g. "hybrid_rerank_dev" for the documentation benchmark, "papers_hybrid_rerank_dev"."""
    prefix = "" if info.get("benchmark", "docs") == "docs" else f"{info['benchmark']}_"
    return f"{prefix}{info['preset']}_{info['split']}"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", help="configs/retrieval/<name>.toml")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="docs")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke runs)")
    parser.add_argument("--final", action="store_true", help="allow the held-out test split")
    parser.add_argument("--judge-from", type=Path, help="judge the answers saved by a full run")
    args = parser.parse_args()

    if args.judge_from:
        path = judge_checkpoint(args.judge_from)
        print(f"\nWrote {path}")
        return
    if not args.preset:
        parser.error("--preset is required")
    if args.split == "test" and not args.final:
        sys.exit(
            "The test split is held out for the final report (Part 8). Tune on dev; "
            "pass --final only for the one final run."
        )
    path = run(args.benchmark, args.preset, args.split, args.mode, args.k, args.limit)
    print(f"\nWrote {path}")
    print(json.dumps(json.loads(path.read_text())["summary"], indent=2))


if __name__ == "__main__":
    main()
