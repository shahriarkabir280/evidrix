"""Lists the failed questions of a results file with what is needed to classify each failure.

Component 8B.

    uv run python -m evals.failures evals/results/<run>_full.json [--text]

A question fails when it is answerable and the answer is not fully correct, unfaithful, or an
abstention, or when it is unanswerable and the system answered. For each one this prints the
labeled evidence, whether retrieval found it, the top retrieved sources, the answer, and the
judge's reasons; --text adds the retrieved passages. The classification itself (vocabulary
mismatch, distractor ranked first, unsupported claim, ...) is done by reading this output.
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from evals.benchmark import PAPERS_QUESTIONS_PATH, QUESTIONS_PATH, load_questions

QUESTION_FILES = {"docs": QUESTIONS_PATH, "papers": PAPERS_QUESTIONS_PATH}


def failed(row: dict[str, Any]) -> list[str]:
    reasons = []
    if not row["answerable"]:
        if not row.get("abstained"):
            reasons.append("missed abstention")
        return reasons
    if row.get("abstained"):
        reasons.append(f"false abstention ({row.get('abstain_reason')})")
    elif (row.get("correctness") or {}).get("score", 1.0) < 1.0:
        reasons.append(f"correctness {row['correctness']['verdict']}")
    faithfulness = (row.get("faithfulness") or {}).get("score")
    if faithfulness is not None and faithfulness < 1.0:
        reasons.append(f"faithfulness {faithfulness:.2f}")
    if row.get("errors"):
        reasons.append(f"error: {row['errors']}")
    return reasons


def show(path: Path, with_text: bool) -> None:
    result = json.loads(path.read_text())
    benchmark = result["provenance"].get("benchmark", "docs")
    questions = {q.id: q for q in load_questions(QUESTION_FILES[benchmark])}
    rows = [(row, failed(row)) for row in result["questions"]]
    failures = [(row, reasons) for row, reasons in rows if reasons]
    print(f"{path.name}: {len(failures)} of {len(rows)} questions failed\n")
    for row, reasons in failures:
        q = questions[row["id"]]
        print(f"## {q.id} ({q.type}) — {'; '.join(reasons)}")
        print(f"Q: {q.question}")
        if q.expected_answer:
            print(f"Expected: {q.expected_answer}")
        for evidence in q.evidence:
            print(f"Evidence [{evidence.source}]: {evidence.quote[:200]!r}")
        if "retrieval" in row:
            r = row["retrieval"]
            print(f"Retrieval: recall {r['recall']:.2f}, mrr {r['mrr']:.2f}")
        for hit in row.get("retrieved", [])[:8]:
            mark = "*" if hit.get("relevant") else " "
            print(f"  {mark}{hit['rank']}. [{hit['score']:.3f}] {hit['source']}")
            if with_text and hit.get("text"):
                print(textwrap.indent(textwrap.fill(hit["text"][:400], 96), "       "))
        print(f"Answer: {row.get('answer', '')[:500]!r}")
        for name in ("correctness", "faithfulness"):
            verdict = row.get(name) or {}
            if verdict.get("reason"):
                print(f"{name}: {verdict['reason'][:400]}")
            for claim in verdict.get("claims", []):
                if not claim.get("supported", True):
                    print(f"  unsupported claim: {claim.get('claim', '')[:200]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--text", action="store_true", help="also print the retrieved passages")
    args = parser.parse_args()
    for path in args.results:
        show(path, args.text)


if __name__ == "__main__":
    main()
