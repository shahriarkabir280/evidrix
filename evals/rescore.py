"""Re-scores the answers that the corrected citation parser reads differently.

Component 8B. A one-off correction, kept for reproducibility.

    uv run python -m evals.rescore evals/results/<run>_full.json evals/results/work/<run>.json

The Part 8 error analysis found that `x[1]` in a code answer was taken for a citation marker:
it was counted as a citation, and the judge, which strips markers, graded `lambda x: x`. For
every row whose stored answer yields other citations under the fixed parser (CITATION_RUN),
this recomputes the citations and citation precision and asks the judge again, with the
passages saved in the run's checkpoint. Everything else in the file is unchanged, and the
provenance records which rows were re-scored, why, and at which commit.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app.rag.generation import CITATION_RUN
from evals.benchmark import PAPERS_QUESTIONS_PATH, QUESTIONS_PATH, load_questions
from evals.judge import get_judge
from evals.run import _git, judge_answer, summarize

QUESTION_FILES = {"docs": QUESTIONS_PATH, "papers": PAPERS_QUESTIONS_PATH}
REASON = "code indexing such as x[1] was read as a citation marker (fixed in b3de835)"
# Brackets glued to a word or ')', as in x[1] or f(x)[0]: the old parser read them as
# citations (and the judge stripped them); the fixed one leaves them in the text.
GLUED_INDEX = re.compile(r"[\w)]\[\d+")


def citations_in(answer: str) -> list[int]:
    return list(
        dict.fromkeys(
            int(n)
            for run in CITATION_RUN.finditer(answer)
            for group in run.group().strip("[]").split("][")
            for n in group.split(",")
        )
    )


def rescore(results_path: Path, checkpoint_path: Path) -> list[str]:
    result = json.loads(results_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint["provenance"]["created_at"] != result["provenance"]["created_at"]:
        raise SystemExit("The checkpoint belongs to another run (provenance differs).")
    benchmark = result["provenance"].get("benchmark", "docs")
    questions = {q.id: q for q in load_questions(QUESTION_FILES[benchmark])}
    sources = dict(zip((r["id"] for r in checkpoint["rows"]), checkpoint["sources"], strict=True))

    judge = None
    rescored = []
    done = {i for entry in result["provenance"].get("rescored", []) for i in entry["ids"]}
    for row in result["questions"]:
        if not row.get("answer") or row.get("abstained") or row["id"] in done:
            continue
        citations = citations_in(row["answer"])
        if citations == row.get("citations") and not GLUED_INDEX.search(row["answer"]):
            continue
        row["citations"] = citations
        cited = [row["retrieved"][n - 1] for n in citations if 0 < n <= len(row["retrieved"])]
        row["citation_precision"] = (
            sum(bool(hit["relevant"]) for hit in cited) / len(cited) if cited else None
        )
        judge = judge or get_judge()
        row["errors"] = []
        judge_answer(row, questions[row["id"]], sources[row["id"]], judge)
        rescored.append(row["id"])

    if rescored:
        result["summary"] = summarize(result["questions"], result["provenance"]["mode"])
        result["provenance"].setdefault("rescored", []).append(
            {
                "ids": rescored,
                "reason": REASON,
                "git_commit": _git("rev-parse", "HEAD"),
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(f"{args.results.name}: re-scored {rescore(args.results, args.checkpoint) or 'nothing'}")


if __name__ == "__main__":
    main()
