"""Chooses the pre-generation abstention threshold for a preset on the dev split.

Component 4F.

    uv run python -m evals.calibrate_abstention \\
        evals/results/<preset>_dev_retrieval.json evals/results/<preset>_dev_full.json

Below the threshold, the top retrieved chunk is considered too weak and the system abstains
without calling the model. The gate can only add abstentions, so the rule is: catch as many
unanswerable questions as possible while blocking no answerable one, and put the threshold in the
middle of the gap that separates them (the largest margin to both sides).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    threshold: float | None
    caught_unanswerable: list[str]  # abstained by the gate, correctly
    blocked_answerable: list[str]  # abstained by the gate, wrongly
    margin: float | None  # distance from the threshold to the nearest score on either side
    separated: bool  # every unanswerable question scores below every answerable one


def calibrate(top_scores: dict[str, float], answerable: dict[str, bool]) -> Calibration:
    """`top_scores` maps question id to the score of its best retrieved chunk."""
    lowest_answerable = min(s for q, s in top_scores.items() if answerable[q])
    below = sorted(
        (s, q) for q, s in top_scores.items() if not answerable[q] and s < lowest_answerable
    )
    unanswerable_scores = [s for q, s in top_scores.items() if not answerable[q]]
    separated = max(unanswerable_scores) < lowest_answerable
    if not below:
        return Calibration(None, [], [], None, separated)
    highest_caught = below[-1][0]
    threshold = round((highest_caught + lowest_answerable) / 2, 2)
    return Calibration(
        threshold=threshold,
        caught_unanswerable=[q for _, q in below],
        blocked_answerable=[],
        margin=round((lowest_answerable - highest_caught) / 2, 2),
        separated=separated,
    )


def top_scores_from(retrieval_results: dict) -> dict[str, float]:
    return {
        row["id"]: row["retrieved"][0]["score"] if row["retrieved"] else float("-inf")
        for row in retrieval_results["questions"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("retrieval_results", type=Path, help="retrieval-mode run (has scores)")
    parser.add_argument("full_results", type=Path, help="full run of the same preset and split")
    args = parser.parse_args()

    retrieval = json.loads(args.retrieval_results.read_text())
    full = {row["id"]: row for row in json.loads(args.full_results.read_text())["questions"]}
    if retrieval["provenance"]["split"] != "dev":
        raise SystemExit("Calibrate on the dev split only.")
    scores = top_scores_from(retrieval)
    answerable = {row["id"]: row["answerable"] for row in retrieval["questions"]}
    result = calibrate(scores, answerable)

    preset = retrieval["provenance"]["preset"]
    print(f"{preset}: unanswerable and answerable scores separated: {result.separated}")
    if result.threshold is None:
        print("No threshold blocks an unanswerable question without blocking an answerable one.")
        return
    model_abstained = [q for q in result.caught_unanswerable if full[q].get("abstained")]
    print(f"threshold {result.threshold} (margin {result.margin} to the nearest score)")
    total_unanswerable = sum(not a for a in answerable.values())
    caught = result.caught_unanswerable
    print(f"  catches {len(caught)} of {total_unanswerable} unanswerable questions: {caught}")
    print(f"  the model already abstained on {len(model_abstained)} of them; the gate saves "
          f"{len(result.caught_unanswerable)} of {len(scores)} LLM calls")  # fmt: skip
    print("  blocks 0 answerable questions")


if __name__ == "__main__":
    main()
