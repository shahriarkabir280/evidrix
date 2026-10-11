"""Measures how often the judge's correctness verdicts match hand-labeled cases.

Component 4E.

    uv run python -m evals.calibrate_judge --prompt judge_correctness_v1

The cases (evals/judge_calibration.jsonl) are written by hand, independent of the benchmark
answers, and cover each verdict plus the failure seen in practice: penalizing correct extra detail.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from evals import judge as judge_module
from evals.judge import get_judge

CASES_PATH = Path(__file__).parent / "judge_calibration.jsonl"


def calibrate(prompt: str) -> dict:
    judge_module.CORRECTNESS_PROMPT = prompt
    judge = get_judge()
    cases = [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        result = judge.correctness(case["question"], case["reference"], case["answer"])
        rows.append({**case, "verdict": result.verdict, "reason": result.reason})
    agreement = sum(r["verdict"] == r["expected"] for r in rows) / len(rows)
    confusion = Counter((r["expected"], r["verdict"]) for r in rows)
    return {"prompt": prompt, "judge_model": judge.model, "agreement": agreement, "rows": rows,
            "confusion": {f"{e}->{v}": n for (e, v), n in sorted(confusion.items())}}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=judge_module.CORRECTNESS_PROMPT)
    args = parser.parse_args()
    report = calibrate(args.prompt)
    print(f"{report['prompt']} on {report['judge_model']}: agreement {report['agreement']:.0%}")
    print("expected->verdict:", report["confusion"])
    for row in report["rows"]:
        if row["verdict"] != row["expected"]:
            print(f"  ✗ {row['case']}: expected {row['expected']}, got {row['verdict']}")
            print(f"      reason: {row['reason']}")


if __name__ == "__main__":
    main()
