"""Turns results files into Markdown tables with bootstrap CIs.

Component 4E.

    uv run python -m evals.report evals/results/A.json evals/results/B.json [...]

The first file is the baseline. Each metric is shown with a 95% bootstrap interval, and every
other file gets a paired-bootstrap difference against the baseline; a difference whose interval
excludes zero is marked with *.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from evals.stats import bootstrap_ci, paired_bootstrap_diff

RETRIEVAL = ("recall", "mrr", "ndcg", "all_evidence")
ANSWERS = ("correctness", "faithfulness")


def per_question(result: dict[str, Any], metric: str) -> dict[str, float]:
    """question id -> value, only for questions the metric applies to."""
    values = {}
    for row in result["questions"]:
        if metric in RETRIEVAL and "retrieval" in row:
            values[row["id"]] = row["retrieval"][metric]
        elif metric == "correctness" and "correctness" in row:
            values[row["id"]] = row["correctness"]["score"]
        elif metric == "faithfulness" and (row.get("faithfulness") or {}).get("score") is not None:
            values[row["id"]] = row["faithfulness"]["score"]
    return values


def label(result: dict[str, Any]) -> str:
    p = result["provenance"]
    return f"{p['preset']} ({p['split']}, {p['mode']})"


def metric_table(results: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[str]:
    baseline = results[0]
    header = "| config | " + " | ".join(metrics) + " |"
    lines = [header, "|" + "---|" * (len(metrics) + 1)]
    for result in results:
        cells = []
        for metric in metrics:
            values = per_question(result, metric)
            if not values:
                cells.append("—")
                continue
            ci = bootstrap_ci(list(values.values()))
            cell = f"{ci.mean:.3f} [{ci.low:.2f}, {ci.high:.2f}]"
            if result is not baseline:
                base = per_question(baseline, metric)
                shared = sorted(set(base) & set(values))
                if shared:
                    diff = paired_bootstrap_diff(
                        [base[q] for q in shared], [values[q] for q in shared]
                    )
                    cell += f"<br>Δ {diff.mean:+.3f} [{diff.low:+.2f}, {diff.high:+.2f}]"
                    cell += "*" if diff.excludes_zero() else ""
            cells.append(cell)
        lines.append(f"| {label(result)} | " + " | ".join(cells) + " |")
    return lines


def report(paths: list[Path]) -> str:
    results = [json.loads(path.read_text()) for path in paths]
    ids = [{row["id"] for row in r["questions"]} for r in results]
    if any(i != ids[0] for i in ids):
        raise SystemExit("All results files must cover the same questions (same split and limit).")

    lines = [f"Baseline: {label(results[0])}. Values: mean [95% bootstrap CI].",
             "Δ = paired difference vs the baseline; * = interval excludes 0.", "",
             "### Retrieval (answerable questions)", ""]  # fmt: skip
    lines += metric_table(results, RETRIEVAL)
    full = [r for r in results if r["provenance"]["mode"] == "full"]
    if full:
        lines += ["", "### Answers (judge: " + full[0]["provenance"]["judge_model"] + ")", ""]
        lines += metric_table(full, ANSWERS)
        lines += ["", "| config | abstention P | R | F1 | false abstention | citation precision |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for r in full:
            a = r["summary"]["answers"]
            ab = a["abstention"]
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(
                f"| {label(r)} | {fmt(ab['precision'])} | {fmt(ab['recall'])} | {fmt(ab['f1'])} "
                f"| {fmt(ab['false_abstention_rate'])} | {fmt(a['citation_precision'])} |"
            )
    lines += [
        "",
        "### Latency p50 / p95 (ms)",
        "",
        "| config | stage | p50 | p95 |",
        "|---|---|---|---|",
    ]
    for r in results:
        for stage, values in r["summary"]["latency_ms"].items():
            lines.append(f"| {label(r)} | {stage} | {values['p50']} | {values['p95']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    print(report(args.results))


if __name__ == "__main__":
    main()
