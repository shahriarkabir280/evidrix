"""Tables for the Part 8 ablations: E1-E5 on the test split, pooled over both benchmarks.

Component 8A.

    uv run python -m evals.ablations            # after scripts/run_ablations.sh
    uv run python -m evals.ablations --split dev  # the same tables for dev runs

Each experiment is found by its results files' provenance (benchmark, preset, split, index
version); the newest file wins. The documentation (24 test questions) and paper (11) benchmarks
are pooled per question, since either alone gives very wide intervals; per-benchmark rows follow.
Differences are paired against E1 (dense), over the questions both runs share.
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.stats import bootstrap_ci, paired_bootstrap_diff

RESULTS_DIR = Path("evals/results")
DEFAULT_INDEX = "bge-small_c400_o64"
SMALL_CHUNKS_INDEX = "bge-small_c256_o42"


@dataclass(frozen=True)
class Experiment:
    id: str
    label: str
    preset: str
    index_version: str = DEFAULT_INDEX


EXPERIMENTS = [
    Experiment("E1", "dense, top 5", "dense"),
    Experiment("E2", "hybrid (RRF), top 5", "hybrid"),
    Experiment("E3", "hybrid + rerank, top 5", "hybrid_rerank"),
    Experiment("E4", "E3 with 256-token chunks", "hybrid_rerank", SMALL_CHUNKS_INDEX),
    Experiment("E5a", "E3 with top 3", "hybrid_rerank_k3"),
    Experiment("E5b", "E3 with top 8", "hybrid_rerank_k8"),
]
BENCHMARKS = ("docs", "papers")
RETRIEVAL = ("recall", "mrr", "ndcg")
ANSWERS = ("correctness", "faithfulness")


def find_results(split: str) -> dict[tuple[str, str], dict[str, Any]]:
    """(experiment id, benchmark) -> the newest matching full-mode results."""
    found: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path in sorted(RESULTS_DIR.glob("*_full.json")):
        result = json.loads(path.read_text())
        p = result["provenance"]
        if p.get("split") != split or p.get("mode") != "full":
            continue
        for experiment in EXPERIMENTS:
            if (p["preset"], p["index_version"]) == (experiment.preset, experiment.index_version):
                key = (experiment.id, p.get("benchmark", "docs"))
                if key not in found or path.name > found[key][0]:
                    found[key] = (path.name, result)
    return {key: result for key, (_, result) in found.items()}


def per_question(result: dict[str, Any], metric: str, benchmark: str) -> dict[str, float]:
    values = {}
    for row in result["questions"]:
        key = f"{benchmark}:{row['id']}"
        if metric in RETRIEVAL and "retrieval" in row:
            values[key] = row["retrieval"][metric]
        elif metric == "correctness" and "correctness" in row:
            values[key] = row["correctness"]["score"]
        elif metric == "faithfulness" and (row.get("faithfulness") or {}).get("score") is not None:
            values[key] = row["faithfulness"]["score"]
    return values


def pooled(results, experiment_id: str, metric: str, benchmarks=BENCHMARKS) -> dict[str, float]:
    values: dict[str, float] = {}
    for benchmark in benchmarks:
        if (experiment_id, benchmark) in results:
            values |= per_question(results[(experiment_id, benchmark)], metric, benchmark)
    return values


def rows_of(results, experiment_id: str, benchmarks=BENCHMARKS) -> list[dict[str, Any]]:
    return [
        row
        for benchmark in benchmarks
        if (experiment_id, benchmark) in results
        for row in results[(experiment_id, benchmark)]["questions"]
    ]


def cell(results, experiment_id: str, metric: str, benchmarks=BENCHMARKS) -> str:
    values = pooled(results, experiment_id, metric, benchmarks)
    if not values:
        return "—"
    ci = bootstrap_ci(list(values.values()))
    text = f"{ci.mean:.3f} [{ci.low:.2f}, {ci.high:.2f}]"
    if experiment_id != "E1":
        base = pooled(results, "E1", metric, benchmarks)
        shared = sorted(set(base) & set(values))
        if shared:
            diff = paired_bootstrap_diff([base[q] for q in shared], [values[q] for q in shared])
            mark = "**" if diff.excludes_zero() else ""
            text += f"<br>{mark}Δ {diff.mean:+.3f} [{diff.low:+.2f}, {diff.high:+.2f}]{mark}"
    return text


def abstention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    caught = sum(bool(r.get("abstained")) for r in unanswerable)
    false = sum(bool(r.get("abstained")) for r in answerable)
    return {"caught": f"{caught}/{len(unanswerable)}", "false": f"{false}/{len(answerable)}"}


def operations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    called = [r for r in rows if (r.get("usage") or {}).get("prompt_tokens")]
    generate = [r["timings_ms"]["generate"] for r in rows if "generate" in r.get("timings_ms", {})]
    precision = [r["citation_precision"] for r in rows if r.get("citation_precision") is not None]
    return {
        "prompt_tokens": round(statistics.mean(r["usage"]["prompt_tokens"] for r in called))
        if called
        else None,
        "generate_p50_s": round(statistics.median(generate) / 1000, 1) if generate else None,
        "citation_precision": round(statistics.mean(precision), 3) if precision else None,
        "errors": sum(bool(r.get("errors")) for r in rows),
    }


def report(split: str) -> str:
    results = find_results(split)
    present = [e for e in EXPERIMENTS if any((e.id, b) in results for b in BENCHMARKS)]
    missing = [f"{e.id}/{b}" for e in EXPERIMENTS for b in BENCHMARKS if (e.id, b) not in results]
    n = {b: len(results[("E1", b)]["questions"]) for b in BENCHMARKS if ("E1", b) in results}
    lines = [
        f"Split: {split}. Questions per benchmark: {n}. Mean [95% bootstrap CI]; Δ is the paired "
        "difference against E1, **bold** when its interval excludes 0.",
    ]
    if missing:
        lines.append(f"Missing runs: {', '.join(missing)}.")

    def table(title: str, metrics, benchmarks=BENCHMARKS) -> None:
        lines.extend(["", f"### {title}", "", "| | Config | " + " | ".join(metrics) + " |"])
        lines.append("|---|---|" + "---|" * len(metrics))
        for e in present:
            cells = [cell(results, e.id, m, benchmarks) for m in metrics]
            lines.append(f"| {e.id} | {e.label} | " + " | ".join(cells) + " |")

    table("Retrieval, pooled (answerable questions; Recall/MRR/nDCG at 5)", RETRIEVAL)
    table("Answers, pooled (judge: gemma3:12b)", ANSWERS)
    lines.extend(
        [
            "",
            "### Abstention and operations, pooled",
            "",
            "| | Config | Unanswerable caught | False abstentions | Citation precision "
            "| Prompt tokens / query | generate p50 | Errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for e in present:
        rows = rows_of(results, e.id)
        a, o = abstention(rows), operations(rows)
        lines.append(
            f"| {e.id} | {e.label} | {a['caught']} | {a['false']} | {o['citation_precision']} "
            f"| {o['prompt_tokens']} | {o['generate_p50_s']} s | {o['errors']} |"
        )
    for benchmark in BENCHMARKS:
        table(f"Per benchmark: {benchmark}", ("recall", "mrr", "correctness"), (benchmark,))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    print(report(args.split))


if __name__ == "__main__":
    main()
