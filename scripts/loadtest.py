"""Replays the dev questions against a running API and reports latency, throughput and memory.

Component 6D · Plan: Part 6, step 7.

    export EVIDRIX_API_KEY=evx_...   # a key for the user that owns the corpus (dev@evidrix.local)
    uv run python -m scripts.loadtest --endpoint chat --concurrency 1
    uv run python -m scripts.loadtest --endpoint retrieve --concurrency 4

The questions are the dev splits of both benchmarks (36 documentation + 13 paper questions),
asked without a document filter, as a user would. Latency is measured twice: by the client
(the whole HTTP round trip, queueing included) and by the server per stage (timings_ms in the
response). While the run lasts, the resident memory (RSS) of the API, the worker and Ollama is
sampled every second.

The server's rate limits are per user per minute, so start it with limits above the load:
RATE_LIMIT_CHAT_PER_MINUTE=1000 RATE_LIMIT_RETRIEVE_PER_MINUTE=1000.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.benchmark import PAPERS_QUESTIONS_PATH, QUESTIONS_PATH, load_questions

RESULTS_DIR = Path("evals/results")
PROCESS_GROUPS = {  # group name -> substring of the command line
    "api": "uvicorn app.main",
    "worker": "rq worker",
    "ollama": "ollama",
}


def dev_questions() -> list[str]:
    paths = (QUESTIONS_PATH, PAPERS_QUESTIONS_PATH)
    return [q.question for path in paths for q in load_questions(path, split="dev")]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest rank, as in evals/run.py, so the numbers are comparable."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(statistics.median(values), 1),
        "p95": round(percentile(values, 0.95), 1),
        "max": round(max(values), 1),
    }


class MemorySampler(threading.Thread):
    """Peak RSS per process group, from `ps` once a second."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak_mb: dict[str, float] = dict.fromkeys(PROCESS_GROUPS, 0.0)
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(1.0)

    def sample(self) -> None:
        output = subprocess.run(
            ["ps", "-axo", "rss=,command="], capture_output=True, text=True, check=False
        ).stdout
        totals = dict.fromkeys(PROCESS_GROUPS, 0.0)
        for line in output.splitlines():
            rss_kb, _, command = line.strip().partition(" ")
            for group, needle in PROCESS_GROUPS.items():
                if needle in command and "loadtest" not in command:
                    totals[group] += int(rss_kb) / 1024
        for group, mb in totals.items():
            self.peak_mb[group] = max(self.peak_mb[group], round(mb, 1))

    def stop(self) -> None:
        self._halt.set()


def docker_memory() -> dict[str, str]:
    output = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return dict(line.split("\t", 1) for line in output.splitlines() if "\t" in line)


def machine() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False
        ).stdout.strip()

    memsize = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_cores": os.cpu_count(),
        "memory_gb": round(int(memsize) / 2**30, 1) if memsize.isdigit() else None,
    }


def ask(client: httpx.Client, endpoint: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(f"/{endpoint}", json={"question": question})
    row: dict[str, Any] = {
        "status": response.status_code,
        "client_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if response.status_code == 200:
        body = response.json()
        row["timings_ms"] = body["timings_ms"]
        if endpoint == "chat":
            row["prompt_tokens"] = body["usage"]["prompt_tokens"]
            row["completion_tokens"] = body["usage"]["completion_tokens"]
            row["cost_usd"] = body["usage"]["cost_usd"]
            row["abstained"] = body["abstained"]
            row["config"] = body["config"]
    else:
        row["error"] = response.json().get("error", {}).get("code")
    return row


def run(base_url: str, api_key: str, endpoint: str, concurrency: int) -> dict[str, Any]:
    questions = dev_questions()
    sampler = MemorySampler()
    sampler.start()
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=600) as client:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(lambda q: ask(client, endpoint, q), questions))
        wall_s = time.perf_counter() - started
    sampler.stop()
    sampler.join()

    ok = [r for r in rows if r["status"] == 200]
    stages = sorted({stage for r in ok for stage in r["timings_ms"]})
    summary: dict[str, Any] = {
        "requests": len(rows),
        "ok": len(ok),
        "errors": {str(r["status"]): r.get("error") for r in rows if r["status"] != 200},
        "wall_s": round(wall_s, 1),
        "throughput_per_min": round(len(ok) / wall_s * 60, 2),
        "client_ms": summarize_ms([r["client_ms"] for r in ok]),
        "server_stage_ms": {
            stage: summarize_ms([r["timings_ms"][stage] for r in ok if stage in r["timings_ms"]])
            for stage in stages
        },
        "peak_rss_mb": sampler.peak_mb,
        "docker_memory": docker_memory(),
    }
    if endpoint == "chat":
        summary["tokens_per_query"] = {
            "prompt_p50": statistics.median(r["prompt_tokens"] for r in ok),
            "completion_p50": statistics.median(r["completion_tokens"] for r in ok),
            "prompt_mean": round(statistics.mean(r["prompt_tokens"] for r in ok), 1),
            "completion_mean": round(statistics.mean(r["completion_tokens"] for r in ok), 1),
        }
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        summary["cost_per_query_usd"] = round(statistics.mean(costs), 6) if costs else None
        summary["abstained"] = sum(r["abstained"] for r in ok)
    config = next((r["config"] for r in ok if "config" in r), None)
    return {
        "provenance": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "endpoint": endpoint,
            "concurrency": concurrency,
            "questions": "dev splits: documentation + papers",
            "machine": machine(),
            "config": config,
        },
        "summary": summary,
        "requests": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", choices=["chat", "retrieve"], default="chat")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    api_key = os.environ.get("EVIDRIX_API_KEY")
    if not api_key:
        raise SystemExit("Set EVIDRIX_API_KEY (uv run python -m app.cli create-key --email ...).")

    result = run(args.url, api_key, args.endpoint, args.concurrency)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"{stamp}_load_{args.endpoint}_c{args.concurrency}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
