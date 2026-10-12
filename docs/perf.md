# Performance

Where `/chat` spends its time, how much memory the stack needs, and what changed because of the
measurements. Part 6, step 7 of the plan. Results files are in `evals/results/*_load_*.json`.

## Setup

| | |
|---|---|
| Machine | MacBook Air M4 (Mac16,12), 10 cores, 16 GB unified memory, **fanless**, on battery |
| OS | macOS 15.7 |
| API | one uvicorn process, sync endpoints in its thread pool; models on MPS (Apple GPU) |
| LLM | `qwen3-vl:8b-instruct` in Ollama 0.35, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=1` |
| Retrieval | `hybrid_rerank`: bge-small-en-v1.5, dense + keyword, RRF, MiniLM cross-encoder, top 5 of 20 |
| Data | 65 documents (FastAPI tutorial, 4 Python HOWTOs, 8 arXiv papers), 1 user |
| Workload | the 49 dev questions of both benchmarks, no document filter, replayed by `scripts/loadtest.py` at concurrency 1 and 4; rate limits raised for the test |
| Memory | physical footprint from `top` (RSS undercounts on macOS; see below) |

Run with:

```bash
RATE_LIMIT_CHAT_PER_MINUTE=1000 RATE_LIMIT_RETRIEVE_PER_MINUTE=1000 make dev
caffeinate -i uv run python -m scripts.loadtest --endpoint chat --concurrency 1
```

## Results

**Retrieval only** (`/retrieve`, no LLM):

| Concurrency | Throughput | Client p50 / p95 | embed p50 | rerank p50 | dense + keyword p50 |
|---|---|---|---|---|---|
| 1 | 304 / min | 191 / 218 ms | 13 ms | 148 ms | 20 ms |
| 4 | 383 / min | 619 / 664 ms | 182 ms | 325 ms | 15 ms |

At concurrency 4 the embed and rerank stages include waiting for the inference lock (below), so
four times the load buys 26% more throughput.

**Answers** (`/chat`), in the order the runs happened:

| Run | Concurrency | Throughput | Errors | Client p50 / p95 | generate p50 / p95 |
|---|---|---|---|---|---|
| A, first | 1 | **6.75 / min** | 0 | 4.7 / 22.8 s | 4.4 / 22.3 s |
| B, LLM queued in Ollama | 4 | 3.45 / min | 1 (`llm_unavailable`), 20 SDK retries | 57.8 / 112.1 s | 56.9 / 110.9 s |
| C, LLM queued in the API | 4 | 3.54 / min | 0, 0 retries | 70.5 / 92.1 s | 69.3 / 91.1 s* |
| D, control, after C | 1 | **3.39 / min** | 0 | 18.4 / 26.9 s | 18.1 / 26.3 s |

\* In run C, `generate` includes waiting for a free model slot.

Per query, in every run: **~1,800 prompt tokens and ~50 completion tokens** (p50 1,828 / 49),
11 of 49 questions abstained (2 of them before the model was called), cost **$0.00** (local
model; `configs/pricing.toml`).

**Where the time goes** (run A, p50): generate 4,356 ms of 4,654 ms total (**~94%**), rerank
164 ms, embed 112 ms, both searches 25 ms. At p95 generation is 98%.

## Memory

| Process | Physical footprint | Notes |
|---|---|---|
| Ollama (qwen3-vl 8B, 8k context) | **10–11.3 GB** | 6.4 GB of weights plus KV cache and buffers |
| API (embedder + reranker on MPS) | **1.5–1.6 GB** | 444–581 MB right after startup, before memory pressure |
| Postgres | 138 MiB | `docker stats` |
| Redis | 22 MiB | `docker stats` |
| Worker | not measured under load | loads the same embedder as the API |

The whole stack needs about 13 GB on a 16 GB machine; swap use rose from 11 to 14 GB during the
tests (VS Code and other apps were also running). `ps` reported 112 MB for the API and 7.2 GB
for Ollama at the same moment: on macOS RSS leaves out compressed and swapped pages and Metal
allocations, so the load test samples the footprint instead.

**Startup and first requests:** loading and warming the models takes ~15 s at startup
(embedder 9.2 s, reranker 5.7 s). Without the warm-up, the first embedding takes 266 ms instead
of 12 ms. With the LLM unloaded (Ollama unloads it after 5 idle minutes), the first `/chat` took
16.3 s and the next 3.9 s.

## What the measurements changed

1. **A crash.** The first concurrency-4 run segfaulted the API after three requests. FastAPI
   runs sync endpoints in a thread pool, and PyTorch's MPS backend is not thread-safe: a
   standalone repro with four threads crashed twice and deadlocked once. A process-wide lock
   around every embed and rerank call (`INFERENCE_LOCK`) fixed it; a real-model test runs six
   threads at once.
2. **Retries that only added work.** In run B, requests queued inside Ollama ran into the 60 s
   timeout and the SDK retried them while Ollama kept generating the abandoned answers. The API
   now holds at most `LLM_MAX_CONCURRENCY` calls in flight (1, like Ollama) and queues the rest
   itself; a call that can't get a slot within 120 s gets `503 llm_busy` with `Retry-After`.
   Run C: no errors, no retries.
3. **…but not the cause of the slowdown.** Run C was no faster than B, so a control run at
   concurrency 1 (D) followed: 3.39 answers a minute, half of run A, with the same code and
   settings. Generation itself had slowed from 4.4 s to 18.1 s (p50) after about two hours of
   sustained load on a fanless laptop on battery under memory pressure. Concurrency doesn't
   change throughput here (Ollama generates one answer at a time); it only adds queueing.
4. **Measurement hygiene.** The Mac went to idle sleep twice during runs (once between two
   `caffeinate` calls), which made two runs useless; runs now go inside one `caffeinate -i`.
   The sampler first reported RSS (see Memory).

## What this means

- **Latency is the LLM's.** Retrieval is ~0.3 s; everything else is generation, and on this
  machine generation varies 4× with thermal and memory state. A dedicated GPU server would be
  steadier; this laptop's numbers are an upper bound on what the code costs, not a capacity plan.
- **Capacity is one answer at a time per LLM.** More API workers or threads don't add
  throughput; a second model replica (or `OLLAMA_NUM_PARALLEL` > 1 with the memory for it, or a
  hosted model) does. `LLM_MAX_CONCURRENCY` must follow that number.
- **Part 7 sizing:** with a local 8B model, at least 16 GB with a GPU, realistically 24 GB;
  without it (hosted LLM), API + worker + Postgres + Redis fit in ~2–3 GB.
- **Caching** wouldn't change this picture: [ADR-0005](adr/0005-no-caching-yet.md).
