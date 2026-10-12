# ADR-0005: No query-embedding or answer cache, for now

**Status:** accepted (2026-10-12)

## Context
The plan (Part 6, step 8) named two candidates and asked for one only if a load test showed a
real benefit: an in-process LRU for query embeddings, and a Redis cache of whole answers. The
load test (`scripts/loadtest.py`, 49 dev questions, concurrency 1, details in
[docs/perf.md](../perf.md)) measured where `/chat` spends its time:

| Stage | p50 | p95 | Share of p50 total |
|---|---|---|---|
| generate (LLM) | 4,356 ms | 22,298 ms | ~94% |
| rerank | 164 ms | 210 ms | ~3.5% |
| embed | 112 ms | 250 ms | ~2.4% |
| dense + keyword search | 25 ms | 48 ms | ~0.5% |
| total | 4,654 ms | 22,704 ms | |

## Decision
Neither cache is built.

- **Query-embedding cache:** a hit saves the embed stage, at most ~2% of a typical `/chat`
  (and 13 ms of a warm, uncontended `/retrieve`). It only hits when the same question is asked
  again, and the answer would still take seconds. Not worth a component.
- **Answer cache:** a hit would save ~95% of the time and the whole LLM cost, so the question is
  only how often it would hit, and whether a hit could be wrong.
  - *Hit rate:* unknown. Evidrix answers questions about a user's own documents; people rarely
    ask the same thing twice, and there is no production traffic to measure yet.
  - *Correctness:* a key of (user, normalized question, document filter, retrieval config,
    index version, prompt version, model) is still not enough. Without a document filter, the
    answer depends on *which documents the user has*: uploading a better source or deleting a
    wrong one must change it, so the key would also need a version of the user's document set.
    Missing any part of the key serves an answer that was right once and is wrong now, which is
    worse than slow for an evidence-grounded system.

## Consequences
- `/chat` latency is the LLM's latency. The levers that matter are the model, its hardware and
  the prompt length (~1,800 prompt tokens per query, so `top_k` is also a latency setting).
- Every chat is stored (`chat_queries`), so the hit rate can be measured instead of guessed.
  Revisit if repeats become common, e.g. more than 10% of a user's questions repeated within a
  day:

  ```sql
  SELECT user_id,
         count(*) FILTER (WHERE repeats > 1) * 1.0 / count(*) AS repeated_share
  FROM (SELECT user_id, lower(btrim(question)) AS q, count(*) AS repeats
        FROM chat_queries
        WHERE created_at > now() - interval '1 day' AND error_code IS NULL
        GROUP BY user_id, q) per_question
  GROUP BY user_id;
  ```

  On the development database this reports 72%, because the load tests asked the same 49
  questions several times. That is the point: replayed benchmarks repeat by construction, so the
  hit rate has to come from real use.
- If an answer cache is ever added, its key must include a document-set version (for example
  the latest `updated_at` of the user's completed documents), and the cache must be invalidated
  when the index version or prompt version changes.
