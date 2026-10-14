# Deployment

How Evidrix runs outside the development setup, and why it runs that way. Part 7 of the plan.

## Two ways to run it

| | Container stack (`docker-compose.prod.yml`) | Public demo (Hugging Face Space) |
|---|---|---|
| What | API, worker, Postgres, Redis, release step | one API container + Neon Postgres + Upstash Redis |
| LLM | Ollama on the host (`/chat` works) | none: `CHAT_ENABLED=false`, `/chat` answers `503 chat_disabled` |
| Ingestion | RQ worker (`INGESTION_MODE=rq`) | in the API process (`INGESTION_MODE=inline`) |
| Uploaded files | Postgres (`BLOB_BACKEND=postgres`) | Postgres |
| Migrations | `migrate` service, before API and worker start | `Deploy` workflow, before the Space is updated |
| Start | `make prod-up` | `Deploy` workflow (below) |

The same image serves both, configured only through environment variables.

## Decisions

**No LLM in the public demo.** A free CPU instance can't run the 8B model that produced the
published answer numbers, and pointing the demo at a different hosted model would mean either
re-running the evaluation with it or showing answers the numbers don't describe. The demo is
retrieval-only; `/chat` runs in the local setup (`make dev` and the `/ui` page), against the
local model the report used.

**Memory.** Measured in the container (`linux/arm64`, CPU torch, 30 `/retrieve` calls):

| Setting | Peak |
|---|---|
| default (rerank 20 pairs per batch) | 687 MiB |
| `MALLOC_ARENA_MAX=2` | 689 MiB (no effect) |
| `RERANKER_BATCH_SIZE=4` | 518 MiB |
| `RERANKER_BATCH_SIZE=4`, `OMP_NUM_THREADS=1` | 488 MiB |
| single container, inline ingestion of a small document, batch 4 | 440 MiB idle, 478 MiB after |

The attention matrices of 20 pairs of 512 tokens alone take ~250 MB, which is why the batch size
matters. 512 MB instances (Render free and Starter) would leave ~25 MB of headroom and crash on
a larger upload. The demo therefore uses a free Hugging Face Docker Space (a CPU Space has
several GB of RAM; check the current limits) with free Neon Postgres (pgvector supported; the
database is ~21 MB plus ~8 MB of files) and free Upstash Redis. Check each provider's current
free-tier limits before relying on them; they change.

**Uploads in Postgres.** Container file systems on these hosts are ephemeral. At demo scale
(files up to 10 MB, tens of MB in total) a `blobs` table is simpler than an object store.

**Inline ingestion.** A free Space runs one container, so there is no separate worker. Jobs run
in one background thread in the API process: same attempts and retry delays, and unfinished jobs
are resubmitted at startup. That recovery is only safe with a single instance.

**Migrations as a release step.** Running `alembic upgrade head` at application startup would
let two replicas migrate at once, and a failing migration would leave a half-started app. In the
container stack a `migrate` service runs first and the API waits for it to succeed; for the
Space, the `Deploy` workflow migrates Neon and only then pushes the new code.

## Deploying the public demo

One-time setup:

1. **Neon:** create a project; copy the connection string and turn it into a SQLAlchemy URL:
   `postgresql+psycopg://USER:PASSWORD@HOST/DBNAME?sslmode=require`.
2. **Upstash:** create a Redis database; copy its `rediss://default:PASSWORD@HOST:6379` URL.
3. **Hugging Face:** create a Space (SDK: Docker, hardware: CPU basic). In its settings, add
   these secrets / variables:

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | the Neon URL from step 1 |
   | `REDIS_URL` | the Upstash URL from step 2 |
   | `BLOB_BACKEND` | `postgres` |
   | `INGESTION_MODE` | `inline` |
   | `CHAT_ENABLED` | `false` |
   | `RERANKER_BATCH_SIZE` | `4` |

4. **GitHub** (repository settings → Secrets and variables → Actions): secrets `DATABASE_URL`
   (the Neon URL), `HF_TOKEN` (a Hugging Face token with write access), and later
   `EVIDRIX_SMOKE_API_KEY`; variable `HF_SPACE` = `<hf-user>/<space-name>`.

Each deploy: **Actions → Deploy → Run workflow.** It migrates Neon, then pushes the app to the
Space, which builds the image and restarts.

Load the corpus and create keys once, from your machine, against Neon:

```bash
export DATABASE_URL='postgresql+psycopg://…?sslmode=require' BLOB_BACKEND=postgres
make corpus
uv run python -m app.cli ingest-dir evals/corpus/raw           # owned by dev@evidrix.local
uv run python -m app.cli create-key --email dev@evidrix.local --name demo
uv run python -m app.cli create-user --email smoke@evidrix.local   # key for the smoke test
```

Then add the smoke key as `EVIDRIX_SMOKE_API_KEY` and run **Actions → Smoke test** with the
Space URL (`https://<hf-user>-<space-name>.hf.space`), or locally:
`make smoke URL=https://… KEY=evx_…`.

Free tiers sleep: a Space that nobody opened for a while, and an idle Neon database, take time
to wake up, so the first request after a pause is slow.

## Verified locally

- `make prod-up`: 7 migrations applied by the `migrate` service, then API and worker started.
- Smoke test against the stack: health, readiness, upload, job completed, answer cites the
  document, unanswerable question abstains, no key gives 401 — PASS.
- `docker compose down` and `up` again (volume kept): no migrations re-run; the document, its
  chunks and the chat history were still there (`--check-persistence`) — PASS.
- The demo configuration (one container, inline ingestion, `/chat` disabled) — smoke PASS via
  `/retrieve`.
