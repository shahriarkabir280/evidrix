.PHONY: up down migrate dev worker test test-live lint fmt psql corpus eval eval-full

up:        ## Start Postgres and Redis in Docker
	docker compose up -d --wait

down:      ## Stop containers (data is kept in the pgdata volume)
	docker compose down

migrate:   ## Apply all migrations to the dev database
	uv run alembic upgrade head

dev:       ## Run the API with auto-reload
	uv run uvicorn app.main:app --reload

worker:    ## Recover jobs a dead worker left behind, then run the ingestion worker
	uv run python -m app.cli reap-jobs
	uv run rq worker --worker-class rq.worker.SimpleWorker --with-scheduler ingestion

test:      ## Run the test suite (everything except tests that call a live LLM)
	uv run pytest -m "not live"

test-live: ## Run tests against the real LLM (start Ollama first)
	uv run pytest -m live

lint:      ## Check lint and formatting
	uv run ruff check . && uv run ruff format --check .

fmt:       ## Auto-format and fix lint
	uv run ruff format . && uv run ruff check --fix .

psql:      ## Open a psql shell on the dev database
	docker compose exec db psql -U postgres -d evidrix

corpus:    ## Download the pinned evaluation corpus and check it against the manifest
	uv run python -m scripts.fetch_corpus && uv run python -m scripts.fetch_corpus --verify

eval:      ## Retrieval metrics on the dev split for every preset (seconds)
	for p in dense hybrid hybrid_rerank; do uv run python -m evals.run --preset $$p --split dev --mode retrieval; done

eval-full: ## Answers + judge on the dev split for PRESET (default hybrid_rerank; minutes, needs Ollama)
	uv run python -m evals.run --preset $${PRESET:-hybrid_rerank} --split dev --mode full
