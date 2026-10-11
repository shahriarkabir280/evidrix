# ADR-0003: Postgres full-text search with the `english` config and OR queries

**Status:** accepted (2026-10-11)

## Context
Dense retrieval matches meaning but can miss exact identifiers (`OAuth2PasswordRequestForm`,
`status_code`). The keyword leg of hybrid search should catch those, without adding a search
engine next to Postgres.

## Decision
- A stored generated column `chunks.tsv` with a GIN index:
  `setweight(to_tsvector('english', coalesce(section_path, '')), 'A') || setweight(to_tsvector('english', text), 'B')`.
  Heading words weigh more than body words; Postgres keeps the column in sync with the text.
- Config `english`, not `simple`. Inspected on corpus identifiers: `english` stems
  ("running" → `run`, "dependencies"/"Depends" → `depend`, `HTTPException` → `httpexcept`), and
  because questions are stemmed the same way, identifiers still match. `simple` would make
  "declaring cookies" miss "declare cookie". Snake_case is split at `_` either way.
- Queries OR the question's lexemes: `replace(plainto_tsquery('english', q)::text, ' & ', ' | ')`.
  On the corpus, the AND form (what `websearch_to_tsquery` produces) matched 10 chunks for
  "How do I declare a cookie parameter?" and 0 once one unknown word was added; OR matched 256
  and lets ranking order them. `plainto_tsquery` accepts arbitrary user text safely.
- Ranking: `ts_rank_cd` (cover density). Only the order matters, because fusion uses ranks.

## Consequences
- Identifier queries that dense search ranks loosely are matched exactly; for "cleanup code after
  the response", keyword search surfaces "Dependencies with yield", which dense search misses.
- `ts_rank_cd` is not BM25 (no IDF saturation tuning); listed as a limitation and future work.
- Stemming conflates `Depends` (the FastAPI function) with the word "dependencies".
- Keyword search returns nothing for words absent from the corpus, which is itself a useful
  "probably unanswerable" signal; dense search always returns something.
