# ADR-0001: PostgreSQL + pgvector for documents and vectors

**Status:** accepted (2026-10-09)

## Context
Evidrix stores users, documents, pages, chunks and job state, and needs nearest-neighbour search
over chunk embeddings. A dedicated vector database (Qdrant, Pinecone) would be a second system to
run, back up and keep consistent with the relational data.

## Decision
Use one PostgreSQL 17 database with the pgvector extension. Chunks live in a normal table with a
`vector(384)` column and an HNSW index (`vector_cosine_ops`, m=16, ef_construction=64). Owner,
status and index-version filters are ordinary SQL `WHERE` clauses in the same query.

## Consequences
- Access control and vector search happen in one query and one transaction, so another user's
  chunks are never candidates (tested by deliberately removing the filter).
- Chunk rows, document rows and their vectors commit or roll back together.
- At the current size (~700 chunks) the planner uses an exact sequential scan (3 ms), so results
  have no approximation error. HNSW and `hnsw.iterative_scan` are in place for when it grows.
- Changing to an embedding model with a different dimension needs a migration and a re-index.
- A dedicated vector database can still be benchmarked later (plan: future work).
