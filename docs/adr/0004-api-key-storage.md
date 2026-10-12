# ADR-0004: API keys stored as sha256 and looked up by hash

**Status:** accepted (2026-10-12)

## Context
Every API request must identify its user (plan: Part 5, step 4). Keys must survive a database
leak (a stolen table must not contain working keys), be checked on every request without
adding noticeable latency, and be revocable. The plan asked for hashed storage and a
constant-time comparison.

## Decision
- **Format:** `evx_` + `secrets.token_urlsafe(32)`, i.e. 256 random bits. The prefix makes a
  leaked key recognisable (for secret scanners and people), and lets the server reject other
  tokens before touching the database.
- **Storage:** only `sha256(key)` (unique index) and a non-secret display prefix
  (`evx_` + 8 characters) to tell keys apart in lists. The key is printed once, at creation.
- **Hash: sha256, not bcrypt/argon2.** Slow hashes exist to make guessing low-entropy
  *passwords* expensive. A 256-bit random key can't be guessed at any hash speed, so a slow hash
  would only add tens of milliseconds to every request. And a fast, unsalted, deterministic hash
  is what makes an indexed lookup possible.
- **Verification: look the hash up, don't compare keys.** The request's key is hashed and
  matched against the unique index (`WHERE key_hash = :h AND revoked_at IS NULL`). A timing
  attack against this comparison could at best reveal something about the *hash* of the
  attacker's own guess, and sha256 preimage resistance means that says nothing about any real
  key. The plan's "constant-time comparison" is needed when a secret is compared directly
  (e.g. `presented == stored_key`); here no secret is ever compared, so the requirement is met
  by construction rather than by `hmac.compare_digest`.
- **Errors:** a missing, malformed, unknown and revoked key all get the same `401` with
  `WWW-Authenticate: Bearer`, so the response doesn't reveal whether a key ever existed.
- **Revocation:** `revoked_at` is set, never deleted, so a key's history stays auditable.

## Consequences
- One indexed lookup per request (well under a millisecond locally); no cache, so a revoked key
  stops working on the very next request.
- No `last_used_at`: it would turn every read request into a write. Can be added later with
  throttled updates if key hygiene needs it.
- Users are created only with the CLI (`create-user`, `create-key`, `revoke-key`). There is no
  sign-up endpoint; this is a single-tenant-operator deployment, not a SaaS.
- The seeded dev user (`dev@evidrix.local`) still owns what the CLI ingests, such as the
  evaluation corpus. The API never falls back to it; to query that corpus through the API,
  create a key for it.
