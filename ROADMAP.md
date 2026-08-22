# Incremental indexing roadmap

## Architecture

Use a local SQLite sidecar as the authoritative snapshot of each fully
completed root scan. It records indexed images plus directory mtimes, so a
subsequent update stats known directories and enumerates only changed or new
subtrees. Qdrant remains the vector source of truth; state is committed only
after its corresponding upserts/deletes succeed.

## Modules

- `IncrementalIndexState`: transactional sidecar storage, validation metadata,
  and snapshot replacement.
- `ImageIndexer`: chooses fast directory-delta scan or a conservative full
  verification, applies exact Qdrant changes, and retains the rebuilt catalog.
- `SearchEngine` / Web: accept the catalog produced by indexing instead of
  invalidating its on-disk cache.

## Risks and safeguards

Directory mtimes do not reliably change when file contents are overwritten.
The fast path therefore has a configurable periodic full verification (24 h by
default), and `force=True` always verifies everything. Full verification still
uses file mtime plus size, so it cannot detect a same-size edit whose mtime was
manually restored; detecting that case requires hashing every file. Missing,
corrupt, dirty, or incomplete sidecar state falls back to a full scan.

## Milestones

1. Complete: add configuration and transactional SQLite state.
2. Complete: implement scandir scans and root delta calculation.
3. Complete: wire catalog handoff to Web and retain API compatibility.
4. Complete: add stub-backed tests and run lint/test verification.
