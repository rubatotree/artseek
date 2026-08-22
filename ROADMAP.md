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

## Random waterfall browsing

Add a lightweight browse mode for serendipitous review of the currently
selected folders. A seeded server-side page returns only metadata for a
random, non-repeating ordering; the existing masonry renderer and infinite
scroll remain responsible for presentation.

1. Complete: add a range-aware random page endpoint with time-filter support.
2. Complete: add an accessible secondary action beside search and reuse gallery paging.
3. Complete: cover pagination, scope boundaries, and empty selection with API tests.

## 收藏夹（Local collections）

Collections are user-owned organization state rather than index state. A small
locked JSON sidecar at `data_dir/collections.json` stores collection metadata
and ordered Qdrant point IDs only; gallery records are projected from the
current index at read time. This keeps collections independent of original
files and preserves stale IDs non-destructively across index removal or a
later re-index.

### Risks and safeguards

- Every read-modify-write operation is guarded by an app-local reentrant lock
  and persisted with a sibling `.tmp` file followed by replacement, so normal
  writes cannot leave a partial JSON document behind.
- Missing or malformed collection files safely behave as an empty list. The
  loader rejects malformed rows and de-duplicates IDs while retaining valid
  insertion order.
- Collection browsing filters unindexed IDs only in the response; it never
  rewrites the stored collection as a side effect.
- Unfiltered gallery pages keep a four-entry LRU of live point IDs and source
  cursors only, capped at 10,240 IDs per collection; Qdrant payloads are never
  retained. Sequential scrolling therefore avoids re-reading earlier scan
  batches while memory remains bounded by the public 10,000-item offset window.
- A missing index is memoized only until an index lifecycle event. The flag is
  reset before and after indexing so a partial failed write is still visible to
  later collection requests.
- Collection page offsets are validated (0–10,000). Each Qdrant retrieval is
  capped to a 256-ID batch; the LRU lock is released before retrieval and only
  a per-collection cursor is advanced during a scan.
- Until an unfiltered request reaches the collection tail, its API total is
  `null` rather than the membership count, so stale IDs cannot create a false
  remaining-image indicator in the gallery.

### Milestones

1. Complete: implement the local persistence module, configuration path, and
   ordered/idempotent collection APIs.
2. Complete: add collection browsing, query pagination, stale-index filtering,
   and focused persistence/API regression coverage.
3. Complete: integrate a third sidebar tab, rename/delete controls, desktop
   drag-and-drop, and card/viewer collection pickers with touch/keyboard paths.
4. Complete: run lint, targeted API tests, static JavaScript syntax validation,
   and the complete test suite.

### Pinned drag interaction milestone

Pinned state is stored beside collection metadata and remains independent from
the image index. The collection API sorts pinned collections ahead of the
stable creation-order remainder, while a partial PATCH can rename and/or pin
in a single atomic sidecar update. Desktop drag uses one fixed dock with a
reference target and pinned collection targets; touch and keyboard operations
stay in the fullscreen viewer. The viewer collection chooser is a top-layer,
direct dialog child rather than a card-local popup, so it cannot be clipped by
the image stage.

Risks: stale list refreshes must not invalidate an active drag, and dialogs
must not leave focus stranded under the modal. The dock therefore renders the
last known pinned list immediately, refreshes asynchronously behind a drag
session token, and keeps its initial clamped position fixed for the drop. The
viewer chooser is a non-modal dialog that clamps again on viewport/soft-keyboard
changes, restores its trigger on Escape, and closes before the active image
changes or on outside interaction/dialog close. Its asynchronous list refresh
captures only focus already inside the chooser and restores a matching control;
typed creation drafts and their cursor positions survive a delayed refresh.

5. Complete: persist compatible pinned collections and partial PATCH updates.
6. Complete: replace waterfall actions with the drag dock and accessible
   fullscreen viewer collection menu.
