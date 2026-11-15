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

### Manual collection ordering milestone

Collection order is an explicit part of the local sidecar's array order; there
is no pinned state or pinned-first sort. A drag operation submits the complete
ordered ID list to a single atomic endpoint. The store rejects incomplete or
stale lists, preventing a delayed UI from dropping a concurrently created or
deleted collection. Legacy `pinned` fields are ignored while reading and are
removed during the next write.

The sidebar uses a dedicated pointer drag handle so rename, browse, and delete
controls remain ordinary buttons. It also supports `Alt` + up/down arrows on
the handle. A refresh is deferred during an active drag; the saved response is
then used as the canonical UI order.

5. Complete: replace pinned metadata and pinned-first sorting with explicit,
   validated persistent ordering.
6. Complete: add pointer and keyboard ordering controls in the sidebar and
   regression coverage for legacy migration and stale order conflicts.

### Undo history for add/delete

Collection delete, adding an image, and removing an image are reversible
commands stored in the same atomic sidecar. The stacks survive refresh; a new
undoable mutation clears redo; history is capped so a large collection cannot
grow the file without bound. Restore keeps the original collection ID, item
order, and list position. Create, rename, and reorder are not undoable.

7. Complete: persist undo/redo stacks beside collection rows and expose undo
   and redo endpoints with history on the collection list.
8. Complete: add sidebar undo/redo controls, keyboard shortcuts, and
   regression coverage for restore, stale entries, and history caps.

### Paged collection grid

The collection order remains one canonical, persisted sequence, but every
collection surface presents it as fixed 2-by-6 pages. The sidebar and both
collection pickers therefore share a 12-item page model and can independently
change pages. Reordering moves the canonical sequence, including across page
boundaries; a drag held over a page control changes the visible page before
drop. Heart taps still open the picker, while quick-add requires a deliberate
drag beyond a pointer-distance threshold before release can activate a tile.

9. Complete: replace the sidebar list and picker lists with fixed-size,
   paged 2-by-6 grids; preserve cross-page ordering and protect touch taps
   from accidental quick-add releases.

10. Complete: separate collection browsing from structural editing so tile
    titles and compact counts remain calm by default, while create, reorder,
    rename, and delete controls appear only in an explicit edit mode.
11. Complete: reserve the full tile width for eight-character Chinese titles,
    move compact counts to the lower-right corner, and consolidate collection
    undo, redo, and edit actions into a bottom icon toolbar.
12. Complete: replace the single edit state with direct create, rename,
    reorder, and guarded-delete tools; align picker tile typography with the
    sidebar and compact the sidebar grid height.
13. Complete: make inline rename a compact blur-to-save interaction, add
    durable rename undo/redo history, and refine the collection action icons.
14. Complete: latch drag pagination by physical button direction so a rendered
    replacement cannot repeatedly advance pages while the pointer remains over it.
15. Complete: create blank-titled collection cards for immediate inline naming
    while preserving empty titles safely in the sidecar and history records.
16. Complete: accept blank titles on both create and rename API paths so older
    clients and the automatic blank-name flow share one validation contract.
17. Complete: treat an automatically created, still-editing card as a temporary
    row and discard it without history when its rename operation is cancelled.
18. Complete: record persistent collection-order transitions for undo/redo
    and make a right-click during an active reorder restore its initial order.
19. Complete: record collection creation in durable history while keeping
    cancellation of a still-pending new card history-free.
20. Complete: provide a visible collection edit-target state and make outside
    focus or right-click cancel an uncommitted edit mode.

## Image notes

Notes are user-owned Markdown, not index state. Each indexed image maps to
one folder in a configurable notes vault, keyed by the file's SHA-256 `content_hash`. Byte-identical copies and
near-duplicate resizes (dHash Hamming distance ≤ 10, similar aspect
ratio) share one folder via aliases stored in `.artseek-index.json`.
The Hamming scan covers only hashes already in the notes vault, not the
whole Qdrant collection. Originals remain read-only.

The in-app editor is Vditor, pinned as a Git submodule at tag `v3.11.3`.
Upstream does not commit `dist/`; this repo does not either. Build once
inside the submodule after `git submodule update --init`. Advanced settings expose the Qdrant directory and the notes vault
path. Path overrides persist in `runtime.json` and are applied by
`load_config`, so CLI indexing and the web server share the same locations.
Changing the Qdrant path requires a process restart; the notes path applies
immediately.

### Risks and safeguards

- Note names and asset paths are contained under `{notes_dir}/{hash}/`.
  `..`, absolute paths, and Windows reserved device names are rejected.
- Writes use a sibling `.tmp` file and replace, with a process-local lock.
- A missing `content_hash` falls back to hashing the current file; a missing
  file cannot open notes. Pixel edits produce a new hash and a new folder.
- Vditor is configured with a local `cdn` path so it does not fetch KaTeX or
  Mermaid from the network. Diagrams stay disabled until those assets are
  vendored.
- Saving a Qdrant path without copying still requires a restart. 「迁移」
  closes the local client, copies the directory (skipping `.lock`), and
  reopens at the destination so the session can keep using the moved
  library. The source directory is left in place. A non-empty destination
  is rejected so an existing library cannot be overwritten.

### Milestones

1. Complete: add overridable `qdrant_path` / `notes_dir` with runtime.json
   precedence.
2. Complete: implement the content-hash note vault (`NoteStore`) and
   persistence tests.
3. Complete: expose notes CRUD, asset upload, and storage-path APIs.
4. Complete: add the viewer notes pane with lazy-loaded Vditor and Advanced
   path fields.
5. Complete: cover sanitization, identity, API, and homepage fingerprints.
6. Complete: persist per-image note order; drag tabs like a browser strip;
   hide the editor scrollbar; name notes from the first line and keep
   timestamp filenames.
7. Complete: add per-path 「迁移」 controls that copy the live Qdrant or
   notes directory to the typed destination, then switch to it.
8. Complete: add cache-directory migrate/save beside the other storage
   paths. Config.toml stays at the platform location.
9. Complete: treat the index database as a sidecar directory (SQLite,
   folder catalog, folder scope, collections, manifest) with migrate/save.
10. Complete: display titles from the first Markdown line; files use unique
    UTC timestamps instead of 「笔记」/untitled names.

## Shareable view URLs

The web UI is still a single `index.html`. FastAPI serves that shell at
`/search`, `/folder`, `/random`, `/c/{id}`, `/i/{point_id}`, and
`/i/{point_id}/notes/{name}` in addition to `/`. The client pushes overlay
entries for the viewer and notes, replaces the URL when flipping images or
switching note tabs, and applies `popstate` without a full reload. Same-origin
links inside Vditor are captured so a pasted `/i/{id}/notes/{name}` opens the
cited note. `/image/{point_id}` remains the original-file route.

The search URL keeps the query text and in-library reference IDs
(`/search?q=…&ref=id[:weight],id`). After a run, the server writes a JSON
snapshot under `cache_dir/search-snapshots/{id}.json` and the client replaces
the address with `&s={id}`. Reopening prefers the snapshot so ranking stays
frozen; a cache miss re-encodes `q` + `ref`. Uploaded file references stay
session-only. Snapshots are capped (96 files, 400 hits) and travel with cache
directory migration.

### Milestones

1. Complete: SPA shell routes, History API sync, in-note link capture, and
   homepage / API fingerprints.
2. Complete: persist search snapshots beside `q`/`ref` in the URL, restore
   them on open, and cover store/API/homepage fingerprints.

## 收藏夹排序（Collection sorting）

Each collection owns a persistent `sort` preference and an optional manual
permutation. `item_ids` stays the immutable insertion sequence, so switching
between "added order" and "manual order" never loses either one. A request-level
`sort` query parameter previews another order without persisting it; the
sidebar tile badge and gallery control show the saved mode.

### Risks and safeguards

- Payload-based sorts (file mtime, name) need every live item's key, so one
  bounded sorted-ID cursor is built per collection and cached; Qdrant payloads
  are not retained. `added`/`manual` sorts keep the existing lazy cursor path.
- Manual order updates are anchor-based (`before_id`/`after_id`) so paginated
  clients never need the full list and concurrent edits get a 409 instead of a
  silent append.
- Entering manual mode seeds the permutation from the order currently on screen
  (`seed_sort`), avoiding a jump back to insertion order.
- Manual reordering is undoable with the full previous/next permutation and
  sort mode; sort-preference changes are not pushed onto the content history.

### Milestones

1. Complete: persistence schema, per-collection sort, temporary overrides.
2. Complete: anchor-based manual ordering with undo/redo and stale-anchor 409.
3. Complete: icon sort menu, tile badges, pointer/keyboard card reordering.
4. Complete: store/API/homepage tests plus a live browser check of the sort
   menu, temporary preview, seeding, drag, keyboard reorder, and undo.

## Image-first gallery cards

Cards are now the image itself: no filename/dimension body, translucent
favourite and removal overlays, and tag hits as a compact pill. Very tall or
very wide images have their display aspect ratio clamped to `0.6…3` and are
cropped with `object-fit: cover`; the masonry height estimate uses the same
value.

A view control beside the collection sort control switches between natural
masonry, square masonry, compact square masonry, and a one-row-per-image list.
The choice persists in `localStorage`. Tablet portrait/landscape use four
columns for natural/square views, six for compact, while phones use two
(three when compact) and list always uses one.

1. Complete: image-only card markup, favourite/removal overlays, drag-gesture
   guards.
2. Complete: clamped card aspect ratios and matching masonry estimates.
3. Complete: flex-column tablet/phone grids with four tablet columns.
4. Complete: persisted view modes, compact mode without favourites, list rows,
   plus full pytest, Ruff, JS syntax, and live browser checks at
   390/768/834/1024/1366px.

### Guarded card actions

Favourite and removal controls stay hidden until a fine-pointer hover or a
380ms touch/pen press. Touch keeps the gesture alive for quick drag: onto the
heart opens the collection panel and can continue to a collection tile; onto
removal and release removes. Desktop removal is drag-only (clicking only shows
a hint), preventing accidental loss. List rows reveal their action group on
hover, long-press, drag, or focus-within.
