"""Initial unified ArtSeek v2 SQLite schema (user_version = 1)."""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_key TEXT UNIQUE NOT NULL,
    root_path TEXT NOT NULL,
    recursive INTEGER NOT NULL DEFAULT 1,
    last_full_verification_ns INTEGER NOT NULL DEFAULT 0,
    dirty INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    folder_key TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0,
    known INTEGER NOT NULL DEFAULT 1
);

-- id is 0-based explicit row index aligned with visual.f32.npy / descriptor.f32.npy.
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    point_id TEXT UNIQUE NOT NULL,
    root_id INTEGER NOT NULL REFERENCES roots(id),
    folder_id INTEGER NOT NULL REFERENCES folders(id),
    path TEXT NOT NULL,
    path_key TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    ext TEXT NOT NULL DEFAULT '',
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    phash TEXT NOT NULL DEFAULT '',
    indexed_at INTEGER NOT NULL DEFAULT 0,
    descriptor_text TEXT NOT NULL DEFAULT '',
    caption_json TEXT,
    deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    category INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS image_tags (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    score REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (image_id, tag_id)
);

CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL,
    sort TEXT NOT NULL DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS collection_items (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    added_position INTEGER NOT NULL,
    manual_position INTEGER,
    PRIMARY KEY (collection_id, item_id)
);

CREATE TABLE IF NOT EXISTS collection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stack TEXT NOT NULL,
    seq INTEGER NOT NULL,
    entry_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    name_key TEXT,
    name_source TEXT,
    avatar_url TEXT,
    note TEXT,
    profile_fetched INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artist_identity (
    kind TEXT NOT NULL,
    identity TEXT NOT NULL,
    display TEXT,
    profile_url TEXT,
    source TEXT,
    artist_id INTEGER NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (kind, identity)
);

CREATE TABLE IF NOT EXISTS artist_legacy (
    legacy_key TEXT PRIMARY KEY,
    artist_id INTEGER NOT NULL REFERENCES artist(id) ON DELETE CASCADE,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS image_source (
    point_id TEXT PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    content_hash TEXT,
    path_key TEXT,
    platform TEXT NOT NULL,
    source_id TEXT NOT NULL,
    page INTEGER,
    url TEXT NOT NULL,
    candidates_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 0.9,
    title TEXT,
    description TEXT,
    author_key TEXT,
    artist_id INTEGER REFERENCES artist(id) ON DELETE SET NULL,
    posted_at TEXT,
    width INTEGER,
    height INTEGER,
    mtime_ns INTEGER,
    page_count INTEGER,
    tags_json TEXT,
    manual_tags_json TEXT,
    auto_tags_json TEXT,
    auto_tag_scores_json TEXT,
    auto_tagged_at INTEGER,
    raw_json TEXT,
    source_origin TEXT NOT NULL DEFAULT 'filename',
    manual_fields TEXT,
    fetched_at INTEGER,
    error TEXT,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS source_tags (
    point_id TEXT NOT NULL REFERENCES image_source(point_id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    tag TEXT NOT NULL,
    tag_norm TEXT NOT NULL,
    origin TEXT NOT NULL,
    score REAL,
    PRIMARY KEY (point_id, origin, tag_norm)
);

CREATE TABLE IF NOT EXISTS fetch_state (
    domain TEXT PRIMARY KEY,
    blocked_until INTEGER,
    last_request REAL,
    day TEXT,
    day_count INTEGER,
    long_remaining INTEGER,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS notes_index (
    content_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (content_hash, name)
);

CREATE TABLE IF NOT EXISTS query_plan_cache (
    query_norm TEXT PRIMARY KEY,
    plan_json TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    summary_json TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_folders_root ON folders(root_id);
CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_id);
CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder_id) WHERE deleted = 0;
CREATE INDEX IF NOT EXISTS idx_images_root ON images(root_id) WHERE deleted = 0;
CREATE INDEX IF NOT EXISTS idx_images_mtime ON images(mtime_ns DESC, id) WHERE deleted = 0;
CREATE INDEX IF NOT EXISTS idx_images_name ON images(name COLLATE NOCASE, id) WHERE deleted = 0;
CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);
CREATE INDEX IF NOT EXISTS idx_images_deleted ON images(deleted);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id, image_id);

CREATE INDEX IF NOT EXISTS idx_collection_items_added
    ON collection_items(collection_id, added_position);
CREATE INDEX IF NOT EXISTS idx_collection_items_manual
    ON collection_items(collection_id, manual_position);
CREATE INDEX IF NOT EXISTS idx_collection_items_image
    ON collection_items(image_id);
CREATE INDEX IF NOT EXISTS idx_collection_history_stack
    ON collection_history(stack, seq);

CREATE INDEX IF NOT EXISTS idx_artist_name_key ON artist(name_key);
CREATE INDEX IF NOT EXISTS idx_artist_image_count ON artist(image_count DESC, id);
CREATE INDEX IF NOT EXISTS idx_artist_identity_artist ON artist_identity(artist_id);
CREATE INDEX IF NOT EXISTS idx_artist_legacy_artist ON artist_legacy(artist_id);

CREATE INDEX IF NOT EXISTS idx_source_image_id ON image_source(image_id);
CREATE INDEX IF NOT EXISTS idx_source_author ON image_source(author_key, image_id);
CREATE INDEX IF NOT EXISTS idx_source_artist_id ON image_source(artist_id, image_id);
CREATE INDEX IF NOT EXISTS idx_source_hash ON image_source(content_hash);
CREATE INDEX IF NOT EXISTS idx_source_platform_status ON image_source(platform, status);

CREATE INDEX IF NOT EXISTS idx_source_tags_norm ON source_tags(tag_norm, point_id);
CREATE INDEX IF NOT EXISTS idx_source_tags_image ON source_tags(tag_norm, image_id);
CREATE INDEX IF NOT EXISTS idx_notes_index_hash ON notes_index(content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at DESC);

-- Triggers maintaining artist.image_count identically to v1
CREATE TRIGGER IF NOT EXISTS trg_artist_count_insert
AFTER INSERT ON image_source
WHEN NEW.author_key IS NOT NULL AND NEW.author_key <> ''
BEGIN
    UPDATE artist
    SET image_count = image_count + 1
    WHERE id = CAST(NEW.author_key AS INTEGER);
END;

CREATE TRIGGER IF NOT EXISTS trg_artist_count_update
AFTER UPDATE OF author_key ON image_source
WHEN OLD.author_key IS NOT NEW.author_key
BEGIN
    UPDATE artist
    SET image_count = MAX(image_count - 1, 0)
    WHERE OLD.author_key IS NOT NULL
      AND OLD.author_key <> ''
      AND id = CAST(OLD.author_key AS INTEGER);
    UPDATE artist
    SET image_count = image_count + 1
    WHERE NEW.author_key IS NOT NULL
      AND NEW.author_key <> ''
      AND id = CAST(NEW.author_key AS INTEGER);
END;

CREATE TRIGGER IF NOT EXISTS trg_artist_count_delete
AFTER DELETE ON image_source
WHEN OLD.author_key IS NOT NULL AND OLD.author_key <> ''
BEGIN
    UPDATE artist
    SET image_count = MAX(image_count - 1, 0)
    WHERE id = CAST(OLD.author_key AS INTEGER);
END;
"""


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
