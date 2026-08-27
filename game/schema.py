"""Game tables.

These live in the same SQLite file as the retrieval index, so a campaign is
still one folder you can copy or back up.  They hold the things that must
never be reconstructed from a vector search: the clock, who exists, how they
feel about each other, and the verbatim transcript.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS characters (
    slug         TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT '',
    sheet        TEXT NOT NULL DEFAULT '',
    present      INTEGER NOT NULL DEFAULT 0,
    protagonist  INTEGER NOT NULL DEFAULT 0,
    first_turn   INTEGER,
    updated_turn INTEGER
);

CREATE TABLE IF NOT EXISTS relationships (
    other        TEXT PRIMARY KEY REFERENCES characters(slug) ON DELETE CASCADE,
    label        TEXT NOT NULL DEFAULT '',
    closeness    INTEGER NOT NULL DEFAULT 5,
    note         TEXT NOT NULL DEFAULT '',
    updated_turn INTEGER
);

CREATE TABLE IF NOT EXISTS flags (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    updated_turn INTEGER
);

CREATE TABLE IF NOT EXISTS turns (
    n            INTEGER PRIMARY KEY,
    played_at    REAL NOT NULL,
    in_date      TEXT NOT NULL,
    in_time      TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    location     TEXT NOT NULL DEFAULT '',
    player_input TEXT NOT NULL DEFAULT '',
    narration    TEXT NOT NULL,
    choices      TEXT NOT NULL DEFAULT '[]',
    words        INTEGER NOT NULL DEFAULT 0,
    summary      TEXT NOT NULL DEFAULT '',
    archived     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS arcs (
    id         INTEGER PRIMARY KEY,
    from_turn  INTEGER NOT NULL,
    to_turn    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""
