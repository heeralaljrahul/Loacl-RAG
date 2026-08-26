"""SQLite-backed store: documents, chunks, keyword index and vectors.

The whole index is one folder you can copy, back up or delete.  There is no
vector-database server to run and no separate index file that can drift out
of sync with the database: vectors live in a BLOB column beside the text
they belong to, and the keyword index is an FTS5 view of the same table kept
current by triggers.

Search is exact brute-force cosine over a float32 matrix.  For a personal
corpus that is the right call — 100k chunks at 384 dimensions is a 150 MB
matrix that scores in about 10 ms, with no approximate-index recall loss and
nothing to rebuild after an edit.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .text import strip_title_prefix

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    ext         TEXT,
    bytes       INTEGER,
    mtime       REAL,
    sha256      TEXT,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    ingested_at REAL
);

CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    doc_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord     INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    page    INTEGER,
    text    TEXT NOT NULL,
    vec     BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading)
    VALUES ('delete', old.id, old.text, old.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text, heading ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading)
    VALUES ('delete', old.id, old.text, old.heading);
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    path: str
    title: str
    heading: str
    page: int | None
    text: str
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    @property
    def label(self) -> str:
        name = Path(self.path).name
        if self.page:
            name = f"{name} p.{self.page}"
        section = strip_title_prefix(self.title, self.heading)
        return f"{name} — {section}" if section else name

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "title": self.title,
            "heading": self.heading,
            "page": self.page,
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 3),
        }


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self.set_meta("schema_version", SCHEMA_VERSION)
        self._matrix: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._matrix_gen: int = -1

    def close(self):
        self.db.close()

    # -- meta -----------------------------------------------------------
    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value):
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.db.commit()

    def bump_generation(self):
        self.set_meta("generation", int(self.get_meta("generation", 0)) + 1)

    # -- documents ------------------------------------------------------
    def document_by_path(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone()

    def all_document_paths(self) -> dict[str, sqlite3.Row]:
        return {r["path"]: r for r in self.db.execute("SELECT * FROM documents")}

    def delete_document(self, doc_id: int):
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.db.commit()
        self.bump_generation()

    def upsert_document(
        self, *, path: str, title: str, ext: str, size: int, mtime: float, sha256: str
    ) -> int:
        existing = self.document_by_path(path)
        if existing:
            self.db.execute("DELETE FROM chunks WHERE doc_id=?", (existing["id"],))
            self.db.execute(
                "UPDATE documents SET title=?, ext=?, bytes=?, mtime=?, sha256=?, "
                "n_chunks=0, ingested_at=? WHERE id=?",
                (title, ext, size, mtime, sha256, time.time(), existing["id"]),
            )
            self.db.commit()
            return int(existing["id"])
        cur = self.db.execute(
            "INSERT INTO documents(path, title, ext, bytes, mtime, sha256, ingested_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (path, title, ext, size, mtime, sha256, time.time()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_chunks(self, doc_id: int, rows: list[tuple], vectors: np.ndarray):
        """rows: (ord, heading, page, text); vectors aligned row-for-row."""
        payload = [
            (doc_id, o, h, p, t, vectors[i].astype(np.float32).tobytes())
            for i, (o, h, p, t) in enumerate(rows)
        ]
        self.db.executemany(
            "INSERT INTO chunks(doc_id, ord, heading, page, text, vec) VALUES(?,?,?,?,?,?)",
            payload,
        )
        self.db.execute(
            "UPDATE documents SET n_chunks=(SELECT COUNT(*) FROM chunks WHERE doc_id=?) "
            "WHERE id=?",
            (doc_id, doc_id),
        )
        self.db.commit()
        self.bump_generation()

    # -- vectors --------------------------------------------------------
    def _ensure_matrix(self):
        generation = int(self.get_meta("generation", 0))
        if self._matrix is not None and self._matrix_gen == generation:
            return
        rows = self.db.execute(
            "SELECT id, vec FROM chunks WHERE vec IS NOT NULL ORDER BY id"
        ).fetchall()
        if not rows:
            self._ids = np.zeros(0, dtype=np.int64)
            self._matrix = np.zeros((0, 1), dtype=np.float32)
        else:
            self._ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
            self._matrix = np.frombuffer(
                b"".join(r["vec"] for r in rows), dtype=np.float32
            ).reshape(len(rows), -1)
        self._matrix_gen = generation

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        self._ensure_matrix()
        if self._matrix is None or len(self._ids) == 0:
            return []
        if self._matrix.shape[1] != query_vec.shape[0]:
            raise RuntimeError(
                f"index was built with {self._matrix.shape[1]}-dim vectors but the "
                f"embedding model produces {query_vec.shape[0]}. Re-ingest with "
                f"`python cli.py reindex` after changing RAG_EMBED_MODEL."
            )
        scores = self._matrix @ query_vec.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._ids[i]), float(scores[i])) for i in top]

    # -- keyword --------------------------------------------------------
    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        expr = fts_query(query)
        if not expr:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(chunks_fts, 1.0, 2.0) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (expr, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() is "more negative is better"; flip it so bigger is better.
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    # -- hydration ------------------------------------------------------
    def hits(self, chunk_ids: list[int]) -> dict[int, Hit]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(
            f"SELECT c.id, c.doc_id, c.heading, c.page, c.text, d.path, d.title "
            f"FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id IN ({marks})",
            chunk_ids,
        ).fetchall()
        return {
            int(r["id"]): Hit(
                chunk_id=int(r["id"]),
                doc_id=int(r["doc_id"]),
                path=r["path"],
                title=r["title"],
                heading=r["heading"],
                page=r["page"],
                text=r["text"],
            )
            for r in rows
        }

    def neighbours(self, hit: Hit, before: int = 1, after: int = 1) -> list[Hit]:
        centre = _ord_of(self.db, hit.chunk_id)
        rows = self.db.execute(
            "SELECT c.id, c.doc_id, c.heading, c.page, c.text, d.path, d.title "
            "FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "WHERE c.doc_id=? AND c.ord BETWEEN ? AND ? ORDER BY c.ord",
            (hit.doc_id, centre - before, centre + after),
        ).fetchall()
        return [
            Hit(int(r["id"]), int(r["doc_id"]), r["path"], r["title"], r["heading"],
                r["page"], r["text"])
            for r in rows
        ]

    # -- stats ----------------------------------------------------------
    def stats(self) -> dict:
        docs = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b FROM documents"
        ).fetchone()
        chunks = self.db.execute("SELECT COUNT(*) n FROM chunks").fetchone()
        by_ext = self.db.execute(
            "SELECT ext, COUNT(*) n FROM documents GROUP BY ext ORDER BY n DESC"
        ).fetchall()
        self._ensure_matrix()
        return {
            "documents": int(docs["n"]),
            "source_bytes": int(docs["b"]),
            "chunks": int(chunks["n"]),
            "dimensions": int(self._matrix.shape[1]) if self._ids is not None and len(self._ids) else 0,
            "embed_model": self.get_meta("embed_model"),
            "index_mb": round(self.cfg.db_path.stat().st_size / 1e6, 1)
            if self.cfg.db_path.exists() else 0.0,
            "by_ext": {r["ext"]: int(r["n"]) for r in by_ext},
        }


def _ord_of(db: sqlite3.Connection, chunk_id: int) -> int:
    row = db.execute("SELECT ord FROM chunks WHERE id=?", (chunk_id,)).fetchone()
    return int(row["ord"]) if row else 0


_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[.\-/][A-Za-z0-9_]+)*")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "do", "does", "did", "what", "which", "who", "how",
    "why", "when", "where", "it", "this", "that", "with", "as", "at", "by",
    "from", "about", "can", "i", "me", "my", "you", "your",
}


def fts_query(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression.

    FTS5's grammar treats most punctuation as syntax, so a raw user question
    is a syntax error more often than not.  Tokens are extracted, quoted and
    OR-ed; ranking (not matching) decides which documents win.
    """
    tokens = [t.lower() for t in _TOKEN.findall(query)]
    keep = [t for t in tokens if t not in _STOP and len(t) > 1]
    if not keep:
        keep = [t for t in tokens if len(t) > 1]
    if not keep:
        return ""
    unique = list(dict.fromkeys(keep))[:32]
    return " OR ".join(f'"{t}"' for t in unique)
