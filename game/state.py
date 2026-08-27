"""Campaign state: the clock, who exists, how they feel, what is true.

Everything in here is injected into every single prompt, verbatim, and is
never retrieved by vector search.  That split is the central design decision
of the whole engine.

If Reina's height lives only in an embedding index, then on any turn where
the conversation is about something else, retrieval will not surface it and
the model will invent a number.  A character sheet is not a memory to be
recalled — it is a fact that is true on every turn, so it is always present
and it always overrides anything the search happened to return.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from .clock import Clock
from .schema import SCHEMA


@dataclass
class Character:
    slug: str
    name: str
    role: str
    sheet: str
    present: bool
    protagonist: bool

    def block(self) -> str:
        head = f"{self.name}" + (f" — {self.role}" if self.role else "")
        return f"{head}\n{self.sheet.strip()}" if self.sheet.strip() else head


@dataclass
class Turn:
    n: int
    in_date: str
    in_time: str
    title: str
    location: str
    player_input: str
    narration: str
    choices: list[str]
    words: int
    summary: str


class GameState:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- campaign meta --------------------------------------------------
    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM campaign WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value):
        self.db.execute(
            "INSERT INTO campaign(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.db.commit()

    @property
    def started(self) -> bool:
        return self.get("title") is not None

    # -- clock ----------------------------------------------------------
    @property
    def clock(self) -> Clock:
        return Clock.parse(self.get("date", "2025-04-14"), self.get("time", "08:30"))

    @clock.setter
    def clock(self, clock: Clock):
        self.set("date", clock.date)
        self.set("time", clock.time)

    @property
    def location(self) -> str:
        return self.get("location", "")

    @location.setter
    def location(self, value: str):
        self.set("location", value)

    # -- characters -----------------------------------------------------
    def upsert_character(self, slug: str, *, name: str, role: str = "",
                         sheet: str = "", present: bool = False,
                         protagonist: bool = False, turn: int = 0):
        self.db.execute(
            "INSERT INTO characters(slug, name, role, sheet, present, protagonist, "
            "first_turn, updated_turn) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, "
            "role=CASE WHEN excluded.role != '' THEN excluded.role ELSE characters.role END, "
            "sheet=CASE WHEN excluded.sheet != '' THEN excluded.sheet ELSE characters.sheet END, "
            "present=excluded.present, updated_turn=excluded.updated_turn",
            (slug, name, role, sheet, int(present), int(protagonist), turn, turn),
        )
        self.db.commit()

    def character(self, slug: str) -> Character | None:
        row = self.db.execute("SELECT * FROM characters WHERE slug=?", (slug,)).fetchone()
        return _character(row) if row else None

    def characters(self, *, present_only: bool = False) -> list[Character]:
        sql = "SELECT * FROM characters"
        if present_only:
            sql += " WHERE present=1"
        sql += " ORDER BY protagonist DESC, name"
        return [_character(r) for r in self.db.execute(sql)]

    @property
    def protagonist(self) -> Character | None:
        row = self.db.execute(
            "SELECT * FROM characters WHERE protagonist=1 LIMIT 1").fetchone()
        return _character(row) if row else None

    def set_present(self, slugs: list[str]):
        """Replace the on-stage cast. Sheets for absent characters stay in the
        database — they are simply not injected while nobody can see them."""
        self.db.execute("UPDATE characters SET present=0")
        for slug in slugs:
            self.db.execute("UPDATE characters SET present=1 WHERE slug=?", (slug,))
        self.db.commit()

    # -- relationships --------------------------------------------------
    def set_relationship(self, other: str, *, label: str | None = None,
                         closeness: int | None = None, note: str | None = None,
                         turn: int = 0):
        existing = self.db.execute(
            "SELECT * FROM relationships WHERE other=?", (other,)).fetchone()
        if existing:
            self.db.execute(
                "UPDATE relationships SET label=?, closeness=?, note=?, updated_turn=? "
                "WHERE other=?",
                (label if label is not None else existing["label"],
                 _clamp(closeness if closeness is not None else existing["closeness"]),
                 note if note is not None else existing["note"], turn, other),
            )
        else:
            self.db.execute(
                "INSERT INTO relationships(other, label, closeness, note, updated_turn) "
                "VALUES(?,?,?,?,?)",
                (other, label or "", _clamp(closeness if closeness is not None else 5),
                 note or "", turn),
            )
        self.db.commit()

    def relationships(self) -> list[sqlite3.Row]:
        return list(self.db.execute(
            "SELECT r.*, c.name FROM relationships r "
            "LEFT JOIN characters c ON c.slug = r.other ORDER BY r.closeness DESC"))

    # -- flags ----------------------------------------------------------
    def set_flag(self, key: str, value, turn: int = 0):
        self.db.execute(
            "INSERT INTO flags(key, value, updated_turn) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_turn=excluded.updated_turn",
            (key, json.dumps(value), turn),
        )
        self.db.commit()

    def flags(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self.db.execute("SELECT * FROM flags")}

    # -- turns ----------------------------------------------------------
    @property
    def turn_count(self) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(n), 0) n FROM turns").fetchone()
        return int(row["n"])

    def add_turn(self, *, in_date: str, in_time: str, title: str, location: str,
                 player_input: str, narration: str, choices: list[str],
                 words: int) -> int:
        n = self.turn_count + 1
        self.db.execute(
            "INSERT INTO turns(n, played_at, in_date, in_time, title, location, "
            "player_input, narration, choices, words) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (n, time.time(), in_date, in_time, title, location, player_input,
             narration, json.dumps(choices), words),
        )
        self.db.commit()
        return n

    def turn(self, n: int) -> Turn | None:
        row = self.db.execute("SELECT * FROM turns WHERE n=?", (n,)).fetchone()
        return _turn(row) if row else None

    def recent_turns(self, limit: int) -> list[Turn]:
        rows = self.db.execute(
            "SELECT * FROM turns ORDER BY n DESC LIMIT ?", (limit,)).fetchall()
        return [_turn(r) for r in reversed(rows)]

    def pending_choices(self) -> list[str]:
        row = self.db.execute(
            "SELECT choices FROM turns ORDER BY n DESC LIMIT 1").fetchone()
        return json.loads(row["choices"]) if row else []

    def set_summary(self, n: int, summary: str):
        self.db.execute("UPDATE turns SET summary=?, archived=1 WHERE n=?", (summary, n))
        self.db.commit()

    def unarchived(self) -> list[Turn]:
        return [_turn(r) for r in
                self.db.execute("SELECT * FROM turns WHERE archived=0 ORDER BY n")]

    # -- arcs -----------------------------------------------------------
    def add_arc(self, from_turn: int, to_turn: int, text: str):
        self.db.execute(
            "INSERT INTO arcs(from_turn, to_turn, text, created_at) VALUES(?,?,?,?)",
            (from_turn, to_turn, text, time.time()),
        )
        self.db.commit()

    def arcs(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM arcs ORDER BY from_turn"))

    def last_arc_turn(self) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(to_turn), 0) t FROM arcs").fetchone()
        return int(row["t"])


def _character(row: sqlite3.Row) -> Character:
    return Character(row["slug"], row["name"], row["role"], row["sheet"],
                     bool(row["present"]), bool(row["protagonist"]))


def _turn(row: sqlite3.Row) -> Turn:
    return Turn(int(row["n"]), row["in_date"], row["in_time"], row["title"],
                row["location"], row["player_input"], row["narration"],
                json.loads(row["choices"]), int(row["words"]), row["summary"])


def _clamp(value: int) -> int:
    return max(0, min(10, int(value)))
