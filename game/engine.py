"""One turn, end to end.

    resolve input → assemble context → narrate → validate → repair
                  → save → distil → apply state → index memory → fold arcs

The validate/repair step is what keeps a small local model usable. A 9B
model hits an exact 800-900 word window maybe two thirds of the time and
occasionally gives four options instead of five. Accepting those silently
degrades every later turn — the transcript teaches the model that four
options is fine, and drift compounds. So the entry is measured, and one
targeted revision pass is spent fixing precisely what was wrong.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from rag.app import App
from rag.llm import LLMError, strip_thinking

from . import format as fmt
from .archivist import Archivist, Distillate
from .clock import Clock
from .config import GameConfig
from .context import build_user_prompt
from .memory import MemoryArchive, recall_query
from .prompts import NARRATOR, REPAIR_CHOICES, REVISE, REVISE_ADVICE
from .state import GameState


@dataclass
class TurnResult:
    n: int
    text: str
    entry: fmt.Entry
    action: str
    recalled: list = field(default_factory=list)
    distillate: Distillate | None = None
    repairs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, app: App, state: GameState, archive: MemoryArchive,
                 cfg: GameConfig | None = None):
        self.app = app
        self.state = state
        self.archive = archive
        self.cfg = cfg or GameConfig.from_env()
        self.archivist = Archivist(app.llm, state)

    # -- durability -----------------------------------------------------
    def catch_up(self) -> list[int]:
        """Archive any turn that was committed but never distilled.

        The entry is written to the transcript before the archivist runs, so
        a crash, a Ctrl-C or an Ollama timeout in between leaves a turn that
        exists in the story but not in memory — invisible to every later
        recall, and silently gone forever. Over a long campaign that is the
        kind of slow corruption that ruins it, so every session repairs the
        gap before playing on.

        State (clock, location, present cast) is only applied for a gap at
        the very end of the transcript. Replaying an older turn's state patch
        would drag the campaign back to where it used to be.
        """
        pending = self.state.unarchived()
        if not pending:
            return []
        latest = self.state.turn_count
        repaired = []
        for turn in pending:
            self._archive_turn(turn.n, turn.player_input, turn.narration,
                               apply_state=(turn.n == latest))
            repaired.append(turn.n)
        return repaired

    # -- the turn -------------------------------------------------------
    def play(self, raw_input: str) -> TurnResult:
        result = None
        for event in self.play_events(raw_input):
            if event["type"] == "turn":
                result = event["result"]
            elif event["type"] == "error":
                raise LLMError(event["message"])
        assert result is not None
        return result

    def play_events(self, raw_input: str) -> Iterator[dict]:
        cfg = self.cfg
        repaired = self.catch_up()
        if repaired:
            yield {"type": "repaired", "turns": repaired}
        action, from_menu = fmt.resolve_input(raw_input, self.state.pending_choices())
        if not action:
            yield {"type": "error", "message": "empty action"}
            return

        # --- recall -----------------------------------------------------
        summaries = [t.summary for t in self.state.recent_turns(3)]
        recalled = self.archive.recall(recall_query(action, summaries), cfg.recall_k)
        yield {"type": "recall", "action": action, "from_menu": from_menu,
               "memories": [h.as_dict() for h in recalled]}

        messages = [
            {"role": "system", "content": NARRATOR.format(
                min_words=cfg.min_words, max_words=cfg.max_words)},
            {"role": "user", "content":
                build_user_prompt(self.state, cfg, action, recalled)},
        ]

        # --- narrate ----------------------------------------------------
        parts: list[str] = []
        try:
            for piece in self.app.llm.chat(messages):
                parts.append(piece)
                yield {"type": "token", "text": piece}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        raw = strip_thinking("".join(parts))
        entry = fmt.parse(raw, min_words=cfg.min_words, max_words=cfg.max_words)
        repairs: list[str] = []

        # --- repair -----------------------------------------------------
        for _ in range(cfg.max_repairs):
            if entry.ok:
                break
            yield {"type": "revising", "problems": entry.problems}
            raw, entry, note = self._repair(messages, raw, entry)
            repairs.append(note)
            yield {"type": "revised", "text": entry.render(self.state.clock),
                   "problems": entry.problems}

        notes: list[str] = []
        if not entry.ok:
            notes.append("kept as written: " + "; ".join(entry.problems))
        entry = self._ensure_five(entry)

        # --- commit -----------------------------------------------------
        clock = self.state.clock
        text = entry.render(clock)
        n = self.state.add_turn(
            in_date=clock.date, in_time=clock.time, title=entry.title,
            location=self.state.location, player_input=action,
            narration=entry.narration, choices=entry.choices, words=entry.words,
        )
        yield {"type": "entry", "n": n, "text": text, "title": entry.title,
               "words": entry.words, "choices": entry.choices,
               "narration": entry.narration,
               "header": text.splitlines()[0] if text else ""}

        # --- remember ---------------------------------------------------
        yield {"type": "archiving"}
        distillate = self._archive_turn(n, action, entry.narration)
        if distillate.error:
            notes.append(f"archivist: {distillate.error}")
        arc = self._maybe_fold_arc()
        if arc:
            yield {"type": "arc", "text": arc}

        result = TurnResult(n=n, text=text, entry=entry, action=action,
                            recalled=recalled, distillate=distillate,
                            repairs=repairs, notes=notes)
        yield {"type": "turn", "result": result}

    # -- repair ---------------------------------------------------------
    def _repair(self, messages: list[dict], raw: str,
                entry: fmt.Entry) -> tuple[str, fmt.Entry, str]:
        cfg = self.cfg
        if len(entry.choices) != 5:
            instruction = REPAIR_CHOICES.format(count=len(entry.choices))
            follow_up = messages + [{"role": "assistant", "content": raw},
                                    {"role": "user", "content": instruction}]
            try:
                block = strip_thinking(self.app.llm.complete(follow_up, max_tokens=400))
            except LLMError:
                return raw, entry, "choice repair failed"
            merged = f"{entry.narration}\n\n{block}"
            rebuilt = fmt.parse(
                f"{_header_stub(entry.title)}\n\n{merged}",
                min_words=cfg.min_words, max_words=cfg.max_words,
                fallback_title=entry.title)
            return merged, rebuilt, f"rebuilt options ({len(entry.choices)} → {len(rebuilt.choices)})"

        direction = "longer" if entry.words < cfg.min_words else "shorter"
        instruction = REVISE.format(words=entry.words, min_words=cfg.min_words,
                                    max_words=cfg.max_words, direction=direction,
                                    advice=REVISE_ADVICE[direction])
        follow_up = messages + [{"role": "assistant", "content": raw},
                                {"role": "user", "content": instruction}]
        try:
            revised = strip_thinking(self.app.llm.complete(follow_up))
        except LLMError:
            return raw, entry, "length repair failed"
        candidate = fmt.parse(revised, min_words=cfg.min_words,
                              max_words=cfg.max_words, fallback_title=entry.title)
        if candidate.problems and not candidate.ok and len(candidate.choices) != 5:
            # A revision that lost the options is worse than the original.
            return raw, entry, f"revision rejected ({direction}, lost options)"
        return revised, candidate, f"{entry.words} → {candidate.words} words ({direction})"

    def _ensure_five(self, entry: fmt.Entry) -> fmt.Entry:
        """Last-resort padding, so the player is never handed a dead end."""
        if len(entry.choices) >= 5:
            return fmt.Entry(entry.title, entry.narration, entry.choices[:5],
                             entry.words, entry.problems)
        filler = ["💬 Say something", "👀 Look around", "🚶 Move on",
                  "🤔 Wait and watch", "✍️ Do something else entirely"]
        padded = list(entry.choices)
        for option in filler:
            if len(padded) >= 5:
                break
            padded.append(option)
        return fmt.Entry(entry.title, entry.narration, padded, entry.words,
                         entry.problems)

    # -- memory ---------------------------------------------------------
    def _archive_turn(self, n: int, action: str, narration: str,
                      *, apply_state: bool = True) -> Distillate:
        distillate = self.archivist.distil(n, action, narration)
        clock = self.state.clock
        turn = self.state.turn(n)
        when = (Clock.parse(turn.in_date, turn.in_time).pretty_date()
                if turn else clock.pretty_date())

        summary = distillate.summary or f"(turn {n}) {action}"
        self.state.set_summary(n, summary)
        self.archive.write("summary", n, summary, when=when)
        self.archive.write_many("fact", n, distillate.facts, when=when)
        self.archive.write_many("event", n, distillate.events, when=when)

        if not apply_state:
            return distillate

        if distillate.present:
            known = {c.slug for c in self.state.characters()}
            self.state.set_present([s for s in distillate.present if s in known])
        for rel in distillate.relationships:
            if self.state.character(rel["other"]):
                self.state.set_relationship(
                    rel["other"], label=rel.get("label"),
                    closeness=rel.get("closeness"), note=rel.get("note"), turn=n)
        for key, value in distillate.flags.items():
            self.state.set_flag(str(key)[:60], value, turn=n)
        if distillate.location:
            self.state.location = distillate.location
        if distillate.title:
            self.state.db.execute("UPDATE turns SET title=? WHERE n=? AND title=''",
                                  (distillate.title, n))
            self.state.db.commit()

        self.state.clock = clock.advance(distillate.minutes_elapsed)
        return distillate

    def _maybe_fold_arc(self) -> str:
        cfg = self.cfg
        last = self.state.last_arc_turn()
        total = self.state.turn_count
        if total - last < cfg.arc_every:
            return ""
        first, final = last + 1, total
        summaries = [t.summary for t in self.state.recent_turns(final - first + 1)
                     if t.n >= first]
        text = self.archivist.fold_arc(first, final, summaries)
        if not text:
            return ""
        self.state.add_arc(first, final, text)
        self.archive.write("arc", final, text, when=f"turns {first}-{final}")
        return text


def _header_stub(title: str) -> str:
    return f"📅 Monday, January 1st, 2000 — 🕛 12:00 PM — {title}"
