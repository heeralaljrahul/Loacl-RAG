"""The archivist pass: narration in, structured memory out.

Runs as its own LLM call with its own prompt, because the qualities that
make good narration — warmth, implication, atmosphere — are exactly the ones
that make bad records.  This call wants a cold literal reader.

Everything it returns is treated as untrusted: a local 9B model will
occasionally emit a fenced block, a trailing comma, prose before the JSON,
or a string where a list belongs.  A malformed archivist response must
degrade to "this turn produced no memory", never to a crash that loses the
turn the player just played.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rag.llm import LLMError, Ollama, strip_thinking

from .prompts import ARC, ARCHIVIST
from .state import GameState


@dataclass
class Distillate:
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    present: list[str] | None = None
    relationships: list[dict] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    minutes_elapsed: int = 10
    location: str = ""
    title: str = ""
    error: str = ""

    @property
    def empty(self) -> bool:
        return not (self.summary or self.facts or self.events)


class Archivist:
    def __init__(self, llm: Ollama, state: GameState):
        self.llm = llm
        self.state = state

    def distil(self, turn_number: int, player_input: str, narration: str) -> Distillate:
        roster = "\n".join(
            f"- {c.slug}: {c.name}" + (f" ({c.role})" if c.role else "")
            for c in self.state.characters()
        ) or "- (none recorded yet)"

        messages = [
            {"role": "system", "content": ARCHIVIST.format(roster=roster)},
            {"role": "user", "content":
                f"TURN {turn_number}\nPLAYER ACTION: {player_input}\n\nNARRATION:\n{narration}"},
        ]
        try:
            raw = self.llm.complete(messages, temperature=0.0, max_tokens=900)
        except LLMError as exc:
            return Distillate(error=str(exc))
        return parse_distillate(raw)

    def fold_arc(self, first: int, last: int, summaries: list[str]) -> str:
        body = "\n".join(f"{i}. {s}" for i, s in enumerate(summaries, start=first))
        try:
            text = self.llm.complete(
                [{"role": "user",
                  "content": ARC.format(first=first, last=last, summaries=body)}],
                temperature=0.2, max_tokens=500)
        except LLMError:
            return ""
        return strip_thinking(text).strip()


def parse_distillate(raw: str) -> Distillate:
    payload = _extract_json(strip_thinking(raw))
    if payload is None:
        return Distillate(error="archivist did not return usable JSON")

    return Distillate(
        summary=_text(payload.get("summary")),
        facts=_string_list(payload.get("facts")),
        events=_string_list(payload.get("events")),
        present=_slug_list(payload["present"]) if "present" in payload else None,
        relationships=_relationships(payload.get("relationships")),
        flags=payload.get("flags") if isinstance(payload.get("flags"), dict) else {},
        minutes_elapsed=_minutes(payload.get("minutes_elapsed")),
        location=_text(payload.get("location")),
        title=_text(payload.get("title")),
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    # Walk to the matching brace: models like to add prose after the object.
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                    except json.JSONDecodeError:
                        return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _text(value) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = _text(item) if not isinstance(item, dict) else _text(item.get("text", ""))
        if len(text) >= 8:
            out.append(text)
    return out[:12]


def _slug_list(value) -> list[str]:
    """Character slugs, not prose.

    These go through a different filter from facts on purpose: the
    minimum-length check that keeps junk out of the fact list would throw
    away every short slug, and quietly dropping "reina" from the present
    cast removes the protagonist's own sheet from the next prompt.
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        slug = _text(item).lower().replace(" ", "_")
        if slug and re.fullmatch(r"[a-z0-9_\-]{2,40}", slug):
            out.append(slug)
    return out[:20]


def _relationships(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict) or not item.get("other"):
            continue
        entry = {"other": _text(item["other"]).lower().replace(" ", "_")}
        if isinstance(item.get("closeness"), (int, float)):
            entry["closeness"] = max(0, min(10, int(item["closeness"])))
        if item.get("label"):
            entry["label"] = _text(item["label"])[:40]
        if item.get("note"):
            entry["note"] = _text(item["note"])[:200]
        out.append(entry)
    return out[:10]


def _minutes(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 10
    return max(0, min(int(value), 60 * 24 * 30))
