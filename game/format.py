"""Parsing and enforcing the entry format.

An entry is three parts: a header, the narration, and exactly five numbered
choices.  All three are validated rather than trusted, because a local 9B
model will occasionally give you four choices, or 1,400 words, or bury the
options inside a paragraph — and a game that silently accepts a broken turn
gets worse every turn after it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .clock import Clock, render_header, strip_header

KEYCAPS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

# Matches "1️⃣ 🗡 Charge in", "1. Charge in", "1) Charge in", "**1.** Charge in"
_CHOICE = re.compile(
    r"^\s*(?:[*_]{0,2})(?P<n>[1-5])(?:️?⃣|[.)\]:]|️⃣)\s*(?:[*_]{0,2})\s*(?P<text>.+?)\s*$",
    re.MULTILINE,
)
_PROMPT_LINE = re.compile(r"^\s*(?:[*_#]{0,3})\s*what will you do\??\s*(?:[*_]{0,3})\s*$",
                          re.IGNORECASE | re.MULTILINE)


@dataclass
class Entry:
    title: str
    narration: str
    choices: list[str]
    words: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self, clock: Clock) -> str:
        lines = [render_header(clock, self.title), "", self.narration.strip(), "",
                 "**What will you do?**", ""]
        lines += [f"{KEYCAPS[i]} {choice}" for i, choice in enumerate(self.choices)]
        return "\n".join(lines)


def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def parse(text: str, *, min_words: int, max_words: int,
          fallback_title: str = "Continued") -> Entry:
    """Split a raw model response into its parts and list what is wrong."""
    from .clock import extract_title

    title = extract_title(text) or fallback_title
    body = strip_header(text)

    prompt = _PROMPT_LINE.search(body)
    if prompt:
        narration = body[:prompt.start()].strip()
        tail = body[prompt.end():]
    else:
        # No "What will you do?" line — fall back to the first numbered option.
        first = _CHOICE.search(body)
        narration = body[:first.start()].strip() if first else body.strip()
        tail = body[first.start():] if first else ""

    choices: list[str] = []
    seen: set[str] = set()
    for match in _CHOICE.finditer(tail):
        number = match.group("n")
        if number in seen:
            continue
        seen.add(number)
        choices.append(_clean_choice(match.group("text")))

    narration = _strip_trailing_choices(narration)
    words = count_words(narration)

    problems: list[str] = []
    if len(choices) != 5:
        problems.append(f"got {len(choices)} choices, need exactly 5")
    if words < min_words:
        problems.append(f"{words} words, needs at least {min_words}")
    elif words > max_words:
        problems.append(f"{words} words, needs at most {max_words}")
    if not narration:
        problems.append("no narration")

    return Entry(title, narration, choices, words, problems)


def _clean_choice(text: str) -> str:
    text = text.strip().strip("*_").strip()
    return re.sub(r"\s{2,}", " ", text)


def _strip_trailing_choices(narration: str) -> str:
    """Drop stray option lines that leaked above the prompt line."""
    lines = narration.splitlines()
    while lines and _CHOICE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def resolve_input(raw: str, pending: list[str]) -> tuple[str, bool]:
    """Turn "3" into the text of option 3.

    Returns (action_text, was_a_listed_choice).  Anything that is not a bare
    1-5 is a custom action and is passed through untouched — the format rules
    make custom input fully canonical, so the engine must never normalise or
    second-guess it.
    """
    stripped = raw.strip()
    if re.fullmatch(r"[1-5]", stripped) and pending:
        index = int(stripped) - 1
        if index < len(pending):
            return pending[index], True
    for index, keycap in enumerate(KEYCAPS):
        if stripped.startswith(keycap) and index < len(pending):
            return pending[index], True
    return stripped, False
