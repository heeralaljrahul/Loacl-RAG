"""A scripted stand-in for the model, so the engine can be tested offline.

It is not a simulation of quality — it produces deterministic, deliberately
dull text. What it does faithfully reproduce is the *shape* of what a real
model returns, including the ways real models get it wrong: a word count
that misses the window, four options instead of five, JSON wrapped in a
markdown fence. That is what the engine has to survive, so that is what the
tests exercise.

Enabled with ``RAG_LLM_BACKEND=fake``. Never used in normal play.
"""

from __future__ import annotations

import json
import re

WORDS = ("rain light window chalk corridor quiet footsteps paper desk warm "
         "shoulder glance breath distant chair sunlight murmur ink page "
         "shadow doorway echo").split()

#: "argument not supplied", so that ``archivist=None`` can mean "go back to
#: the default payload". Using None for both is how a test ends up quietly
#: reusing the previous test's script and passing for the wrong reason.
UNSET = object()

_DEFAULTS = {"words": 850, "choices": 5, "archivist": None}
_state = dict(_DEFAULTS, calls=0)


def configure(*, words=UNSET, choices=UNSET, archivist=UNSET):
    """Steer the fake: entry length, how many options, archivist payload.

    Pass ``archivist=None`` to go back to the default payload; omit the
    argument to leave it as it is.
    """
    if words is not UNSET:
        _state["words"] = words
    if choices is not UNSET:
        _state["choices"] = choices
    if archivist is not UNSET:
        _state["archivist"] = archivist
    _state["calls"] = 0


def reset():
    """Back to a known state. Every test should start here."""
    _state.update(_DEFAULTS, calls=0)


def calls() -> int:
    return _state["calls"]


def fake_response(messages: list[dict]) -> str:
    _state["calls"] += 1
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    last_user = next((m["content"] for m in reversed(messages)
                      if m["role"] == "user"), "")

    if "record-keeper" in system:
        return _archivist(last_user)
    if last_user.startswith("Fold these turn summaries"):
        return "The class went about its week. Nothing was resolved about Thursday."
    if "Rewrite only the closing block" in last_user:
        return _choice_block(5)
    if last_user.startswith("That entry was"):
        target = _target_from(last_user)
        return _entry(target, 5)
    return _entry(_state["words"], _state["choices"])


# --------------------------------------------------------------------------


def _target_from(instruction: str) -> int:
    numbers = re.findall(r"between (\d+) and (\d+)", instruction)
    if numbers:
        low, high = int(numbers[0][0]), int(numbers[0][1])
        return (low + high) // 2
    return 850


def _entry(words: int, choices: int) -> str:
    body = " ".join(WORDS[i % len(WORDS)] for i in range(words))
    # Give it sentence shape so the word counter meets realistic input.
    body = re.sub(r"((?:\w+ ){11})", r"\1. ", body).replace(" .", ".")
    return (
        "📅 Monday, April 14th, 2025 — 🕛 3:35 PM — 🧹 A Quiet Room\n\n"
        f"{body}\n\n"
        "**What will you do?**\n\n" + _choice_block(choices).split("\n", 2)[-1]
    )


def _choice_block(count: int) -> str:
    keycaps = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    options = ["🔔 Wait for the bell", "🧹 Take the broom anyway",
               "🍱 Suggest going somewhere", "💫 Ask Min about Thursday",
               "🚪 Head for the genkan"]
    lines = [f"{keycaps[i]} {options[i]}" for i in range(min(count, 5))]
    return "**What will you do?**\n\n" + "\n".join(lines)


def _archivist(user: str) -> str:
    scripted = _state["archivist"]
    if isinstance(scripted, str):
        return scripted
    payload = scripted or {
        "summary": "The class cleaned around Reina while she read at the podium.",
        "facts": ["Mirajane and Louis refuse to let Reina do cleaning duty."],
        "events": ["Cleaning duty was carried out by Group Four."],
        "present": ["reina", "mirajane", "louis"],
        "relationships": [],
        "flags": {},
        "minutes_elapsed": 25,
        "location": "Class 1-A",
        "title": "The Podium",
    }
    # Real models fence their JSON about half the time; make the tests eat it.
    return "```json\n" + json.dumps(payload) + "\n```"
