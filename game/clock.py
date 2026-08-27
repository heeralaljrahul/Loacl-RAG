"""The in-story clock.

The date and time are *state*, not prose.  A model asked to keep track of
the date across 300 entries will drift — it will repeat a Tuesday, skip a
week, or quietly reset to the opening scene.  So the engine owns the clock,
advances it, formats the header, and repairs whatever the model writes to
match.  The model chooses only the mini-title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


@dataclass
class Clock:
    when: datetime

    @classmethod
    def parse(cls, date: str, time: str) -> "Clock":
        return cls(datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M"))

    @property
    def date(self) -> str:
        return self.when.strftime("%Y-%m-%d")

    @property
    def time(self) -> str:
        return self.when.strftime("%H:%M")

    @property
    def weekday(self) -> str:
        return WEEKDAYS[self.when.weekday()]

    def pretty_date(self) -> str:
        return (f"{self.weekday}, {MONTHS[self.when.month - 1]} "
                f"{ordinal(self.when.day)}, {self.when.year}")

    def pretty_time(self) -> str:
        hour = self.when.hour % 12 or 12
        return f"{hour}:{self.when.minute:02d} {'AM' if self.when.hour < 12 else 'PM'}"

    def advance(self, minutes: int) -> "Clock":
        return Clock(self.when + timedelta(minutes=max(0, int(minutes))))

    def jump_to(self, date: str | None, time: str | None) -> "Clock":
        when = self.when
        if date:
            try:
                parsed = datetime.strptime(date, "%Y-%m-%d")
                when = when.replace(year=parsed.year, month=parsed.month, day=parsed.day)
            except ValueError:
                pass
        if time:
            try:
                parsed = datetime.strptime(time, "%H:%M")
                when = when.replace(hour=parsed.hour, minute=parsed.minute)
            except ValueError:
                pass
        return Clock(when)


CLOCK_EMOJI = {range(0, 5): "🌙", range(5, 8): "🌅", range(8, 12): "🕘",
               range(12, 17): "🕛", range(17, 20): "🌆", range(20, 24): "🌙"}


def hour_emoji(hour: int) -> str:
    for span, emoji in CLOCK_EMOJI.items():
        if hour in span:
            return emoji
    return "🕛"


HEADER_RE = re.compile(
    r"^\s*(?:📅\s*)?(?P<date>[^—\n]+?)\s*—\s*(?:[^\s—]*\s*)?(?P<time>\d{1,2}:\d{2}\s*[APap]\.?[Mm]\.?)"
    r"\s*—\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)


def render_header(clock: Clock, title: str) -> str:
    """Build the canonical header line from tracked state."""
    return (f"📅 {clock.pretty_date()} — {hour_emoji(clock.when.hour)} "
            f"{clock.pretty_time()} — {title.strip()}")


def extract_title(text: str) -> str | None:
    """Pull just the mini-title out of whatever header the model wrote.

    Everything else in the header is rebuilt from the clock, so a model that
    hallucinates the wrong Tuesday costs nothing.
    """
    match = HEADER_RE.search(text.split("\n\n")[0] if "\n\n" in text else text)
    if not match:
        return None
    title = match.group("title").strip().strip("*_ ")
    return title or None


def strip_header(text: str) -> str:
    """Remove the model's header line, leaving the narration body."""
    lines = text.lstrip().splitlines()
    for index, line in enumerate(lines[:3]):
        if HEADER_RE.match(line.strip()):
            return "\n".join(lines[index + 1:]).lstrip()
    return text.lstrip()
