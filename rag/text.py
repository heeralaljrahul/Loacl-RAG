"""Small shared text helpers."""

from __future__ import annotations

SEP = " > "


def strip_title_prefix(title: str, heading: str) -> str:
    """Drop a leading breadcrumb component that just repeats the title.

    ``_title()`` picks a document's own first heading when it has one, which
    is usually the same string that starts every breadcrumb.  Left alone,
    every chunk embeds and displays "Handbook 2026 > Handbook 2026 > Leave" —
    wasted tokens in the embedding, a duplicated label in the UI, and noise
    in front of the cross-encoder.
    """
    if not heading:
        return ""
    parts = heading.split(SEP)
    if title and parts and parts[0].casefold() == title.casefold():
        parts = parts[1:]
    return SEP.join(parts)


def trail(title: str, heading: str) -> str:
    """The breadcrumb line placed in front of a chunk: ``Title > A > B``."""
    rest = strip_title_prefix(title, heading)
    return SEP.join(part for part in (title, rest) if part)
