"""Structure-aware chunking.

Two things here matter more than the exact chunk size:

1. **Heading breadcrumbs.**  Every chunk carries the trail of headings it
   sits under ("Handbook > Leave policy > Carry-over").  That trail is
   prepended to the text that gets embedded, so a chunk whose body says
   "up to five days may be carried over" still matches a query about
   *leave*, even though the word never appears in the body.

2. **Never split mid-paragraph if it can be helped.**  Paragraphs are the
   unit; chunks are built by accumulating them, and only an oversized
   paragraph is cut, at sentence boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Block
from .text import trail

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass
class Chunk:
    text: str
    heading: str
    page: int | None
    ord: int


def chunk_blocks(
    blocks: list[Block],
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 300,
    min_chunk_chars: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, text)

    buf: list[str] = []
    buf_len = 0
    buf_heading = ""
    buf_page: int | None = None

    def breadcrumb() -> str:
        return " > ".join(text for _, text in stack)

    def flush(carry: bool = True):
        nonlocal buf, buf_len, buf_heading, buf_page
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append(Chunk(body, buf_heading, buf_page, len(chunks)))
        tail = _tail(body, overlap_chars) if (carry and body) else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        buf_heading = breadcrumb()
        buf_page = None

    for block in blocks:
        for para in _paragraphs(block.text):
            match = _HEADING.match(para)
            if match:
                level, text = len(match.group(1)), match.group(2)
                # A heading starts a new chunk: content under it is a new
                # topic, and overlap must not bleed across the boundary.
                # Note this flush is unconditional — a section shorter than
                # min_chunk_chars is still text, and dropping it here would
                # silently lose whole subsections from the index.
                flush(carry=False)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                buf_heading = breadcrumb()
                continue

            if buf_page is None:
                buf_page = block.page
            if not buf_heading:
                buf_heading = breadcrumb()

            for piece in _split_oversized(para, chunk_chars):
                if buf_len and buf_len + len(piece) > chunk_chars:
                    flush()
                    if buf_page is None:
                        buf_page = block.page
                buf.append(piece)
                buf_len += len(piece) + 2

    flush(carry=False)
    return _merge_tiny(chunks, min_chunk_chars, chunk_chars)


def _merge_tiny(chunks: list[Chunk], min_chars: int, chunk_chars: int) -> list[Chunk]:
    """Fold an undersized chunk back into its predecessor.

    Only when the two share a heading — that case is a pure size artefact of
    where the paragraph boundaries fell.  A short chunk under a heading of
    its own is left alone: it is a real, precise passage, and merging it
    would blur the section it belongs to.  Nothing is ever discarded.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        if (out and len(chunk.text) < min_chars
                and out[-1].heading == chunk.heading
                and len(out[-1].text) + len(chunk.text) <= chunk_chars * 1.5):
            previous = out[-1]
            out[-1] = Chunk(
                f"{previous.text}\n\n{chunk.text}",
                previous.heading,
                previous.page if previous.page is not None else chunk.page,
                previous.ord,
            )
        else:
            out.append(chunk)
    for index, chunk in enumerate(out):
        chunk.ord = index
    return out


def embed_text(title: str, chunk: Chunk) -> str:
    """The string actually handed to the embedding model."""
    crumbs = trail(title, chunk.heading)
    return f"{crumbs}\n\n{chunk.text}" if crumbs else chunk.text


# --------------------------------------------------------------------------


def _paragraphs(text: str) -> list[str]:
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        # A heading glued to following lines has to be separated out so the
        # heading logic above can see it.
        lines = block.splitlines()
        current: list[str] = []
        for line in lines:
            if _HEADING.match(line.strip()):
                if current:
                    out.append("\n".join(current).strip())
                    current = []
                out.append(line.strip())
            else:
                current.append(line)
        if current and "\n".join(current).strip():
            out.append("\n".join(current).strip())
    return out


def _split_oversized(para: str, limit: int) -> list[str]:
    if len(para) <= limit:
        return [para]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(para):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current.strip())
            current = sentence
        elif len(sentence) > limit:
            # A single monstrous "sentence" (minified code, a table row).
            if current.strip():
                pieces.append(current.strip())
                current = ""
            for i in range(0, len(sentence), limit):
                pieces.append(sentence[i:i + limit])
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _tail(text: str, n: int) -> str:
    """Last ~n chars of text, snapped back to a sentence boundary."""
    if n <= 0 or len(text) <= n:
        return ""
    tail = text[-n:]
    match = _SENTENCE.search(tail)
    return tail[match.end():].strip() if match else tail.strip()
