"""Long-term memory.

**Raw narration is never embedded.**  That is the rule that decides whether
a campaign still works at turn 300.

If you index the transcript, then within a few sessions your index is mostly
"Reina nodded", "the rain kept falling", "Min laughed" — a thousand chunks
of atmosphere that match every query weakly and the right one never wins.
Retrieval returns noise, and the more you play the worse it gets.

So an archivist pass reads the new turns and returns a distillate: one
summary, a handful of standalone facts, and any events that actually
happened.  Only the distillate is indexed.  A fact is written to stand on
its own — "Min asked Reina to come to the Thursday showcase" — because a
memory retrieved on turn 300 arrives with no surrounding context and has to
be intelligible cold.

Above that sits a second tier.  Every `arc_every` turns the summaries are
folded into one arc summary, so recalling "day one" at turn 400 does not
mean out-ranking four hundred sibling chunks — it means matching one of
forty arc summaries, and those are always injected anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.embed import Embedder
from rag.retrieve import Retriever
from rag.store import Hit, Store

KINDS = ("summary", "fact", "event", "arc", "lore")


@dataclass
class Memory:
    kind: str
    turn: int
    text: str
    when: str = ""

    @property
    def path(self) -> str:
        return f"memory://{self.kind}/{self.turn:05d}"


class MemoryArchive:
    def __init__(self, store: Store, embedder: Embedder, retriever: Retriever):
        self.store = store
        self.embedder = embedder
        self.retriever = retriever

    def write(self, kind: str, turn: int, text: str, *, when: str = "",
              index: int = 0) -> bool:
        """Index one distilled memory. Returns False for junk that is not
        worth an index slot."""
        text = " ".join(text.split())
        if len(text) < 12 or kind not in KINDS:
            return False

        path = f"memory://{kind}/{turn:05d}/{index:02d}"
        if self.store.document_by_path(path):
            return False

        title = f"Turn {turn}" if turn else "Lore"
        heading = f"{when} > {kind}" if when else kind
        doc_id = self.store.upsert_document(
            path=path, title=title, ext=".memory", size=len(text),
            mtime=float(turn), sha256=f"{kind}:{turn}:{index}",
        )
        vectors = self.embedder.embed_documents([f"{title} > {heading}\n\n{text}"])
        self.store.add_chunks(doc_id, [(0, heading, None, text)], vectors)
        return True

    def write_many(self, kind: str, turn: int, texts: list[str], *,
                   when: str = "") -> int:
        return sum(self.write(kind, turn, text, when=when, index=i)
                   for i, text in enumerate(texts))

    def recall(self, query: str, top_k: int) -> list[Hit]:
        return self.retriever.search(query, top_k=top_k).hits

    def count(self) -> dict[str, int]:
        rows = self.store.db.execute(
            "SELECT c.heading, COUNT(*) n FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "WHERE d.ext='.memory' GROUP BY c.heading")
        totals: dict[str, int] = {}
        for row in rows:
            kind = row["heading"].split(" > ")[-1]
            totals[kind] = totals.get(kind, 0) + int(row["n"])
        return totals


def recall_query(action: str, recent_summaries: list[str]) -> str:
    """What to search the archive with.

    The player's action alone is a thin query — "3" resolves to a one-line
    option that may share no vocabulary with the memory that matters.  Recent
    summaries are appended to carry the current situation into the search,
    which is what surfaces the callback to something forty turns ago.
    """
    parts = [action.strip()]
    parts += [s for s in recent_summaries[-2:] if s]
    return " ".join(parts)[:1200]
