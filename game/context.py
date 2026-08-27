"""Assembling the prompt for one turn.

The ordering here is the whole trick, and it is a budget problem.  Prompt
space is finite; the question is what gets it first.

  1. World state, sheets, relationships, flags — always, verbatim, whole.
     These are cheap and they are what the model gets wrong without.
  2. Arc summaries — the compressed spine of the campaign, oldest to newest.
     This is what lets turn 400 refer back to day one.
  3. Recalled memories — whatever hybrid search surfaced for this action.
  4. The last entry, verbatim, so prose style and immediate continuity hold.
  5. The player's action, last, closest to the model's own output.

Retrieval spends what is left over, never the other way round.  A model that
has forgotten Reina's height is broken; a model that failed to recall one
memory from turn 12 has merely missed a callback.
"""

from __future__ import annotations

from rag.store import Hit

from .config import GameConfig
from .state import GameState


def build_user_prompt(state: GameState, cfg: GameConfig, action: str,
                      recalled: list[Hit]) -> str:
    blocks: list[str] = []
    clock = state.clock

    # 1 — authoritative state
    lines = [f"Now: {clock.pretty_date()} — {clock.pretty_time()}"]
    if state.location:
        lines.append(f"Place: {state.location}")
    lines.append(f"Entry number: {state.turn_count + 1}")
    blocks.append(_section("WORLD STATE (authoritative)", "\n".join(lines)))

    protagonist = state.protagonist
    if protagonist:
        blocks.append(_section("PROTAGONIST", protagonist.block()))

    present = [c for c in state.characters(present_only=True) if not c.protagonist]
    if present:
        blocks.append(_section("PRESENT IN THE SCENE",
                               "\n\n".join(c.block() for c in present)))

    standing = state.relationships()
    if standing:
        rows = [
            f"{r['name'] or r['other']} — {r['label'] or 'known'} ({r['closeness']}/10)"
            + (f". {r['note']}" if r["note"] else "")
            for r in standing
        ]
        who = protagonist.name if protagonist else "the protagonist"
        blocks.append(_section(f"STANDING WITH {who.upper()}", "\n".join(rows)))

    flags = state.flags()
    if flags:
        blocks.append(_section("ONGOING",
                               "\n".join(f"{k}: {v}" for k, v in sorted(flags.items()))))

    # 2 — the spine
    arcs = state.arcs()[-cfg.arc_context:]
    if arcs:
        rows = [f"(turns {a['from_turn']}-{a['to_turn']}) {a['text']}" for a in arcs]
        blocks.append(_section("THE STORY SO FAR", "\n\n".join(rows)))

    # 3 — retrieved
    if recalled:
        rows, spent = [], 0
        for index, hit in enumerate(recalled, start=1):
            row = f"[m{index}] {hit.heading} — {hit.text.strip()}"
            if spent + len(row) > cfg.recall_chars and rows:
                break
            rows.append(row)
            spent += len(row)
        blocks.append(_section("RECALLED FROM EARLIER", "\n".join(rows)))

    # 4 — recent transcript
    recent = state.recent_turns(cfg.verbatim_turns + cfg.summary_turns)
    verbatim = recent[-cfg.verbatim_turns:] if cfg.verbatim_turns else []
    older = recent[:len(recent) - len(verbatim)]
    if older:
        rows = [f"Turn {t.n}: {t.summary or t.title or '(not yet distilled)'}"
                for t in older]
        blocks.append(_section("EARLIER THIS SESSION", "\n".join(rows)))
    for turn in verbatim:
        body = turn.narration.strip()
        if len(body) > cfg.narration_chars:
            body = "…" + body[-cfg.narration_chars:]
        header = f"Turn {turn.n} — the player chose: {turn.player_input}" \
            if turn.player_input else f"Turn {turn.n}"
        blocks.append(_section("THE PREVIOUS ENTRY (verbatim)", f"{header}\n\n{body}"))

    # 5 — the move
    blocks.append(_section("THE PLAYER'S ACTION", action.strip()))
    blocks.append("Write the next entry now.")
    return "\n\n".join(blocks)


def _section(title: str, body: str) -> str:
    return f"=== {title} ===\n{body}"
