"""Web front end for playing a campaign.

Turns stream over server-sent events. The memories recalled for a turn are
sent *before* the first token, so the side panel fills in while the entry is
still being written — which is what makes it usable as a diagnostic. When an
entry contradicts something established forty turns ago, that panel tells
you whether the memory was never retrieved or was retrieved and ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .campaign import Campaign

WEB = Path(__file__).parent / "web"


class TurnRequest(BaseModel):
    action: str


class RecallRequest(BaseModel):
    query: str
    top_k: int = 8


def create_app(campaign: Campaign) -> FastAPI:
    api = FastAPI(title="Campaign", version="1.0.0")

    @api.get("/")
    def index():
        return FileResponse(WEB / "play.html")

    @api.get("/api/state")
    def state():
        status = campaign.status()
        last = campaign.state.turn(campaign.state.turn_count)
        status["choices"] = campaign.state.pending_choices()
        status["relationships"] = [
            {"name": r["name"] or r["other"], "label": r["label"],
             "closeness": r["closeness"], "note": r["note"]}
            for r in campaign.state.relationships()
        ]
        status["last_entry"] = _entry_text(campaign, last) if last else ""
        status["unarchived"] = len(campaign.state.unarchived())
        return status

    @api.get("/api/history")
    def history(limit: int = 12):
        return {"turns": [
            {"n": t.n, "title": t.title, "when": f"{t.in_date} {t.in_time}",
             "player_input": t.player_input, "summary": t.summary,
             "words": t.words, "text": _entry_text(campaign, t)}
            for t in campaign.state.recent_turns(limit)
        ]}

    @api.post("/api/turn")
    def turn(request: TurnRequest):
        def events():
            for event in campaign.engine.play_events(request.action):
                if event["type"] == "turn":
                    result = event["result"]
                    event = {"type": "done", "n": result.n,
                             "notes": result.notes, "repairs": result.repairs}
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/api/recall")
    def recall(request: RecallRequest):
        hits = campaign.archive.recall(request.query, request.top_k)
        return {"memories": [h.as_dict() for h in hits]}

    return api


def _entry_text(campaign: Campaign, turn) -> str:
    from .clock import Clock, render_header
    from .format import KEYCAPS

    clock = Clock.parse(turn.in_date, turn.in_time)
    lines = [render_header(clock, turn.title or "Continued"), "",
             turn.narration.strip()]
    if turn.choices:
        lines += ["", "**What will you do?**", ""]
        lines += [f"{KEYCAPS[i]} {c}" for i, c in enumerate(turn.choices)
                  if i < len(KEYCAPS)]
    return "\n".join(lines)
