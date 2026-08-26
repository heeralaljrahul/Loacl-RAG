"""FastAPI app: one page, one streaming endpoint.

Answers stream over server-sent events so tokens appear as the model
produces them.  The sources for a turn are sent *first*, before any text,
which is what makes the right-hand panel usable as a debugging tool: you can
see what was retrieved while the answer is still being written, and tell the
two distinct failures apart — the passage was never retrieved, or it was
retrieved and the model ignored it.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .app import App

WEB = Path(__file__).parent / "web"


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    history: list[dict] = []


class IngestRequest(BaseModel):
    paths: list[str] = []
    force: bool = False


def create_app(app: App | None = None) -> FastAPI:
    rag = app or App()
    api = FastAPI(title="Local RAG", version="1.0.0")

    @api.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @api.get("/api/stats")
    def stats():
        info = rag.stats()
        try:
            info["ollama"] = {"reachable": True, "loaded": rag.llm.running(),
                             "has_model": rag.llm.has_model(rag.cfg.llm_model)}
        except Exception as exc:  # noqa: BLE001
            info["ollama"] = {"reachable": False, "error": str(exc)}
        return info

    @api.post("/api/search")
    def search(request: AskRequest):
        result = rag.search(request.question, top_k=request.top_k)
        return {"sources": [h.as_dict() for h in result.hits],
                "timings": result.timings, "counts": result.counts}

    @api.post("/api/ask")
    def ask(request: AskRequest):
        def events():
            for event in rag.stream(request.question, top_k=request.top_k,
                                    history=request.history):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post("/api/ingest")
    def ingest(request: IngestRequest):
        if request.paths:
            report = rag.ingestor.ingest(request.paths, force=request.force)
        else:
            report = rag.ingestor.rescan(force=request.force)
        return {"summary": report.summary(), "added": report.added,
                "updated": report.updated, "unchanged": report.unchanged,
                "removed": report.removed, "chunks": report.chunks,
                "failures": report.failures}

    return api


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
