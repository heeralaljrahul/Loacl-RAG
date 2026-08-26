#!/usr/bin/env python
"""Command line entry point.

    python cli.py ingest C:\\Users\\me\\Documents
    python cli.py ask "what does the warranty cover?"
    python cli.py chat
    python cli.py search "carry-over days"     # retrieval only, no model
    python cli.py stats
    python cli.py doctor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.app import App  # noqa: E402
from rag.config import CONFIG  # noqa: E402
from rag.llm import LLMError  # noqa: E402

DIM, BOLD, CYAN, YELLOW, GREEN, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"
)


def _no_color():
    global DIM, BOLD, CYAN, YELLOW, GREEN, RED, RESET
    DIM = BOLD = CYAN = YELLOW = GREEN = RED = RESET = ""


# -- commands --------------------------------------------------------------


def cmd_ingest(args, app: App) -> int:
    printer = (lambda line: print(line)) if not args.quiet else (lambda _: None)
    if args.paths:
        report = app.ingestor.ingest(args.paths, force=args.force,
                                     prune=not args.no_prune, progress=printer)
    else:
        print(f"{DIM}rescanning remembered folders…{RESET}")
        report = app.ingestor.rescan(force=args.force, prune=not args.no_prune,
                                     progress=printer)
    print(f"\n{GREEN}{report.summary()}{RESET}")
    for path, error in report.failures[:20]:
        print(f"{RED}  failed{RESET} {path}: {error}")
    if len(report.failures) > 20:
        print(f"{DIM}  … and {len(report.failures) - 20} more{RESET}")
    return 0


def cmd_reindex(args, app: App) -> int:
    print(f"{YELLOW}rebuilding the whole index from scratch…{RESET}")
    report = app.ingestor.reindex(progress=lambda line: print(line) if not args.quiet else None)
    print(f"\n{GREEN}{report.summary()}{RESET}")
    return 0


def cmd_search(args, app: App) -> int:
    result = app.search(" ".join(args.query), top_k=args.top_k)
    if not result.hits:
        print(f"{YELLOW}nothing matched.{RESET}")
        return 1
    for i, hit in enumerate(result.hits, start=1):
        print(f"\n{BOLD}[{i}] {hit.label}{RESET}  {DIM}score={hit.score:.3f} "
              f"dense=#{hit.dense_rank} bm25=#{hit.bm25_rank}{RESET}")
        print(f"{DIM}{hit.path}{RESET}")
        print(_indent(hit.text[:500]))
    print(f"\n{DIM}{_timing_line(result)}{RESET}")
    return 0


def cmd_ask(args, app: App) -> int:
    question = " ".join(args.query)
    return _answer(app, question, args)


def cmd_chat(args, app: App) -> int:
    print(f"{BOLD}Local RAG{RESET} — {app.cfg.llm_model} over "
          f"{app.stats()['chunks']} chunks. Ctrl-C or 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input(f"{CYAN}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit", ":q"):
            return 0
        if question.lower() in ("reset", "clear"):
            history.clear()
            print(f"{DIM}history cleared{RESET}\n")
            continue
        text = _answer(app, question, args, history=history, return_text=True)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text or ""})
        print()


def cmd_stats(args, app: App) -> int:
    info = app.stats()
    print(f"{BOLD}index{RESET}      {info['documents']} documents, "
          f"{info['chunks']} chunks, {info['dimensions']}-dim vectors")
    print(f"{BOLD}on disk{RESET}    {info['index_mb']} MB "
          f"(from {info['source_bytes'] / 1e6:.1f} MB of source files)")
    print(f"{BOLD}embedding{RESET}  {info['embed_model']}")
    print(f"{BOLD}reranker{RESET}   {info['rerank_model'] or 'disabled'}")
    print(f"{BOLD}llm{RESET}        {info['llm_model']} @ {app.cfg.ollama_url} "
          f"(num_ctx={app.cfg.num_ctx})")
    if info["by_ext"]:
        kinds = ", ".join(f"{ext or '?'}×{n}" for ext, n in info["by_ext"].items())
        print(f"{BOLD}file types{RESET} {kinds}")
    for root in info["roots"]:
        print(f"{DIM}  root: {root}{RESET}")
    return 0


def cmd_doctor(args, app: App) -> int:
    ok = True
    print(f"{BOLD}checking your setup{RESET}\n")

    # 1. Ollama
    try:
        tags = app.llm.tags()
        print(f"{GREEN}  ok{RESET}   Ollama reachable at {app.cfg.ollama_url} "
              f"({len(tags)} model{'s' if len(tags) != 1 else ''} installed)")
        if app.llm.has_model(app.cfg.llm_model):
            print(f"{GREEN}  ok{RESET}   model '{app.cfg.llm_model}' is installed")
        else:
            ok = False
            print(f"{RED}  FAIL{RESET} model '{app.cfg.llm_model}' not installed "
                  f"→ ollama pull {app.cfg.llm_model}")
            if tags:
                print(f"{DIM}         installed: {', '.join(tags[:8])}{RESET}")
    except LLMError as exc:
        ok = False
        print(f"{RED}  FAIL{RESET} {exc}")

    # 2. Where the model is actually running — the thing people get wrong.
    for model in app.llm.running():
        total = model.get("size", 0) or 0
        gpu = model.get("size_vram", 0) or 0
        share = (gpu / total * 100) if total else 0
        flag = GREEN if share > 98 else RED
        print(f"{flag}  {'ok' if share > 98 else 'WARN'}{RESET}   "
              f"{model.get('name')} loaded: {share:.0f}% on GPU "
              f"({gpu / 1e9:.1f} of {total / 1e9:.1f} GB)")
        if share <= 98:
            ok = False
            print(f"{DIM}         CPU spill costs ~10× speed. Lower RAG_NUM_CTX "
                  f"or use a smaller model.{RESET}")

    # 3. Embeddings
    try:
        vec = app.embedder.embed_query("smoke test")
        print(f"{GREEN}  ok{RESET}   embeddings: {app.embedder.name} → {len(vec)} dims")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"{RED}  FAIL{RESET} embedding model failed: {exc}")

    # 4. Reranker
    if app.cfg.rerank:
        try:
            app.reranker.score("smoke test", ["a passage"])
            print(f"{GREEN}  ok{RESET}   reranker: {app.cfg.rerank_model}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"{RED}  FAIL{RESET} reranker failed: {exc} "
                  f"(set RAG_RERANK=0 to run without it)")

    # 5. Index
    info = app.stats()
    if info["chunks"]:
        print(f"{GREEN}  ok{RESET}   index: {info['documents']} documents, "
              f"{info['chunks']} chunks")
    else:
        print(f"{YELLOW}  WARN{RESET} index is empty → python cli.py ingest <folder>")

    stored = info["embed_model"]
    if stored and stored != app.embedder.name:
        ok = False
        print(f"{RED}  FAIL{RESET} index built with '{stored}', config says "
              f"'{app.embedder.name}' → python cli.py reindex")

    print(f"\n{(GREEN + 'all good') if ok else (YELLOW + 'fix the items above')}{RESET}")
    return 0 if ok else 1


def cmd_serve(args, app: App) -> int:
    import uvicorn

    from rag.server import create_app

    print(f"{BOLD}Local RAG{RESET} → http://{args.host}:{args.port}")
    uvicorn.run(create_app(app), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_forget(args, app: App) -> int:
    removed = 0
    for raw in args.paths:
        target = str(Path(raw).expanduser().resolve())
        rows = app.store.db.execute(
            "SELECT id, path FROM documents WHERE path=? OR path LIKE ?",
            (target, target.rstrip("/\\") + "%"),
        ).fetchall()
        for row in rows:
            app.store.delete_document(int(row["id"]))
            removed += 1
            print(f"{DIM}  removed {row['path']}{RESET}")
    roots = [r for r in (app.store.get_meta("roots", []) or [])
             if r not in {str(Path(p).expanduser().resolve()) for p in args.paths}]
    app.store.set_meta("roots", roots)
    print(f"{GREEN}{removed} document(s) removed from the index{RESET}")
    return 0


# -- shared ----------------------------------------------------------------


def _answer(app: App, question: str, args, history=None, return_text=False):
    sources: list[dict] = []
    text = ""
    printed_header = False
    for event in app.stream(question, top_k=args.top_k, history=history):
        if event["type"] == "sources":
            sources = event["sources"]
            if not args.quiet:
                names = ", ".join(f"[{i}] {s['label']}" for i, s in enumerate(sources, 1))
                print(f"{DIM}{names or 'no sources'}{RESET}\n")
        elif event["type"] == "token":
            if not printed_header:
                printed_header = True
            sys.stdout.write(event["text"])
            sys.stdout.flush()
        elif event["type"] == "error":
            print(f"\n{RED}error: {event['message']}{RESET}")
            return "" if return_text else 1
        elif event["type"] == "done":
            text = event["text"]
    print()
    if args.show_sources and sources:
        print(f"\n{BOLD}sources{RESET}")
        for i, source in enumerate(sources, start=1):
            print(f"{DIM}[{i}]{RESET} {source['label']}  {DIM}{source['path']}{RESET}")
    return text if return_text else 0


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.strip().splitlines())


def _timing_line(result) -> str:
    timings = " ".join(f"{k.replace('_ms', '')}={v:.0f}ms" for k, v in result.timings.items())
    counts = " ".join(f"{k}={v}" for k, v in result.counts.items())
    return f"{timings}  |  {counts}"


# Shared flags are attached to the top level *and* to every subcommand, with
# SUPPRESS defaults so whichever side the user typed them on wins. Without
# this, `cli.py ask "..." --show-sources` — the way everyone actually types
# it — is an argparse error.
GLOBAL_DEFAULTS = {"top_k": None, "quiet": False, "no_color": False,
                   "show_sources": False}


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top-k", type=int, default=argparse.SUPPRESS,
                        help=f"chunks fed to the model (default {CONFIG.top_k})")
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="less chatter")
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--show-sources", action="store_true",
                        default=argparse.SUPPRESS,
                        help="print full source paths after the answer")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Local RAG over your own documents.",
        parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, **kwargs):
        return sub.add_parser(name, parents=[common], **kwargs)

    p = add("ingest", help="add or update folders and files")
    p.add_argument("paths", nargs="*", help="folders/files; omit to rescan remembered ones")
    p.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    p.add_argument("--no-prune", action="store_true", help="keep deleted files in the index")
    p.set_defaults(func=cmd_ingest)

    p = add("reindex", help="wipe and rebuild (after changing embed model)")
    p.set_defaults(func=cmd_reindex)

    p = add("ask", help="ask one question")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_ask)

    p = add("chat", help="interactive session with short-term memory")
    p.set_defaults(func=cmd_chat)

    p = add("search", help="retrieval only — no model, no generation")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_search)

    p = add("serve", help="web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=cmd_serve)

    p = add("stats", help="what is in the index")
    p.set_defaults(func=cmd_stats)

    p = add("doctor", help="check Ollama, GPU placement, models, index")
    p.set_defaults(func=cmd_doctor)

    p = add("forget", help="remove files or folders from the index")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_forget)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key, value in GLOBAL_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.no_color or not sys.stdout.isatty():
        _no_color()
    with App() as app:
        try:
            return args.func(args, app)
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 130
        except (RuntimeError, LLMError) as exc:
            print(f"{RED}error:{RESET} {exc}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
