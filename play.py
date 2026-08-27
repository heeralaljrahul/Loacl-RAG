#!/usr/bin/env python
"""Play a campaign.

    python play.py new reina --seed seeds/reina.json
    python play.py play reina
    python play.py status reina
    python play.py lore reina my_character_bible/
    python play.py recall reina "what happened on the first day"
    python play.py serve reina
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game.campaign import Campaign, campaign_dir, list_campaigns  # noqa: E402
from game.format import KEYCAPS  # noqa: E402
from rag.llm import LLMError  # noqa: E402

DIM, BOLD, CYAN, YELLOW, GREEN, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m")


def _no_color():
    global DIM, BOLD, CYAN, YELLOW, GREEN, RED, RESET
    DIM = BOLD = CYAN = YELLOW = GREEN = RED = RESET = ""


def cmd_new(args, _) -> int:
    target = campaign_dir(args.slug)
    if (target / "index.sqlite3").exists() and not args.force:
        print(f"{RED}campaign '{args.slug}' already exists{RESET} — "
              f"--force to re-seed it, or pick another name")
        return 1
    with Campaign(args.slug) as campaign:
        counts = campaign.seed(args.seed)
        print(f"{GREEN}seeded '{args.slug}'{RESET}: " +
              ", ".join(f"{v} {k}" for k, v in counts.items() if v))
        if args.opening:
            text = Path(args.opening).read_text(encoding="utf-8")
            n = campaign.open_with(text, title=args.title or "Opening")
            print(f"{DIM}opening prose stored as turn {n} and distilled into memory{RESET}")
        print(f"{DIM}{target}{RESET}")
        print(f"\nnext: python play.py play {args.slug}")
    return 0


def cmd_play(args, campaign: Campaign) -> int:
    status = campaign.status()
    print(f"{BOLD}{status['title']}{RESET}  {DIM}turn {status['turns']} · "
          f"{status['when']}{RESET}")
    if status["location"]:
        print(f"{DIM}{status['location']}{RESET}")

    pending = campaign.state.pending_choices()
    if pending:
        _print_choices(pending)
    elif status["turns"] == 0:
        print(f"{DIM}\nno entries yet — describe how the scene opens, or type "
              f"'begin'{RESET}")
    print(f"{DIM}\ntype 1-5, or write anything you like. 'quit' to stop.{RESET}\n")

    while True:
        try:
            raw = input(f"{CYAN}› {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw.lower() in ("quit", "exit", ":q"):
            return 0
        if raw.lower() == "status":
            _print_status(campaign.status())
            continue
        print()
        if _run_turn(campaign, raw, args) != 0:
            return 1
    return 0


def cmd_turn(args, campaign: Campaign) -> int:
    return _run_turn(campaign, " ".join(args.action), args)


def _run_turn(campaign: Campaign, raw: str, args) -> int:
    streamed: list[str] = []
    revised = False
    for event in campaign.engine.play_events(raw):
        kind = event["type"]
        if kind == "repaired":
            print(f"{DIM}· recovered memory for turn(s) "
                  f"{', '.join(str(n) for n in event['turns'])}{RESET}")
        elif kind == "recall" and not args.quiet:
            memories = event["memories"]
            if memories:
                names = " · ".join(_memory_tag(m) for m in memories[:6])
                print(f"{DIM}recalled: {names}{RESET}\n")
        elif kind == "token":
            sys.stdout.write(event["text"])
            sys.stdout.flush()
            streamed.append(event["text"])
        elif kind == "revising":
            revised = True
            print(f"\n\n{YELLOW}· revising: {'; '.join(event['problems'])}{RESET}\n")
        elif kind == "entry":
            # The text was already streamed above, so reprinting it wholesale
            # would show the player 850 words twice. Only the narration is
            # compared: a header rebuilt from the clock is a one-line
            # correction and does not justify a full reprint.
            body_matches = event["narration"].strip() in "".join(streamed)
            if revised or not body_matches:
                print("\n" + DIM + "─" * 60 + RESET)
                print("\n" + event["text"] + "\n")
            else:
                print()
                if event["header"] not in "".join(streamed):
                    print(f"{DIM}· {event['header']}{RESET}")
            print(f"{DIM}turn {event['n']} · {event['words']} words{RESET}")
        elif kind == "archiving" and not args.quiet:
            print(f"{DIM}· archiving…{RESET}")
        elif kind == "arc":
            print(f"{DIM}· arc summary written{RESET}")
        elif kind == "error":
            print(f"\n{RED}error: {event['message']}{RESET}")
            return 1
        elif kind == "turn":
            result = event["result"]
            for note in result.notes:
                print(f"{YELLOW}· {note}{RESET}")
            if result.distillate and not args.quiet:
                d = result.distillate
                bits = []
                if d.facts:
                    bits.append(f"{len(d.facts)} facts")
                if d.events:
                    bits.append(f"{len(d.events)} events")
                if d.relationships:
                    bits.append(f"{len(d.relationships)} relationships")
                if bits:
                    print(f"{DIM}· remembered: {', '.join(bits)}{RESET}")
            print()
    return 0


def cmd_status(args, campaign: Campaign) -> int:
    _print_status(campaign.status())
    return 0


def cmd_lore(args, campaign: Campaign) -> int:
    print(campaign.ingest_lore(args.paths))
    return 0


def cmd_recall(args, campaign: Campaign) -> int:
    hits = campaign.archive.recall(" ".join(args.query), args.top_k or 8)
    if not hits:
        print(f"{YELLOW}nothing recalled{RESET}")
        return 1
    for index, hit in enumerate(hits, start=1):
        print(f"\n{BOLD}[{index}] {hit.heading}{RESET} {DIM}({hit.label}, "
              f"score {hit.score:.2f}){RESET}")
        print("    " + hit.text.strip()[:400])
    print()
    return 0


def cmd_history(args, campaign: Campaign) -> int:
    turns = campaign.state.recent_turns(args.limit)
    for turn in turns:
        print(f"{BOLD}turn {turn.n}{RESET} {DIM}{turn.in_date} {turn.in_time} · "
              f"{turn.words}w{RESET}  {turn.title}")
        if turn.player_input:
            print(f"    {CYAN}› {turn.player_input[:100]}{RESET}")
        if turn.summary:
            print(f"    {DIM}{turn.summary[:160]}{RESET}")
    return 0


def cmd_list(args, _) -> int:
    names = list_campaigns()
    if not names:
        print(f"{DIM}no campaigns yet — python play.py new <name> "
              f"--seed seeds/reina.json{RESET}")
        return 0
    for name in names:
        with Campaign(name) as campaign:
            status = campaign.status()
            print(f"{BOLD}{name}{RESET}  {DIM}turn {status['turns']} · "
                  f"{status['when']}{RESET}")
    return 0


def cmd_serve(args, campaign: Campaign) -> int:
    import uvicorn

    from game.server import create_app

    print(f"{BOLD}{campaign.status()['title']}{RESET} → http://{args.host}:{args.port}")
    uvicorn.run(create_app(campaign), host=args.host, port=args.port,
                log_level="warning")
    return 0


# -- helpers ---------------------------------------------------------------


def _memory_tag(memory: dict) -> str:
    kind = memory["heading"].split(" > ")[-1]
    return f"{kind}·{memory['title'].lower()}"


def _print_choices(choices: list[str]):
    print(f"\n{BOLD}What will you do?{RESET}")
    for index, choice in enumerate(choices):
        print(f"{KEYCAPS[index] if index < len(KEYCAPS) else index + 1} {choice}")


def _print_status(status: dict):
    print(f"{BOLD}{status['title']}{RESET} ({status['slug']})")
    print(f"  {status['when']}")
    if status["location"]:
        print(f"  {status['location']}")
    print(f"  turn {status['turns']} · {status['arcs']} arcs · "
          f"{status['characters']} characters")
    if status["present"]:
        print(f"  present: {', '.join(status['present'])}")
    if status["memories"]:
        print("  memory: " + ", ".join(f"{v} {k}" for k, v in
                                       sorted(status["memories"].items())))
    for key, value in status["flags"].items():
        print(f"  {DIM}{key}: {value}{RESET}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--top-k", type=int, default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(prog="play.py", parents=[common],
                                     description="Play a long-memory RPG campaign.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, **kwargs):
        return sub.add_parser(name, parents=[common], **kwargs)

    p = add("new", help="create and seed a campaign")
    p.add_argument("slug")
    p.add_argument("--seed", required=True, help="seed JSON, e.g. seeds/reina.json")
    p.add_argument("--opening", help="text file of prose to store as turn 1")
    p.add_argument("--title", help="title for that opening entry")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_new, needs_campaign=False)

    p = add("play", help="interactive session")
    p.add_argument("slug")
    p.set_defaults(func=cmd_play, needs_campaign=True)

    p = add("turn", help="play exactly one turn and exit")
    p.add_argument("slug")
    p.add_argument("action", nargs="+")
    p.set_defaults(func=cmd_turn, needs_campaign=True)

    p = add("status", help="clock, cast, relationships, memory counts")
    p.add_argument("slug")
    p.set_defaults(func=cmd_status, needs_campaign=True)

    p = add("history", help="recent turns")
    p.add_argument("slug")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_history, needs_campaign=True)

    p = add("recall", help="search the campaign's memory directly")
    p.add_argument("slug")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_recall, needs_campaign=True)

    p = add("lore", help="index reference documents into this campaign")
    p.add_argument("slug")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_lore, needs_campaign=True)

    p = add("serve", help="play in the browser")
    p.add_argument("slug")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.set_defaults(func=cmd_serve, needs_campaign=True)

    p = add("list", help="all campaigns")
    p.set_defaults(func=cmd_list, needs_campaign=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key, value in {"quiet": False, "no_color": False, "top_k": None}.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.no_color or not sys.stdout.isatty():
        _no_color()

    if not args.needs_campaign:
        return args.func(args, None)

    if args.slug not in list_campaigns():
        print(f"{RED}no campaign '{args.slug}'{RESET}. Existing: "
              f"{', '.join(list_campaigns()) or 'none'}")
        return 1
    with Campaign(args.slug) as campaign:
        try:
            return args.func(args, campaign)
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 130
        except LLMError as exc:
            print(f"{RED}error:{RESET} {exc}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
