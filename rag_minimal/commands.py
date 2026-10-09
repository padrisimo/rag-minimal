import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence

from .chunking import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP
from .index import DEFAULT_TOP_K, RAG, Hit

PREVIEW_CHARS = 320
QUIT_COMMANDS = frozenset({"/quit", "/exit", "/q", "salir"})

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = PROJECT_DIR / "docs"


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _resolve_index(args: argparse.Namespace) -> Path:
    if args.index:
        return Path(args.index)
    root = Path(args.path)
    return root / "index.json" if root.is_dir() else root.parent / "index.json"


def _die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _load(path: Path) -> RAG:
    if not path.exists():
        _die(f"No index at {path}. Run first: python -m rag_minimal index <dir>")
    return RAG.load(path)


def _print_hits(hits: Sequence[Hit], question: str, top_k: int) -> None:
    print(f"Question: {question}\n")
    for position, hit in enumerate(hits, 1):
        title = f'"{hit.heading}"' if hit.heading else ""
        print(f"[{position}] score={hit.score:.3f}  {hit.source} {title}".rstrip())
        print(f"    {_preview(hit.text)}\n")
    if len(hits) == top_k:
        print(f"Showing the top {top_k}. Use -k for more.\n")


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.path)
    bot = RAG.index(root, chunk_chars=args.chunk_chars, overlap=args.overlap)
    if not len(bot):
        _die(f"No .txt/.md found in {root}. Empty index.")

    destination = Path(args.output) if args.output else root if root.is_dir() else root.parent
    out = bot.save(destination / "index.json")
    print(f"{len(bot)} chunks · {len(bot.idf)} terms · index written to {out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    bot = _load(_resolve_index(args))
    hits = bot.search(args.question, top_k=args.top, min_score=args.min_score)

    if args.json:
        print(json.dumps({"question": args.question, "hits": [h.to_dict() for h in hits]}, ensure_ascii=False, indent=2))
        return 0 if hits else 1

    if not hits:
        print(f"No matches for: {args.question}")
        print("Hint: without bigrams or synonyms, a question worded very differently from the text will find nothing.")
        return 1

    _print_hits(hits, args.question, args.top)
    return 0


def _help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  /sources        list the indexed documents, with chunk counts",
            "  /top N          change how many chunks each question returns",
            "  /min-score N    drop hits scoring below N (0.00 - 1.00)",
            "  /help           show this help",
            "  /quit           leave (Ctrl-D or Ctrl-C works too)",
            "",
            "Anything else is treated as a question.",
        ]
    )


def _sources(bot: RAG) -> str:
    counts: dict[str, int] = {}
    for chunk in bot.chunks:
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    lines = [f"{len(bot)} chunks from {len(counts)} files:"]
    lines += [f"  {source}  ({count} chunks)" for source, count in sorted(counts.items())]
    return "\n".join(lines)


def _number(raw: str, low: float, high: float, label: str) -> Optional[float]:
    try:
        value = float(raw)
    except ValueError:
        print(f"{label} must be a number, got: {raw!r}")
        return None
    if not low <= value <= high:
        print(f"{label} must be between {low:g} and {high:g}, got: {value:g}")
        return None
    return value


def cmd_chat(args: argparse.Namespace) -> int:
    index_path = _resolve_index(args)
    bot = _load(index_path)
    top_k, min_score = args.top, args.min_score

    print(f"Loaded {len(bot)} chunks from {len(bot.idf)} terms ({index_path}).")
    print(f"Type /help for commands, /quit to leave.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not line:
            continue
        if line.lower() in QUIT_COMMANDS:
            print("Bye.")
            return 0

        if line.startswith("/"):
            command, _, argument = line.partition(" ")
            command, argument = command.lower(), argument.strip()
            if command == "/help":
                print(_help_text())
            elif command == "/sources":
                print(_sources(bot))
            elif command == "/top":
                value = _number(argument, 1, 50, "top")
                if value is not None:
                    top_k = int(value)
                    print(f"top_k = {top_k}")
            elif command == "/min-score":
                value = _number(argument, 0.0, 1.0, "min-score")
                if value is not None:
                    min_score = value
                    print(f"min_score = {min_score:g}")
            else:
                print(f"Unknown command: {command} (try /help)")
            continue

        hits = bot.search(line, top_k=top_k, min_score=min_score)
        if not hits:
            print("No matches. Try different wording, or /sources to see what is indexed.\n")
        else:
            _print_hits(hits, line, top_k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RAG with TF-IDF: retrieves chunks from .txt/.md files")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("path", type=Path, help="directory with .txt/.md files (or a single file)")
    p_index.add_argument("-o", "--output", type=Path, help="directory where index.json is written")
    p_index.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="maximum chunk size")
    p_index.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="overlap between chunks")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="search for the chunks relevant to a question")
    p_ask.add_argument("question", help="the question, in natural language")
    p_ask.add_argument("path", nargs="?", type=Path, default=DEFAULT_DOCS, help="folder containing index.json")
    p_ask.add_argument("--index", type=Path, help="explicit path to the index")
    p_ask.add_argument("-k", "--top", type=int, default=DEFAULT_TOP_K, help="number of chunks to return")
    p_ask.add_argument("--min-score", type=float, default=0.0, help="drop chunks scoring below this")
    p_ask.add_argument("--json", action="store_true", help="JSON output")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive session: ask questions in a loop")
    p_chat.add_argument("path", nargs="?", type=Path, default=DEFAULT_DOCS, help="folder containing index.json")
    p_chat.add_argument("--index", type=Path, help="explicit path to the index")
    p_chat.add_argument("-k", "--top", type=int, default=DEFAULT_TOP_K, help="chunks per question (default 3)")
    p_chat.add_argument("--min-score", type=float, default=0.0, help="drop chunks scoring below this")
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)