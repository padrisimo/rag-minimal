"""CLI del RAG mínimo: `index` construye el índice, `ask` consulta.

    python cli.py index docs/                 # indexa y guarda docs/index.json
    python cli.py ask "cada cuánto regar"       # carga el índice y busca
    python cli.py ask "..." --json              # salida JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rag import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, RAG

PREVIEW_CHARS = 320


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.path)
    bot = RAG.index(root, chunk_chars=args.chunk_chars, overlap=args.overlap)
    if not len(bot):
        print(f"No encontré .txt/.md en {root}. Index vacío.", file=sys.stderr)
        return 1

    destination = Path(args.output) if args.output else root if root.is_dir() else root.parent
    out = bot.save(destination / "index.json")
    vocab = len(bot.idf)
    print(f"{len(bot)} chunks · {vocab} términos · índice en {out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index_path = Path(args.index) if args.index else Path(args.path) / "index.json" if Path(args.path).is_dir() else Path(args.path).parent / "index.json"

    if not index_path.exists():
        print(f"No existe el índice {index_path}. Ejecuta primero: python cli.py index <carpeta>", file=sys.stderr)
        return 1

    bot = RAG.load(index_path)
    hits = bot.search(args.question, top_k=args.top, min_score=args.min_score)

    if args.json:
        import json

        print(json.dumps({"question": args.question, "hits": [h.to_dict() for h in hits]}, ensure_ascii=False, indent=2))
        return 0 if hits else 1

    if not hits:
        print(f"Sin coincidencias para: {args.question}")
        print("Pista: sin bigrams ni sinónimos, una pregunta muy distinta al texto no encontrará nada.")
        return 1

    print(f'Pregunta: {args.question}\n')
    for position, hit in enumerate(hits, 1):
        title = f'"{hit.heading}"' if hit.heading else ""
        print(f"[{position}] score={hit.score:.3f}  {hit.source} {title}".rstrip())
        print(f"    {_preview(hit.text)}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG mínimo con TF-IDF: recupera fragmentos de .txt/.md")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="indexa un archivo o directorio")
    p_index.add_argument("path", type=Path, help="directorio con .txt/.md (o un solo archivo)")
    p_index.add_argument("-o", "--output", type=Path, help="directorio donde escribir index.json")
    p_index.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="tamaño máximo de chunk")
    p_index.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="solape entre chunks")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="busca fragmentos relevantes para una pregunta")
    p_ask.add_argument("question", help="la pregunta en lenguaje natural")
    p_ask.add_argument("path", nargs="?", type=Path, default=Path("docs"), help="carpeta con index.json")
    p_ask.add_argument("--index", type=Path, help="ruta explícita del índice")
    p_ask.add_argument("-k", "--top", type=int, default=DEFAULT_TOP_K, help="número de fragmentos")
    p_ask.add_argument("--min-score", type=float, default=0.0, help="score mínimo para considerar un hit")
    p_ask.add_argument("--json", action="store_true", help="salida en JSON")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())