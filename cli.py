#!/usr/bin/env -S uv run --script
from rag_minimal.commands import main

if __name__ == "__main__":
    raise SystemExit(main())