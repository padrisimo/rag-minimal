#!/usr/bin/env -S uv run --script
from src.commands import main

if __name__ == "__main__":
    raise SystemExit(main())