# MEMORY

Working notes for this repository: decisions already made, and the traps worth
remembering next time something here is changed.

## What this project is

A minimal RAG over TF-IDF. Reads `.txt`/`.md`, returns the chunks most relevant
to a question. Stdlib only, no LLM. It exists as a base to later swap in real
embeddings, so the retrieval interface matters more than the retrieval quality.

## Hard constraints

- **Zero runtime dependencies.** This is a deliberate feature, not an accident.
  The README advertises it. Do not add a dependency to silence an inconvenience.
- **Python 3.9+ must keep working**, despite `.python-version` pinning 3.12 for
  uv. Both interpreters are available locally and both are tested.
- **No comments or docstrings in the code.** Explicit user instruction. The only
  exception already in the tree is the two-word shebang. Module and function
  names carry the meaning instead: `text` / `chunking` / `index` / `commands`.
- **English in code and docs**, with `README.es.md` kept as the Spanish
  translation. Both must be updated together when either changes.

## Environment

- `python3` on PATH is the system 3.9.6; uv provides 3.12.15 in `.venv`.
- Run tests with `uv run python -m unittest`, or `python3 -m unittest` to check
  3.9 compatibility. Do both before calling anything done.
- `uv` is installed via Homebrew, not the curl installer. `/opt/homebrew/bin` is
  already on PATH, so `~/.zshrc` was never modified.
- `~/.local/bin` is still not on PATH, so `python3.12` does not work directly.
  Not a problem: `uv run` does not need it.
- System `git` is 2.15.0 (2018), so `git branch --show-current` fails. Use
  `git rev-parse --abbrev-ref HEAD`.

## Architecture

The package is split by pipeline stage, and the split is load-bearing:

```
text.py       tokenization only; knows nothing about chunks
chunking.py   reads files, splits into chunks; knows nothing about vectors
index.py      TF-IDF + cosine search + persistence; the only place the three meet
commands.py   argparse front end (index / ask / chat)
__main__.py   3 lines, enables `python -m rag_minimal`
```

Root `cli.py` is a 4-line executable wrapper with a shebang, so `./cli.py` works.

**To swap TF-IDF for real embeddings, rewrite `index.py` and nothing else.**
That is the whole point of the layout. Do not let vector logic leak into
`chunking.py`.

**Do not add a root-level `rag_minimal.py`.** A file and a package with the same
base name collide: Python always resolves the package, so the module would be
silently dead code. This was tried and rejected.

## Entry points

All four are equivalent and all four are expected to keep working:

```bash
./cli.py ask "..."                       # shebang resolves the interpreter
python3 cli.py ask "..."                 # plain interpreter
python -m rag_minimal ask "..."          # as a module
uv run python -m rag_minimal ask "..."   # uv, honours .python-version
```

`./cli.py` must work from any directory, hence `DEFAULT_DOCS` in `commands.py`
resolving against `Path(__file__).parent.parent`, not the cwd.

## Traps

- **`cli.py` must stay executable and keep its shebang.** Two tests assert this.
  A `git clone` preserves the exec bit, but any rewrite of the file does not.
- **Tests must never depend on `docs/index.json` existing.** That file is
  gitignored and regenerable, so a test relying on it passes locally and fails
  after a clean. `test_runs_as_an_executable_from_any_directory` got this wrong
  and was fixed; index inside the test instead.
- **Read bytes, not rendered tool output, when a value looks wrong.** A dict
  comprehension once shipped as `if c in idf` instead of `if t in idf`, which
  silently emptied every vector and made search return nothing. The file view
  showed the correct-looking text while the bytes on disk were wrong.
  `dis.dis(fn.__code__)` settled it immediately. When behaviour contradicts the
  source, disassemble.
- **The Spanish-tuned stemmer applied to English is approximate but consistent.**
  `_stem` is pure and applied identically to query and document, so collisions
  are symmetric and retrieval still works. English plurals are pinned by a test.
- **Avoid `env -S` if a script must run on old POSIX.** macOS 10.15+ and modern
  Linux are fine. A `#!/bin/sh` wrapper is the portable fallback.
- **Docstrings were once lost between commits** (`poc v0` had 17 in `rag.py`,
  `wip v1` had 3). They are gone on purpose now, but be aware that large diffs
  between your own commits can lose content silently.
- **Do not grep for `\"\"\"` to prove the package is comment-free.** The
  `STOPWORDS` frozenset is written as a multiline string literal, so it matches.
  Use `ast` and check for a string `Expr` in first position instead.

## Tests

35 tests in `test_rag.py`, grouped by concern: tokenizer, chunker, vectors,
search, persistence, document loading, CLI, chat, executable entry point. All
the subprocess-driven ones share `CLITestCase` and a `sample_index()` context
manager.

Spanish fixtures in the tokenizer tests are intentional: they pin the Spanish
path and back the bilingual claim in the README. Do not "clean them up".

## Known limitations

Documented in both READMEs, and worth restating before promising recall:

- No synonyms, no semantic matching. "how often to water" will not find text
  that only says "watering frequency".
- Stemming is a suffix-stripper, not Porter or Snowball.
- No reranking or MMR, so near-duplicate chunks can fill the top-k.
- Whole index in memory, JSON on disk. Fine to hundreds of thousands of chunks,
  then switch to SQLite or FAISS.

## Docs

`README.md` is canonical; `README.es.md` mirrors it. When editing one, check the
other: they have drifted before (the sample output was stale after the CLI
messages were translated).

Captured examples of CLI output in the READMEs were copied from real runs. Keep
that discipline, or the docs will describe behaviour the program does not have.