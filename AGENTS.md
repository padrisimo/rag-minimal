# AGENTS.md

Notebook for future sessions on this repo. `MEMORY.md` holds the longer
rationale; this file is the short operational version. If they disagree, trust
the code.

## Commands

There is no lint, typecheck, formatter, or CI. Testing is the only check.

```bash
uv run python -m unittest           # primary: Python 3.12 from .venv
python3 -m unittest                 # must also pass: system Python 3.9
python3 -m unittest test_rag.TestSearch              # one class
python3 -m unittest test_rag.TestSearch.test_top_k_respected   # one test
```

Run **both** interpreters before calling anything done. `requires-python` is
`>=3.9` while `.python-version` pins 3.12, so the 3.9 run is the one that catches
the mistakes that matter.

## Constraints

- **Zero runtime dependencies.** Advertised in the README. Do not add one to
  dodge an inconvenience.
- **No comments or docstrings.** The only `#` in the tree is the shebang.
  Module and function names carry the meaning.
- **English in code and docs.** `README.es.md` mirrors `README.md`; update both,
  and keep the CLI output examples copied from real runs.
- No formatter is configured, so match surrounding style by hand.

## Layout

Split by pipeline stage, and the split is load-bearing:

```
text.py       tokenization; knows nothing about chunks
chunking.py   reads files, splits chunks; knows nothing about vectors
index.py      TF-IDF + cosine + persistence; the only place the three meet
commands.py   argparse front end (index / ask / chat)
```

Swapping TF-IDF for real embeddings means rewriting `index.py` alone. Keep vector
logic out of `chunking.py`.

Never add a root-level `rag_minimal.py`: it collides with the package, and Python
always resolves the package, leaving the module as dead code.

## Entrypoints

All four must keep working:

```bash
./cli.py ask "..."                 # executable; shebang needs uv on PATH
python3 cli.py ask "..."           # plain interpreter
python -m rag_minimal ask "..."    # as a module
uv run python -m rag_minimal ...   # uv
```

`./cli.py` must work from any directory, which is why `DEFAULT_DOCS` in
`commands.py` resolves against `Path(__file__).parent.parent`.

## Gotchas

- **`cli.py` must keep its exec bit and shebang.** `git clone` preserves it; a
  rewrite of the file does not. `chmod +x cli.py` if a test complains.
- **`TestExecutableEntryPoint` needs `uv` on `PATH`.** It invokes `./cli.py`, so
  it fails with `env: uv: No such file or directory` under a stripped PATH. Not
  a code bug.
- **Never let a test depend on `docs/index.json` existing.** It is gitignored and
  regenerable, so such a test passes locally and fails after a clean. Index
  inside the test. This already happened once.
- **Trust bytes over rendered file views.** A dict comprehension once shipped as
  `if c in idf` instead of `if t in idf`, emptying every vector so search
  returned nothing, while the file view showed plausible text.
  `dis.dis(fn.__code__)` settled it. When behaviour contradicts the source,
  disassemble rather than re-reading.
- **Grepping `"""` to prove the code is comment-free gives a false positive**:
  `STOPWORDS` is a multiline string literal. Use `ast` and look for a string
  `Expr` in first position.
- **`env -S` in the shebang** needs macOS 10.15+ or modern Linux. Fine locally;
  use a `#!/bin/sh` wrapper if a script must run on older POSIX.
- **System `git` is 2.15.0**, so `git branch --show-current` fails. Use
  `git rev-parse --abbrev-ref HEAD`.
- Spanish fixtures in the tokenizer tests are intentional; they pin the Spanish
  path behind the README's bilingual claim. Do not normalise them away.

## Tests

35 tests in one file, grouped by concern. The subprocess-driven groups share
`CLITestCase` and a `sample_index()` context manager — reuse it rather than
recreating fixtures.

## Known limitations

TF-IDF is lexical: no synonyms, no semantic matching. Stemming is a
suffix-stripper, not Porter. No reranking or MMR. Whole index in memory. See the
READMEs for the honest version; do not promise recall beyond that.