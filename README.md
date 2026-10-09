# rag-minimal

> English. La versión en español está en [README.es.md](README.es.md).

A minimal RAG in Python, **with no external dependencies and no LLM**: it indexes
`.txt`/`.md` files and returns the chunks most relevant to a question.

Retrieval is **TF-IDF + cosine similarity** (unigrams + bigrams, Spanish/English
stopwords, light stemming, accent-insensitive). It is the base you can later add
real embeddings or an LLM to, without touching anything else.

Stdlib only: runs on Python 3.9+, needs neither `pip install` nor a connection.

### Running it

`cli.py` is executable, and its shebang makes uv resolve the interpreter for
you, so this just works from any directory:

```bash
./cli.py ask "how often should I water the cactus"
```

Without uv, or if you prefer to be explicit:

```bash
python3 cli.py ask "how often should I water the cactus"     # any Python 3.9+
uv run python cli.py ask "how often should I water the cactus"  # uv, honours .python-version
```

Tests run the same way: `uv run python -m unittest`.

## Usage

```bash
./cli.py index docs/                    # index and save docs/index.json
./cli.py ask "how often should I water the cactus"
./cli.py ask "..." -k 5 --json          # 5 chunks, JSON output
./cli.py chat docs/                     # interactive session
```

Typical output:

```
$ ./cli.py index docs/
6 chunks · 242 terms · index written to docs/index.json

$ ./cli.py ask "how often should I water the cactus"
Question: how often should I water the cactus

[1] score=0.235  watering.md "Watering the cactus"
    A cactus needs very little water. Water it every two weeks in summer and once a month in winter. The soil must dry out completely between waterings: if the substrate stays wet, the root rots.

[2] score=0.073  watering.md "Substrates and drainage"
    Succulents store water in their leaves, so they tolerate drought. The substrate must drain well: a mix of perlite, coarse sand and peat in a 2:1:1 ratio.
```

Every hit carries `source`, `heading` and `text`, so you can always cite which
chunk the information came from.

Note: the CLI messages and the bundled sample docs in `docs/` are in English,
but the tokenizer is bilingual — it strips accents and filters both Spanish and
English stopwords, so Spanish documents are handled too (there is a Spanish test
for exactly that).

### Options

| Command | Option | Default | What it does |
|---|---|---|---|
| `index` | `--chunk-chars` | 600 | Maximum chunk size |
| `index` | `--overlap` | 120 | Overlap between consecutive chunks |
| `index` | `-o` | input dir | Folder where `index.json` is written |
| `ask` | `-k` | 3 | Number of chunks to return |
| `ask` | `--min-score` | 0.0 | Drop chunks scoring below this |
| `ask` | `--index` | `docs/index.json` | Explicit path to the index |
| `ask` | `--json` | — | JSON output, for integrations |
| `chat` | `-k` | 3 | Chunks per question |
| `chat` | `--min-score` | 0.0 | Drop chunks scoring below this |

### Interactive session

`chat` loads the index once and keeps it in memory, so follow-up questions are
instant:

```
$ ./cli.py chat docs/
Loaded 6 chunks from 242 terms (docs/index.json).
Type /help for commands, /quit to leave.

you> how often should I water the cactus
Question: how often should I water the cactus

[1] score=0.235  watering.md "Watering the cactus"
    A cactus needs very little water. Water it every two weeks in summer and once a month in winter. …

you> /sources
6 chunks from 2 files:
  meeting-notes.txt  (1 chunks)
  watering.md  (5 chunks)

you> /quit
Bye.
```

| Command | What it does |
|---|---|
| `/sources` | List the indexed documents, with chunk counts |
| `/top N` | Change how many chunks each question returns |
| `/min-score N` | Drop hits scoring below N (`0.00`–`1.00`) |
| `/help` | Show the command list |
| `/quit` | Leave (`Ctrl-D` and `Ctrl-C` work too) |

Anything that is not a command is treated as a question. Bad arguments report
the problem and leave the setting unchanged, rather than killing the session.

## As a library

```python
from pathlib import Path
from rag_minimal import RAG

bot = RAG.index(Path("docs"))
for hit in bot.search("what temperature do they need", top_k=3):
    print(f"{hit.score:.3f}  {hit.location()}")
    print(hit.text)

bot.save(Path("docs/index.json"))     # reindexing is cheap: just index again
```

## How it works

The code is split along the four pipeline stages, each in its own module:

| Stage | Module | Does |
|---|---|---|
| Tokenize | `text.py` | Lowercase, strip diacritics, drop stopwords, stem, add bigrams |
| Load + chunk | `chunking.py` | Read `.txt`/`.md`, split into overlapping chunks |
| Vectorize + search | `index.py` | TF-IDF, cosine similarity, `save()`/`load()` |
| Command line | `cli.py` | The `index` / `ask` / `chat` commands |

1. **Load** — walk the directory and read the `.txt`/`.md` files (UTF-8, with a
   fallback).
2. **Chunk** — group paragraphs up to `--chunk-chars` and overlap `--overlap`
   characters with the previous chunk, so an idea is not split in half. `#`
   headings do not go into the text: they are kept as a `heading` metadata field.
3. **Vectorize** — TF `(1 + log tf)` × IDF over unigrams and bigrams, L2
   normalized. Since both vectors are normalized, the dot product *is* the cosine.
4. **Search** — score every chunk and return the best `k`. It iterates the
   shorter of the two dicts, so it stays reasonably fast.

`index.json` stores only the chunk text (readable, diff-friendly); vectors and
IDF are recomputed on load.

### Entry points

The CLI can be started four ways, all equivalent:

```bash
./cli.py ask "cactus"          # executable, shebang resolves the interpreter
python3 cli.py ask "cactus"    # any Python 3.9+
python -m rag_minimal ask ...  # as a module
uv run python -m rag_minimal ...  # uv, honours .python-version
```

## Tests

```bash
python3 -m unittest -v
```

36 `unittest` tests: tokenization, stemming, chunking, L2 normalization,
ordering, persistence round-trip, plus the CLI, the interactive session, and
the executable entry point, all driven end-to-end through subprocess.

**Requires Python 3.9+.** Tested on 3.9.6 and 3.12.15.

## Limitations (on purpose, so you know when to scale up)

- **No synonyms, no semantic understanding**: "how often should I water" will not
  retrieve a text that only talks about "watering frequency". TF-IDF compares
  literal words.
- **Very simple stemming**: it unifies plurals and many suffixes (`plant`/`plants`
  do match, `watered` does not). It is Spanish-tuned, so it is approximate on
  English too. It is not Porter, not Snowball.
- **No reranking, no MMR**: it returns the best `k` even if they are nearly
  identical.
- **Disk round-trip**: for hundreds of thousands of chunks, loading the JSON
  starts to hurt; that is when you reach for SQLite or FAISS.

When the hits stop being good enough: the biggest single jump is swapping
`_vectorize` for real embeddings (sentence-transformers or Ollama) while keeping
the `RAG.search` interface; and to produce the final answer, join the hits as
context and call an LLM with them.

## Layout

```
rag_minimal/
  __init__.py     public API: RAG, Hit, Chunk, tokenize, chunk_text, ...
  __main__.py     enables `python -m rag_minimal`
  text.py         tokenization: folding, stemming, stopwords, bigrams
  chunking.py     document loading and chunking
  index.py        TF-IDF index, cosine search, persistence
  commands.py     command-line interface (index / ask / chat)
cli.py         thin executable wrapper: ./cli.py
test_rag.py    tests, using unittest
MEMORY.md      project notes: constraints, decisions, known traps
AGENTS.md      operational notes for coding agents
docs/          sample documents (.md and .txt)
pyproject.toml project metadata (no dependencies)
.python-version interpreter pinned by uv
```

The split is by responsibility, not size: `text.py` knows nothing about chunks,
`chunking.py` knows nothing about vectors, and `index.py` is the only place the
three stages meet. Swapping TF-IDF for real embeddings means rewriting
`index.py` alone.