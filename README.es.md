# rag-minimal

> Español. The English version is in [README.md](README.md).

RAG mínimo en Python, **sin dependencias externas y sin LLM**: indexa archivos
`.txt`/`.md` y devuelve los fragmentos más relevantes para una pregunta.

Retirada por **TF-IDF + similitud coseno** (unigrams + bigrams, stopwords
español/inglés, stemming ligero, sin acentos). Es la base sobre la que luego
puedes añadir embeddings reales o un LLM sin tocar nada más.

Solo stdlib: funciona con Python 3.9+ y no necesita `pip install` ni conexión.

### Cómo ejecutarlo

`cli.py` es ejecutable, y su shebang hace que uv resuelva el intérprete, así que
esto funciona desde cualquier directorio:

```bash
./cli.py ask "how often should I water the cactus"
```

Sin uv, o si prefieres ser explícito:

```bash
python3 cli.py ask "how often should I water the cactus"     # cualquier Python 3.9+
uv run python cli.py ask "how often should I water the cactus"  # uv, respeta .python-version
```

Los tests se ejecutan igual: `uv run python -m unittest`.

## Uso

```bash
./cli.py index docs/                    # indexa y guarda docs/index.json
./cli.py ask "how often should I water the cactus"
./cli.py ask "..." -k 5 --json          # 5 fragmentos, salida JSON
./cli.py chat docs/                     # sesión interactiva
```

Salida típica:

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

Cada hit trae `source`, `heading` y `text`, así que siempre puedes citar de qué
fragmento salió la información.

Nota: los mensajes de la CLI y los documentos de ejemplo de `docs/` están en
inglés, pero el tokenizador es bilingüe: quita acentos y filtra stopwords tanto
en español como en inglés, así que también funciona con documentos en español
(hay un test en español precisamente para comprobarlo).

### Opciones

| Comando | Opción | Default | Qué hace |
|---|---|---|---|
| `index` | `--chunk-chars` | 600 | Tamaño máximo de chunk |
| `index` | `--overlap` | 120 | Solape entre chunks consecutivos |
| `index` | `-o` | dir de entrada | Carpeta donde escribir `index.json` |
| `ask` | `-k` | 3 | Número de fragmentos a devolver |
| `ask` | `--min-score` | 0.0 | Corta fragmentos por debajo del score |
| `ask` | `--index` | `docs/index.json` | Ruta explícita del índice |
| `ask` | `--json` | — | Salida JSON para integraciones |
| `chat` | `-k` | 3 | Chunks por pregunta |
| `chat` | `--min-score` | 0.0 | Descarta chunks por debajo del score |

### Sesión interactiva

`chat` carga el índice una vez y lo mantiene en memoria, así que las preguntas
seguidas son instantáneas:

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

| Comando | Qué hace |
|---|---|
| `/sources` | Lista los documentos indexados, con el número de chunks |
| `/top N` | Cambia cuántos chunks devuelve cada pregunta |
| `/min-score N` | Descarta hits por debajo de N (`0.00`–`1.00`) |
| `/help` | Muestra la lista de comandos |
| `/quit` | Sale (`Ctrl-D` y `Ctrl-C` también funcionan) |

Todo lo que no sea un comando se trata como una pregunta. Los argumentos
inválidos avisan del problema y dejan el valor anterior intacto, en lugar de
matar la sesión.

## Como librería

```python
from pathlib import Path
from rag_minimal import RAG

bot = RAG.index(Path("docs"))
for hit in bot.search("what temperature do they need", top_k=3):
    print(f"{hit.score:.3f}  {hit.location()}")
    print(hit.text)

bot.save(Path("docs/index.json"))     # reindexar es barato: vuelve a indexar
```

## Cómo funciona

El código está partido según las cuatro etapas del pipeline, cada una en su
módulo:

| Etapa | Módulo | Qué hace |
|---|---|---|
| Tokenizar | `text.py` | Minúsculas, sin acentos, quita stopwords, stemmer, bigrams |
| Cargar + trocear | `chunking.py` | Lee `.txt`/`.md` y trocea con solape |
| Vectorizar + buscar | `index.py` | TF-IDF, similitud coseno, `save()`/`load()` |
| Línea de comandos | `cli.py` | Los comandos `index` / `ask` / `chat` |

1. **Carga** — recorre el directorio y lee los `.txt`/`.md` (UTF-8, con fallback).
2. **Troceo** — agrupa párrafos hasta `--chunk-chars` y solapa `--overlap`
   caracteres con el chunk anterior, para no partir una idea por la mitad.
   Un párrafo más largo que el tope se parte por palabras.
   Los encabezados `#` no entran en el texto: se guardan como metadato `heading`.
3. **Vectorización** — TF `(1 + log tf)` × IDF sobre unigrams y bigrams,
   normalizado en L2. Como ambos vectores están normalizados, el producto
   punto *es* el coseno.
4. **Búsqueda** — puntúa todos los chunks y devuelve los `k` mejores.
   Recorre el diccionario más corto de los dos, así que es bastante rápido.

`index.json` guarda solo el texto de los chunks (legible, versionable); los
vectores e IDF se recalculan al cargar.

### Puntos de entrada

La CLI se puede arrancar de cuatro formas equivalentes:

```bash
./cli.py ask "cactus"          # ejecutable; el shebang resuelve el intérprete
python3 cli.py ask "cactus"    # cualquier Python 3.9+
python -m rag_minimal ask ...  # como módulo
uv run python -m rag_minimal ...  # uv, respeta .python-version
```

## Tests

```bash
python3 -m unittest -v
```

39 tests con `unittest`: tokenización, stemming, troceo, normalización L2,
ordenación, persistencia (ida y vuelta), más la CLI, la sesión interactiva y el
punto de entrada ejecutable, todo probado end-to-end por subprocess.

**Requiere Python 3.9+.** Probado en 3.9.6 y 3.12.15.

## Limitaciones (a propósito, para saber cuándo escalarlo)

- **Sin sinónimos ni comprensión semántica**: "how often should I water" no recupera un
  texto que solo hable de "watering frequency". TF-IDF compara palabras literales.
- **Stemming muy simple**: unifica plurales y muchos sufijos (`plant`/`plants` sí
  coinciden, `watered` no). Está afinado para español, así que en inglés también es
  aproximado. No es Porter ni Snowball.
- **Sin reranking ni MMR**: devuelve los `k` mejores aunque sean casi idénticos.
- **Ida y vuelta a disco**: para cientos de miles de chunks, la carga del JSON
  se nota; ahí toca SQLite o FAISS.

Cuando no lleguen los hits: el salto de mayor rendimiento es cambiar `_vectorize`
por embeddings reales (sentence-transformers u Ollama) manteniendo la interfaz de
`RAG.search`; y para generar la respuesta final, concatenar los hits como
contexto y llamarlos con un LLM.

## Estructura

```
rag_minimal/
  __init__.py     API pública: RAG, Hit, Chunk, tokenize, chunk_text, ...
  __main__.py     habilita `python -m rag_minimal`
  text.py         tokenización: folding, stemming, stopwords, bigrams
  chunking.py     carga de documentos y troceo
  index.py        índice TF-IDF, búsqueda por coseno, persistencia
  commands.py     interfaz de línea de comandos (index / ask / chat)
cli.py         wrapper ejecutable fino: ./cli.py
test_rag.py    tests con unittest
MEMORY.md      notas del proyecto: restricciones, decisiones, trampas conocidas
AGENTS.md      notas operativas para agentes de código
docs/          documentos de ejemplo (.md y .txt)
pyproject.toml metadatos del proyecto (sin dependencias)
.python-version intérprete fijado por uv
```

La división es por responsabilidad, no por tamaño: `text.py` no sabe nada de
chunks, `chunking.py` no sabe nada de vectores, y `index.py` es el único punto
donde se encuentran las tres etapas. Cambiar TF-IDF por embeddings reales exige
reescribir solo `index.py`.