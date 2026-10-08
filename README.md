# rag-minimal

RAG mínimo en Python, **sin dependencias externas y sin LLM**: indexa archivos
`.txt`/`.md` y devuelve los fragmentos más relevantes para una pregunta.

Retirada por **TF-IDF + similitud coseno** (unigrams + bigrams, stopwords
español/inglés, stemming ligero, sin acentos). Es la base sobre la que luego
puedes añadir embeddings reales o un LLM sin tocar nada más.

Solo stdlib: funciona con Python 3.9+ y no necesita `pip install` ni conexión.

## Uso

```bash
python3 cli.py index docs/              # indexa y guarda docs/index.json
python3 cli.py ask "cada cuánto regar el cactus"
python3 cli.py ask "..." -k 5 --json    # 5 fragmentos, salida JSON
```

Salida típica:

```
Pregunta: cada cuánto tengo que regar el cactus

[1] score=0.167  riego.md "Riego del cactus"
    El cactus necesita muy poca agua. Se riega cada dos semanas en verano…
```

Cada hit trae `source`, `heading` y `text`, así que siempre puedes citar de qué
fragmento salió la información.

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

## Como librería

```python
from pathlib import Path
from rag import RAG

bot = RAG.index(Path("docs"))
for hit in bot.search("qué temperatura necesitan", top_k=3):
    print(f"{hit.score:.3f}  {hit.location()}")
    print(hit.text)

bot.save(Path("docs/index.json"))     # reindexar es barato: vuelve a indexar
```

## Cómo funciona

1. **Carga** — recorre el directorio y lee los `.txt`/`.md` (UTF-8, con fallback).
2. **Troceo** — agrupa párrafos hasta `--chunk-chars` y solapa `--overlap`
   caracteres con el chunk anterior, para no partir una idea por la mitad.
   Los encabezados `#` no entran en el texto: se guardan como metadato `heading`.
3. **Vectorización** — TF `(1 + log tf)` × IDF sobre unigrams y bigrams,
   normalizado en L2. Como ambos vectores están normalizados, el producto
   punto *es* el coseno.
4. **Búsqueda** — puntúa todos los chunks y devuelve los `k` mejores.
   Recorre el diccionario más corto de los dos, así que es bastante rápido.

`index.json` guarda solo el texto de los chunks (legible, versionable); los
vectores e IDF se recalculan al cargar.

## Tests

```bash
python3 -m unittest -v
```

23 tests con `unittest`: tokenización, stemming, troceo, normalización L2,
ordenación, persistencia (ida y vuelta) y un test end-to-end de la CLI.

## Limitaciones (a propósito, para saber cuándo escalarlo)

- **Sin sinónimos ni comprensión semántica**: "cada cuánto regar" no recupera un texto
  que solo hable de "frecuencia de riego". TF-IDF compara palabras literales.
- **Stemming muy simple**: unifica plurales y muchos sufijos (`riega`/`riego` sí
  coinciden, `regar` no). No es Porter ni Snowball.
- **Sin reranking ni MMR**: devuelve los `k` mejores aunque sean casi idénticos.
- **Ida y vuelta a disco**: para cientos de miles de chunks, la carga del JSON
  se nota; ahí toca SQLite o FAISS.

Cuando no lleguen los hits: el salto de mayor rendimiento es cambiar `_vectorize`
por embeddings reales (sentence-transformers u Ollama) manteniendo la interfaz de
`RAG.search`; y para generar la respuesta final, concatenar los hits como
contexto y llamarlos con un LLM.

## Estructura

```
rag.py        núcleo: tokenización, troceo, TF-IDF, búsqueda, persistencia
cli.py        interfaz de línea de comandos (index / ask)
test_rag.py   tests con unittest
docs/         documentos de ejemplo (.md y .txt)
```