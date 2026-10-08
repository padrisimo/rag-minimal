"""RAG mínimo sobre TF-IDF: indexa .txt/.md y recupera los fragmentos relevantes.

Cero dependencias externas y sin LLM. La similitud coseno entre vectores
TF-IDF (unigrams + bigrams, normalizados en L2) determina qué chunks se
parecen más a la pregunta.

Uso como librería:

    from rag import RAG
    bot = RAG.index(Path("docs"))
    for hit in bot.search("cada cuanto regar", top_k=3):
        print(hit.score, hit.source, hit.text[:80])
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Chunk", "Hit", "RAG", "chunk_text", "tokenize", "load_documents"]

DOC_EXTENSIONS = frozenset({".txt", ".md"})

DEFAULT_CHUNK_CHARS = 600
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Sufijos para un stemmer muy simple (el de Porter es mucho más elaborado).
# Se recorren de más largo a más corto y nunca se deja un tallo de <4 letras,
# así 'agua' y 'dos' quedan intactos. No unifica todas las conjugaciones
# verbales: 'riega' y 'riego' sí coinciden, pero 'regar' no. Ver README.
_SUFFIXES = (
    "amientos",
    "imientos",
    "amiento",
    "imiento",
    "aciones",
    "acion",
    "adores",
    "adora",
    "mente",
    "ando",
    "iendo",
    "ar",
    "er",
    "ir",
    "as",
    "es",
    "s",
    "a",
    "o",
)
_MIN_STEM = 4

# Stopwords (español + inglés) escritas ya normalizadas sin diacríticos.
_STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque
    bajo bien cada como con contra cual cuales cuando de del desde donde dos
    durante e el ella ellas ellos en entre era eran eres es esa esas ese eso
    esta estaba estan estas este esto estos estoy fue fueron ha habia hacen
    hasta hay la las le les lo los mas me mi mis mientras mucho muy nada ni
    no nos nosotros o os otra otras otro otros para pero poco por porque que
    quien quienes se sea segun ser si sido sin sobre solo son su sus tambien
    tanto te tiene tienen toda todas todo todos tras tu tus un una unas uno
    unos usted ustedes va vamos van ver vez y ya yo
    a about after all also an and any are as at be because been before being
    but by can did do does doing for from further had has have having he her
    here hers him his how i if in into is it its itself just more most my
    no nor not now of off on once only or other our out over own same she
    should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where
    which while who why will with would you your
    """.split()
)


def _fold(text: str) -> str:
    """Minúsculas y sin diacríticos, para que 'intención' ~ 'intencion'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    """Quita el sufijo más largo que deje un tallo de al menos 4 letras."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text: str, ngram_max: int = 2) -> List[str]:
    """Divide en tokens útiles: sin stopwords, sin 1-char, stemmed, con bigrams.

    Los bigrams ('riego_cactus') evitan que una pregunta corta hecha de
    palabras genéricas compita con todos los chunks del documento.
    """
    words = [
        _stem(w)
        for w in _TOKEN_RE.findall(_fold(text))
        if len(w) > 1 and w not in _STOPWORDS
    ]
    words = [w for w in words if w]
    grams: List[str] = list(words)
    if ngram_max >= 2:
        grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


@dataclass(frozen=True)
class Chunk:
    """Un fragmento indexable, con su procedencia para poder citarlo."""

    id: int
    source: str  # ruta relativa al directorio indexado
    heading: str  # encabezado markdown activo, o "" si no hay
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "heading": self.heading, "text": self.text}

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":
        return cls(id=int(raw["id"]), source=raw["source"], heading=raw.get("heading", ""), text=raw["text"])


def _carry_over(text: str, overlap: int) -> str:
    """Últimos `overlap` caracteres del chunk previo, sin cortar a media palabra."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    if 0 <= space < overlap // 2:
        tail = tail[space + 1 :]
    return tail


def chunk_text(
    text: str,
    source: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Trocea por párrafos: llena hasta `max_chars` y solapa con el anterior.

    Los encabezados markdown no entran en el cuerpo del chunk; se guardan como
    metadato `heading` para que cada hit salga con su sección de origen.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    blocks: List[Tuple[str, str]] = []
    heading = ""
    buf: List[str] = []
    buf_len = 0

    for para in paragraphs:
        if para.startswith("#"):
            if buf:
                blocks.append((heading, "\n\n".join(buf)))
                buf, buf_len = [], 0
            heading = para.lstrip("#").strip() or heading
            continue

        sep = 2 if buf else 0
        if buf and buf_len + sep + len(para) > max_chars:
            blocks.append((heading, "\n\n".join(buf)))
            carry = _carry_over("\n\n".join(buf), overlap)
            buf = [carry] if carry else []
            buf_len, sep = len(carry), 2

        buf.append(para)
        buf_len += sep + len(para)

    if buf:
        blocks.append((heading, "\n\n".join(buf)))

    chunks: List[Chunk] = []
    for heading, body in blocks:
        body = body.strip()
        if body:
            chunks.append(Chunk(id=len(chunks), source=source, heading=heading, text=body))
    return chunks


def load_documents(root: Path, extensions: Iterable[str] = DOC_EXTENSIONS) -> List[Tuple[str, str]]:
    """Lee recursivamente los .txt/.md de `root`. Devuelve (ruta_relativa, texto)."""
    root = Path(root)
    if root.is_file():
        return [(root.name, root.read_text(encoding="utf-8", errors="replace"))]
    if not root.is_dir():
        raise FileNotFoundError(f"no existe la ruta: {root}")

    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    docs: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            rel = path.relative_to(root).as_posix()
            docs.append((rel, path.read_text(encoding="utf-8", errors="replace")))
    return docs


@dataclass(frozen=True)
class Hit:
    """Un chunk recuperados, con su score de similitud coseno."""

    score: float
    chunk: Chunk

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def heading(self) -> str:
        return self.chunk.heading

    @property
    def text(self) -> str:
        return self.chunk.text

    def location(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "source": self.source,
            "heading": self.heading,
            "text": self.text,
        }


class RAG:
    """Índice TF-IDF en memoria, con búsqueda por coseno.

    Construir con `RAG.index(Path("docs"))` y persistir con `save()`.
    """

    INDEX_VERSION = 1

    def __init__(self, chunks: Sequence[Chunk], vectors: Sequence[Dict[str, float]], idf: Dict[str, float], ngram_max: int = 2):
        self.chunks = list(chunks)
        self.vectors = list(vectors)
        self.idf = idf
        self.ngram_max = ngram_max

    # ---------- construcción ----------

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], ngram_max: int = 2) -> "RAG":
        """Calcula TF-IDF y los vectores L2-normalizados de cada chunk."""
        tokenized = [tokenize(c.text, ngram_max) for c in chunks]
        n = len(chunks) or 1

        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        # IDF suavizado (variante de sklearn): siempre positivo, sin df == 0.
        idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
        vectors = [cls._vectorize(tokens, idf) for tokens in tokenized]
        return cls(chunks, vectors, idf, ngram_max)

    @classmethod
    def index(cls, root: Path, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP, ngram_max: int = 2) -> "RAG":
        """Indexa un archivo o un directorio entero de .txt/.md."""
        chunks: List[Chunk] = []
        for source, text in load_documents(root):
            chunks.extend(chunk_text(text, source, chunk_chars, overlap))
        return cls.from_chunks(chunks, ngram_max)

    @staticmethod
    def _vectorize(tokens: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
        """TF (1+log) * IDF sobre los términos del vocabulario, ya normalizado en L2."""
        raw = Counter(tokens)
        vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in raw.items() if t in idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    # ---------- búsqueda ----------

    def search(self, question: str, top_k: int = DEFAULT_TOP_K, min_score: float = 0.0) -> List[Hit]:
        """Top-k chunks por coseno contra la pregunta (orden descendente)."""
        query = self._vectorize(tokenize(question, self.ngram_max), self.idf)
        if not query:
            return []

        scored: List[Tuple[float, int]] = []
        for i, vec in enumerate(self.vectors):
            # Recorre el dict más corto: los bigrams hacen casi siempre la conexión.
            terms = vec.keys() & query.keys() if len(vec) < len(query) else query.keys() & vec.keys()
            score = sum(query[t] * vec[t] for t in terms)
            if score > min_score:
                scored.append((score, i))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(score=score, chunk=self.chunks[i]) for score, i in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------- persistencia ----------

    def to_dict(self) -> dict:
        return {
            "version": self.INDEX_VERSION,
            "ngram_max": self.ngram_max,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RAG":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("version") != cls.INDEX_VERSION:
            raise ValueError(f"índice incompatible (v{raw.get('version')}, esperado v{cls.INDEX_VERSION}); vuelve a indexar")
        chunks = [Chunk.from_dict(c) for c in raw["chunks"]]
        return cls.from_chunks(chunks, int(raw.get("ngram_max", 2)))