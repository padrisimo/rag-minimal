import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunking import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP, Chunk, chunk_text, load_documents
from .text import tokenize

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Hit:
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
    INDEX_VERSION = 1

    def __init__(self, chunks: Sequence[Chunk], vectors: Sequence[dict[str, float]], idf: dict[str, float], ngram_max: int = 2):
        self.chunks = list(chunks)
        self.vectors = list(vectors)
        self.idf = idf
        self.ngram_max = ngram_max

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], ngram_max: int = 2) -> "RAG":
        tokenized = [tokenize(c.text, ngram_max) for c in chunks]
        n = len(chunks) or 1

        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
        vectors = [cls._vectorize(tokens, idf) for tokens in tokenized]
        return cls(chunks, vectors, idf, ngram_max)

    @classmethod
    def index(cls, root: Path, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP, ngram_max: int = 2) -> "RAG":
        chunks: list[Chunk] = []
        for source, text in load_documents(root):
            chunks.extend(chunk_text(text, source, chunk_chars, overlap))
        return cls.from_chunks(chunks, ngram_max)

    @staticmethod
    def _vectorize(tokens: Sequence[str], idf: dict[str, float]) -> dict[str, float]:
        raw = Counter(tokens)
        vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in raw.items() if t in idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def search(self, question: str, top_k: int = DEFAULT_TOP_K, min_score: float = 0.0) -> list[Hit]:
        query = self._vectorize(tokenize(question, self.ngram_max), self.idf)
        if not query:
            return []

        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(self.vectors):
            terms = vec.keys() & query.keys() if len(vec) < len(query) else query.keys() & vec.keys()
            score = sum(query[t] * vec[t] for t in terms)
            if score > min_score:
                scored.append((score, i))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(score=score, chunk=self.chunks[i]) for score, i in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.chunks)

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
            raise ValueError(f"incompatible index (v{raw.get('version')}, expected v{cls.INDEX_VERSION}); reindex it")
        chunks = [Chunk.from_dict(c) for c in raw["chunks"]]
        return cls.from_chunks(chunks, int(raw.get("ngram_max", 2)))