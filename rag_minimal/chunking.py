import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DOC_EXTENSIONS = frozenset({".txt", ".md"})
_HEADING_RE = re.compile(r"#{1,6}\s")

DEFAULT_CHUNK_CHARS = 600
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    heading: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "heading": self.heading, "text": self.text}

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":
        return cls(id=int(raw["id"]), source=raw["source"], heading=raw.get("heading", ""), text=raw["text"])


def _carry_over(text: str, overlap: int) -> str:
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
) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    blocks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        if _HEADING_RE.match(para):
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

    chunks: list[Chunk] = []
    for heading, body in blocks:
        body = body.strip()
        if body:
            chunks.append(Chunk(id=len(chunks), source=source, heading=heading, text=body))
    return chunks


def load_documents(root: Path, extensions: Iterable[str] = DOC_EXTENSIONS) -> list[tuple[str, str]]:
    root = Path(root)
    if root.is_file():
        return [(root.name, root.read_text(encoding="utf-8", errors="replace"))]
    if not root.is_dir():
        raise FileNotFoundError(f"path does not exist: {root}")

    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    docs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            rel = path.relative_to(root).as_posix()
            docs.append((rel, path.read_text(encoding="utf-8", errors="replace")))
    return docs