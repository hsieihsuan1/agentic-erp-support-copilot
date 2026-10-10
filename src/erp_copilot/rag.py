"""Minimal retrieval-augmented generation core: section chunking + BM25 ranking.

Pure Python on purpose: the retriever is deterministic, has no model downloads,
and is easy to evaluate (see ``erp-copilot eval`` retrieval metrics).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB_DIR = Path(__file__).parent / "data" / "kb"

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "our the this to was we what when where which why with you your me get".split()
)


def _stem(token: str) -> str:
    """Tiny suffix stripper (not Porter): enough to match release/released/releases."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token[:-1] if token.endswith("e") and len(token) > 3 else token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{re.sub(r'[^a-z0-9]+', '-', self.section.lower()).strip('-')}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float = 0.0  # share of query term weight found in this chunk (0..1)


def load_chunks(kb_dir: Path = DEFAULT_KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        title, section = path.stem, "overview"
        buf: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                sections.append((section, buf))
                section, buf = line[3:].strip(), []
            else:
                buf.append(line)
        sections.append((section, buf))
        for name, lines in sections:
            body = " ".join(ln.strip() for ln in lines if ln.strip())
            if body:
                chunks.append(Chunk(path.stem, title, name, body))
    return chunks


class Retriever:
    """BM25 over section chunks. The title is indexed with each chunk."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_dir(cls, kb_dir: Path = DEFAULT_KB_DIR) -> Retriever:
        return cls(load_chunks(kb_dir))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokenize(query)
        # Terms never seen in the corpus count at the highest IDF, so off-topic questions
        # get low coverage even when one generic word happens to match.
        max_idf = max(self._idf.values(), default=1.0)
        total_weight = sum(self._idf.get(t, max_idf) for t in q) or 1.0
        hits: list[Hit] = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lens, strict=True):
            score = 0.0
            matched = 0.0
            for term in q:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self._avg))
                score += self._idf.get(term, 0.0) * norm
                matched += self._idf.get(term, 0.0)
            if score > 0:
                hits.append(Hit(chunk, round(score, 4), round(matched / total_weight, 3)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
