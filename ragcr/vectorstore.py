"""Tiny in-memory vector store: paragraph chunking + cosine search."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    source: str
    text: str


def load_chunks(docs_dir: str, max_words: int = 90) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        for para in path.read_text(encoding="utf-8").split("\n\n"):
            words = para.split()
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i : i + max_words]).strip()
                if piece:
                    chunks.append(Chunk(path.name, piece))
    return chunks


class VectorStore:
    def __init__(self, embedder, chunks: list[Chunk]):
        self.chunks = chunks
        self.matrix = embedder.embed([c.text for c in chunks])  # rows are unit vectors

    def search(self, q_vec: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = self.matrix @ q_vec
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]
