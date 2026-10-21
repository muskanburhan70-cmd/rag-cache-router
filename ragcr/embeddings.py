"""Embedders. FastEmbedder is the real one; HashEmbedder is an offline fallback."""
import hashlib
import os
import re

import numpy as np

STOP = set("a an the is are was were be to of in on for and or what how why do does did can i you it this that with as by at from".split())


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashEmbedder:
    """Signed hashed word + char-trigram features. No downloads, weaker than a real model."""

    name = "hash-ngram"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if w not in STOP]
            feats = list(toks)
            for w in toks:
                p = f"#{w}#"
                feats += [p[j : j + 3] for j in range(len(p) - 2)]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(out)


class FastEmbedder:
    name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed(texts)), dtype=np.float32))


def get_embedder():
    """RAGCR_EMBEDDER=hash forces the offline fallback; otherwise try fastembed first."""
    if os.getenv("RAGCR_EMBEDDER") != "hash":
        try:
            return FastEmbedder()
        except Exception as e:  # not installed, or model download blocked
            print(f"[embeddings] fastembed unavailable ({type(e).__name__}); using hash fallback")
    return HashEmbedder()
