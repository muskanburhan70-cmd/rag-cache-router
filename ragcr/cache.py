"""Semantic cache: reuse an answer when a new query is close enough in embedding space."""
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheEntry:
    query: str
    answer: str
    vec: np.ndarray
    sources: list[str]
    in_tokens: int
    out_tokens: int
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(self, threshold: float = 0.90, ttl_seconds: float = 3600, max_size: int = 500):
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.entries: list[CacheEntry] = []

    def _evict_expired(self):
        now = time.time()
        self.entries = [e for e in self.entries if now - e.created < self.ttl]

    def lookup(self, q_vec: np.ndarray) -> tuple[CacheEntry | None, float]:
        """Return (entry, similarity). Entry is None if best similarity < threshold."""
        self._evict_expired()
        if not self.entries:
            return None, 0.0
        sims = np.stack([e.vec for e in self.entries]) @ q_vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= self.threshold:
            e = self.entries[best]
            e.hits += 1
            e.last_used = time.time()
            return e, score
        return None, score

    def store(self, entry: CacheEntry):
        if len(self.entries) >= self.max_size:  # evict least recently used
            self.entries.remove(min(self.entries, key=lambda e: e.last_used))
        self.entries.append(entry)
