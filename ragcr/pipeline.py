"""query -> embed -> semantic cache -> (miss) retrieve -> route -> generate -> cache + cost ledger."""
import time
from dataclasses import dataclass, field

from .cache import CacheEntry, SemanticCache
from .embeddings import get_embedder
from .llm import est_tokens, get_llm
from .router import TIERS, CostAwareRouter
from .vectorstore import VectorStore, load_chunks

SYSTEM = "Answer using only the provided context. If the context is insufficient, say so briefly."


@dataclass
class Result:
    query: str
    answer: str
    cache_hit: bool
    similarity: float          # best cache similarity (hit or miss)
    tier: str                  # "cache" on a hit
    model: str
    cost: float                # actual USD spent on this query
    baseline_cost: float       # what the large model with no cache would have cost
    latency_ms: float
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, docs_dir: str = "data/docs", cache_threshold: float = 0.90,
                 budget_usd: float | None = None, top_k: int = 3, embedder=None, llm=None):
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.store = VectorStore(self.embedder, load_chunks(docs_dir))
        self.cache = SemanticCache(threshold=cache_threshold)
        self.router = CostAwareRouter(budget_usd=budget_usd)
        self.top_k = top_k
        self.history: list[Result] = []

    def ask(self, query: str) -> Result:
        t0 = time.perf_counter()
        q_vec = self.embedder.embed([query])[0]

        entry, sim = self.cache.lookup(q_vec)
        if entry:
            large = TIERS["large"]
            res = Result(query, entry.answer, True, sim, "cache", "-", 0.0,
                         large.cost(entry.in_tokens, entry.out_tokens),
                         (time.perf_counter() - t0) * 1000, entry.sources,
                         [f"matched cached query: {entry.query!r}"])
            self.history.append(res)
            return res

        hits = self.store.search(q_vec, self.top_k)
        context = "\n\n".join(c.text for c, _ in hits)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        prompt_tokens = est_tokens(SYSTEM + prompt)

        decision = self.router.choose(query, prompt_tokens, hits[0][1])
        out = self.llm.generate(decision.model, SYSTEM, prompt)
        spec = TIERS[decision.tier]
        cost = spec.cost(out.in_tokens, out.out_tokens)
        self.router.record(cost)

        sources = sorted({c.source for c, _ in hits})
        self.cache.store(CacheEntry(query, out.text, q_vec, sources, out.in_tokens, out.out_tokens))
        res = Result(query, out.text, False, sim, decision.tier, decision.model, cost,
                     TIERS["large"].cost(out.in_tokens, out.out_tokens),
                     (time.perf_counter() - t0) * 1000, sources, decision.reasons)
        self.history.append(res)
        return res

    def summary(self) -> dict:
        h = self.history
        if not h:
            return {}
        actual, base = sum(r.cost for r in h), sum(r.baseline_cost for r in h)
        tiers: dict[str, int] = {}
        for r in h:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
        return {
            "queries": len(h),
            "cache_hit_rate": sum(r.cache_hit for r in h) / len(h),
            "actual_cost_usd": actual,
            "baseline_cost_usd": base,
            "savings_pct": (1 - actual / base) * 100 if base else 0.0,
            "tier_counts": tiers,
            "avg_latency_ms": sum(r.latency_ms for r in h) / len(h),
        }
