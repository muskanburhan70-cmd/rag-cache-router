"""Run: python demo.py   (set ANTHROPIC_API_KEY to use real models; otherwise a free mock LLM is used)"""
from ragcr import RAGPipeline

QUERIES = [
    "What is semantic caching?",
    "Explain semantic caching",                                   # paraphrase -> should hit cache
    "How does a semantic cache decide to reuse an answer?",       # paraphrase -> may hit
    "What is RAG?",
    "Why does chunk size matter in RAG?",
    "Compare a high versus a low similarity threshold and explain the trade-off for cost and correctness",
    "What is model routing?",
    "What does model routing do?",                                # paraphrase
    "What is semantic caching?",                                  # exact repeat -> hit
]

if __name__ == "__main__":
    rag = RAGPipeline(budget_usd=0.05)
    print(f"embedder={rag.embedder.name}  llm={rag.llm.name}\n")
    print(f"{'HIT':<4} {'sim':>5} {'tier':<7} {'cost($)':>9}  query")
    for q in QUERIES:
        r = rag.ask(q)
        print(f"{'yes' if r.cache_hit else 'no':<4} {r.similarity:5.2f} {r.tier:<7} {r.cost:9.6f}  {q[:70]}")
    s = rag.summary()
    print(f"\nqueries={s['queries']}  hit rate={s['cache_hit_rate']:.0%}  tiers={s['tier_counts']}")
    print(f"cost=${s['actual_cost_usd']:.5f} vs always-large no-cache=${s['baseline_cost_usd']:.5f}  "
          f"-> saved {s['savings_pct']:.0f}%")
