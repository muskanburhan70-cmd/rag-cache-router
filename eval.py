"""Evaluate the semantic-cache threshold.   Run: python eval.py

Idea: a cache should say HIT for paraphrases (same meaning) and MISS for look-alike
questions that ask something different. We measure both at many thresholds.

  hit rate on paraphrases  = how many repeat questions we save money on  (higher is better)
  false-hit rate           = how often we wrongly return a cached answer (must stay ~0)

No LLM or API key is needed: only embeddings are compared.
Add your own pairs to the two lists below - more pairs = a more trustworthy result.
"""
import csv

import numpy as np

from ragcr.embeddings import get_embedder

# Same meaning, different words  -> should be a cache HIT
PARAPHRASES = [
    ("What is semantic caching?", "Explain semantic caching"),
    ("How does a semantic cache work?", "Can you describe how semantic caching works?"),
    ("What is model routing?", "What does model routing do?"),
    ("Why does chunk size matter in RAG?", "How does chunk size affect a RAG system?"),
    ("What is retrieval-augmented generation?", "Explain RAG in simple terms"),
    ("How do I reduce LLM costs?", "Ways to lower the cost of language model calls"),
    ("What is a similarity threshold?", "Explain the similarity threshold in a cache"),
    ("How do embeddings work?", "Explain what text embeddings are"),
    ("What is a budget guard in a router?", "How does the budget guard in routing work?"),
    ("Why does RAG reduce hallucination?", "How does RAG help stop hallucinations?"),
    ("What is cosine similarity?", "Explain cosine similarity"),
    ("When should cached answers expire?", "What is the time to live of a cache entry for?"),
]

# Look similar, but ask something DIFFERENT -> must be a cache MISS
DIFFERENT = [
    ("What is semantic caching?", "What is exact-match caching?"),
    ("What is model routing?", "What is model fine-tuning?"),
    ("Why does chunk size matter in RAG?", "Why does the embedding model matter in RAG?"),
    ("How does a semantic cache work?", "How does a vector database work?"),
    ("What is a similarity threshold?", "What is a context window?"),
    ("How do I reduce LLM costs?", "How do I reduce LLM latency?"),
    ("What is retrieval-augmented generation?", "What is reinforcement learning?"),
    ("What is cosine similarity?", "What is Euclidean distance?"),
    ("When should cached answers expire?", "When should I re-embed my documents?"),
    ("What is a budget guard in a router?", "What is cascading in a router?"),
    ("Why does RAG reduce hallucination?", "Why does RAG increase input token cost?"),
    ("How do embeddings work?", "How do transformers work?"),
]

THRESHOLDS = [round(float(t), 2) for t in np.arange(0.60, 0.99, 0.02)]


def pair_similarities(embedder, pairs):
    a = embedder.embed([p[0] for p in pairs])
    b = embedder.embed([p[1] for p in pairs])
    return np.sum(a * b, axis=1)  # vectors are unit length, so dot product = cosine similarity


def main():
    embedder = get_embedder()
    print(f"embedder: {embedder.name}\n")
    sim_para = pair_similarities(embedder, PARAPHRASES)
    sim_diff = pair_similarities(embedder, DIFFERENT)

    rows = []
    for t in THRESHOLDS:
        hit = float(np.mean(sim_para >= t))
        false_hit = float(np.mean(sim_diff >= t))
        rows.append({"threshold": t, "paraphrase_hit_rate": hit, "false_hit_rate": false_hit})

    print(f"{'threshold':>9} | {'paraphrase hits':>15} | {'false hits':>10}")
    for r in rows:
        print(f"{r['threshold']:>9.2f} | {r['paraphrase_hit_rate']:>14.0%} | {r['false_hit_rate']:>9.0%}")

    # Recommend: the LOWEST threshold with zero false hits (that saves the most money safely).
    safe = [r for r in rows if r["false_hit_rate"] == 0]
    best = min(safe, key=lambda r: r["threshold"]) if safe else rows[-1]
    print(f"\nRecommended threshold: {best['threshold']:.2f}  "
          f"(catches {best['paraphrase_hit_rate']:.0%} of paraphrases, {best['false_hit_rate']:.0%} false hits)")
    if not safe:
        print("No threshold had zero false hits - add more distinct examples or try a better embedding model.")

    print("\nParaphrases the cache would MISS first (lowest similarity):")
    for i in np.argsort(sim_para)[:3]:
        print(f"  {sim_para[i]:.2f}  {PARAPHRASES[i][0]!r} vs {PARAPHRASES[i][1]!r}")
    print("Different questions the cache would wrongly MATCH first (highest similarity):")
    for i in np.argsort(-sim_diff)[:3]:
        print(f"  {sim_diff[i]:.2f}  {DIFFERENT[i][0]!r} vs {DIFFERENT[i][1]!r}")

    with open("eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r["threshold"] for r in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(ts, [r["paraphrase_hit_rate"] * 100 for r in rows], marker="o", label="Paraphrase hit rate (savings)")
        plt.plot(ts, [r["false_hit_rate"] * 100 for r in rows], marker="s", label="False-hit rate (wrong answers)")
        plt.axvline(best["threshold"], linestyle="--", color="gray", label=f"Recommended {best['threshold']:.2f}")
        plt.xlabel("Cache similarity threshold")
        plt.ylabel("%")
        plt.title(f"Semantic cache threshold sweep ({embedder.name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig("eval_results.png", dpi=150)
        print("\nSaved eval_results.csv and eval_results.png")
    except ImportError:
        print("\nSaved eval_results.csv (install matplotlib for the chart: pip install matplotlib)")


if __name__ == "__main__":
    main()
