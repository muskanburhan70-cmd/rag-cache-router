# RAG with Semantic Caching & Cost-Aware Routing

A small, readable RAG pipeline that cuts LLM cost two ways:

1. **Semantic cache** – if a new question is close enough (cosine similarity) to one already answered, return the stored answer with zero LLM calls.
2. **Cost-aware router** – on a cache miss, send the question to the cheapest model tier likely to handle it (Haiku / Sonnet / Opus), with a hard budget guard.

```
query → embed → semantic cache ──hit──▶ cached answer ($0)
                    │miss
                    ▼
            retrieve top-k chunks → router (complexity + budget) → LLM tier → store in cache
```

## Run it

```bash
pip install -r requirements.txt
python demo.py                 # works with no API key (mock LLM)
export ANTHROPIC_API_KEY=...   # use real Claude models
streamlit run app.py           # dashboard
```

Works in GitHub Codespaces / Colab. `fastembed` uses ONNX, so no PyTorch or GPU is needed.
Force the offline embedder with `RAGCR_EMBEDDER=hash`.

## Layout

| File | Job |
|---|---|
| `ragcr/embeddings.py` | fastembed (bge-small) + offline hash fallback |
| `ragcr/vectorstore.py` | chunking + cosine search |
| `ragcr/cache.py` | semantic cache: threshold, TTL, LRU eviction |
| `ragcr/router.py` | complexity scoring, tier choice, budget guard, price table |
| `ragcr/llm.py` | Claude backend + mock backend, token usage |
| `ragcr/pipeline.py` | wires everything, cost ledger vs. "always-large, no-cache" baseline |
| `demo.py`, `app.py` | CLI demo and Streamlit dashboard |

## Tuning notes

- **Cache threshold** is the key knob. Too low returns wrong answers to look-alike questions; too high misses paraphrases. Sweep it on a labelled list of paraphrase / non-paraphrase pairs and report precision.
- **Prices** in `router.py` are placeholders. Copy current numbers from the Anthropic pricing page.
- The router is a heuristic. A good upgrade is to log (query, tier, answer quality) and train a small classifier, or add cascading (small model first, escalate on low confidence).
- The cache keys on the query only. If your documents change, clear the cache (TTL helps).

## Ideas to extend (good portfolio material)

- Evaluation script: hit rate, cost saved, and answer-quality (LLM-as-judge) at different thresholds.
- Swap the numpy store for FAISS or Qdrant; persist the cache in Redis/SQLite.
- Cache per-user or per-document-set namespaces.
