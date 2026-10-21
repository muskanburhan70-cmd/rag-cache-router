"""Dashboard: streamlit run app.py"""
import pandas as pd
import streamlit as st

from ragcr import RAGPipeline

st.set_page_config(page_title="RAG: semantic cache + cost-aware routing", layout="wide")
st.title("RAG with semantic caching & cost-aware routing")

with st.sidebar:
    threshold = st.slider("Cache similarity threshold", 0.70, 0.99, 0.90, 0.01)
    budget = st.number_input("Budget (USD, 0 = unlimited)", 0.0, 100.0, 0.05, 0.01)
    if st.button("Reset session"):
        st.session_state.clear()

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(budget_usd=budget or None, cache_threshold=threshold)
rag: RAGPipeline = st.session_state.rag
rag.cache.threshold = threshold
rag.router.budget = budget or None

st.caption(f"embedder: {rag.embedder.name} · llm: {rag.llm.name}")
q = st.chat_input("Ask something about RAG, semantic caching or model routing")
if q:
    rag.ask(q)

left, right = st.columns([3, 2])
with left:
    for r in reversed(rag.history):
        with st.chat_message("user"):
            st.write(r.query)
        with st.chat_message("assistant"):
            st.write(r.answer)
            badge = f"CACHE HIT (sim {r.similarity:.2f})" if r.cache_hit else f"{r.tier.upper()} · {r.model}"
            st.caption(f"{badge} · ${r.cost:.6f} · {r.latency_ms:.0f} ms · sources: {', '.join(r.sources)}")
            if r.reasons:
                st.caption("why: " + "; ".join(r.reasons))
with right:
    s = rag.summary()
    if s:
        c1, c2 = st.columns(2)
        c1.metric("Cache hit rate", f"{s['cache_hit_rate']:.0%}")
        c2.metric("Cost saved vs always-large", f"{s['savings_pct']:.0f}%")
        c1.metric("Actual cost", f"${s['actual_cost_usd']:.5f}")
        c2.metric("Baseline cost", f"${s['baseline_cost_usd']:.5f}")
        st.bar_chart(pd.Series(s["tier_counts"], name="queries"))
