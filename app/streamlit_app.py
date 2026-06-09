import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pickle
import json
import time
import subprocess
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import pandas as pd
from datetime import datetime
from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_COLLECTION, CHUNKS_DIR, RAW_PDF_DIR,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL,
    LLM_MODEL
)

st.set_page_config(page_title="DRHP Intelligence", page_icon="📄", layout="wide")
from google import genai as google_genai
gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)

# ── Load indexes ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_indexes(version=0):
    with open(os.path.join(CHUNKS_DIR, "bm25_index.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(CHUNKS_DIR, "embeddings.pkl"), "rb") as f:
        emb_data = pickle.load(f)
    qc = QdrantClient(":memory:")
    qc.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    chunks = data["chunks"]
    embeddings = emb_data["embeddings"]
    points = [PointStruct(id=i, vector=embeddings[i], payload=chunks[i]) for i in range(len(chunks))]
    for i in range(0, len(points), 100):
        qc.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])
    return qc, data["bm25"], chunks

def get_companies(chunks):
    return sorted(set(c["company"] for c in chunks))

# ── Core RAG functions ─────────────────────────────────────────────
def embed_query(query, retries=3):
    for attempt in range(retries):
        try:
            result = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[query]
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 5
                st.toast(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e

def dense_search(qc, query_vector, company_filter=None):
    search_filter = None
    if company_filter:
        search_filter = Filter(must=[FieldCondition(key="company", match=MatchValue(value=company_filter))])
    results = qc.query_points(collection_name=QDRANT_COLLECTION, query=query_vector,
                               limit=TOP_K_DENSE, query_filter=search_filter).points
    return [{"text": r.payload["text"], "company": r.payload["company"],
              "section": r.payload["section"], "score": r.score} for r in results]

def bm25_search(bm25, chunks, query, company_filter=None):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked:
        if score == 0: continue
        chunk = chunks[idx]
        if company_filter and chunk["company"] != company_filter: continue
        results.append({"text": chunk["text"], "company": chunk["company"],
                         "section": chunk["section"], "score": float(score)})
        if len(results) >= TOP_K_BM25: break
    return results

def rrf(dense, bm25, k=60):
    scores, texts = {}, {}
    for rank, r in enumerate(dense):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        texts[key] = r
    for rank, r in enumerate(bm25):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        texts[key] = r
    return [texts[k] for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def generate_answer(query, chunks, extra_instruction="", retries=3):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[{i+1}] Company: {chunk['company'].title()} | Section: {chunk['section']}\n{chunk['text']}\n"
    prompt = f"""You are a financial analyst assistant specializing in Indian IPO documents (DRHPs).
{extra_instruction}
Answer based ONLY on the context below. Always cite company and section.
If not found, say "I couldn't find this information in the provided documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 10
                st.toast(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "Rate limit reached. Please wait and try again."
                        
def run_query(question, company_filter, qc, bm25, chunks):
    # check cache first
    cache_key = f"{question}_{company_filter}"
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if cache_key in st.session_state.query_cache:
        return st.session_state.query_cache[cache_key]
    
    bm25_res = bm25_search(bm25, chunks, question, company_filter)
    time.sleep(1)
    qv = embed_query(question)
    dense = dense_search(qc, qv, company_filter)
    fused = rrf(dense, bm25_res)
    top = fused[:TOP_K_FINAL]
    # filter out low quality chunks
    top = [c for c in top if len(c["text"].split()) > 30]
    time.sleep(2)
    answer = generate_answer(question, top)
    
    # save to cache
    result = (answer, top)
    st.session_state.query_cache[cache_key] = result
    return result

# ── Feature: Comparison mode ───────────────────────────────────────
def run_comparison(question, companies, qc, bm25, chunks):
    results = {}
    for company in [c.lower() for c in companies]:
        qv = embed_query(question)
        dense = dense_search(qc, qv, company.lower())
        bm25_res = bm25_search(bm25, chunks, question, company.lower())
        fused = rrf(dense, bm25_res)
        top = fused[:8]
        answer = generate_answer(question, top,
            extra_instruction=f"Focus only on {company.title()}.")
        results[company] = {"answer": answer, "sources": top}
        time.sleep(0.5)
    return results

# ── Feature: Document summary ──────────────────────────────────────
def generate_summary(company, qc, bm25, chunks):
    sections = ["Business Overview", "Risk Factors", "Objects Of The Issue",
                "Financial Information", "Management"]
    summary_parts = {}
    for section in sections:
        company_chunks = [c for c in chunks if c["company"] == company
                         and c["section"].lower() == section.lower()]
        if not company_chunks:
            company_chunks = [c for c in chunks if c["company"] == company][:3]
        if not company_chunks:
            continue
        context = "\n".join([c["text"] for c in company_chunks[:3]])
        prompt = f"""Summarize the following {section} section from {company.title()}'s DRHP in 3-4 bullet points.
Be concise and factual.

CONTEXT:
{context}

SUMMARY:"""
        response = gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        summary_parts[section] = response.text
        time.sleep(0.3)
    return summary_parts

# ── Feature: Financial extractor ──────────────────────────────────
def extract_financials(company, chunks):
    fin_chunks = [c for c in chunks if c["company"] == company and
                  any(kw in c["section"].lower() for kw in
                      ["financial", "statement", "management discussion"])]
    if not fin_chunks:
        fin_chunks = [c for c in chunks if c["company"] == company][:5]
    context = "\n".join([c["text"] for c in fin_chunks[:6]])
    prompt = f"""Extract key financial metrics from this {company.title()} DRHP text.
Return ONLY a JSON object with these keys (use null if not found):
{{
  "revenue_fy24": "value with unit",
  "revenue_fy23": "value with unit",
  "net_profit_fy24": "value with unit",
  "net_profit_fy23": "value with unit",
  "ebitda_fy24": "value with unit",
  "total_assets": "value with unit",
  "ipo_size": "value with unit",
  "price_band": "value",
  "market_cap": "value with unit"
}}
Return ONLY the JSON, no explanation.

CONTEXT:
{context}"""
    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return {}

# ── Feature: PDF upload & reindex ─────────────────────────────────
def process_uploaded_pdf(uploaded_file):
    os.makedirs(RAW_PDF_DIR, exist_ok=True)
    safe_name = uploaded_file.name.replace(" ", "_").lower()
    save_path = os.path.join(RAW_PDF_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path, safe_name.replace(".pdf", "")

# ── Chat history helpers ───────────────────────────────────────────
def export_chat_csv():
    if not st.session_state.messages:
        return None
    rows = []
    for msg in st.session_state.messages:
        rows.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", "")
        })
    return pd.DataFrame(rows).to_csv(index=False)

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

with st.spinner("Loading indexes..."):
    if "index_version" not in st.session_state:
        st.session_state.index_version = 0
    qc, bm25, chunks = load_indexes(st.session_state.index_version)
    companies = get_companies(chunks)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DRHP Intelligence")
    st.divider()
    if st.button("🔄 Refresh indexes", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    mode = st.radio("Mode", ["💬 Chat", "⚖️ Compare", "📋 Summary", "💰 Financials", "📤 Upload PDF"])
    st.divider()

    if mode == "💬 Chat":
        company_filter = st.selectbox("Filter by company",
            ["All Companies"] + [c.title() for c in companies])
        st.divider()
        st.markdown("**Sample questions:**")
        samples = ["What are the key risk factors?", "What is the objects of the issue?",
                   "Who are the promoters?", "What is the business overview?",
                   "What are the financial highlights?", "What is the competitive landscape?"]
        for s in samples:
            if st.button(s, use_container_width=True, key=f"sample_{s}"):
                st.session_state.sample_q = s

    if mode == "💬 Chat" and st.session_state.get("messages"):
        st.divider()
        csv = export_chat_csv()
        if csv:
            st.download_button("⬇️ Export chat as CSV", csv,
                file_name=f"drhp_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main area ──────────────────────────────────────────────────────

# CHAT MODE
if mode == "💬 Chat":
    st.title("💬 Ask the DRHPs")
    st.caption("Powered by hybrid RAG — dense search + BM25 + reranking")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("View sources"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**[{i+1}] {src['company'].title()} — {src['section']}**")
                        st.caption(src["text"][:300] + "...")

    question = st.chat_input("Ask anything about the DRHPs...")
    if "sample_q" in st.session_state and not question:
        question = st.session_state.pop("sample_q")

    if question:
        cf = None if company_filter == "All Companies" else company_filter.lower()
        st.session_state.messages.append({
            "role": "user", "content": question,
            "timestamp": datetime.now().isoformat()
        })
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                answer, sources = run_query(question, cf, qc, bm25, chunks)
            st.markdown(answer)
            with st.expander("View sources"):
                for i, src in enumerate(sources):
                    st.markdown(f"**[{i+1}] {src['company'].title()} — {src['section']}**")
                    st.caption(src["text"][:300] + "...")
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
            "timestamp": datetime.now().isoformat()
        })

# COMPARE MODE
elif mode == "⚖️ Compare":
    st.title("⚖️ Company Comparison")
    st.caption("Ask the same question across multiple companies simultaneously")

    selected = st.multiselect("Select companies to compare",
        [c.title() for c in companies], default=[c.title() for c in companies[:2]])
    question = st.text_input("Question to compare across companies",
        placeholder="e.g. What are the key risk factors?")

    if st.button("🔍 Compare", type="primary", disabled=not (selected and question)):
        selected_lower = [s.replace(" ", "_").lower() for s in selected]
        cols = st.columns(len(selected))
        with st.spinner(f"Querying {len(selected)} companies..."):
            results = run_comparison(question, selected_lower, qc, bm25, chunks)
        for i, company in enumerate(selected_lower):
            with cols[i]:
                st.subheader(company.replace("_", " ").title())
                st.markdown(results[company]["answer"])
                with st.expander("Sources"):
                    for src in results[company]["sources"]:
                        st.caption(f"**{src['section']}** — {src['text'][:200]}...")

# SUMMARY MODE
elif mode == "📋 Summary":
    st.title("📋 Document Summary")
    st.caption("One-click structured summary of any DRHP")

    company = st.selectbox("Select company", [c.title() for c in companies])
    if st.button("Generate Summary", type="primary"):
        with st.spinner(f"Summarizing {company}'s DRHP..."):
            summary = generate_summary(company.lower(), qc, bm25, chunks)
        for section, text in summary.items():
            with st.expander(f"📌 {section}", expanded=True):
                st.markdown(text)
        summary_text = "\n\n".join([f"## {s}\n{t}" for s, t in summary.items()])
        st.download_button("⬇️ Download summary",
            summary_text,
            file_name=f"{company.lower()}_summary.txt",
            mime="text/plain")

# FINANCIALS MODE
elif mode == "💰 Financials":
    st.title("💰 Financial Data Extractor")
    st.caption("Extract key financial metrics from DRHPs automatically")

    selected_cos = st.multiselect("Select companies",
        [c.title() for c in companies], default=[c.title() for c in companies])

    if st.button("Extract Financials", type="primary", disabled=not selected_cos):
        all_data = {}
        progress = st.progress(0)
        for i, company in enumerate(selected_cos):
            with st.spinner(f"Extracting {company}..."):
                data = extract_financials(company.lower(), chunks)
                all_data[company] = data
            progress.progress((i + 1) / len(selected_cos))

        st.subheader("Extracted Metrics")
        rows = []
        for company, metrics in all_data.items():
            row = {"Company": company}
            row.update({k.replace("_", " ").title(): v for k, v in metrics.items()})
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows).set_index("Company")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download as CSV",
                df.to_csv(),
                file_name=f"drhp_financials_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv")

# UPLOAD MODE
elif mode == "📤 Upload PDF":
    st.title("📤 Upload New DRHP")
    st.caption("Add a new IPO document — it will be parsed and indexed automatically")

    uploaded = st.file_uploader("Drop a DRHP PDF here", type=["pdf"])
    if uploaded:
        st.info(f"File: {uploaded.name} ({uploaded.size / 1024 / 1024:.1f} MB)")
        if st.button("Process & Index", type="primary"):
            st.info("⏳ This will take 5-10 minutes depending on PDF size. Please keep this tab open and do not refresh.")
            with st.spinner("Saving file..."):
                save_path, company_name = process_uploaded_pdf(uploaded)
                st.success(f"Saved to {save_path}")
            with st.spinner("Parsing PDF..."):
                result = subprocess.run(
                    [sys.executable, "parser/pdf_parser.py"],
                    capture_output=True, text=True,
                    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                )
                st.code(result.stdout[-500:] if result.stdout else result.stderr[-500:] if result.stderr else "Done")

            with st.spinner("Embedding and indexing (this takes a few minutes)..."):
                result = subprocess.run(
                    [sys.executable, "ingestion/embedder.py"],
                    capture_output=True, text=True,
                    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                )
            st.code(result.stdout[-500:] if result.stdout else result.stderr[-500:] if result.stderr else "Done")

            st.success("Done! New document indexed successfully.")
            st.session_state.index_version = st.session_state.get("index_version", 0) + 1
            st.cache_resource.clear()
            if st.button("🔄 Reload app"):
                st.rerun()