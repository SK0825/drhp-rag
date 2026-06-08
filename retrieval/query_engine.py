import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi
from config import (
    COHERE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_COLLECTION, CHUNKS_DIR,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL,
    RERANK_MODEL, LLM_MODEL
)

co = cohere.ClientV2(api_key=COHERE_API_KEY)

def load_qdrant_and_bm25():
    import pickle
    from qdrant_client.models import VectorParams, Distance, PointStruct

    index_path = os.path.join(CHUNKS_DIR, "bm25_index.pkl")
    emb_path = os.path.join(CHUNKS_DIR, "embeddings.pkl")

    with open(index_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    with open(emb_path, "rb") as f:
        emb_data = pickle.load(f)
    embeddings = emb_data["embeddings"]

    qc = QdrantClient(":memory:")
    qc.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    points = [
        PointStruct(id=i, vector=embeddings[i], payload=chunks[i])
        for i in range(len(chunks))
    ]
    for i in range(0, len(points), 100):
        qc.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])
    print(f"Loaded {len(points)} vectors into Qdrant from saved embeddings")
    return qc, bm25, chunks

def embed_query(query):
    result = co.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query",
        embedding_types=["float"]
    )
    return result.embeddings.float[0]

def dense_search(qc, query_vector, company_filter=None, top_k=TOP_K_DENSE):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    search_filter = None
    if company_filter:
        search_filter = Filter(
            must=[FieldCondition(
                key="company",
                match=MatchValue(value=company_filter)
            )]
        )
    results = qc.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter
    ).points
    return [{"text": r.payload["text"], "company": r.payload["company"],
             "section": r.payload["section"], "score": r.score,
             "source": "dense"} for r in results]

def bm25_search(bm25, chunks, query, company_filter=None, top_k=TOP_K_BM25):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked:
        if score == 0:
            continue
        chunk = chunks[idx]
        if company_filter and chunk["company"] != company_filter:
            continue
        results.append({
            "text": chunk["text"],
            "company": chunk["company"],
            "section": chunk["section"],
            "score": float(score),
            "source": "bm25"
        })
        if len(results) >= top_k:
            break
    return results

def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    texts = {}
    for rank, r in enumerate(dense_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        texts[key] = r
    for rank, r in enumerate(bm25_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        texts[key] = r
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [texts[key] for key, _ in ranked]

def rerank(query, candidates, top_n=TOP_K_FINAL):
    docs = [c["text"] for c in candidates]
    result = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=top_n
    )
    reranked = []
    for r in result.results:
        candidate = candidates[r.index].copy()
        candidate["rerank_score"] = r.relevance_score
        reranked.append(candidate)
    return reranked

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[{i+1}] Company: {chunk['company'].title()} | Section: {chunk['section']}\n"
        context += chunk["text"] + "\n"
    prompt = f"""You are a financial analyst assistant specializing in Indian IPO documents (DRHPs).

Answer the user's question based ONLY on the context provided below.
Always cite which company and section your answer comes from.
If the answer is not in the context, say "I couldn't find this information in the provided documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def generate_answer(prompt):
    response = co.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content[0].text

def query(question, company_filter=None, qc=None, bm25=None, chunks=None):
    print(f"\nQuery: {question}")
    if company_filter:
        print(f"Filter: {company_filter} only")

    query_vector = embed_query(question)
    dense_results = dense_search(qc, query_vector, company_filter)
    bm25_results = bm25_search(bm25, chunks, question, company_filter)
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    top_chunks = rerank(question, fused[:30])

    print(f"Retrieved: {len(dense_results)} dense, {len(bm25_results)} bm25 → reranked to {len(top_chunks)}")

    prompt = build_prompt(question, top_chunks)
    answer = generate_answer(prompt)

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    print("="*60)
    return answer, top_chunks

def interactive_session():
    print("Loading indexes...")
    qc, bm25, chunks = load_qdrant_and_bm25()
    print("Ready! Type your questions. Commands: 'filter:<company>' to filter, 'quit' to exit.\n")

    company_filter = None
    while True:
        user_input = input("\nQuestion: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower().startswith("filter:"):
            company_filter = user_input.split(":", 1)[1].strip().lower()
            print(f"Filter set to: {company_filter}")
            continue
        if user_input.lower() == "filter:clear":
            company_filter = None
            print("Filter cleared")
            continue
        query(user_input, company_filter, qc, bm25, chunks)

if __name__ == "__main__":
    interactive_session()