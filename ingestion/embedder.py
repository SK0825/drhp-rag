import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pickle
import cohere
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)
from rank_bm25 import BM25Okapi
from config import (
    COHERE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_COLLECTION, CHUNKS_DIR
)

co = cohere.ClientV2(api_key=COHERE_API_KEY)
qc = QdrantClient(":memory:")

def load_all_chunks():
    all_chunks = []
    for f in os.listdir(CHUNKS_DIR):
        if f.endswith("_chunks.json"):
            with open(os.path.join(CHUNKS_DIR, f), "r", encoding="utf-8") as fp:
                chunks = json.load(fp)
                all_chunks.extend(chunks)
    print(f"Loaded {len(all_chunks)} chunks total")
    return all_chunks

def embed_chunks(texts, batch_size=50):
    from google import genai
    from config import GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            all_embeddings.extend([e.values for e in result.embeddings])
            time.sleep(0.5)
        except Exception as e:
            print(f"Embedding error on batch {i}: {e}")
            time.sleep(10)
            all_embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))
    return all_embeddings

def save_embeddings(chunks, embeddings):
    path = os.path.join(CHUNKS_DIR, "embeddings.pkl")
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"Embeddings saved to {path}")

def load_embeddings():
    path = os.path.join(CHUNKS_DIR, "embeddings.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['embeddings'])} saved embeddings")
    return data["chunks"], data["embeddings"]

def setup_qdrant():
    existing = [c.name for c in qc.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        qc.delete_collection(QDRANT_COLLECTION)
    qc.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION}' created")

def upsert_to_qdrant(chunks, embeddings):
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk_id": chunk["chunk_id"],
                "company": chunk["company"],
                "section": chunk["section"],
                "text": chunk["text"],
                "word_count": chunk["word_count"]
            }
        ))
    for i in tqdm(range(0, len(points), 100), desc="Upserting to Qdrant"):
        qc.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i:i+100]
        )
    print(f"Upserted {len(points)} vectors to Qdrant")

def build_bm25_index(chunks):
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    index_path = os.path.join(CHUNKS_DIR, "bm25_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved to {index_path}")
    return bm25

def run_ingestion():
    print("=== Phase 4: Embedding + Indexing ===\n")

    # check if embeddings already exist
    saved_chunks, saved_embeddings = load_embeddings()
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith("_chunks.json")]
    saved_companies = set(c["company"] for c in saved_chunks) if saved_chunks else set()
    all_companies = set(f.replace("_chunks.json","") for f in chunk_files)
    new_companies = all_companies - saved_companies

    if saved_chunks and saved_embeddings and not new_companies:
        print("Found saved embeddings — skipping API calls.")
        chunks = saved_chunks
        embeddings = saved_embeddings
    elif saved_chunks and saved_embeddings and new_companies:
        print(f"New companies detected: {new_companies}. Embedding only new chunks...")
        new_chunks = []
        for company in new_companies:
            with open(os.path.join(CHUNKS_DIR, f"{company}_chunks.json"), encoding="utf-8") as f:
                import json
                new_chunks.extend(json.load(f))
        print(f"Embedding {len(new_chunks)} new chunks...")
        new_embeddings = embed_chunks([c["text"] for c in new_chunks])
        chunks = saved_chunks + new_chunks
        embeddings = saved_embeddings + new_embeddings
        save_embeddings(chunks, embeddings)
    else:
        chunks = load_all_chunks()
        texts = [c["text"] for c in chunks]
        print("\nEmbedding chunks (this takes a few minutes)...")
        embeddings = embed_chunks(texts)
        save_embeddings(chunks, embeddings)

    print("\nSetting up Qdrant...")
    setup_qdrant()

    print("\nUploading to Qdrant...")
    upsert_to_qdrant(chunks, embeddings)

    print("\nBuilding BM25 keyword index...")
    build_bm25_index(chunks)

    info = qc.get_collection(QDRANT_COLLECTION)
    print(f"\n=== Ingestion Complete ===")
    print(f"  Vectors in Qdrant: {info.points_count}")
    print(f"  Collection: {QDRANT_COLLECTION}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"\nPhase 4 complete!")

    return qc, chunks

if __name__ == "__main__":
    run_ingestion()