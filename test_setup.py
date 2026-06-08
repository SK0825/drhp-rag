import sys
print(f"Python: {sys.version}")

print("\nTesting imports...")
from config import *
print("  config.py loaded")

import pdfplumber
print("  pdfplumber imported")

import fitz
print("  pymupdf imported")

print("\nTesting Cohere embeddings...")
import cohere
co = cohere.ClientV2(api_key=COHERE_API_KEY)
result = co.embed(
    texts=["test DRHP document"],
    model=EMBEDDING_MODEL,
    input_type="search_document",
    embedding_types=["float"]
)
dim = len(result.embeddings.float[0])
print(f"  Embedding dim: {dim}")

print("\nTesting Cohere LLM...")
response = co.chat(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": "Say exactly: setup complete"}]
)
print(f"  Response: {response.message.content[0].text.strip()}")

print("\nTesting Cohere reranker...")
rr = co.rerank(
    model=RERANK_MODEL,
    query="what are the risk factors",
    documents=["company faces market risk", "revenue grew 20%"],
    top_n=2
)
print(f"  Rerank results: {len(rr.results)} results")

print("\nTesting Qdrant (in-memory)...")
from qdrant_client import QdrantClient
qc = QdrantClient(":memory:")
print("  Qdrant in-memory running")

print("\n*** Phase 1 complete. All systems go. ***")