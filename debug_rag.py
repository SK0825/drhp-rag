import pickle
import cohere
from config import COHERE_API_KEY, EMBEDDING_MODEL, CHUNKS_DIR

co = cohere.ClientV2(api_key=COHERE_API_KEY)

# load BM25
with open('data/chunks/bm25_index.pkl', 'rb') as f:
    data = pickle.load(f)
bm25 = data['bm25']
chunks = data['chunks']

print(f"Total chunks in index: {len(chunks)}")
print(f"Companies: {set(c['company'] for c in chunks)}")

# test BM25
query = "what are the key risk factors"
scores = bm25.get_scores(query.lower().split())
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 BM25 results:")
for idx, score in ranked:
    c = chunks[idx]
    print(f"  score={score:.2f} | {c['company']} | {c['section']}")
    print(f"  {c['text'][:120]}")
    print()

# test embedding
print("Testing Cohere embed...")
result = co.embed(
    texts=[query],
    model=EMBEDDING_MODEL,
    input_type="search_query",
    embedding_types=["float"]
)
print(f"Embedding dim: {len(result.embeddings.float[0])}")
print("\nDebug complete.")