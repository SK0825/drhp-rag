import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "drhp_chunks")

CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 64
EMBEDDING_MODEL    = "models/gemini-embedding-001"
EMBEDDING_DIM      = 3072
LLM_MODEL          = "gemini-2.5-flash"

TOP_K_DENSE        = 20
TOP_K_BM25         = 20
TOP_K_FINAL        = 10

DATA_DIR           = "data"
RAW_PDF_DIR        = "data/raw_pdfs"
PARSED_DIR         = "data/parsed"
CHUNKS_DIR         = "data/chunks"
DB_PATH            = "db/metadata.db"