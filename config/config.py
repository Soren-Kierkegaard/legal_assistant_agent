CHROMA_DIR  = "./chroma_db"
BM25_DIR  = "./bm25_db"

COLLECTION  = "legal_docs"
#EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"
TOP_K       = 6        # nombre de chunks récupérés
MIN_SCORE   = 0.5      # seuil de similarité minimum (cosinus, 0→1)

# ingestor
DOCS_DIR       = "./documents"
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "legal_docs"
#EMBED_MODEL    = "nomic-embed-text"
EMBED_MODEL    = "bge-m3" #8192 Tokens
CHUNK_SIZE     = 1024    # caractères par chunk
CHUNK_OVERLAP  = 128    # chevauchement entre chunks (préserve le contexte)