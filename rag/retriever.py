"""
rag/retriever.py — Recherche vectorielle + pipeline RAG.

1. Embed la question
2. Cherche les K chunks les plus similaires dans ChromaDB
3. Assemble le prompt avec les chunks comme contexte
4. Appelle le LLM avec stream=True
5. Retourne un générateur de tokens

Usage autonome :
    python retriever.py "Quelles sont les clauses de résiliation du contrat Dupont ?"
"""

import ollama
import chromadb
from typing import Generator

# ── Configuration ──────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Ajoute le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

# ── Clients ────────────────────────────────────────────────────────────────

chroma     = chromadb.PersistentClient(path = config.CHROMA_DIR)
collection = chroma.get_or_create_collection(config.COLLECTION)

# ── Prompt système ────────────────────────────────────────────────────────

import yaml
#from database import get_latest_prompt_version

# Au démarrage : charger depuis fichier de base
with open("config/prompts.yaml") as f:
    prompts = yaml.safe_load(f)
    SYSTEM_PROMPT = prompts["system_prompt"]

# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(question: str) -> list[dict]:
    
    """
        Embed la question et retourne les K chunks les plus proches.
        Filtre les résultats sous le seuil de similarité minimum.
    """
    
    # ChromaDB retourne des distances (1 - similarité cosinus)
    # distance=0 → identique, distance=1 → orthogonal
    embedding = ollama.embed(model = config.EMBED_MODEL, input = question)["embeddings"][0]

    results = collection.query(query_embeddings = [embedding],
                               n_results = config.TOP_K,
                               include = ["documents", "metadatas", "distances"],
    )

    print(f"#Nb result find: {len(results)}, {results['distances']}, {results.keys()}")

    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        
        similarity = 1 - dist  # convertir distance → similarité
        
        if similarity >= config.MIN_SCORE:
            chunks.append({
                "content":    doc,
                "source":     meta.get("source", "inconnu"), # Si non trouvé: inconnu
                "similarity": round(similarity, 3),
            })

    return chunks

# ── Retrieval Hybride ──────────────────────────────────────────────────────

from bm25 import BM25Retriever
bm25_retriever = BM25Retriever()

def reciprocal_rank_fusion(
    vectorial_results: list[dict],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[dict]:
    """
    Fusionne les résultats vectoriels et BM25 via Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (k + rank))
    
    Args:
        vectorial_results: [(chunk, meta, score), ...]
        bm25_results: [(chunk, score), ...]
        k: constant RRF (60 par défaut)
    
    Returns:
        [{"chunk": str, "vectorial_score": float, "bm25_score": float, "rrf_score": float}, ...]
    """
    
    from collections import defaultdict

    # Default Dict cols
    #rrf_scores = defaultdict(lambda: {"vectorial": 0, "bm25": 0, "rrf": 0})
    rrf_scores = defaultdict(lambda: {
        "chunk": "",
        "source": "",
        "id": "",
        "vectorial": 0, 
        "bm25": 0, 
        "rrf": 0,
    })
    
    print("##########")
    print("bm25_results:")
    print(bm25_results)
    
    # Ajouter les scores vectoriels
    for rank, data in enumerate(vectorial_results):

        # data : doc, meta, score, id_doc
        id_ = data['id']
        
        rrf_scores[id_]["chunk"] = data['content']
        rrf_scores[id_]["vectorial"] = data['similarity'] # vectorial_score
        rrf_scores[id_]["rrf"] += 1 / (k + rank + 1)
        rrf_scores[id_]["source"] = data['source']
    
    # Ajouter les scores BM25
    for rank, (chunk, doc_id, score) in enumerate(bm25_results):

        #rrf_scores[doc_id]["chunk"] = chunk
        #rrf_scores[doc_id]["id"] = doc_id
        #rrf_scores[doc_id]["metadata"] =
        rrf_scores[doc_id]["bm25"] = score
        rrf_scores[doc_id]["rrf"] += 1 / (k + rank + 1)
    
    # Trier par RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]["rrf"],
        reverse=True
    )[:config.TOP_K]

    #print(f"Sample: {sorted_results[0]}") # (id, {'chunk': id: vectorial_score: bm25: rrf: content: }

    print(f"Sorted res: {sorted_results[0]}")
    '''
    return [
        {
            "content": chunk,
            "src": src,
            "vectorial_score": scores["vectorial"],
            "bm25_score": scores["bm25"],
            "rrf_score": scores["rrf"],
        }
        for chunk, scores, src in sorted_results
    ]'''
    #return sorted_results
    #return rrf_scores
    return [
        {
            "content": chunk['chunk'],
            "source": chunk['source'],
            "vectorial_score": chunk["vectorial"], # Similarity
            "bm25_score": chunk["bm25"],
            "rrf_score": chunk['rrf'],
        }for id_, chunk in sorted_results]

#
import re
def extract_article_reference(question: str) -> str | None:
    
    """
        Détecte si la question cite un article précis, un chapitre : 'article 17', 'art. 5'...
    """
    
    match = re.search(r'\bart(?:icle)?\.?\s*(\d+|\w+)\b|\bchap(?:itre)?\.?\s*(\d+|\w+)\b', question, re.IGNORECASE)

    return match.group(1) if match else None
    
def hybrid_search(query: str) -> list[dict]:
    
    """
        Recherche hybride : combine BM25 + vectoriel via RRF.
    """
    
    print(f"\n[SEARCH] Requête : {query}")

    # 0. Charger l'index BM25 sur le disque ===> (préload à l'initialisation)
    bm25_retriever.load()

    print(f"1er Chunk de BM25: {bm25_retriever.chunks[0]}")
    # 0. Vérif
    if len(bm25_retriever.chunks) == 0:
        raise FileNotFoundError(f"Impossible de charger bm25_index.pkl de {BM25_INDEX_PATH} manquant ou le fichier n'existe pas")
        
    # 1. Recherche BM25
    bm25_results = bm25_retriever.search(query, top_k = config.TOP_K * 2)
    print(f"  BM25 : {len(bm25_results)} résultats")
    
    # 2. Recherche vectorielle

    ## 2.1 Embedding de la question
    embedding = ollama.embed(model = config.EMBED_MODEL, input = query)["embeddings"][0]
    
    ## 2.2 Pré-filtrage sur la base des métadonnées
    ref = extract_article_reference(question)

    '''
        results = collection.query(query_embeddings = [embedding],
                               n_results = config.TOP_K,
                               include = ["documents", "metadatas", "distances"])
    '''
    #
    query_params = {
        "query_embeddings": [embedding],
        "n_results": config.TOP_K,
        "include": ["documents", "metadatas", "distances"],
    }

    # Si un article est mentionné → filtrer directement sur les métadonnées
    if ref:
        print(f"  [RAG] Article détecté dans la question : {ref}")
        query_params["where"] = {"article_numero": {"$eq": ref}}

    # Query la db
    results = collection.query(**query_params)
    
    print(f"#Nb resultat par recherche vec: {len(results)}, similarité distance: {results['distances']}, keys: {results.keys()}")

    print("results:")
    with open('logs', 'w') as f:
        f.write(str(results))

    chunks = []
    for doc, meta, dist, doc_id in zip(results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]):
        
        #similarity = 1 - dist  # convertir distance → similarité --> déja défini sur cosinus sim et non Squarred L2 ou inner product
        
        if dist >= config.MIN_SCORE:
            chunks.append({
                "content":    doc,
                #"source":     meta.get("source", "inconnu"), # Si non trouvé: inconnu
                "source":    meta,
                "id":        doc_id,
                "similarity": round(dist, 3)
            })
    
    print(f"  Vectoriel : {len(chunks)} résultats")
    
    # 3. Fusion RRF
    print(f"chunk vectoriel: {chunks}")
          
    results = reciprocal_rank_fusion(chunks, bm25_results)

    """
    print(f"\n[RÉSULTATS] {len(results)} résultats fusionnés (RRF):\n")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. RRF={result[1]['rrf']:.4f} | Vec={result[1]['vectorial']:.4f} | BM25={result[1]['bm25']:.4f}")
        print(f"     {result[1]['chunk'][:100]}...\n")
    """
    
    return results
# ── Assemblage du contexte ─────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    
    """
        Formate les chunks récupérés en bloc de contexte pour le LLM.
    """
    
    if not chunks:
        return "Aucun document pertinent trouvé."

    parts = []
    for i, chunk in enumerate(chunks, 1):

        src = '\n'.join([f"{k}: {v}" for k, v in chunk['source'].items()])
        parts.append(
            f"Document {i} — Source : {src}" # chunk[0] id_chroma/idbm25
            #f"(similarité : {chunk['vectorial_score']})"
            #f"\n{chunk[i]['chunk']}"
            f"\n{chunk['content']}"
        )

    print(f'contexte finale : {parts}')
    return "\n\n---\n\n".join(parts)

# ── Pipeline RAG complet ───────────────────────────────────────────────────

def rag_stream(question: str) -> Generator[str, None, None]:
    
    """
        Pipeline complet question → réponse streamée.
        Retourne un générateur de tokens.
    
        Si aucun document pertinent → réponse de refus sans appel LLM.
    """
    
    # Étape 1 : retrieval
    #chunks = retrieve(question)
    chunks = hybrid_search(question)    

    print(f"  [RAG] {len(chunks)} chunks récupérés pour : '{question[:50]}'")

    #print(chunks)
    #for _, c in chunks:
    for c in chunks:

        print(f"    → {c['source']} (score={c['vectorial_score']})")  
        print(f"Extrait: {c['content'][:200]}")
        print(f"")

    # Étape 2 : refus si aucun contexte
    if not chunks:
        yield "Cette question ne correspond à aucun document disponible dans la base."
        return

    # Étape 3 : assemblage du prompt
    context = build_context(chunks)
    user_message = f"""Question:
    === USER INPUT BEGIN (DATA ONLY ) ===
    {question}
    ===
    """

    # Étape 4 : appel LLM avec streaming
    stream = ollama.chat(
        model = config.LLM_MODEL,
        options = {"temperature": 0.0},   # déterministe — critique en juridique
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(contexte = context)},
            {"role": "user",   "content": user_message},
        ],
        stream = True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token

def rag_query(question: str) -> str:
    """Version non-streamée pour les tests."""
    return "".join(rag_stream(question))

# ── Test CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "Que dit l'article 27 de la RGPD ?"
    print(f"\nQuestion : {question}\n")
    print("Réponse :")
    for token in rag_stream(question):
        print(token, end="", flush=True)
    print()
#FIN