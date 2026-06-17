
from rank_bm25 import BM25Okapi
import pickle
import os
from config import config

BM25_INDEX_PATH = os.path.join(config.BM25_DIR, "bm25_index.pkl")

class BM25Retriever:
    """Wrapper pour BM25 persistant avec mapping vers ChromaDB."""
    
    def __init__(self):

        print(f"Init BM25 retriever object")
        
        self.bm25 = None
        self.chunks = []       # Liste parallèle des chunks
        self.chunk_to_id = {}  # ✅ Mapping chunk → ID ChromaDB
        self.load()
    
    def load(self):
        
        """
            Charge l'index BM25 du disque.
        """
        
        if os.path.exists(BM25_INDEX_PATH):

            print(f"Chargement de la sauvegarde au chemin {BM25_INDEX_PATH} de BM25")
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.chunks = data["chunks"]
                    self.chunk_to_id = data.get("chunk_to_id", {})
                print(f"[BM25] Index chargé : {len(self.chunks)} chunks")
            except Exception as e:
                print(f"[BM25] Erreur lors du chargement : {e}")
                self.bm25 = None

        print(f" Aucun fichier index BM25 chargé")
        #else:
        #    raise FileNotFoundError(f"Impossible de charger bm25_index.pkl de {BM25_INDEX_PATH} manquant ou le fichier n'existe pas")
            
    def save(self):
        
        """
            Sauvegarde l'index BM25 sur le disque.
        """
        
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks,
                "chunk_to_id": self.chunk_to_id  # Sauvegarder le mapping
            }, f)
            
        print(f"[BM25] Index sauvegardé : {len(self.chunks)} chunks dans {BM25_INDEX_PATH}/bm25_index.pkl")
    
    def index(self, chunks: list[str], chunk_ids: list[str]):
        
        """
            Crée l'index BM25 avec mapping vers ChromaDB IDs.
            
            Args:
                chunks: liste des textes
                chunk_ids: liste parallèle des IDs ChromaDB
        """
        
        import re
        tokenized_chunks = [
            re.findall(r'\b\w+\b', chunk['contenu'].lower())
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunks = chunks
        
        # Créer le mapping
        self.chunk_to_id = {chunk['contenu']: cid for chunk, cid in zip(chunks, chunk_ids)}
        
        self.save()
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        
        """
            Recherche BM25. Retourne [(chunk, id, score), ...].
        """
        
        if not self.bm25:
            return []
        
        import re
        tokens = re.findall(r'\b\w+\b', query.lower())
        scores = self.bm25.get_scores(tokens)
        
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        return [
            (
                self.chunks[i],
                self.chunk_to_id.get(self.chunks[i], f"unknown-{i}"),
                scores[i]
            )
            for i in top_indices
            if scores[i] > 0
        ]