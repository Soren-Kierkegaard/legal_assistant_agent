# Implémentation basée sur (https://medium.com/@jacobrcasey135/how-to-use-local-embedding-models-and-sentence-transformers-c0bf80a00ce2)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Assumer qu'Ollama tourne sur localhost:11434 (plus de GPU/CPU)
import requests
import numpy as np

def get_embedding_from_model(texte: str, normalized = False):

    try:
        response = requests.post('http://localhost:11434/api/embeddings', json={
            'model': 'llama3.2',
            'prompt': texte
        })
        
        embedding = response.json()['embedding']
        
        print(f"Embedding ({len(embedding)}): {embedding[:10]}")

        if normalized:
            # Normalisation L2
            return embedding / np.linalg.norm(embedding)

        return embedding
        
    except Exception as e:
        raise(f"Erreur : {e}") 
        
class SemanticCache:
    
    def __init__(self, model_name = "llama3.2", similarity_threshold = 0.90):

        """
            model_name: str
                Type de modèle pour le calcule

            similarity_threshold: float 
                Avec 80-85%, On capture les variations formelles tout en évitant les faux positifs.
                90% c'est trop restrictif un seuil trop haut est contre intuitif
        """

        self.name = model_name
        self.model = get_embedding_from_model # call external func get_embedding_from_model
        self.similarity_threshold = similarity_threshold
        self.entries = [] # Mémoire
 
    def embed(self, text):

        """
            Vectorise d'après le modèle la requête
        """
        
        #return self.model.encode([text], normalize_embeddings=True)[0]
        
        return self.model(text, normalized = True)
 
    def lookup(self, query):

         """
             Cherche une entrée dans le cache si existe, si oui retourner le score et la réponse
             Si score trop faible < seuil (80%) 
         """
         if not self.entries:
             #return None
             return {"hit": False, "score": float(0.0)}

         # Obtenir l'emb de la requete
         query_vec = self.embed(query)

         # Get all
         cached_vectors = np.array([entry["embedding"] for entry in self.entries])

         # Calcule Sim Score
         print(f"{cached_vectors.shape}, {query_vec.shape}")
         similarities = cosine_similarity([query_vec], cached_vectors)[0]

         # Trouver indice argmax & à partir de la le best score
         best_idx = np.argmax(similarities)
         best_score = similarities[best_idx]

         # Si > au seuil fixé
         if best_score >= self.similarity_threshold:
             return {
             "hit": True,
             "score": float(best_score),
             "matched_query": self.entries[best_idx]["query"],
             "result": self.entries[best_idx]["result"]
             }
         
         return {"hit": False, "score": float(best_score)}
        
    def add(self, query, result):
         
         embedding = self.embed(query)
         
         self.entries.append({"query": query, "embedding": embedding, "result": result})
#FIN