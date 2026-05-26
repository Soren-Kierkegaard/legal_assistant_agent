# Repertoire (Arborescence)
```
legal_assistant/
├── main.py           # API FastAPI
├── streaming.py      # SSE
├── limiter.py        # Rate limiting
├── cache.py          # Semantic cache
├── security.py       # Guards
├── bm25.py           # Best Match 25 implementation
├── rag/
│   ├── ingestor.py   # Indexation des documents
│   ├── retriever.py  # Recherche vectorielle
│   └── pipeline.py   # Assemblage question + docs → LLM
├── documents/        # PDFs, DOCX à indexer
├── config/
    └── prompts.yaml  # Stockage du system prompt et rag instruction
└── ui/
    └── index.html    # Interface chat avec streaming
```

# Pré-requis

Avoir ollama d'installer en local.

Le modèle de base configuré est llama3.2 (Màj de Avril 2026) [voir config.py]

# Architecture

Tout est fait pour être paramètrable et customizable ; il faut commencer par ajouter les différent document dans le répértoire ```document```. Format *.pdf*, *.txt*, *.md*

## Créer la base chromeDB

```bash
python3 rag/ingestor.py
```

Cela créer les répertoire **chrome_db** et **bm25_db**

## Tester la recherche hybride (RAG+BM25)

```bash
python3 rag/retriever.py "Quel est l'intitulé de l'article 27 de la RGPD ?
```

# Lancer l'Application (Interface)

```bash
uvicorn main:app --reload
```

# ! Axe d'Amélioration !

1. La recherche Sémantique est encore assez perfectible. Il faut encore travailler sur la manière de découper le texte vis-à-vis du format particulier des textes de loi (normé mais variablité). 
Pour l'instant cela se fait de manière assez trivial sur une fenêtre de contexte

2. Tester d'autre modèle d'embedding (notamment multilingue), pour l'instant testé "nomic-embed-text" et "bge-m3". 

3. Fine-tuned un modèle d'embedding sur des paires (query text, chunks text pertinents)

4. Ajout d'un reranker

5. Fine-tuning des poids RRF 

6. Tokenisation avancée (spaCy)

# Side Note Technique

Peut-être trivial pour certain, mais cela a été une source de problème et confusion.

## Les permissions CORS
CORS (Cross-Origin Resource Sharing) est le mécanisme qui contrôle si un navigateur a le droit de faire des requêtes vers un serveur situé sur une origine différente (protocole, domaine ou port différent). 

Par exemple, si un frontend tourne sur http://localhost:8080 et l' API FastAPI sur http://localhost, ce sont deux origines différentes. 
Le navigateur va alors demander la permission au serveur avant d'envoyer la vraie requête — c'est ce qu'on appelle une requête de preflight.

le navigateur envoie automatiquement : "Est-ce que je suis autorisé à faire un POST depuis cette origine ?"
OPTIONS /chat/stream
Origin: http://localhost:8080
Access-Control-Request-Method: POST

## Le rôle du middleware CORS ?
Un middleware est une couche de code qui s'exécute avant et/ou après chaque requête, de manière transparente, sans avoir à modifier chaque route.

Il joue deux rôles :

1. Intercepte les requêtes OPTIONS (preflight) du navigateur et répond avec les bons headers CORS (200 ou 400), sans que la requête n'atteigne vos routes.
2. Ajoute les headers CORS aux réponses normales (requêtes avec un header Origin).

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # qui peut appeler l'API
    allow_methods=["*"],                      # quelles méthodes HTTP (Get, POST, ...
    allow_headers=["*"],                      # (Accept, Accept-Language, Content-Language et Content-Type sont ""toujours"" autorisés pour les requêtes CORS simples)
                                              # ⚠️ Ne peut pas être ["*"] si allow_credentials=True — doit être spécifié explicitement
)
```

Exemple de Hearder dans une reqête curl: 
  * header API: -H 'x-api-key: secret-key-user1' \
  * header accept: 'accept: application/json' \
  * header content-type 'Content-Type: application/json' \

==>    Sans ce middleware, FastAPI ne sait pas répondre aux requêtes OPTIONS → votre navigateur bloque tout <==

Header	          Rôle
Authorization	  Transmet les credentials d'authentification (ex: Bearer <token>, Basic ...)
Content-Type	  Indique le format du corps de la requête (ex: application/json)
Accept	Indique   les formats de réponse acceptés par le client
Accept-Language	  Indique la langue préférée du client
Content-Language  Indique la langue du contenu envoyé