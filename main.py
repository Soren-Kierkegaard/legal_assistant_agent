"""
main.py — API FastAPI minimale exposant un LLM Ollama local.
 
Couvre :
- Authentification par API key (header X-API-Key)
- Gestion des utilisateurs et de leurs permissions
- Routes : /chat, /health, /models, /chatsream
- Middleware de sécurité
- Gestion propre des erreurs
 
Lancer : uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import ollama
import time
import logging
import json

from rag.retriever import rag_stream

from security import HOMOGLYPH_MAP, INJECTION_PATTERNS, PII_PATTERNS
from security import normalize, is_injection, has_pii_leak

# ── FASTAPP Objet  ─────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
app = FastAPI(title="LLM API", version="1.0.0")

# ── Générateur SSE ─────────────────────────────────────────────────────────

from fastapi.responses import StreamingResponse

#from streaming import router

#app.include_router(router)

# ── Middleware de sécurité (réutilisé du module 3) ─────────────────────────

from security import normalize, is_injection

# ── RateLimiter ────────────────────────────────────────────────────────────

from limiter import RateLimiter, count_tokens

limiter = RateLimiter()

# ── Cache Semantique ───────────────────────────────────────────────────────

from cache import SemanticCache

cache = SemanticCache()

# ── Permissions CORS ───────────────────────────────────────────────────────

"""
Le client (eg. navigateur) envoie automatiquement une requête OPTIONS avant le POST pour vérifier les permissions CORS. 
C'est le comportement standard des navigateurs pour les requêtes cross-origin.

Par exemple, si un frontend tourne sur http://localhost:8080 et l' API FastAPI sur http://localhost, ce sont deux origines différentes. 
Le navigateur va alors demander la permission au serveur avant d'envoyer la vraie requête — c'est ce qu'on appelle une requête de preflight.

le navigateur envoie automatiquement : "Est-ce que je suis autorisé à faire un POST depuis cette origine ?"
OPTIONS /chat/stream
Origin: http://localhost:8080
Access-Control-Request-Method: POST

### Le rôle du middleware CORS
Un middleware est une couche de code qui s'exécute avant et/ou après chaque requête, de manière transparente, sans que vous ayez à modifier chaque route.

Il joue deux rôles :

1. Intercepte les requêtes OPTIONS (preflight) du navigateur et répond avec les bons headers CORS (200 ou 400), sans que la requête n'atteigne vos routes.
2. Ajoute les headers CORS aux réponses normales (requêtes avec un header Origin).

==>    Sans ce middleware, FastAPI ne sait pas répondre aux requêtes OPTIONS → votre navigateur bloque tout <==

Header	          Rôle
Authorization	  Transmet les credentials d'authentification (ex: Bearer <token>, Basic ...)
Content-Type	  Indique le format du corps de la requête (ex: application/json)
Accept	Indique   les formats de réponse acceptés par le client
Accept-Language	  Indique la langue préférée du client
Content-Language  Indique la langue du contenu envoyé
"""
from fastapi.middleware.cors import CORSMiddleware

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_credentials = False, # concerne spécifiquement les cookies/en-têtes Authorization (au sens HTTP credentials). Une API key transmise via un header personnalisé comme X-API-Key n'est pas considérée comme un "credential" au sens CORS.
    allow_origins= origins,  # ou ["*"] pour tout autoriser
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key"], # Note: les headers Accept, Accept-Language, Content-Language et Content-Type resteront autorisés, même si non mentionné explicitement.
                                 #       dans le code source de CORSMiddleware: allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers)) ces headers font partie des SAFELISTED_HEADERS fuse avec 
)

# ── Authentification ───────────────────────────────────────────────────────
# En production : stocker dans une DB, pas en dur dans le code.
# Ici : dictionnaire minimal pour illustrer le principe.

# Le modèle est lié à l'utilisateur — chaque clé API a son model assigné. 
# Ça permet de donner un modèle rapide et économique aux utilisateurs standards, et un modèle plus capable aux admins.
from dotenv import load_dotenv
import os

load_dotenv()  # charge le fichier .env, ce fichier n'est jamaid commit - committe uniquement .env.example avec des valeurs fictives pour montrer la structure aux autres développeurs

# Vérification au démarrage — l'API refuse de démarrer si une clé manque
for var in ["API_KEY_ADMIN", "API_KEY_USER1", "API_KEY_DEMO"]:
    if not os.getenv(var):
        raise RuntimeError(f"Variable d'environnement manquante : {var}")

API_KEYS = {
    os.getenv("API_KEY_ADMIN"): {"user": "admin",  "role": "admin",    "model": "llama3.2"},
    os.getenv("API_KEY_USER1"): {"user": "alice",  "role": "user",     "model": "llama3.2"},
    os.getenv("API_KEY_DEMO"):  {"user": "demo",   "role": "readonly", "model": "llama3.2"},
}

# Rôles et permissions associées
PERMISSIONS = {
    "admin":    ["chat", "list_models", "health"],
    "user":     ["chat", "health"],
    "readonly": ["health"],  # peut juste pinger l'API
}

# APIKeyHeader extrait la valeur du header X-API-Key, passe en **requête (** cette requête est transparente HTTP (GET, ...) et circule en coulisse, gérée entièrement par FastAPI
# Cet objet :
#
# 1. Mémorise le nom du header à surveiller ("X-API-Key")
# 2. Est callable — c'est-à-dire que FastAPI peut l'appeler comme une fonction: api_key_header() APIKeyHeader.__call__() missing 1 required positional argument: 'request' <==  n'est jamais à manipuler directement. transparent pour FASTAPI mais peut-être sorc de confusion pour moi
#
#  FastAPI appelle l'objet api_key_header en lui passant la requête HTTP
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
 
def get_current_user(api_key: str = Depends(api_key_header)) -> dict:
    
    """
        Dépendance FastAPI : vérifie la clé API et retourne l'utilisateur.
        Injectée dans chaque route qui nécessite une auth.

        api_key: Prend la valeur du header HTTP X-API-Key envoyé par le client dans la requête. 
                 Ce que FastAPI fait en coulisse :: Appelle api_key_header(request) et injecte le résultat
    """
    
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="API key invalide ou manquante")
        
    return API_KEYS[api_key]
 
def require_permission(permission: str):
    
    """
        Factory de dépendance : vérifie qu'un utilisateur a la permission requise.
        Usage : Depends(require_permission("chat"))
    """
    
    def check(user: dict = Depends(get_current_user)):
        if permission not in PERMISSIONS.get(user["role"], []):
            raise HTTPException(
                status_code=403,
                detail=f"Rôle '{user['role']}' insuffisant pour '{permission}'"
            )
        return user
        
    return check

# ── System Prompt ──────────────────────────────────────────────────────────

import yaml
#from database import get_latest_prompt_version

# Au démarrage : charger depuis fichier de base
with open("config/prompts.yaml") as f:
    prompts = yaml.safe_load(f)
    system_prompt = prompts["system_prompt"]
    
# ── Schémas Pydantic ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = system_prompt
    temperature: float = 0.7
 
class ChatResponse(BaseModel):
    response: str
    model: str
    user: str
    latency_ms: int

class StreamRequest(BaseModel):
    message: str
    system_prompt: str = "Tu es un assistant utile et concis."
    temperature: float = 0.7
    
# ── Routes ─────────────────────────────────────────────────────────────────

# Depends() est la fonction centrale du système d'injection de dépendances de FastAPI. 
# Elle permet de déclarer qu'une path operation function a besoin d'un autre callable (une fonction, par exemple) 
# qui doit être exécuté avant elle, et dont le résultat lui sera automatiquement fourni
# Depends() est utile pour :
#
# * Partager de la logique commune entre plusieurs endpoints (ex. pagination, filtres).
# * Partager des connexions à une base de données.
# * Appliquer de la sécurité : authentification, vérification de rôles, validation de clés API.
# * Gérer des ressources avec nettoyage (ex. sessions DB), via des dépendances avec yield.
    
@app.get("/health")
def health(user: dict = Depends(require_permission("health"))):
    
    """
        Route publique (avec auth minimale) pour vérifier que l'API est vivante.
    """
    
    return {"status": "ok", "user": user["user"]}
 
 
@app.get("/models")
def list_models(user: dict = Depends(require_permission("list_models"))):
    """Liste les modèles disponibles sur Ollama. Admin uniquement."""
    try:
        models = ollama.list()
        return {"models": [m["name"] for m in models.get("models", [])]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama inaccessible : {e}")
 
 
@app.post("/chat", response_model = ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    user: dict = Depends(require_permission("chat")),
):
    """
    Route principale de chat.
    Pipeline : validation → sécurité → LLM → réponse
    """
    # 1. Validation basique
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide") # chaine vide
    if len(req.message) > 4000:
        raise HTTPException(status_code=400, detail="Message trop long (max 4000 chars)") # chaine trop longue
 
    # 2. Détection d'injection
    if is_injection(req.message):
        logger.warning(f"[SECURITY] Injection détectée | user={user['user']} | ip={request.client.host}")
        raise HTTPException(status_code=400, detail="Requête refusée")

    # 2.bis Vérifier le cache
    # .....
    
    # 3. Appel LLM
    start = time.time()
    try:

        limiter.check(user["user"], user["role"], estimated_tokens = len(req.message) // 4)
        
        response = ollama.chat(
            model=user["model"],  # chaque utilisateur a son modèle assigné
            options={"temperature": req.temperature},
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user",   "content": req.message},
            ]
        )

        # limiter update
        limiter.record_usage( user["user"], tokens_used = sum([ 1 for token in reponse["message"]["content"] ]) )
        
    except Exception as e:
        logger.error(f"Erreur Ollama : {e}")
        raise HTTPException(status_code=503, detail="Modèle indisponible")
 
    latency = int((time.time() - start) * 1000)
    answer = response["message"]["content"]
 
    logger.info(f"[CHAT] user={user['user']} | latency={latency}ms | chars={len(answer)}")
 
    return ChatResponse(
        response=answer,
        model=user["model"],
        user=user["user"],
        latency_ms=latency,
    )

@app.post("/chat/stream")
def chat_stream(
    req: StreamRequest,
    user: dict = Depends(require_permission("chat")),
):

    """
        Rôle de chaque clé (côté client navigateur - javascript)
        Clé	Rôle	 Utilisation client
        token	     Le contenu à afficher	display += json.token
        done	     Signal de fin du stream	if (json.done) { stop_spinner(); }
        cached	     Indique que c'était du cache	if (json.cached) { show_badge("From Cache"); }

        voir index.html boucle while:102
    """
    
    # Même pipeline de sécurité que /chat
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")
    if is_injection(req.message):
        raise HTTPException(status_code=400, detail="Requête refusée")

    message = req.message.strip()
    
    # Regarder le crédit
    limiter.check(user["user"], user["role"], estimated_tokens = len(message) // 4)

    # Ensuite regarder si une répons en cache existe
    cached = cache.lookup(message)

    # Si réponse en cache ....
    if cached["hit"]: # True

        logger.info(f"[CACHE HIT] user={user['user']}, précédente query matché: {cached['matched_query']}, réponse: {cached['result']} ")
        #print(f"score: {cached['score']}, précédente query matché: {cached['matched_query']}, réponse: {cached['result']} ")

        # ... Définir un générateur (func) pour rejouer la réponsé mise en cache token par token
        def stream_cache():

            for token in cached['result'].split():
                yield f"data {json.dumps({'token': token + ' ', 'done': False})}\n\n"
            
            yield f"data {json.dumps({'token': '', 'done': True, 'cached': True})}\n\n"
        #fed

        return StreamingResponse(cached_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    else:

        # Pipeline RAG + streaming SSE générateur
        def rag_sse_stream():
            
            full_response = []
            
            start = time.time()
            
            try:
                for token in rag_stream(message):
                    
                    full_response.append(token)
                    
                    # Output validation sur chaque token
                    if has_pii_leak(token):
                        yield f"data: {json.dumps({'token': '', 'done': True, 'error': 'Réponse bloquée'})}\n\n"
                        return
                        
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
    
                # Fin du stream
                answer = "".join(full_response)
                latency = int((time.time() - start) * 1000)
                tokens_used = count_tokens(message + answer)
    
                # Mettre en cache la réponse + enregistrer usage
                cache.add(message, answer)
                limiter.record_usage(user["user"], tokens_used)
    
                logger.info(f"[CHAT] user={user['user']} | {latency}ms | {tokens_used} tokens")
                yield f"data: {json.dumps({'token': '', 'done': True, 'latency_ms': latency})}\n\n"
    
            except Exception as e:
                logger.error(f"Erreur RAG : {e}")
                yield f"data: {json.dumps({'token': '', 'done': True, 'error': 'Erreur interne'})}\n\n"
            
    # return (generateur, media-type, headers)
    return StreamingResponse(
        rag_sse_stream(),
        media_type="text/event-stream",
        headers={
            # Headers SSE obligatoires
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # CORS si le client est un navigateur
            "Access-Control-Allow-Origin": "*",
        }
    )

# ── Gestion globale des erreurs ────────────────────────────────────────────
 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all : ne jamais exposer les stack traces en production.
    Logger l'erreur complète, renvoyer un message générique.
    """
    logger.error(f"Erreur non gérée : {exc}", exc_info=True)
    return {"detail": "Erreur interne du serveur"}, 500
 
#> uvicorn main:app --reload
#ou
#> fastapi dev 