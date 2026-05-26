"""
streaming.py — Streaming SSE (Server-Sent Events) avec FastAPI + Ollama.
 
SSE : connexion HTTP persistante où le serveur envoie des événements
au client au fur et à mesure. Format standard :
    data: {"token": "Bon"}\n\n
    data: {"token": "jour"}\n\n
    data: [DONE]\n\n
 
Ajouter à main.py : app.include_router(router)
"""
 
#from fastapi import APIRouter, Depends, HTTPException
#from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json
import time
 
# Réutilise l'auth de main.py
from main import require_permission, is_injection
 
router = APIRouter()

# ── RateLimiter ────────────────────────────────────────────────────────────

#from limiter import RateLimiter, count_tokens

#limiter = RateLimiter()

# ── Générateur SSE ─────────────────────────────────────────────────────────
 
def token_generator(user: str, role: str, message: str, system_prompt: str, temperature: float, model: str):
    """
    Générateur Python qui yield des chunks SSE.
    FastAPI + StreamingResponse consomme ce générateur
    et envoie chaque chunk au client dès qu'il est disponible.
 
    Format SSE strict :
    - Chaque événement commence par "data: "
    - Chaque événement se termine par "\n\n" (double saut de ligne)
    - L'événement final est "data: [DONE]"
    """
    start = time.time()
    token_count = 0
    
    try:
        
        # stream=True : Ollama retourne un itérateur de chunks
        stream = ollama.chat(
            model=model,
            options={"temperature": temperature},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": message},
            ],
            stream=True, # A chaque Token généré ....
        )
 
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                token_count += 1
                # Format SSE : "data: <json>\n\n"
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n" # ... générer une sortie
 
        # Événement final avec les métriques
        latency = int((time.time() - start) * 1000)
        meta = json.dumps({"latency_ms": latency, "tokens": token_count})

        # limiter update
        limiter.record_usage(user, tokens_used = token_count)
        
        yield f"data: {meta}\n\n"
        yield "data: [DONE]\n\n"
 
    except Exception as e:
        # En cas d'erreur en cours de stream : envoyer l'erreur puis fermer
        error = json.dumps({"error": str(e)})
        yield f"data: {error}\n\n"
        yield "data: [DONE]\n\n"
 
# ── Route streaming ────────────────────────────────────────────────────────

"""
@router.post("/chat/stream")
def chat_stream(
    req: StreamRequest,
    user: dict = Depends(require_permission("chat")),
):
    # Même pipeline de sécurité que /chat
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")
    if is_injection(req.message):
        raise HTTPException(status_code=400, detail="Requête refusée")
        
    return StreamingResponse(
        token_generator(
            message=req.message.strip(),
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            model=user["model"],
        ),
        media_type="text/event-stream",
        headers={
            # Headers SSE obligatoires
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # CORS si le client est un navigateur
            "Access-Control-Allow-Origin": "*",
        }
    )
"""
#FIN