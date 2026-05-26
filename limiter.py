#limiter.py — Rate limiting par utilisateur + comptage de tokens.
""" 
Deux niveaux de limitation :
- RPM (Requests Per Minute) : nombre d'appels par minute
- TPD (Tokens Per Day)      : budget de tokens journalier par utilisateur
 
Stockage : en mémoire (dict) — suffisant pour un seul process.
En production multi-process : remplacer par Redis (voir cache.py).
 
Ajouter à main.py :
    from limiter import RateLimiter, count_tokens
    limiter = RateLimiter()
 
    # Dans la route /chat, avant l'appel LLM :
    limiter.check(user["user"], estimated_tokens=len(req.message) // 4)
"""
 
import time
import re
from collections import defaultdict
from fastapi import HTTPException
 
# ── Configuration des limites par rôle ────────────────────────────────────
 
LIMITS = {
    "admin":    {"rpm": 60,  "tpd": 500_000}, # pas plus de 60 requête/min , le tout ne dépassant pas 500.000 token par jour. 
    "user":     {"rpm": 3,  "tpd": 100_000},
    #"readonly": {"rpm": 2,   "tpd": 10_000}, Pas de chat pour toi
}

# ── Comptage de tokens (approximation sans librairie externe) ──────────────

def count_tokens(text: str) -> int:
    """
    Approximation du nombre de tokens sans tiktoken ni sentencepiece.
    Règle empirique : ~4 caractères par token en anglais, ~3 en français
    (les mots français sont plus courts mais les accents comptent).
 
    Pour une précision production : utiliser tiktoken (OpenAI) ou
    les tokenizers HuggingFace correspondant au modèle utilisé.

    "Bonjour, monde!" serait tokenisé en :
    Bonjour (via w+),
    , (via [^ws]),
    monde (via w+),
    ! (via [^ws])

    Pourquoi les caract_re accentué coûte plus cher que leur équivalent ASCII ?

        Les accents et caractères spéciaux sont moins fréquents dans les données d'entraînement
        Les modèles de langage (LLM) sont principalement entraînés sur des corpus en anglais ou dans des langues utilisant l'alphabet latin sans accents (ex. : anglais, néerlandais, indonésien).
        Les caractères accentués (é, è, ç, ü, ñ, etc.) sont moins fréquents dans ces corpus, donc :
        Ils sont moins susceptibles d'être fusionnés en un seul token par BPE.
        Ils sont souvent décomposés en plusieurs tokens (ex. : é → e + un token spécial pour l'accent, ou ç → c + ¸).

        Donc il forment un seul token, donc augmente la taille du voca

        voir explication plus bas
    """
    
    # Découpage grossier : mots + ponctuation - Tokeniser un texte en séparant les mots (\w+) des caractères spéciaux ou de ponctuation ([^\w\s])
    tokens = re.findall(r"\w+|[^\w\s]", text) # \w : Correspond à tout caractère alphanumérique 1,+ OU Crochets avec un ^ à l'intérieur. Cela signifie "tout caractère **qui n'est pas** mot/esapce
    
    # Pénalité pour les mots longs (souvent découpés en plusieurs tokens) => ça vient du fonctionnement interne des tokenizers (voir: BPE (Byte Pair Encoding))
    # C'est pourquoi count_tokens() applique une pénalité sur les mots longs — c'est une approximation de ce phénomène sans charger un vrai tokenizer
    penalty = sum(1 for t in tokens if len(t) > 8)

    def decoupage_llama(text: str, model: str = None):
        
        # Vrai découpage d'une phrase, vous pouvez utiliser le tokenizer Llama directement :
        from transformers import AutoTokenizer
    
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        text = "anticonstitutionnellement"
        tokens = tokenizer.tokenize(text)
        print(tokens)        # ['▁anti', 'con', 'sti', 'tu', 'tion', 'nellement']
        print(len(tokens))   # 6
    
    return len(tokens) + penalty

# ── Rate Limiter ───────────────────────────────────────────────────────────
 
class RateLimiter:
    
    def __init__(self):
        
        # RPM : timestamps des N dernières requêtes par utilisateur
        # Structure : {"alice": [timestamp1, timestamp2, ...]}
        self._rpm_windows: dict[str, list[float]] = defaultdict(list)
 
        # TPD : {user: {"date": "2025-01-15", "tokens": 4200}}
        self._tpd_usage: dict[str, dict] = defaultdict(
            lambda: {"date": "", "tokens": 0}
        )
 
    def _today(self) -> str:
        
        return time.strftime("%Y-%m-%d")
 
    def _check_rpm(self, user: str, role: str) -> None:
        
        """Vérifie la limite de requêtes par minute (fenêtre glissante de 60s)."""

        # Pour ce 'role' quel est la lilmite fixé ?
        limit = LIMITS[role]["rpm"]

        # Prendre le temps actuel (Return the current time in seconds depuis l'époque Unix (le 1er janvier 1970 à 00:00:00 UTC)
        now   = time.time()

        # fenêtre de 60 secondes t - 60
        window_start = now - 60
 
        # Garder uniquement les timestamps dans la fenêtre courante
        self._rpm_windows[user] = [
            ts for ts in self._rpm_windows[user] if ts > window_start
        ]
 
        if len(self._rpm_windows[user]) >= limit:
            oldest = self._rpm_windows[user][0]
            retry_in = int(60 - (now - oldest)) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Limite RPM atteinte ({limit} req/min). Réessayez dans {retry_in}s.",
                headers={"Retry-After": str(retry_in)},
            )
 
        self._rpm_windows[user].append(now)
 
    def _check_tpd(self, user: str, role: str, tokens: int) -> None:
        
        """Vérifie le budget de tokens journalier."""
        
        limit  = LIMITS[role]["tpd"]
        today  = self._today()
        usage  = self._tpd_usage[user]
 
        # Reset si nouveau jour
        if usage["date"] != today:
            usage["date"]   = today
            usage["tokens"] = 0
 
        if usage["tokens"] + tokens > limit:
            remaining = limit - usage["tokens"]
            raise HTTPException(
                status_code=429,
                detail=f"Budget journalier atteint ({limit} tokens/jour). "
                       f"Restant : {remaining} tokens. Reset à minuit.",
            )
 
    def check(self, user: str, role: str, estimated_tokens: int = 100) -> None:
        
        """Point d'entrée unique — appeler avant chaque requête LLM."""
        
        self._check_rpm(user, role)
        self._check_tpd(user, role, estimated_tokens)
 
    def record_usage(self, user: str, tokens_used: int) -> None:
        
        """Appeler APRÈS la réponse LLM avec le vrai nombre de tokens."""
        
        self._tpd_usage[user]["tokens"] += tokens_used
 
    def get_stats(self, user: str, role: str) -> dict:
        
        """Retourne les stats de consommation d'un utilisateur."""
        
        today  = self._today()
        usage  = self._tpd_usage[user]
        limits = LIMITS[role]
 
        tokens_today = usage["tokens"] if usage["date"] == today else 0
        now          = time.time()
        rpm_count    = len([ts for ts in self._rpm_windows[user] if ts > now - 60])
 
        return {
            "rpm":        {"used": rpm_count,    "limit": limits["rpm"]},
            "tpd":        {"used": tokens_today, "limit": limits["tpd"]},
            "tpd_percent": round(tokens_today / limits["tpd"] * 100, 1),
        }


"""  Pourquoi les caract_re accentué coûte plus cher que leur équivalent ASCII ?
4. Les caractères spéciaux sont souvent traités comme des tokens séparés
            Les symboles (-, _, @, #, etc.) et les accents sont souvent décomposés en tokens individuels.
            Exemple :
            "Jean-François" → ["Jean", "-", "Fran", "çois"] (4 tokens),
            "e-mail" → ["e", "-", "mail"] (3 tokens).
            Pourquoi ?
            
            Ces caractères sont peu fréquents dans les données d'entraînement, donc BPE ne les fusionne pas avec d'autres tokens.
            
        5. Impact sur les performances du modèle
            Plus de tokens = plus de calculs :
            Chaque token doit être traité par le modèle (embedding, attention, etc.).
            Un mot comme "anticonstitutionnellement" (6 tokens) prend 6 fois plus de ressources qu'un mot comme "chat" (1 token).
            Plus de mémoire nécessaire :
            Le modèle doit stocker des embeddings pour chaque token, y compris les tokens rares ou spéciaux.

        6. Biais dans les architectures et hyperparamètres (Biais anglo-centré)
            a) Taille des context windows
            La plupart des LLM ont des fenêtres de contexte limitées (ex. : 2048 tokens pour GPT-3.5, 4096 pour Llama 2).
            Problème :
            Les langues à mots longs (allemand, finnois) ou à caractères complexes (chinois, japonais) consomment plus de tokens pour le même texte.
            Exemple :
            La phrase "Je suis un étudiant en informatique" (français) pourrait prendre plus de tokens que son équivalent anglais "I am a computer science student", réduisant l'efficacité du modèle.

        7. Solutions pour réduire le biais anglo-centré
            a) Utiliser des modèles multilingues
            Certains LLM sont spécifiquement entraînés pour plusieurs langues :
            Mistral 7B : Bon support du français.
            BLOOM : Entraîné sur 46 langues.
            XLM-RoBERTa : Optimisé pour le multilingue.
            Avantage :
            Vocabulaire adapté aux langues non anglaises.
            Meilleure tokenisation pour les mots longs ou accentués.
            b) Normalisation des textes
            Étape de prétraitement :
            Remplacer les accents par leurs équivalents ASCII (é → e).
            Convertir les nombres en mots ("2024" → "deux mille vingt-quatre").
            Inconvénient :
            Peut causer des pertes d'information (ex. : "été" vs "ete").
            c) Fine-tuning sur des corpus locaux
            Adapter un modèle existant à une langue spécifique :
            Exemple : Fine-tuner un modèle comme Llama 2 sur un corpus en français.
            Outils :
            LoRA (Low-Rank Adaptation) pour un fine-tuning efficace.
            Hugging Face Transformers pour entraîner des modèles multilingues.
            d) Utiliser des tokenizers spécialisés
            Certains tokenizers gèrent mieux les langues non anglaises :
            SentencePiece (utilisé par Mistral) : Meilleure gestion des caractères spéciaux.
            FastText : Optimisé pour les langues à caractères complexes (ex. : chinois, japonais).


"""