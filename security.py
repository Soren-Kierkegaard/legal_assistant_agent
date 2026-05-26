import re, unicodedata, json
 
INJECTION_PATTERNS = [
    r"ignore\s+(toutes?\s+)?tes\s+instructions",
    r"(oublie|efface)\s+ce\s+qui\s+précède",
    r"répète\s+(ton\s+)?system\s+prompt",
    r"tu\s+es\s+maintenant",
    r"new\s+instructions?\s*:",
]
 
HOMOGLYPH_MAP = {
    'І':'I','і':'i','е':'e','Е':'E','а':'a','А':'A',
    'о':'o','О':'O','р':'p','с':'c','С':'C',
    'ο':'o','Ο':'O','α':'a',
}

PII_PATTERNS = [
    r"sk-[a-zA-Z0-9\-]{10,}",
    r"[A-Z]{2}\d{2}[\s]?(\d{4}[\s]?){4,}",   # IBAN
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",          # téléphone
]
 
def normalize(text: str) -> str:
    
    text = unicodedata.normalize("NFKC", text)
    
    return ''.join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)
 
def is_injection(text: str) -> bool:
    
    normalized = normalize(text).lower()
    
    return any(re.search(p, normalized) for p in INJECTION_PATTERNS)

def has_pii_leak(text: str) -> bool:
    return any(re.search(p, text) for p in PII_PATTERNS)