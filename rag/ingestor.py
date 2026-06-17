"""
rag/ingestor.py — Indexation de documents dans ChromaDB.

Pipeline offline qui transforme les PDFs en vecteurs stockés dans ChromaDB. 

Charge des fichiers texte/PDF, les découpe en chunks,
génère les embeddings via Ollama, stocke dans ChromaDB.

pip install chromadb pypdf2 ollama

ollama pull bge-m3               # 8192 tokens — Pour résoudre le problème de limite définitivement

Usage :
    python ingestor.py                        # indexe tout le dossier ./documents
    python ingestor.py --file contrat.pdf     # indexe un fichier spécifique
    python ingestor.py --reset                # vide la base et réindexe
"""

import re
import os
import hashlib
import argparse
import ollama
import chromadb
from rank_bm25 import BM25Okapi
import pickle

# ── Configuration ──────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Ajoute le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

DOCS_DIR       = config.DOCS_DIR
CHROMA_DIR     = config.CHROMA_DIR
COLLECTION     = config.COLLECTION
EMBED_MODEL    = config.EMBED_MODEL   # On utilisera un modèle d'embedding =/ car Llama3.2 est entraîné pour la génération de tokens — sa représentation interne optimise la prédiction du token suivant, pas la similarité sémantique
CHUNK_SIZE     = config.CHUNK_SIZE    # caractères par chunk
CHUNK_OVERLAP  = config.CHUNK_OVERLAP     # chevauchement entre chunks (préserve le contexte)

# ── Client ChromaDB ────────────────────────────────────────────────────────

client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},  # similarité cosinus pour les embeddings
)

# ── BM25 Index ─────────────────────────────────────────────────────────────

from bm25 import BM25Retriever

# Instance globale
bm25_retriever = BM25Retriever()

# ── Chunking ───────────────────────────────────────────────────────────────

#MAX_TOKENS = 400  # marge de sécurité sous la limite de 512 pour nomic-embed-text
#MAX_TOKENS = 5000  # marge de sécurité sous la limite de 8000 pour bge-m3

"""
def truncate_to_token_limit(text: str, max_tokens: int = MAX_TOKENS) -> str:
    
    '''
        Estimation : ~3 chars par token pour du français juridique.
        On tronque au dernier espace avant la limite.
    '''
    
    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text

    print(f"Truncated text")
    truncated = text[:max_chars]
    
    # Reculer jusqu'au dernier espace pour ne pas couper un mot
    last_space = truncated.rfind(' ')
    
    return truncated[:last_space] if last_space > 0 else truncated
    
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    '''
        Découpe le texte en chunks avec chevauchement.
        Le chevauchement évite de couper une phrase en deux parties
        qui se retrouveraient dans des chunks distincts.
    '''
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + size
        chunk = text[start:end]
        
        # Reculer jusqu'au dernier espace pour ne pas couper un mot
        if end < len(text) and ' ' in chunk:
            end   = start + chunk.rfind(' ')
            chunk = text[start:end]
            
        chunks.append(chunk.strip())
        start = end - overlap
        
    return [c for c in chunks if len(c) > 50]  # ignorer les micro-chunks

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    
    '''
        Chunking structuré : découpe en priorité sur les délimiteurs naturels
        du texte juridique avant de découper sur la longueur brute.
    '''
    import re

    # Nettoyer le bruit du PDF (en-têtes répétés, numéros de page, tirets de césure)
    text = re.sub(r'\n{3,}', '\n\n', text)                    # espaces multiples
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)              # tirets de coupure de ligne
    text = re.sub(r'\b\d{1,3}\b\s*\n', '\n', text)            # numéros de page isolés
    text = re.sub(r'(Journal officiel[^\n]*\n)', '', text)     # en-têtes JO répétés
    text = re.sub(r'(L \d+/\d+[^\n]*\n)', '', text)           # références de page JO

    # Essayer de découper sur les délimiteurs structurels du RGPD
    # Par ordre de priorité : article > alinéa numéroté > paragraphe > longueur brute
    DELIMITERS = [
        r'\n(?=Article\s+\d+)',           # "Article 17"
        r'\n(?=Article\s+\w+)',           # "Article Premier"
        r'\n(?=\(\d+\)\s)',               # "(1) Les États membres..."
        r'\n(?=[a-z]\)\s)',               # "a) la durée..."
        r'\n\n',                          # double saut de ligne
    ]

    chunks = []
    current = ""

    # Découper sur le premier délimiteur qui produit des chunks de bonne taille
    for delimiter in DELIMITERS:
        segments = re.split(delimiter, text)
        if len(segments) > 5:  # délimiteur utile s'il produit suffisamment de segments
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                if len(current) + len(seg) < size:
                    current += "\n\n" + seg if current else seg
                else:
                    if current:
                        chunks.append(current.strip())
                    # Overlap : garder la fin du chunk précédent
                    current = current[-overlap:] + "\n\n" + seg if overlap and current else seg
            if current:
                chunks.append(current.strip())
            break

    # Fallback : découpage brut si aucun délimiteur trouvé
    if not chunks:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end].strip())
            start = end - overlap

    return [c for c in chunks if len(c) > 100]
"""

INTRO_PATTERN    = r'RÈGLEMENTS'
CHAPITRE_PATTERN = r'^.*CHAPITRE\s+([IVX]+|\d+)$'
SECTION_PATTERN  = r'^.*Section\s+(\d+|[IVX]+)$'
ARTICLE_PATTERN  = r'^.*Article\s+(\d+|\w+)$'
HEADER = r'^4.5.2016 L 119/50 Journal officiel de l'Union européenne FR$'

def parsing_text(texte: str, pattern = None) -> list[dict]:

    '''
        New parsing method

        Sortie attendu:
            [{'numero': 'I', 'titre': 'Dispositions générales', 'type': 'chapitre', 'sections': [], 'articles': [
                {'numero': 'premier', 'titre': 'Objet et objectifs', 'type': 'article', 'contenu': "1. Le présent règlement établit blabla"}, {...}]
             {'numero': 'II', 'titre': 'Dispositions secondaire', 'type': 'chapitre', 'sections': [], 'articles': [ ....]}]
    '''
    
    #p = re.compile(r'^Chapitre\s+([IVX]+|\d+)\s*[:-]?\s*(.*)$').search(texte.lower()) # 2 groupes
    #print(p.group(0))
    #print(p.group(1))

    lines = texte.split('\n')
    chapitres = []
    current_chapitre = None
    current_section = None

    i = 0
    while i < len(lines):
        
        line = lines[i].strip()

        # Nettoyer le texte
        line = re.sub(r'\s+', ' ', line)

        if not line:
            i += 1
            continue
        
        # Vérifier si c'est un chapitre
        match_chapitre = re.match(CHAPITRE_PATTERN, line)
        if match_chapitre:

            # lire le titre à la ligne suivante
            titre = lines[i + 1].strip() if i + 1 < len(lines) else ''
            
            current_chapitre = {
                'numero': match_chapitre.group(1),
                'titre': titre,
                'type': 'chapitre',
                'sections': [],
                'articles': []
            }

            #print(match_chapitre)
            #print(current_chapitre)   
            
            current_section = None
            chapitres.append(current_chapitre)
            i += 2

            continue

        # Section
        match_section = re.match(SECTION_PATTERN, line)
        
        if match_section and current_chapitre:
            titre = lines[i + 1].strip() if i + 1 < len(lines) else ''
            current_section = {
                'numero': match_section.group(1),
                'titre': titre,
                'type': 'section',
                'articles': []
            }
            
            current_chapitre['sections'].append(current_section)
            i += 2
            
            continue
        
        # Article
        match_article = re.match(ARTICLE_PATTERN, line)
        
        if match_article and current_chapitre:
            titre = lines[i + 1].strip() if i + 1 < len(lines) else ''
            article = {
                'numero': match_article.group(1),
                'titre': titre,
                'type': 'article',
                'contenu': ''
            }
            
            if current_section:
                current_section['articles'].append(article)
            else:
                current_chapitre['articles'].append(article)
            
            i += 2
            continue
        
        # Contenu de l'article
        if current_chapitre:
            if current_section and current_section['articles']:
                current_section['articles'][-1]['contenu'] += line + ' '
            elif current_chapitre['articles']:
                current_chapitre['articles'][-1]['contenu'] += line + ' '
        
        i += 1
    
    return chapitres

# MAx token par chunk
MAX_TOKEN_CHUNK = 5000

def divide_content_v2(chapitre, article, section = None) -> list[dict]:

    '''
        Args:
            chapitre, article, section: Texte a spliter si nb de token > MAX_TOKEN_CHUNK
        Return(s):
            chunks list

        Amelioration_ eviter les chunk trop petit <
        Coupe aux frontières de phrases quand possible.
    '''

    chunks  = []
    # Taille totale EN TOKEN
    if section:
        size_chunk = len(article['contenu'].strip().split(' ')) + len(chapitre['titre'].split(' ')) + len(section['titre'].split(' ')) + len(article['titre'].split(' ')) + 3 #num chap, art, sec

        # Max len contenu sans texte enrichi
        max_content_token = MAX_TOKEN_CHUNK - (len(chapitre['titre'].split(' ')) + len(section['titre'].split(' ')) + len(article['titre'].split(' ')) + 3)
    else:
        size_chunk = len(article['contenu'].strip().split(' ')) + len(chapitre['titre'].split(' ')) + len(article['titre'].split(' ')) + 2 #num chap, art

        max_content_token = MAX_TOKEN_CHUNK - (len(chapitre['titre'].split(' ')) + len(article['titre'].split(' ')) + 2)

    print(f"Estimé {size_chunk} / {MAX_TOKEN_CHUNK} ...")
    
    # Si taille > Limite ==> il faut div
    if size_chunk > MAX_TOKEN_CHUNK: # +3 numéro/id

        print(f"...splitting texte avec un contenu <= {max_content_token}")

        # Chercher a couper au frontière de phrase en utilisant un Lookbehind positif (cherche une position précédée par ., ! ou ?) en excluant les numéro de liste (1. blla, 2. blal
        phrases = re.split(r'(?<=[.!?])\s*\n\s*(?=[0-9]+\.|[A-Z])', article['contenu'].strip()) # prec: r'(?<=[.!?])\s+'
        chunk_actuel = ''
        idx = 1
        
        for phrase in phrases:
        #do
            tokens_phrase = len(phrase.split(' '))
            tokens_chunk  = len(chunk_actuel.split(' '))

            if tokens_chunk + tokens_phrase <= max_content_token:
                chunk_actuel += phrase + ' '
            else:
                if chunk_actuel.strip():

                    print(f"Un chunk créer idx: {idx} avec contenu {chunk_actuel[:10]}")
                    
                    chunk = {
                    'contenu': chunk_actuel.strip(),
                    'metadata': { # ChromeDB attent, des str ou des ints aucun objet complexe: list, ...
                            'split': idx,
                            'chapitre_numero': chapitre['numero'],
                            'chapitre_titre': chapitre['titre'],
                            'section_numero': section['numero'] if section is not None else "None",
                            'section_titre': section['titre'] if section is not None else "None",
                            'article_numero': article['numero'],
                            'article_titre': article['titre']
                                    }
                    }
                    chunks.append(chunk)
                    idx += 1

                # Nouveau départ
                chunk_actuel = phrase + ' '
        #done
        
        if chunk_actuel.strip():                  
            chunk = {
                    'contenu': chunk_actuel.strip(),
                    'metadata': {
                            'split': idx,
                            'chapitre_numero': chapitre['numero'],
                            'chapitre_titre': chapitre['titre'],
                            'section_numero': section['numero'] if section is not None else "None",
                            'section_titre': section['titre'] if section is not None else "None",
                            'article_numero': article['numero'],
                            'article_titre': article['titre']
                                    }
            }
            chunks.append(chunk)
    else:

        chunk = {
                'contenu': article['contenu'].strip(),
                'metadata': {
                        'split' : 0,
                        'chapitre_numero': chapitre['numero'],
                        'chapitre_titre': chapitre['titre'],
                        'section_numero': section['numero'] if section is not None else "None",
                        'section_titre': section['titre'] if section is not None else "None",
                        'article_numero': article['numero'],
                        'article_titre': article['titre']
                            }
        }
        chunks.append(chunk)

    return chunks
    
def prepare_chunks_for_indexing(chapitres: list[dict]) -> list[dict]:
    
    """
        Transforme la structure parsée en chunks avec métadonnées

        Sortie Attendue:
            {'contenu': "1. Le présent règlement établit des règles relatives à la protection des personnes physiques à l'égard du traitement des données à caractère personnel et des règles relatives à la
            libre circulation de ces données. 2. Le (...) données à caractère personnel.",
             'metadata': {'split': 1,
              'chapitre_numero': 'I',
              'chapitre_titre': 'Dispositions générales',
              'section_numero': None,
              'section_titre': None,
              'article_numero': 'premier',
              'article_titre': 'Objet et objectifs'}}
    """
    
    chunks = []
    
    for chapitre in chapitres:
        if chapitre['sections']:
            for section in chapitre['sections']:
                for article in section['articles']:

                    parts = divide_content_v2(chapitre, article, section)

                    for chunk in parts:
                    #do
                        chunks.append(chunk)
                    #done
        else:
            for article in chapitre['articles']:

                parts = divide_content_v2(chapitre, article, None)

                for chunk in parts:
                #do
                    chunks.append(chunk)
                #done
    
    return chunks
# ── Lecture de fichiers ────────────────────────────────────────────────────

def read_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
"""
def read_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("  [WARN] PyPDF2 non installé — pip install pypdf2")
        return ""
"""

import fitz
def read_pdf(path: str) -> str:

    """
        Version greedy, bcp de texte en mémoire
    """
    
    doc = fitz.open(path)

    contenu = ""
    for page_num in range(len(doc)):
        
        page = doc[page_num]
        
        # Extraire avec option de découpage
        texte = page.get_text()
        
        # Supprimer les zones d'en-têtes/pieds (en haut/bas de page)
        lignes = texte.split('\n')
        contenu = contenu+'\n'.join(lignes[1:-1])  # Ignorer les 1 premières et dernières lignes corrspondant au pied de page et entête (peut ne pas toujours être régulier
        
        if page_num + 1 < 2: print(f"Page {page_num + 1}:\n{contenu}\n")

    #print(f"Doc size: {len(contenu)}")
    return contenu
    
def read_file(path: str) -> str:

    # Split the extension from a pathname. - Extension is everything from the last dot to the end
    ext = os.path.splitext(path)[1].lower() # if path blabla/aaaa/corax.pdf => (blabla/aaaa/, pdf)
    if ext == ".pdf":   return read_pdf(path)
    if ext == ".txt":   return read_txt(path)
    if ext == ".md":    return read_txt(path)
    print(f"  [SKIP] Format non supporté : {ext}")
    return ""

# ── Embedding ──────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    """
        Génère les embeddings en batch via Ollama.
    """

    # Tronquer texte si nécessaire
    #texts = [truncate_to_token_limit(text) for text in texts]
    texts = [text for text in texts]
    
    result = ollama.embed(model=EMBED_MODEL, input = texts)

    #print(f"Embeddings: {len(result['embeddings']), result['embeddings'][:10]}")
    return result["embeddings"]

# ── Indexation ─────────────────────────────────────────────────────────────

def doc_id(filepath: str, chunk_idx: int) -> str:
    
    """
        ID unique = hash du chemin + index du chunk.
    """
    
    h = hashlib.md5(filepath.encode()).hexdigest()[:8]
    
    return f"{h}-{chunk_idx}"

def build_embed_text(chunk: dict, meta: dict) -> str:
    
    """
        Enrichir le texte pour l'embedding — combine métadonnées + contenu.
        Le contenu brut reste séparé pour l'affichage au LLM.
    """

    parts = [f"{k} : {v}" for k, v in meta.items()]
    #parts = [f"Chapitre {meta['chapitre_numero']} : {meta['chapitre_titre']}"]

    #if meta["section_numero"] != "None":
    #    parts.append(f"Section {meta['section_numero']} : {meta['section_titre']}")

    #parts.append(f"Article {meta['article_numero']} : {meta['article_titre']}")
    parts.append(chunk[0])

    return "\n".join(parts)
    
def ingest_file(path: str) -> int:
    
    """
        Indexe un fichier. Retourne le nombre de chunks ajoutés.
    """
    
    print(f"\n[INGEST] {os.path.basename(path)}")

    text = read_file(path)
    if not text.strip():
        print("  Fichier vide ou illisible.")
        return 0

    print(f"len text {len(text)}")
    
    # Parser le texte du document
    parsing = parsing_text(text)

    print(f"len parsing {len(parsing)}")

    # Obtenir une list de chunk et leur metadonnées --> return list[dict]
    chunks = prepare_chunks_for_indexing(parsing)

    print(f"  {len(chunks)} chunks générés")

    # BM25
    bm25_retriever.chunks.extend(chunks)

    all_chunk_ids = []  # Collecter les IDs
    
    # Traiter par batch de 10 (limite de l'API Ollama en local) et vectoriser le 'contenue'
    batch_size = 10
    total = 0
    """
    for i in range(0, len(chunks), batch_size):
        
        batch   = chunks[i:i + batch_size]
        ids     = [doc_id(path, i + j) for j in range(len(batch))]
        vectors = embed(batch)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=batch,
            metadatas=[{
                "source":   os.path.basename(path),
                "chunk":    i + j,
                "filepath": path,
            } for j in range(len(batch))],
        )
        
        all_chunk_ids.extend(ids)  # Stocker les IDs
        
        total += len(batch)
        print(f"  Batch {i // batch_size + 1} : {len(batch)} chunks indexés")
    """

    
    for id_, chunk in enumerate(chunks):
        
        # Ajouter à ChromeDB
        collection.add(
            ids        = [str(id_)],
            documents  = [chunk['contenu']],  # Texte brut stocké
            #embeddings = [embed([chunk['contenu']])[0]],      # Embedding du contexte
            embeddings = [ embed(build_embed_text( [chunk['contenu']], chunk['metadata'] ) )[0] ],      # Embedding du contexte
            metadatas  = [chunk['metadata']]  # Métadonnées stockées
        )
        
        # Up ids
        #all_chunk_ids.extend(id_)
        all_chunk_ids.append(id_)

    print(f"All_chunk_id: {all_chunk_ids}")
    # Créer l'index BM25 avec IDs
    bm25_retriever.index(chunks, all_chunk_ids)
    
    return len(chunks)

def ingest_directory(directory: str) -> None:
    
    """Indexe tous les fichiers supportés d'un dossier."""
    
    supported = {".pdf", ".txt", ".md"}
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"Aucun fichier supporté dans {directory}")
        return

    print(f"[INGEST] {len(files)} fichiers trouvés dans {directory}")
    total_chunks = sum(ingest_file(f) for f in files)
    print(f"\n[DONE] {total_chunks} chunks indexés au total")
    print(f"[STATS] Collection : {collection.count()} documents")

    # Créer l'index BM25 après ingestion
    print(f"\n[BM25] Création de l'index...")

# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  help="Indexer un fichier spécifique")
    parser.add_argument("--reset", action="store_true", help="Vider la base avant indexation")
    args = parser.parse_args()

    if args.reset:
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(COLLECTION)
        print("[RESET] Base vectorielle vidée")

    if args.file:
        ingest_file(args.file)
    else:
        os.makedirs(DOCS_DIR, exist_ok=True)
        ingest_directory(DOCS_DIR)
