"""
Search module for RAG FastAPI Application
Handles keyword search, semantic search, hybrid search, query processing, and LLM generation
"""
import re
import math
import requests
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from config import (
    TOP_K, SIMILARITY_THRESHOLD, RERANK_BOOST_RECENT, RERANK_BOOST_LENGTH,
    MISTRAL_API_KEY, MISTRAL_MODEL, STOP_WORDS, QUERY_TRANSFORMATIONS,
    PII_PATTERNS, LEGAL_KEYWORDS, MEDICAL_KEYWORDS, GREETING_KEYWORDS
)
from ingestion import embed_query


# ---------- Tokenization ----------
def tokenize(text: str) -> List[str]:
    """Tokenize text into words, removing stopwords"""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]


# ---------- TF-IDF Search ----------
def build_idf(chunks: List[Dict]) -> Dict[str, float]:
    """Build IDF scores for all terms in the corpus"""
    doc_count = len(chunks)
    term_doc_count = Counter()
    
    for ch in chunks:
        unique_terms = set(tokenize(ch["text"]))
        for term in unique_terms:
            term_doc_count[term] += 1
    
    idf = {}
    for term, count in term_doc_count.items():
        idf[term] = math.log(doc_count / count)
    
    return idf


def compute_tf(text: str) -> Dict[str, float]:
    """Compute term frequency for a text"""
    tokens = tokenize(text)
    if not tokens:
        return {}
    
    term_count = Counter(tokens)
    max_count = max(term_count.values())
    
    tf = {}
    for term, count in term_count.items():
        tf[term] = count / max_count
    
    return tf


def tfidf_score(query_tf: Dict[str, float], doc_tf: Dict[str, float], idf: Dict[str, float]) -> float:
    """Compute TF-IDF similarity score between query and document"""
    score = 0.0
    for term in query_tf:
        if term in doc_tf and term in idf:
            score += query_tf[term] * doc_tf[term] * idf[term]
    return score


def search_keyword(query: str, chunks: List[Dict], idf: Dict[str, float], top_k: int = TOP_K) -> List[Dict]:
    """Perform keyword-based search using TF-IDF"""
    query_tf = compute_tf(query)
    
    results = []
    for ch in chunks:
        doc_tf = compute_tf(ch["text"])
        score = tfidf_score(query_tf, doc_tf, idf)
        if score > 0:
            results.append({
                "id": ch["id"],
                "file": ch["file"],
                "page": ch["page"],
                "text": ch["text"],
                "score": score,
                "snippet": ch["text"][:200]
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------- Semantic Search ----------
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def search_semantic(query: str, chunks: List[Dict], embeddings: Dict[str, List[float]], top_k: int = TOP_K) -> List[Dict]:
    """Perform semantic search using embeddings"""
    if not embeddings:
        return []
    
    query_vec = embed_query(query)
    
    results = []
    for ch in chunks:
        if ch["id"] not in embeddings:
            continue
        
        doc_vec = embeddings[ch["id"]]
        score = cosine_similarity(query_vec, doc_vec)
        
        if score > 0:
            results.append({
                "id": ch["id"],
                "file": ch["file"],
                "page": ch["page"],
                "text": ch["text"],
                "score": score,
                "snippet": ch["text"][:200]
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------- Hybrid Search ----------
def rerank_results(results: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Re-rank results based on additional signals"""
    chunk_map = {ch["id"]: ch for ch in chunks}
    
    for r in results:
        ch = chunk_map.get(r["id"])
        if not ch:
            continue
        
        # Boost for more recent pages (assuming higher page numbers are more recent)
        page_boost = (ch["page"] / 100) * RERANK_BOOST_RECENT
        
        # Boost for longer, more detailed chunks
        length_boost = (ch.get("word_count", 0) / 100) * RERANK_BOOST_LENGTH
        
        r["score"] += page_boost + length_boost
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def hybrid_search(query: str, chunks: List[Dict], idf: Dict[str, float], embeddings: Dict[str, List[float]], top_k: int = TOP_K) -> List[Dict]:
    """Perform hybrid search combining keyword and semantic search"""
    # Get results from both methods
    keyword_results = search_keyword(query, chunks, idf, top_k * 2)
    semantic_results = search_semantic(query, chunks, embeddings, top_k * 2)
    
    # Combine results by averaging scores
    combined = {}
    for r in keyword_results:
        combined[r["id"]] = {
            "id": r["id"],
            "file": r["file"],
            "page": r["page"],
            "text": r["text"],
            "snippet": r["snippet"],
            "keyword_score": r["score"],
            "semantic_score": 0.0
        }
    
    for r in semantic_results:
        if r["id"] in combined:
            combined[r["id"]]["semantic_score"] = r["score"]
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "file": r["file"],
                "page": r["page"],
                "text": r["text"],
                "snippet": r["snippet"],
                "keyword_score": 0.0,
                "semantic_score": r["score"]
            }
    
    # Average the scores
    results = []
    for cid, data in combined.items():
        avg_score = (data["keyword_score"] + data["semantic_score"]) / 2
        results.append({
            "id": data["id"],
            "file": data["file"],
            "page": data["page"],
            "text": data["text"],
            "score": avg_score,
            "snippet": data["snippet"]
        })
    
    # Re-rank and return top-k
    results = rerank_results(results, chunks)
    return results[:top_k]


# ---------- Query Processing ----------
def detect_intent(query: str) -> str:
    """Detect the intent of the query"""
    query_lower = query.lower().strip()
    
    # Check for greetings
    if any(word in query_lower for word in GREETING_KEYWORDS):
        return "greeting"
    
    # Check for legal/medical queries
    if any(keyword in query_lower for keyword in LEGAL_KEYWORDS):
        return "legal"
    if any(keyword in query_lower for keyword in MEDICAL_KEYWORDS):
        return "medical"
    
    # Check for PII
    for pattern in PII_PATTERNS:
        if re.search(pattern, query):
            return "pii"
    
    return "general"


def transform_query(query: str) -> str:
    """Transform query to improve retrieval"""
    # Expand contractions
    for contraction, expansion in QUERY_TRANSFORMATIONS.items():
        query = re.sub(r'\b' + contraction + r'\b', expansion, query, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query


def process_query(query: str) -> Tuple[str, str]:
    """Process query: detect intent and transform"""
    intent = detect_intent(query)
    transformed_query = transform_query(query)
    return intent, transformed_query


# ---------- LLM Generation ----------
def generate_response(query: str, context_chunks: List[Dict]) -> str:
    """Generate a response using Mistral AI"""
    if not context_chunks:
        return "I don't have enough information to answer that question."
    
    # Build context from chunks
    context = "\n\n".join([
        f"[Source: {ch['file']}, Page {ch['page']}]\n{ch['text']}"
        for ch in context_chunks
    ])
    
    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information to answer, say so
- Be concise and accurate
- Cite the source (file and page) when possible

Answer:"""
    
    # Call Mistral API
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.mistral.ai/v1/chat/completions"
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating response: {str(e)}"


def should_refuse_query(intent: str, top_score: float) -> Tuple[bool, str]:
    """Determine if query should be refused based on intent and confidence"""
    # Refuse greetings
    if intent == "greeting":
        return True, "Hello! I'm here to answer questions about the documents. Please ask me a specific question."
    
    # Refuse legal queries
    if intent == "legal":
        return True, "I cannot provide legal advice. Please consult with a qualified attorney for legal matters."
    
    # Refuse medical queries
    if intent == "medical":
        return True, "I cannot provide medical advice. Please consult with a qualified healthcare professional for medical matters."
    
    # Refuse PII queries
    if intent == "pii":
        return True, "I cannot process queries containing personal identifiable information (PII)."
    
    # Refuse if confidence is too low
    if top_score < SIMILARITY_THRESHOLD:
        return True, "I don't have enough relevant information to answer that question confidently."
    
    return False, ""
