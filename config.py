"""
Configuration file for RAG FastAPI Application
Contains all constants, paths, and configuration parameters
"""
from pathlib import Path

# ---------- Paths ----------
PDF_DIR = Path(".")  # Directory containing PDF files
OUT_PATH = PDF_DIR / "chunks.jsonl"  # Output path for chunks
EMB_PATH = PDF_DIR / "embeddings.jsonl"  # Output path for embeddings

# ---------- Chunking Parameters ----------
CHUNK_SIZE = 800  # Characters per chunk
OVERLAP = 150  # Overlap between chunks

# ---------- Search Parameters ----------
TOP_K = 5  # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum score to answer confidently
RERANK_BOOST_RECENT = 0.1  # Boost for more recent pages
RERANK_BOOST_LENGTH = 0.05  # Boost for longer, more detailed chunks

# ---------- API Configuration ----------
MISTRAL_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your Mistral API key
MISTRAL_MODEL = "mistral-small-latest"  # LLM model for generation
EMBED_MODEL = "mistral-embed"  # Model for embeddings
EMBED_BATCH_SIZE = 32  # Batch size for embedding generation

# ---------- Server Configuration ----------
RUN_MODE = "serve"  # Options: "ingest" | "embed" | "search" | "serve"
HOST = "127.0.0.1"
PORT = 8000

# ---------- Stopwords for Tokenization ----------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "it", "that", "this", "as", "at", "from", "be", "been", "but", "if", "not",
    "we", "you", "your", "our"
}

# ---------- Query Transformations ----------
QUERY_TRANSFORMATIONS = {
    "what's": "what is",
    "it's": "it is",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not"
}

# ---------- Refusal Policies ----------
# PII detection patterns
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b\d{16}\b',  # Credit card
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
]

# Legal/medical keywords
LEGAL_KEYWORDS = ['legal advice', 'lawsuit', 'sue', 'attorney', 'lawyer', 'court case']
MEDICAL_KEYWORDS = ['medical advice', 'diagnosis', 'treatment', 'medication', 'prescription', 'doctor']

# Greeting keywords
GREETING_KEYWORDS = {"hi", "hello", "thanks", "thank you", "bye", "goodbye"}
