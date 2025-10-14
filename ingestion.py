"""
Ingestion module for RAG FastAPI Application
Handles PDF processing, text extraction, chunking, and embedding generation
"""
import json
import uuid
import re
import time
import requests
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict

from config import (
    PDF_DIR, OUT_PATH, EMB_PATH, CHUNK_SIZE, OVERLAP,
    MISTRAL_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE
)


# ---------- Text Normalization ----------
def normalize(text: str) -> str:
    """Remove PDF artifacts like headers, footers, page numbers"""
    t = re.sub(r'\n\d+\n', '\n', text)  # Remove page numbers
    t = re.sub(r'^\s*Page \d+\s*$', '', t, flags=re.MULTILINE)  # Remove "Page X" lines
    t = re.sub(r'^\s*Chapter \d+.*$', '', t, flags=re.MULTILINE)  # Remove chapter headers
    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


# ---------- PDF Processing ----------
def extract_pages(pdf_path: Path) -> List[Dict]:
    """Extract text from each page of a PDF"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize(text)
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: List[Dict], file_name: str) -> List[Dict]:
    """Chunk pages into smaller pieces using sentence-based chunking"""
    chunks = []
    for p in pages:
        txt = p["text"]
        # Sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', txt)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > CHUNK_SIZE:
                if current_chunk:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "file": file_name,
                        "page": p["page"],
                        "start": len(txt) - len(current_chunk) - len(sentence),
                        "end": len(txt) - len(sentence),
                        "text": current_chunk.strip(),
                        "word_count": len(current_chunk.split()),
                        "sentence_count": len(re.findall(r'[.!?]', current_chunk))
                    })
                    current_chunk = sentence
                else:
                    # If sentence is too long, split it
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "file": file_name,
                        "page": p["page"],
                        "start": len(txt) - len(sentence),
                        "end": len(txt),
                        "text": sentence.strip(),
                        "word_count": len(sentence.split()),
                        "sentence_count": len(re.findall(r'[.!?]', sentence))
                    })
            else:
                current_chunk += " " + sentence
                
        if current_chunk:
            chunks.append({
                "id": str(uuid.uuid4()),
                "file": file_name,
                "page": p["page"],
                "start": len(txt) - len(current_chunk),
                "end": len(txt),
                "text": current_chunk.strip(),
                "word_count": len(current_chunk.split()),
                "sentence_count": len(re.findall(r'[.!?]', current_chunk))
            })
    return chunks


def ingest_folder() -> None:
    """Ingest all PDFs from the PDF directory"""
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")
    
    all_chunks = []
    for pdf in PDF_DIR.glob("*.pdf"):
        pages = extract_pages(pdf)
        chunks = chunk_pages(pages, pdf.name)
        all_chunks.extend(chunks)
        print(f"{pdf.name}: {len(chunks)} chunks")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    
    print(f"Total chunks: {len(all_chunks)}")
    print("Wrote:", OUT_PATH.resolve())


# ---------- Embedding Generation ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using Mistral API"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.mistral.ai/v1/embeddings"
    out = []
    
    start_time = time.time()
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        r = requests.post(url, headers=headers, json={"model": EMBED_MODEL, "input": batch})
        r.raise_for_status()
        out.extend([row["embedding"] for row in r.json()["data"]])
        time.sleep(0.1)
    
    end_time = time.time()
    print(f"Embedded {len(texts)} texts in {end_time - start_time:.2f} seconds")
    return out


def save_embeddings(ids: List[str], vecs: List[List[float]], path: Path = EMB_PATH) -> None:
    """Save embeddings to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid, v in zip(ids, vecs):
            f.write(json.dumps({"id": cid, "vec": v}) + "\n")


def load_chunks(path: Path) -> List[Dict]:
    """Load chunks from disk"""
    chunks = []
    if not path.exists():
        return chunks
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_embeddings(path: Path) -> Dict[str, List[float]]:
    """Load embeddings from disk"""
    if not path.exists():
        return {}
    embs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            embs[row["id"]] = row["vec"]
    return embs


def build_embeddings_from_chunks() -> None:
    """Build embeddings for all chunks"""
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"No chunks to embed at {OUT_PATH}")
    
    chunks = load_chunks(OUT_PATH)
    ids = [ch["id"] for ch in chunks]
    texts = [ch["text"] for ch in chunks]
    vecs = embed_texts(texts)
    save_embeddings(ids, vecs)
    print(f"Embedded {len(vecs)} chunks. Wrote to {EMB_PATH}")


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query"""
    return embed_texts([query])[0]
