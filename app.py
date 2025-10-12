# app.py
# Minimal one file RAG scaffold, ingest + keyword search + optional API
# How to use:
# 1) Set RUN_MODE = "ingest" to build chunks.jsonl from your PDFs
# 2) Set RUN_MODE = "search" to run a default search without typing input
# 3) Set RUN_MODE = "serve" to start a FastAPI server and test /docs or /demo
import os
from pathlib import Path
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re, json, uuid, math
from collections import Counter, defaultdict
import requests, time

# ---------- paths and knobs ----------
PDF_DIR = Path(".")  # put your PDFs here
OUT_PATH = PDF_DIR / "chunks.jsonl"  # save next to your PDFs
EMB_PATH = PDF_DIR / "embeddings.jsonl"
CHUNK_SIZE = 800
OVERLAP = 150
RUN_MODE = "serve"         # choose: "ingest" | "embed" | "search" | "serve"
TOP_K = 5

# ---------- ingestion ----------
def normalize(text: str) -> str:
    # Remove PDF artifacts like headers, footers, page numbers
    t = re.sub(r'\n\d+\n', '\n', text)  # Remove page numbers
    t = re.sub(r'^\s*Page \d+\s*$', '', t, flags=re.MULTILINE)  # Remove "Page X" lines
    t = re.sub(r'^\s*Chapter \d+.*$', '', t, flags=re.MULTILINE)  # Remove chapter headers
    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize(text)
        if text:
            pages.append({"page": i, "text": text})
    return pages

def chunk_pages(pages, file_name):
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
                        "start": len(txt) - len(current_chunk) - len(sentence),  # approximate start
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

def ingest_folder():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")
    all_chunks = []
    for pdf in PDF_DIR.glob("*.pdf"):
        pages = extract_pages(pdf)
        chunks = chunk_pages(pages, pdf.name)
        all_chunks.extend(chunks)
        print(f"{pdf.name}: {len(chunks)} chunks")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # overwrite for clean runs
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"Total chunks: {len(all_chunks)}")
    print("Wrote:", OUT_PATH.resolve())

def dedupe_chunks_on_disk(path: Path):
    """Optional: remove duplicates if you ever accidentally append twice."""
    if not path.exists():
        print("No chunks file to dedupe.")
        return
    seen = set()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            key = (ch["file"], ch["page"], ch.get("start"), ch.get("end"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(ch)
    tmp = path.with_suffix(".dedup.jsonl")
    with tmp.open("w", encoding="utf-8") as out:
        for ch in rows:
            out.write(json.dumps(ch, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"Deduped. Kept {len(rows)} chunks. Updated {path}")

# ---------- keyword search (no external search libs) ----------
STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are",
    "was","were","it","that","this","as","at","from","be","been","but","if","not",
    "we","you","your","our"
}
_tok_findall = re.compile(r"[A-Za-z0-9]+").findall

def tokenize(t: str):
    return [w.lower() for w in _tok_findall(t) if w.lower() not in STOP]

def load_chunks(path: Path):
    chunks = []
    if not path.exists():
        return chunks
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# NOTE: fix — load the same key we save ("vec"); also handle missing file gracefully

def load_embeddings(path: Path):
    if not path.exists():
        return {}
    embs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            embs[row["id"]] = row["vec"]
    return embs

def build_idf(chunks):
    N = len(chunks)
    df = defaultdict(int)
    for ch in chunks:
        for t in set(tokenize(ch["text"])):
            df[t] += 1
    # classic smoothed idf
    return {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

def tfidf(tokens, idf):
    tf = Counter(tokens)
    return {t: c * idf[t] for t, c in tf.items() if t in idf}

def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a)
    na = math.sqrt(sum(w*w for w in a.values()))
    nb = math.sqrt(sum(w*w for w in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)

# ---------- semantic embeddings (Mistral) ----------
MISTRAL_API_KEY = "OvaGGKbmryAJjIo3rg3vRumbJelS6nsk"


def embed_texts(texts):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.mistral.ai/v1/embeddings"
    out = []
    B = 32
    start_time = time.time()
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        r = requests.post(url, headers=headers, json={"model": "mistral-embed", "input": batch})
        r.raise_for_status()
        out.extend([row["embedding"] for row in r.json()["data"]])
        time.sleep(0.1)
    end_time = time.time()
    print(f"Embedded {len(texts)} texts in {end_time - start_time:.2f} seconds")
    return out

def save_embeddings(ids, vecs, path=EMB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid, v in zip(ids, vecs):
            f.write(json.dumps({"id": cid, "vec": v}) + "\n")

def build_embeddings_from_chunks():
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"No chunks to embed at {OUT_PATH}")
    chunks = load_chunks(OUT_PATH)
    ids = [ch["id"] for ch in chunks]
    texts = [ch["text"] for ch in chunks]
    vecs = embed_texts(texts)
    save_embeddings(ids, vecs)
    print(f"Embedded {len(vecs)} chunks. Wrote to {EMB_PATH}")

def embed_query(query):
    return embed_texts([query])[0]


def generate_response(query, context_chunks):
    """Generate a response using Mistral chat API based on query and retrieved chunks."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.mistral.ai/v1/chat/completions"

    # Prepare context from chunks
    context = "\n".join([f"Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(context_chunks[:3])])  # Limit to top 3 for brevity

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    data = {
        "model": "mistral-medium",  # or another available model
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7
    }

    r = requests.post(url, headers=headers, json=data)
    r.raise_for_status()
    response = r.json()["choices"][0]["message"]["content"]
    return response


def search_keyword(query, chunks, idf, k=TOP_K):
    qv = tfidf(tokenize(query), idf)
    scored = []
    for ch in chunks:
        dv = tfidf(tokenize(ch["text"]), idf)
        s = cosine(qv, dv)
        if s > 0:
            scored.append((s, ch))
    scored.sort(reverse=True, key=lambda x: x[0])

    # small result dedupe so overlap does not show duplicates
    seen = set()
    results = []
    for s, ch in scored:
        key = (ch["file"], ch["page"], ch["text"][:160])
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "score": round(s, 4),
            "file": ch["file"],
            "page": ch["page"],
            "id": ch["id"],
            "snippet": ch["text"][:200].replace("\n", " ")
        })
        if len(results) == k:
            break
    return results


def search_semantic(query, chunks, embeddings, k=TOP_K):
    qvec = embed_query(query)
    scored = []
    for ch in chunks:
        cid = ch["id"]
        if cid not in embeddings:
            continue
        score = cosine_similarity(qvec, embeddings[cid])
        scored.append((score, ch))

    scored.sort(reverse=True, key=lambda x: x[0])
    results = []
    seen = set()

    for score, ch in scored:
        key = (ch["file"], ch["page"], ch["text"][:160])
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "score": round(score, 4),
            "file": ch["file"],
            "page": ch["page"],
            "id": ch["id"],
            "snippet": ch["text"][:200].replace("\n", " ")
        })
        if len(results) == k:
            break

    return results


def search_hybrid(query, chunks, idf, embeddings, k=TOP_K):
    kw_results = search_keyword(query, chunks, idf, k * 2)  # Get more for hybrid
    sem_results = search_semantic(query, chunks, embeddings, k * 2)

    # Combine scores: average of keyword and semantic
    combined = {}
    for res in kw_results + sem_results:
        cid = res["id"]
        if cid not in combined:
            combined[cid] = {"score": 0, "count": 0, "data": res}
        combined[cid]["score"] += res["score"]
        combined[cid]["count"] += 1

    # Average the scores
    for cid in combined:
        combined[cid]["score"] /= combined[cid]["count"]

    # Sort by combined score
    sorted_combined = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

    results = []
    seen = set()
    for item in sorted_combined:
        res = item["data"]
        key = (res["file"], res["page"], res["text"][:160])
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "score": round(item["score"], 4),
            "file": res["file"],
            "page": res["page"],
            "id": res["id"],
            "snippet": res["snippet"]
        })
        if len(results) == k:
            break
    return results

# ---------- FastAPI server ----------
CHUNKS = None
IDF = None
EMBEDDINGS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CHUNKS, IDF, EMBEDDINGS
    if OUT_PATH.exists():
        CHUNKS = load_chunks(OUT_PATH)
        IDF = build_idf(CHUNKS)
        EMBEDDINGS = load_embeddings(EMB_PATH)
        print(f"Loaded {len(CHUNKS)} chunks for serving")
    else:
        print("Warning: chunks.jsonl not found at startup. Run RUN_MODE='ingest' first.")
    yield

app = FastAPI(title="RAG demo", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "RAG API is running. Visit /docs for API documentation, /demo for a demo query."}

class QueryIn(BaseModel):
    query: str
    top_k: int = TOP_K

@app.post("/query")
def query_api(body: QueryIn):
    if not CHUNKS or not IDF or not EMBEDDINGS:
        return {"results": [], "note": "Missing data. Run ingestion and ensure embeddings are loaded."}

    q = body.query.strip()

    if q.lower() in {"hi", "hello", "thanks", "thank you"} or len(q.split()) < 2:
        return {"results": [], "note": "Ask a question about your PDFs."}

    hybrid_results = search_hybrid(q, CHUNKS, IDF, EMBEDDINGS, body.top_k)

    return {
        "query": q,
        "hybrid_results": hybrid_results
    }

@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Upload one or more PDF files for ingestion."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_chunks = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

        # Save the uploaded file
        temp_path = PDF_DIR / file.filename
        with temp_path.open("wb") as f:
            f.write(await file.read())

        # Process the PDF
        pages = extract_pages(temp_path)
        chunks = chunk_pages(pages, file.filename)
        all_chunks.extend(chunks)
        print(f"{file.filename}: {len(chunks)} chunks")

    # Save chunks
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # Embed chunks
    ids = [ch["id"] for ch in all_chunks]
    texts = [ch["text"] for ch in all_chunks]
    vecs = embed_texts(texts)
    save_embeddings(ids, vecs)

    # Update global data
    global CHUNKS, IDF, EMBEDDINGS
    CHUNKS = all_chunks
    IDF = build_idf(CHUNKS)
    EMBEDDINGS = load_embeddings(EMB_PATH)

    return {"message": f"Uploaded and processed {len(files)} PDF(s). Total chunks: {len(all_chunks)}"}

@app.get("/demo")
def demo_api():
    if not CHUNKS or not IDF:
        return {"query": "What items are included in the property sale by default?", "results": [], "note": "No chunks loaded. Run ingestion first."}
    return {"query": "What items are included in the property sale by default?", "results": search_keyword("What items are included in the property sale by default?", CHUNKS, IDF, TOP_K)}

# ---------- simple runner ----------
if __name__ == "__main__":
    # if RUN_MODE == "ingest":
    #     ingest_folder()
        # dedupe_chunks_on_disk(OUT_PATH)

    if RUN_MODE == "embed":
        build_embeddings_from_chunks()

    elif RUN_MODE == "search":
        if not OUT_PATH.exists():
            raise FileNotFoundError(f"Could not find {OUT_PATH}. Set RUN_MODE='ingest' first.")
        chunks = load_chunks(OUT_PATH)
        idf = build_idf(chunks)
        print(f"Loaded chunks: {len(chunks)}")
        print(f'Default query: "{DEFAULT_QUERY}"')
        for r in search_keyword(DEFAULT_QUERY, chunks, idf, TOP_K):
            print(f"[{r['score']}] {r['file']} p.{r['page']}  {r['snippet']}")

    elif RUN_MODE == "serve":
        if not OUT_PATH.exists():
            print("Warning: no chunks file yet. Start server anyway, but /demo will be empty.")
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

    else:
        raise ValueError("RUN_MODE must be one of: 'ingest', 'search', 'serve'")