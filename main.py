"""
Main FastAPI application for RAG system
Handles API endpoints and server configuration
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
from pathlib import Path

# Import from our modules
from config import OUT_PATH, EMB_PATH, TOP_K, RUN_MODE, HOST, PORT, PDF_DIR
from ingestion import (
    extract_pages, chunk_pages, load_chunks, load_embeddings,
    embed_texts, save_embeddings, build_embeddings_from_chunks
)
from search import (
    build_idf, hybrid_search, process_query, generate_response, should_refuse_query
)


# ---------- Global State ----------
CHUNKS = []
IDF = {}
EMBEDDINGS = {}


# ---------- Pydantic Models ----------
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    query: str
    intent: str
    answer: str
    citations: List[dict]
    note: str = ""


# ---------- Lifespan Management ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    global CHUNKS, IDF, EMBEDDINGS
    
    if OUT_PATH.exists():
        CHUNKS = load_chunks(OUT_PATH)
        IDF = build_idf(CHUNKS)
        print(f"Loaded {len(CHUNKS)} chunks for serving")
    else:
        print("Warning: No chunks file found. Upload PDFs via /upload endpoint.")
    
    if EMB_PATH.exists():
        EMBEDDINGS = load_embeddings(EMB_PATH)
        print(f"Loaded {len(EMBEDDINGS)} embeddings")
    else:
        print("Warning: No embeddings file found. Semantic search will be unavailable.")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG FastAPI Application",
    description="Retrieval-Augmented Generation system with PDF ingestion and hybrid search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")


# ---------- API Endpoints ----------
@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse("index.html")


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    global CHUNKS, IDF, EMBEDDINGS
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    all_chunks = []
    
    # Process each PDF
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Save temporarily
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
        import json
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    
    # Embed chunks
    ids = [ch["id"] for ch in all_chunks]
    texts = [ch["text"] for ch in all_chunks]
    vecs = embed_texts(texts)
    save_embeddings(ids, vecs)
    
    # Update global data
    CHUNKS = all_chunks
    IDF = build_idf(CHUNKS)
    EMBEDDINGS = load_embeddings(EMB_PATH)
    
    return {"message": f"Uploaded and processed {len(files)} PDF(s). Total chunks: {len(all_chunks)}"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Query the RAG system"""
    if not CHUNKS or not IDF:
        return QueryResponse(
            query=req.query,
            intent="error",
            answer="",
            citations=[],
            note="No documents loaded. Please upload PDFs first via /upload endpoint."
        )
    
    # Process query
    intent, transformed_query = process_query(req.query)
    
    # Perform hybrid search
    results = hybrid_search(transformed_query, CHUNKS, IDF, EMBEDDINGS, req.top_k)
    
    # Check if we should refuse
    top_score = results[0]["score"] if results else 0.0
    should_refuse, refusal_message = should_refuse_query(intent, top_score)
    
    if should_refuse:
        return QueryResponse(
            query=req.query,
            intent=intent,
            answer="",
            citations=[],
            note=refusal_message
        )
    
    # Generate response
    answer = generate_response(req.query, results)
    
    # Format citations
    citations = [
        {
            "file": r["file"],
            "page": r["page"],
            "snippet": r["snippet"],
            "score": round(r["score"], 3)
        }
        for r in results
    ]
    
    return QueryResponse(
        query=req.query,
        intent=intent,
        answer=answer,
        citations=citations
    )


@app.get("/demo")
def demo_api():
    """Demo endpoint with a sample query"""
    if not CHUNKS or not IDF:
        return {
            "query": "What items are included in the property sale by default?",
            "results": [],
            "note": "No chunks loaded. Run ingestion first."
        }
    
    from search import search_keyword
    return {
        "query": "What items are included in the property sale by default?",
        "results": search_keyword("What items are included in the property sale by default?", CHUNKS, IDF, TOP_K)
    }


# ---------- Main Runner ----------
if __name__ == "__main__":
    if RUN_MODE == "embed":
        build_embeddings_from_chunks()
    
    elif RUN_MODE == "search":
        if not OUT_PATH.exists():
            raise FileNotFoundError(f"Could not find {OUT_PATH}. Set RUN_MODE='ingest' first.")
        
        from search import search_keyword
        chunks = load_chunks(OUT_PATH)
        idf = build_idf(chunks)
        
        DEFAULT_QUERY = "What items are included in the property sale by default?"
        print(f"Loaded chunks: {len(chunks)}")
        print(f'Default query: "{DEFAULT_QUERY}"')
        
        for r in search_keyword(DEFAULT_QUERY, chunks, idf, TOP_K):
            print(f"[{r['score']}] {r['file']} p.{r['page']}  {r['snippet']}")
    
    elif RUN_MODE == "serve":
        if not OUT_PATH.exists():
            print("Warning: no chunks file yet. Start server anyway, but /demo will be empty.")
        
        import uvicorn
        uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
    
    else:
        raise ValueError("RUN_MODE must be one of: 'embed', 'search', 'serve'")
