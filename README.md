# RAG FastAPI Application

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI and Mistral AI, featuring hybrid search (semantic + keyword), intelligent query processing, and a user-friendly chat interface.

## 🎯 Overview

This system enables users to upload PDF documents and ask questions about their content using natural language. It combines traditional keyword search with modern semantic search to retrieve relevant information, then uses Mistral AI to generate accurate, contextual answers with citations.

## 💡 Example Questions to Try

Questions to ask:
- "What items are included in the property sale by default?"
- "What are the seller's obligations under this agreement?"

**Note**: The system will refuse to answer if:
- The question is too vague or conversational (e.g., "hello", "thanks")
- The retrieved information doesn't meet the confidence threshold
- The query contains personal identifiable information (PII)
- The question asks for legal or medical advice

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (HTML/JS Chat Interface)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                            │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │  POST /upload    │              │   POST /query    │        │
│  │  (Ingestion)     │              │   (Retrieval)    │        │
│  └────────┬─────────┘              └────────┬─────────┘        │
│           │                                  │                   │
│           ▼                                  ▼                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │            Document Processing Pipeline              │       │
│  │  1. PDF Text Extraction (pypdf)                     │       │
│  │  2. Sentence-based Chunking (800 chars, 150 overlap)│       │
│  │  3. Embedding Generation (Mistral API)              │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Query Processing Pipeline               │       │
│  │  1. Intent Detection (greeting vs. question)        │       │
│  │  2. Query Transformation (stopwords, normalization) │       │
│  │  3. Hybrid Search (Keyword TF-IDF + Semantic)       │       │
│  │  4. Result Re-ranking & Deduplication               │       │
│  │  5. LLM Generation with Citations                   │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  • chunks.jsonl (Document chunks with metadata)                 │
│  • embeddings.jsonl (Vector embeddings for semantic search)     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│                    Mistral AI API                                │
│  • Embedding Generation (mistral-embed)                         │
│  • Answer Generation (mistral-small-latest)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Functionality
- ✅ **PDF Ingestion**: Upload multiple PDF files via REST API
- ✅ **Intelligent Chunking**: Sentence-based chunking with configurable overlap
- ✅ **Hybrid Search**: Combines TF-IDF keyword search with semantic embeddings
- ✅ **Query Processing**: Intent detection and query transformation
- ✅ **LLM Generation**: Context-aware answers using Mistral AI
- ✅ **Citations**: Automatic source attribution with page numbers
- ✅ **Web UI**: Interactive chat interface for easy interaction

### Advanced Features
- ✅ **No External Dependencies**: Custom implementation of search and RAG (no LangChain, LlamaIndex, etc.)
- ✅ **No Vector Database**: Embeddings stored locally in JSONL format
- ✅ **CORS Support**: Frontend-backend communication enabled
- ✅ **Error Handling**: Comprehensive error handling with detailed logging
- ✅ **Similarity Threshold**: Refuses to answer when confidence is too low
- ✅ **Re-ranking**: Advanced result scoring and deduplication

## 📁 Project Structure

```
rag-fastapi-app/
├── main.py              # FastAPI application and API endpoints
├── config.py            # Configuration constants and parameters
├── ingestion.py         # PDF processing, chunking, and embeddings
├── search.py            # Search algorithms and query processing
├── index.html           # Web UI chat interface
├── chunks.jsonl         # Stored document chunks (generated)
├── embeddings.jsonl     # Stored embeddings (generated)
├── test_query.py        # API testing script
└── README.md            # This file
```

### Module Descriptions

**main.py** - FastAPI Application
- API endpoint definitions (`/upload`, `/query`, `/demo`)
- CORS middleware configuration
- Server startup and lifespan management
- Request/response models (Pydantic)

**config.py** - Configuration
- All constants and parameters (paths, API keys, thresholds)
- Stopwords and query transformations
- Refusal policies (PII, legal, medical keywords)

**ingestion.py** - Document Processing
- PDF text extraction and normalization
- Sentence-based chunking with overlap
- Embedding generation via Mistral API
- Storage utilities (load/save chunks and embeddings)

**search.py** - Search & Query Processing
- Tokenization and TF-IDF keyword search
- Semantic search with cosine similarity
- Hybrid search combining both methods
- Query intent detection and transformation
- LLM response generation with Mistral AI
- Result re-ranking and deduplication

## 📋 Requirements

### Python Dependencies
```
fastapi>=0.104.0
uvicorn>=0.24.0
pypdf>=3.17.0
pydantic>=2.5.0
requests>=2.31.0
```

### External Services
- **Mistral AI API**: For embeddings and text generation
  - Requires API key (configure in `config.py`)

## ️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-fastapi-app
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn pypdf pydantic requests
```

4. **Set up Mistral API Key**
Edit `config.py` and update the `MISTRAL_API_KEY` variable:
```python
MISTRAL_API_KEY = "your-api-key-here"
```

## 🎮 Usage

### Starting the Server

```bash
python main.py
```

The server will start at `http://127.0.0.1:8000`

### Using the Web Interface

1. Open your browser and navigate to: `http://127.0.0.1:8000`
2. Upload PDF files using the `/upload` endpoint (via API or future UI enhancement)
3. Ask questions in the chat interface
4. Receive answers with citations

### API Endpoints

#### 1. Upload PDFs
**Endpoint**: `POST /upload`

**Description**: Upload one or more PDF files for ingestion and indexing.

**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response**:
```json
{
  "message": "Uploaded and processed 2 PDF(s). Total chunks: 150"
}
```

#### 2. Query the System
**Endpoint**: `POST /query`

**Description**: Ask questions about the uploaded documents.

**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What items are included in the property sale?",
    "top_k": 5
  }'
```

**Response**:
```json
{
  "query": "What items are included in the property sale?",
  "answer": "According to the documents, items included in the property sale by default include...",
  "citations": [
    {
      "file": "Purchase_Agreement.pdf",
      "page": 1,
      "snippet": "The following items are included in the sale..."
    }
  ]
}
```

#### 3. Demo Endpoint
**Endpoint**: `GET /demo`

**Description**: Test the system with a predefined query.

**Request**:
```bash
curl http://127.0.0.1:8000/demo
```

#### 4. API Documentation
**Endpoint**: `GET /docs`

**Description**: Interactive API documentation (Swagger UI).

Access at: `http://127.0.0.1:8000/docs`

## 🧠 How It Works

### 1. Document Ingestion Pipeline

```python
PDF Upload → Text Extraction → Normalization → Chunking → Embedding → Storage
```

**Text Extraction**:
- Uses `pypdf.PdfReader` to extract text from each page
- Removes headers, footers, and page numbers
- Normalizes whitespace and line breaks

**Chunking Strategy**:
- **Method**: Sentence-based chunking
- **Size**: 800 characters per chunk
- **Overlap**: 150 characters between chunks
- **Rationale**: Preserves semantic coherence while ensuring context continuity

**Embedding Generation**:
- Model: `mistral-embed` via Mistral AI API
- Dimension: 1024-dimensional vectors
- Batch processing for efficiency

### 2. Query Processing Pipeline

```python
User Query → Intent Detection → Query Transformation → Hybrid Search → Re-ranking → LLM Generation
```

**Intent Detection**:
- Identifies greetings (hello, hi, hey) vs. actual questions
- Prevents unnecessary searches for conversational inputs

**Query Transformation**:
- Removes stopwords (the, is, are, etc.)
- Converts to lowercase
- Preserves important query terms

**Hybrid Search**:
1. **Keyword Search (TF-IDF)**:
   - Computes term frequency-inverse document frequency
   - Scores chunks based on term overlap
   - Fast and interpretable

2. **Semantic Search (Embeddings)**:
   - Generates query embedding using Mistral API
   - Computes cosine similarity with chunk embeddings
   - Captures semantic meaning beyond keywords

3. **Score Combination**:
   - Averages keyword and semantic scores
   - Normalizes scores to [0, 1] range
   - Configurable weighting (currently 50/50)

**Re-ranking**:
- Deduplicates similar chunks
- Sorts by combined score
- Returns top-k results (default: 5)

### 3. Answer Generation

**Prompt Template**:
```
You are a helpful assistant. Answer the user's question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{retrieved_chunks}

Question: {user_query}

Answer:
```

**Generation**:
- Model: `mistral-small-latest`
- Temperature: 0.7 (balanced creativity/accuracy)
- Max tokens: 500
- Includes citations with file names and page numbers

### 4. Similarity Threshold & Refusal

The system refuses to answer when:
- Top result score < 0.3 (insufficient evidence)
- No chunks retrieved
- Intent is conversational (greeting)

Response: `"I don't have enough information in the knowledge base to answer this question confidently."`

## 📊 Technical Decisions & Considerations

### Chunking Strategy
**Decision**: Sentence-based chunking with 800 char size and 150 char overlap

**Rationale**:
- Preserves semantic coherence (complete sentences)
- 800 chars ≈ 2-3 sentences, optimal for context
- 150 char overlap ensures no information loss at boundaries
- Balances retrieval precision vs. context completeness

**Alternatives Considered**:
- Fixed-size chunking: Faster but breaks sentences
- Paragraph-based: Too large, reduces precision
- Sliding window: More chunks but redundant

### Hybrid Search
**Decision**: Combine TF-IDF keyword search with semantic embeddings

**Rationale**:
- Keyword search: Fast, interpretable, good for exact matches
- Semantic search: Captures meaning, handles paraphrasing
- Hybrid: Best of both worlds, more robust

**Implementation**:
- Equal weighting (50/50) between keyword and semantic
- Cosine similarity for semantic comparison
- TF-IDF with custom IDF calculation

### No External Libraries
**Decision**: Custom implementation of search and RAG

**Rationale**:
- Full control over algorithms
- No dependency on LangChain, LlamaIndex, etc.
- Demonstrates understanding of underlying concepts
- Easier to debug and customize

### Storage Format
**Decision**: JSONL files for chunks and embeddings

**Rationale**:
- Simple, human-readable format
- No database setup required
- Easy to inspect and debug
- Sufficient for moderate-scale applications
- Line-by-line processing for large files

## 🔧 Configuration

Key parameters in `main.py`:

```python
CHUNK_SIZE = 800        # Characters per chunk
OVERLAP = 150           # Overlap between chunks
TOP_K = 5              # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum score to answer
MISTRAL_MODEL = "mistral-small-latest"  # LLM model
EMBED_MODEL = "mistral-embed"           # Embedding model
```

## 📚 Libraries & Technologies Used

### Core Framework
- **FastAPI** (0.104+): Modern web framework for building APIs
  - [Documentation](https://fastapi.tiangolo.com/)
  - Automatic API documentation with Swagger UI
  - Type validation with Pydantic

### PDF Processing
- **pypdf** (3.17+): PDF text extraction
  - [Documentation](https://pypdf.readthedocs.io/)
  - Pure Python, no external dependencies
  - Supports various PDF formats

### LLM & Embeddings
- **Mistral AI API**: Text generation and embeddings
  - [Documentation](https://docs.mistral.ai/)
  - Models: `mistral-small-latest`, `mistral-embed`
  - REST API with Python requests library

### Web Server
- **Uvicorn**: ASGI server for FastAPI
  - [Documentation](https://www.uvicorn.org/)
  - Auto-reload for development
  - Production-ready performance

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **HTML5/CSS3**: Modern, responsive UI
- **Fetch API**: Async communication with backend

## 🧪 Testing

### Manual Testing
1. Start the server: `python main.py`
2. Upload test PDFs via `/upload` endpoint
3. Query via web UI or `/query` endpoint
4. Verify citations and answer quality

### Automated Testing
```bash
python test_query.py
```

Tests the `/query` endpoint with sample questions.

## 🐛 Troubleshooting

### Issue: Server won't start
**Solution**: Check if port 8000 is already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <process_id>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Issue: CORS errors in browser
**Solution**: Ensure CORS middleware is enabled in `main.py`
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

### Issue: Mistral API errors
**Solution**: 
1. Check API key is valid
2. Verify internet connection
3. Check Mistral API status: https://status.mistral.ai/

### Issue: No results returned
**Solution**:
1. Verify PDFs were uploaded successfully
2. Check `chunks.jsonl` and `embeddings.jsonl` exist
3. Lower `SIMILARITY_THRESHOLD` if too restrictive

## 📈 Performance Considerations

- **Embedding Generation**: Batched for efficiency (up to 100 texts per request)
- **Search**: O(n) complexity where n = number of chunks (acceptable for <10k chunks)
- **Memory**: Embeddings loaded into memory for fast retrieval
- **Scalability**: For >100k chunks, consider:
  - Vector database (Pinecone, Weaviate, Qdrant)
  - Approximate nearest neighbor search (FAISS, Annoy)
  - Distributed processing

## 📝 License

This project is provided as-is for educational and evaluation purposes.

## 👥 Contributing

This is a demonstration project. For questions or suggestions, please open an issue.

## 🙏 Acknowledgments

- Mistral AI for providing the LLM and embedding APIs
- FastAPI team for the excellent web framework
- pypdf maintainers for PDF processing capabilities
