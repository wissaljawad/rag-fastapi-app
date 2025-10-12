# TODO List for Full RAG Pipeline Implementation

## Core Requirements
- [x] Data Ingestion: Implement text extraction and chunking from PDF files (sentence-based chunking done)
- [x] Data Ingestion: Develop API endpoint to upload PDF files (/upload endpoint)
- [x] Query Processing: Intent detection (basic check for greetings done, enhance)
- [x] Query Processing: Transform query to improve retrieval (e.g., expand abbreviations, remove stop words)
- [x] Semantic Search: Design search mechanism using processed query (hybrid search done)
- [x] Semantic Search: Combine semantic and keyword results (hybrid search averages scores)
- [ ] Post-processing: Merge and re-rank results (enhance hybrid search with better ranking)
- [x] Generation: Call LLM with prompt template to generate answers (generate_response done)
- [ ] UI: Implement a chat UI to interact with the system

## FastAPI Endpoints
- [x] Ingestion: POST /upload - Upload one or more PDF files
- [x] Querying: POST /query - Query the system with user questions (hybrid search)
- [ ] Querying: Enhance /query to include generation and citations

## Bonus Features
- [ ] Citations: Refuse to answer if top-k chunks don't meet similarity threshold
- [ ] Answer shaping: Switch templates by intent; structured outputs
- [ ] Hallucination filters: Post-hoc evidence check
- [ ] Query refusal policies: PII, legal/medical disclaimers

## Miscellaneous
- [ ] Update MISTRAL_API_KEY to "CF2DvjIoshzasO0mtBkPj44fo2nXDwPk" (skipped as user said it doesn't work)
- [ ] Add README.md with system design, diagrams, how to run
- [ ] Implement UI (e.g., simple HTML/JS chat interface)
- [ ] Test full pipeline: ingestion via API, querying, generation
