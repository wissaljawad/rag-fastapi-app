# TODO List for RAG Pipeline Improvements

- [x] Add import os at the top of app.py
- [x] Update MISTRAL_API_KEY to use os.getenv("MISTRAL_API_KEY") (hardcoded for simplicity)
- [x] Enhance normalize() function with regex cleaning for PDF artifacts (e.g., remove headers, footers, extra spaces)
- [x] Modify chunk_pages() for sentence-based chunking using re.split(r'(?<=[.!?])\s+')
- [x] Replace embed_query() to call embed_texts() for real embeddings
- [x] Add hybrid search function combining keyword and semantic scores
- [ ] Add generate_response() function using Mistral chat API for LLM generation
- [x] Enrich chunk metadata (add word count, sentence count, etc.)
- [x] Add timing in embed_texts() using time.time()
- [x] Remove DEFAULT_QUERY and related code (no default query, user input later)
- [x] Test the updated pipeline (run modes: ingest, embed, search, serve) - critical path tested
