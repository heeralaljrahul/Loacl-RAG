"""Role-playing game engine with long-term memory.

The retrieval half is shared with the document RAG (`rag/`): story memories
are written into the same hybrid index as pseudo-documents, so they get the
same dense+BM25 fusion, cross-encoder rerank and relevance floor.

What is *different* here, and what makes a 500-turn campaign hold together,
is what gets indexed and what never does.  See `memory.py`.
"""
