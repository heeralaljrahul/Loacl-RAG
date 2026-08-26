@echo off
REM ===================================================================
REM  Settings for an RTX 3080 (10 GB) + 16 GB RAM + i9-11900K.
REM  Edit here, not in the Python.
REM ===================================================================

REM --- Generation model (must be pulled: ollama pull <name>) ---------
REM  3080 10 GB  ->  qwen3.5:9b   (default, 6.6 GB)  NUM_CTX 8192
REM  3080 12 GB  ->  qwen3.5:9b   with NUM_CTX 16384
REM  spilling to CPU? ->  qwen3.5:4b  (3.4 GB)
REM  Check placement after any change:  ollama ps   (must say 100% GPU)
set RAG_LLM_MODEL=qwen3.5:9b
set RAG_NUM_CTX=8192

REM --- Retrieval -----------------------------------------------------
set RAG_TOP_K=6
set RAG_RERANK=1
set RAG_CONTEXT_BUDGET_CHARS=9000

REM --- Embeddings (CPU; keeps VRAM free for the model) ---------------
REM  BAAI/bge-small-en-v1.5  fast, 384-dim   (default)
REM  BAAI/bge-base-en-v1.5   sharper, 768-dim, ~3x slower to ingest
REM  Changing this needs:  python cli.py reindex
set RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5

REM  Leave 4 threads for the OS and Ollama; the i9-11900K has 16.
set RAG_EMBED_THREADS=12
