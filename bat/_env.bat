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

REM ===================================================================
REM  Story engine (play.py). Only read when playing a campaign.
REM ===================================================================

REM --- Entry shape (the format rules) --------------------------------
set GAME_MIN_WORDS=800
set GAME_MAX_WORDS=900
REM  Revision passes spent forcing an entry into the window. 1 is the
REM  right trade on a 9B model: it fixes most misses and costs one extra
REM  generation. 0 is faster and lets short entries through.
set GAME_MAX_REPAIRS=1

REM --- Memory --------------------------------------------------------
REM  How many recalled memories reach the prompt each turn.
set GAME_RECALL_K=6
REM  Fold turn summaries into an arc summary this often. Lower = the
REM  early game stays sharper for longer, at the cost of prompt space.
set GAME_ARC_EVERY=10
set GAME_ARC_CONTEXT=6

REM --- Context budget ------------------------------------------------
REM  Previous entries kept word-for-word. One is right on an 8K context:
REM  an 850-word entry is ~1200 tokens, and the model still has to write
REM  another one. Raise this only with NUM_CTX raised too.
set GAME_VERBATIM_TURNS=1
set GAME_SUMMARY_TURNS=6
