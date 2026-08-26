"""Local RAG: fully offline retrieval-augmented question answering.

Dense vectors + BM25, fused and reranked on CPU; generation via a local
Ollama model. No cloud services, no API keys, no telemetry.
"""

__version__ = "1.0.0"
