# Local RAG

Two things share one retrieval engine, both running entirely on your own
machine — no API keys, no accounts, no telemetry, no network at query time.

**A story engine** (`play.py`) — a role-playing game that remembers a
campaign across hundreds of turns. **→ [STORY.md](STORY.md)**

**Document Q&A** (`cli.py`) — ask questions about your own files, answered
with citations. Described below.

They share the hybrid search, the reranker and the storage layer; what
differs is what gets indexed, and that difference is the whole design. See
[STORY.md](STORY.md) for why a story cannot simply be poured into a document
index.

```
python cli.py ingest "C:\Users\you\Documents"
python cli.py ask "what is the carry-over limit on annual leave?"
```

```
handbook.md — Annual leave > Carry-over, travel.md — Hotels

Up to five days may be carried into the following year; anything beyond
that lapses on 31 March and is not paid out on departure [1].
```

---

## How it works

```
your files ──► loaders ──► chunker ──► embeddings (CPU) ──┐
   .pdf .docx .md .txt        │                            ├─► SQLite
   .html .csv .json .epub     └────────────────────────────┘   ├ text
   source code                                                 ├ FTS5 keyword index
                                                               └ float32 vectors
                                            ┌──────────────────────┘
   question ──► dense search ──┐            │
           └──► BM25 search ───┴─► RRF ──► rerank (CPU) ──► floor ──► prompt
                                                                        │
                                                          Ollama (GPU) ─┘
                                                                        │
                                                        answer with [1] citations
```

Five decisions carry most of the quality. Each is defended below, and each
is measurable with `tools/eval_retrieval.py` on your own documents.

### 1. Hybrid retrieval, because the two methods fail differently

Vector search understands paraphrase and is hopeless at exact tokens: ask
about error `E-4471` and a 384-dimensional embedding returns things
*about* errors. BM25 nails identifiers, names and version numbers, and is
useless when the question shares no words with the answer ("how much
holiday" vs "annual leave"). Their failure modes barely overlap, so both
run on every query.

Results are merged with **reciprocal rank fusion** rather than a weighted
score blend, because cosine similarity and BM25 scores are not on a
comparable scale and their ranges shift with corpus and query length. RRF
reads only *ranks*:

```
score(chunk) = Σ over both lists  1 / (60 + rank_in_that_list)
```

No per-corpus calibration, nothing to retune when the corpus grows.

### 2. Reranking, which is where the accuracy actually comes from

Retrieval scores a query and a chunk that never met — each was embedded
alone. A cross-encoder reads them *together*, which is far more accurate
and far too slow to run over a corpus. So it only ever sees the ~40
candidates that survived fusion.

Measured on the sample corpus (34 labelled questions, `tests/fixtures/eval.jsonl`):

| configuration | hit@1 | hit@3 | MRR | ms/query |
|---|---|---|---|---|
| dense only | 0.91 | 0.94 | 0.929 | 45 |
| BM25 only | 0.91 | 0.94 | 0.932 | 0 |
| hybrid (RRF) | 0.91 | 0.94 | 0.932 | 38 |
| **hybrid + rerank** | **0.94** | **0.97** | **0.963** | 228 |

Two caveats worth stating plainly: 27 chunks is a small corpus and these
numbers are close to saturation, and the differences are one or two
questions wide. Re-run it on your documents before trusting any of it.
That is what the tool is for.

Reranker choice was measured too, not assumed — `BAAI/bge-reranker-base` is
13× larger and scored *worse* here (hit@1 0.79, 1806 ms) than the 80 MB
MiniLM default.

### 3. A relevance floor, so the prompt isn't padded with noise

Cross-encoder scores are informative in absolute terms: a chunk that
answers the question lands above roughly 0, irrelevant text below -7. Some
questions genuinely have one relevant chunk. Returning six anyway spends
prompt budget on material the model has to ignore, and hands it
plausible-looking text to blend into the answer.

So chunks scoring more than `rerank_margin` below the best one are dropped,
with at least one always kept:

| margin | coverage | chunks sent per question |
|---|---|---|
| off | 1.00 | 6.0 |
| **12 (default)** | **1.00** | **3.9** |
| 8 | 0.97 | 2.6 |
| 6 | 0.95 | 2.0 |
| 4 | 0.90 | 1.7 |

35% less prompt at no measured cost in coverage — which on a 3080 is also
straightforwardly faster.

### 4. Heading breadcrumbs on every chunk

Each chunk carries the trail of headings it sits under, and that trail is
prepended to the text that gets embedded:

```
Employee Handbook 2026 > Annual leave > Carry-over

Up to five days may be carried into the following year…
```

Without it, a chunk whose body never says "leave" cannot be found by a
question about leave. The chunker also never splits mid-paragraph unless a
paragraph exceeds the budget on its own, and **never discards text** — a
section shorter than the minimum is merged, not dropped.

### 5. Embeddings and reranking on the CPU, deliberately

On a 10 GB card every megabyte of VRAM belongs to the language model.
bge-small quantised to int8 through ONNX Runtime handles ~600 chunks/s on
an i9-11900K while the GPU stays entirely free. Putting the embedder on the
GPU would cost context length and buy nothing. There is no PyTorch in the
dependency list at all.

---

## Grounding

Three things separate this from a model that has read some notes and is now
guessing:

- **Numbered sources and required `[n]` citations.** A claim without a
  number is visibly unsupported, so hallucination becomes legible instead
  of invisible.
- **Refusing is named as an acceptable answer.** Models default to
  helpfulness and will bridge a gap with invention unless told that "the
  documents don't cover this" is a correct response.
- **Retrieved text is fenced and marked as reference material, never
  instructions.** Otherwise a document containing "ignore previous
  instructions" is a prompt injection against your own notes.

The web UI shows exactly what was retrieved for the turn you just took,
with scores and both ranks. That panel is the debugging tool: when an
answer is wrong it tells you whether the passage was never retrieved or was
retrieved and ignored — different problems, different fixes.

---

## Install

**Windows** — see **[SETUP-WINDOWS.md](SETUP-WINDOWS.md)** for the
step-by-step version, with settings pre-tuned for an RTX 3080. Short form:

```
bat\setup.bat
bat\ingest.bat "C:\Users\you\Documents"
bat\serve.bat
```

**Anything else:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py ingest sample_docs
python cli.py serve
```

Ollama must be running with a model pulled. `python cli.py doctor` checks
all of it and names the fix for whatever is broken.

---

## Commands

| command | what it does |
|---|---|
| `ingest [paths…]` | add or update folders; no paths = rescan everything already added |
| `ask "question"` | one question, streamed, with citations |
| `chat` | interactive, with short-term conversation memory |
| `search "query"` | retrieval only — no model, no generation |
| `serve` | web UI on http://localhost:8080 |
| `stats` | what is in the index |
| `doctor` | check Ollama, GPU placement, models, index consistency |
| `forget <path>` | remove files or a folder from the index |
| `reindex` | wipe and rebuild (needed after changing the embedding model) |

Re-running `ingest` is cheap. Each file is checked in three widening steps —
size+mtime, then SHA-256, then actual parsing — so editing one file in a
folder of two thousand re-embeds one file. Deleted files are pruned.

---

## Supported files

`.pdf` `.docx` `.md` `.txt` `.html` `.epub` `.csv` `.tsv` `.json` `.jsonl`
and source code in most common languages.

PDFs keep page numbers, so citations read `handbook.pdf p.14`. Scanned PDFs
have no text layer and are reported as such rather than silently indexed
empty — OCR them first (`ocrmypdf`). CSVs repeat their header in every
block so a chunk is never orphaned from its column names. DOCX headings
become real headings, so breadcrumbs work.

---

## Tuning

Every setting is an environment variable prefixed `RAG_`; on Windows edit
`bat\_env.bat`. The ones that matter:

| variable | default | when to change it |
|---|---|---|
| `RAG_LLM_MODEL` | `qwen3.5:9b` | see the setup guide's VRAM table |
| `RAG_NUM_CTX` | `8192` | lower it first if `ollama ps` shows CPU spill |
| `RAG_TOP_K` | `6` | more chunks, slower answers |
| `RAG_RERANK_MARGIN` | `12.0` | lower = shorter prompts, eventually drops real answers |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | `BAAI/bge-base-en-v1.5` for recall over speed; needs `reindex` |
| `RAG_RERANK` | `1` | `0` saves ~250 ms/query and costs real accuracy |
| `RAG_CHUNK_CHARS` | `1800` | ~450 tokens; smaller = more precise, less context per hit |

**Measure, don't guess.** Write questions with a phrase you know is in the
answer:

```jsonl
{"query": "how much holiday do I get", "expect": "25 days of paid annual leave"}
{"query": "per diem and the London hotel cap", "expect": ["45 EUR", "260 EUR in London"]}
```

```bash
python tools/eval_retrieval.py --corpus my_docs --queries my_questions.jsonl \
    --embed-models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5
```

Thirty questions is enough to tell a real change from a rearrangement.

---

## Known limitations

- **Vocabulary gaps are real.** "How much holiday do I get" against a
  document that only ever says "annual leave" ranks 4th on the sample
  corpus and can fall out of the top 6 when combined with a second
  question. Both the embedder and the reranker miss the synonym. A larger
  embedding model (`bge-base`) closes part of it; nothing here closes all
  of it.
- **Clause splitting for compound questions is implemented but off.** It
  measured as no gain and ~20% slower on the sample corpus, and it splits
  "black and white" into two nonsense clauses. Set
  `RAG_EXPAND_CLAUSES=1` to A/B it on your own documents.
- **Brute-force vector search** is exact and needs no index build, which is
  the right trade to about a million chunks. Past that it wants a real ANN
  index.
- **English-tuned defaults.** Both default models are English. For other
  languages use `bge-m3` or a multilingual reranker and re-run the eval.
- **No OCR, no images, no tables-as-structure.** A table becomes pipe-
  separated text, which retrieves acceptably and reasons poorly.

---

## Layout

```
play.py                 story engine commands
game/                   campaign state, memory, narrator + archivist
cli.py                  document Q&A commands
rag/config.py           all settings, env-overridable
rag/loaders.py          file → text blocks
rag/chunk.py            structure-aware chunking, heading breadcrumbs
rag/embed.py            CPU embeddings + cross-encoder rerank
rag/store.py            SQLite: text, FTS5 keyword index, vectors
rag/retrieve.py         hybrid search, RRF, rerank, relevance floor
rag/answer.py           prompt assembly and grounding rules
rag/server.py           FastAPI + SSE streaming
rag/web/index.html      the UI, one file
tools/eval_retrieval.py measure retrieval quality
tests/                  100 tests, no network and no Ollama required
```

`python -m pytest tests/ -q`
