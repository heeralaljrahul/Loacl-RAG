# Setup — Windows, RTX 3080, 16 GB RAM

Tuned for an i9-11900K / 16 GB DDR4 / RTX 3080. About 40 minutes, most of
it downloads.

---

## First: which 3080 do you have?

The RTX 3080 shipped in a 10 GB and a 12 GB version and it decides your
model. **Task Manager → Performance → GPU → Dedicated GPU memory.**

| your card | model | `RAG_NUM_CTX` | download |
|---|---|---|---|
| 3080 10 GB | `qwen3.5:9b` | 8192 | 6.6 GB |
| 3080 12 GB | `qwen3.5:9b` | 16384 | 6.6 GB |
| spilling to CPU either way | `qwen3.5:4b` | 8192 | 3.4 GB |

The 3080's memory bandwidth is excellent, which is what sets tokens/sec
once a model fits. Capacity is the only real gate.

**Your 16 GB of system RAM is the tighter constraint.** Ollama, the
embedding model, Python and Windows all share it. This is why embeddings
run on the CPU as int8 ONNX rather than PyTorch — roughly 400 MB resident
instead of several GB, and it leaves your VRAM entirely to the model. If
you ever go to 32 GB it is a cheap upgrade on that board, and it is the
first one worth making.

---

## 1. Install Ollama

Download from <https://ollama.com/download/windows> and run it. It adds a
tray icon and a background server on `localhost:11434`.

```powershell
ollama --version
```

## 2. Set two environment variables

These roughly halve what the context window costs in VRAM — on a 10 GB card
that is the difference between the context sitting on the GPU and spilling
to CPU. They apply to the Ollama *server*, so they must be set permanently
and the server restarted.

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_KEEP_ALIVE 30m
```

Now **right-click the Ollama tray icon → Quit**, and start it again from
the Start menu. Variables only apply to a freshly started server. This is
the step people skip and then wonder why everything is slow.

(`bat\setup.bat` sets these for you, but you still have to restart Ollama.)

## 3. Pull the model

```powershell
ollama pull qwen3.5:9b
```

Then check where it actually runs — this matters more than the model choice:

```powershell
ollama run qwen3.5:9b "Write two sentences of plain technical prose."
ollama ps
```

`ollama ps` must say **100% GPU**. Any percentage on CPU takes you from
~40 tokens/sec to ~4. If it shows CPU, lower `RAG_NUM_CTX` in
`bat\_env.bat`, or drop to `qwen3.5:4b`.

Tags move. If a pull 404s, check <https://ollama.com/library> rather than
fighting it.

## 4. Install Python

Python 3.11 or 3.12 from <https://www.python.org/downloads/windows/> — not
3.13, some ML wheels still lag on it. **Tick "Add python.exe to PATH"** on
the first installer screen; easy to miss, annoying to fix later.

```powershell
python --version
```

## 5. Install the engine

Unzip or clone the project somewhere sensible, e.g. `C:\local-rag`. Then
double-click **`bat\setup.bat`**, or:

```powershell
cd C:\local-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

About 300 MB and a few minutes. There is no PyTorch and no CUDA toolkit in
that list — deliberately.

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 6. Check everything before ingesting anything

```powershell
bat\doctor.bat
```

It verifies Ollama is reachable, the model is installed, **what fraction of
it is on the GPU**, that the embedding and rerank models load, and whether
the index matches the configured embedding model. It names the fix for
whatever is broken. Run this first whenever something is wrong later.

The first run downloads the embedding and rerank models, about 130 MB.

## 7. Try it on the sample documents

Before pointing it at your own files, confirm the whole loop works:

```powershell
.venv\Scripts\python cli.py ingest sample_docs
.venv\Scripts\python cli.py ask "what is the carry-over limit on annual leave?"
```

You should get an answer citing `[1]`, sourced from `handbook.md`.

## 8. Ingest your own documents

```powershell
bat\ingest.bat "C:\Users\you\Documents\notes"
```

Add as many folders as you like; each is remembered, and plain
`bat\ingest.bat` afterwards rescans all of them and picks up changes.

Rough throughput on your CPU: 500–1500 pages per minute for text and
markdown, slower for PDFs, which are dominated by extraction rather than
embedding.

## 9. Use it

```powershell
bat\serve.bat
```

Opens <http://localhost:8080>. The right-hand panel shows what was
retrieved for each question, with scores. `bat\ask.bat` gives you the same
thing in a terminal.

---

## The 10-minute sanity check

Worth doing before you trust it with anything.

1. Ask something whose answer you know is in one specific file. Check the
   citation points at that file.
2. Ask something that is **definitely not** in your documents. It should
   say so rather than inventing an answer. If it invents, your model is too
   small — that is the single best test of a RAG setup.
3. Ask something phrased in your own words rather than the document's.
   Watch the panel: if the right passage is absent, that is a retrieval
   problem (try `bge-base`); if it is present but ignored, that is a model
   problem (try a bigger one).
4. Edit a file, re-run `bat\ingest.bat`, ask again. The answer should
   change.

---

## When something is wrong

| symptom | cause | fix |
|---|---|---|
| `cannot reach Ollama` | server not running | relaunch the tray app |
| answers take 60+ seconds | model spilled to CPU | `ollama ps`; lower `RAG_NUM_CTX` or use `qwen3.5:4b` |
| whole PC crawls, disk thrashes | 16 GB RAM exhausted | close Chrome; don't run two models |
| first question slow, rest fast | model loading into VRAM | expected; `OLLAMA_KEEP_ALIVE=30m` keeps it warm |
| answers ignore an obvious passage | it wasn't retrieved | check the panel, then `cli.py search "…"` |
| `index was built with '<other>'` | embedding model changed | `cli.py reindex` |
| `no extractable text (scanned PDF?)` | PDF is images | OCR it: `ocrmypdf in.pdf out.pdf` |
| citations point at the wrong section | headings not detected | check the file actually uses headings |
| `<think>` text in answers | reasoning model | already stripped; report if it leaks |

---

## What to expect on this hardware

- 20–45 tokens/sec generating, so a 200-word answer lands in 6–12 seconds.
- Retrieval adds about 250 ms — roughly 40 ms of search and 200 ms of
  reranking, all on the CPU, none of it touching VRAM.
- Ingestion is a few hundred pages a minute and only ever processes files
  that changed.

## Upgrades, in order of value

1. **16 → 32 GB DDR4.** Cheapest meaningful upgrade on that board. Removes
   the memory pressure that constrains everything else here.
2. **A 16 GB+ GPU.** Only then do 24B-class models and their noticeably
   better reasoning over retrieved text come into range.

Do neither until you have used it for a week and know what actually annoys
you. It is at least as likely to be the chunk size or the prompt as the
hardware.
