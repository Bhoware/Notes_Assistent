# Notes Assistant 2.0

A RAG (Retrieval-Augmented Generation) API that lets you ask natural-language questions about your PDF notes and get grounded, cited answers. It loads PDFs, chunks and embeds them into a FAISS vector store, retrieves the most relevant passages for a query, and sends them to an LLM to produce an answer.

## Features

- **PDF ingestion** — recursively loads all `*.pdf` files from a directory using `PyPDFLoader`.
- **Chunking** — splits documents with `RecursiveCharacterTextSplitter` (default 1000 chars, 200 overlap).
- **Sentence-Transformer embeddings** — encodes chunks with `all-MiniLM-L6-v2` via `sentence-transformers`.
- **FAISS vector store** — stores embeddings locally; supports save/load to disk (`faiss.index` + `metadata.pkl`).
- **LLM-powered answers** — retrieves top-k chunks, builds a prompt, and calls the Groq-hosted `openai/gpt-oss-120b` model via `langchain-groq`.
- **FastAPI REST endpoint** — single `POST /query` endpoint that accepts a question and returns an answer.
- **CI pipeline** — GitHub Actions workflow runs `ruff` linting and `pytest` on Python 3.10/3.11.

## How It Works

```
User ──POST /query──▶ app.py (FastAPI)
                         │
                         ▼
                 Search.search()          ← src/search.py
                    │         │
     embed query    │         │  build prompt + call LLM
     ┌──────────────┘         └──────────────┐
     ▼                                       ▼
FaissVectorStore.query()              ChatGroq (Groq API)
  src/vectorstore.py                   src/search.py
     │                                       │
     │  top-k results                        │  answer
     └───────────────────▶ prompt ◀──────────┘
```

**Step-by-step:**

1. **`app.py`** — creates a `FastAPI` app, loads the FAISS index on startup, and exposes `POST /query`.
2. **`src/data_loader.py` → `load_docs(data_dir)`** — finds all `*.pdf` files in the given directory and loads them page-by-page with `PyPDFLoader`.
3. **`src/embedding.py` → `Embeddings`** — chunks documents with `RecursiveCharacterTextSplitter`, then encodes chunks using `SentenceTransformer("all-MiniLM-L6-v2")`.
4. **`src/vectorstore.py` → `FaissVectorStore`** — wraps a `faiss.IndexFlatL2` index. `build_from_documents()` runs the full embed pipeline and persists the index. `query(text, k)` encodes the query text and returns the top-k nearest neighbours with their metadata.
5. **`src/search.py` → `Search`** — loads the vector store, queries it, assembles retrieved chunks into a context string, and invokes `ChatGroq` with a prompt that instructs the LLM to answer using only the provided context.

## Tech Stack

| Category | Library / Service |
|---|---|
| Web framework | [FastAPI](https://fastapi.tiangolo.com/) + [Pydantic](https://docs.pydantic.dev/) |
| PDF parsing | [pypdf](https://github.com/py-pdf/pypdf) (via `langchain-community` `PyPDFLoader`) |
| Text splitting | [langchain-text-splitters](https://python.langchain.com/) (`RecursiveCharacterTextSplitter`) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) (CPU, `IndexFlatL2`) |
| LLM | [Groq API](https://groq.com/) via [langchain-groq](https://python.langchain.com/) (`openai/gpt-oss-120b`) |
| Prompt orchestration | [LangChain Core](https://python.langchain.com/) (`PromptTemplate`, LCEL pipe) |
| Env management | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Linting | [Ruff](https://docs.astral.sh/ruff/) |
| CI | GitHub Actions (Python 3.10, 3.11) |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Bhoware/Notes_Assistent.git
cd Notes_Assistent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file with your Groq API key
cp .env.example .env
# Then edit .env and replace the placeholder:
```

### Required environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com/) |

### Preparing your documents

Place your PDF files in `data/Documents/`. On the first run (if no FAISS index exists), the app will automatically build the vector store from these PDFs.

To manually rebuild the vector store:

```bash
python -m src.vectorstore
```

This runs the `__main__` block in `vectorstore.py`, which loads PDFs from `data/Documents/`, builds embeddings, and saves the index to `faiss_store/`.

## Usage

Start the API server:

```bash
uvicorn app:app --reload
```

The server runs at `http://127.0.0.1:8000`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the attention mechanism?", "k": 3}'
```

### Example response

```
"The attention mechanism is a technique that allows models to focus on specific parts
of the input sequence when producing an output. According to the notes (Unit 5),
it assigns different weights to different input tokens based on their relevance..."
```

*(Actual content depends on the PDFs in `data/Documents/`.)*

## Project Structure

```
Notes_Assist_2.0/
├── app.py                  # FastAPI entry point; POST /query endpoint
├── src/
│   ├── __init__.py         # Makes src a Python package
│   ├── data_loader.py      # load_docs(): loads PDFs from a directory
│   ├── embedding.py        # Embeddings: chunks docs + generates embeddings
│   ├── search.py           # Search: retrieves context + calls Groq LLM
│   └── vectorstore.py      # FaissVectorStore: FAISS index CRUD + query
├── data/
│   └── Documents/          # Place your PDF files here
├── faiss_store/
│   ├── faiss.index         # Persisted FAISS index (auto-generated)
│   └── metadata.pkl        # Chunk metadata (auto-generated)
├── .env.example            # Template for environment variables
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI (ruff + pytest)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Pyright config (extraPaths)
└── .gitignore
```

## Limitations

- **Text-based PDFs only** — uses `pypdf` under the hood, which cannot extract text from scanned/image-based PDFs (no OCR).
- **No authentication** — the FastAPI endpoint has no auth; anyone with network access can query it.
- **Brute-force search** — uses `faiss.IndexFlatL2` (exact L2 search), which works well for small-to-medium corpora but will slow down with very large document sets.
- **Single document directory** — PDFs must be in `data/Documents/`; there is no upload endpoint or multi-directory support.
- **No streaming** — the LLM response is returned in full; there is no streaming/SSE support.
- **Missing dependencies in `requirements.txt`** — several packages the code actually imports (`fastapi`, `uvicorn`, `python-dotenv`, `sentence-transformers`, `faiss-cpu`, `numpy`, `langchain-groq`, `langchain-text-splitters`) are not listed in `requirements.txt`.
- **Groq API rate limits** — the free Groq tier has rate limits; heavy use may result in `429` errors.
- **No tests** — the CI pipeline runs `pytest` but the repo contains no test files yet.

## Live Demo

> TODO: add deployed link

## Author

**Anshul** — [github.com/Bhoware](https://github.com/Bhoware)
