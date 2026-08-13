# Self-Verifying Standards Compliance Assistant

**Tech Stack:** Python · LangChain · Groq API · Streamlit · ChromaDB

An agentic RAG system that answers questions from technical documents with
cited, **self-verified** answers — and checks whether a paper's claims are
still accurate against a reference source, suggesting corrections when
they're not.

Unlike a basic RAG chatbot, every answer passes through a **groundedness
check** before being shown: the system verifies its own answer against the
retrieved source text, and marks it "Unverified" rather than presenting an
unsupported claim as fact.

---

## Demo

Three modes, all in one app:

| Mode | What it does |
|---|---|
| **Ask (built-in doc)** | Ask questions about the pre-loaded reference document |
| **Ask (upload your PDF)** | Upload any PDF, ask questions about it on the spot |
| **Check a Paper** | Paste text from a paper; each factual claim is checked against the reference document and flagged if outdated, with a suggested correction |

---

## Features

- 🔎 **Hybrid retrieval** — combines semantic (embedding) search with
  keyword (BM25) search, so exact terms and numbers aren't missed by
  meaning-based search alone
- ✅ **Relevance grading** — an LLM call filters out retrieved chunks that
  aren't actually relevant before they reach the answer generator
- 🛡️ **Groundedness verification** — every generated answer is checked
  against the retrieved source text; unsupported answers are flagged, not
  hidden
- 📄 **Ask your own PDF** — upload any PDF and ask questions about it,
  indexed on the fly for that session
- 📝 **Claim verification with corrections** — paste text from a paper or
  report; each factual claim is extracted, checked, and — if outdated or
  unsupported — the system suggests what it should say instead
- ⚡ **Fast, free inference** — runs on Groq's free-tier API; no OpenAI
  key, no per-request cost

---

## Architecture

```
Document (PDF)
      │
Ingestion: load → split into chunks → embed → store (Chroma vector DB)
      │
      ├── Path A: User Question ─────────────┐
      │                                       │
      └── Path B: Pasted Paper Text           │
             → extract factual claims ────────┤
                                               ▼
                                   Hybrid Retrieval (semantic + BM25)
                                               │
                                     LLM Relevance Grading
                                     (discard, re-retrieve if weak)
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                   Answer Generation (Path A)         Claim Comparison (Path B)
                              │                                 │
                              └────── Groundedness Check ───────┘
                                   (verify against source text)
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                  Verified Answer + Citation         Verified / Outdated /
                  or "Unverified"                    Unsupported + Suggested Fix
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| LLM (answering, grading, verification) | Groq API (`llama-3.1-8b-instant`) |
| Embeddings | Ollama (`nomic-embed-text`, local) |
| Vector store | ChromaDB |
| Keyword search | rank_bm25 |
| Web UI | Streamlit |
| PDF parsing | PyPDF |

> The LLM layer can also run fully offline via local Ollama instead of
> Groq — toggle `USE_GROQ = False` in `src/config.py`.

---

## Project Structure

```
├── data/                       # built-in reference document(s)
├── eval_questions.json         # evaluation test set
├── requirements.txt
└── src/
    ├── config.py                # models, paths, tunable parameters
    ├── ingest.py                 # load → split → embed → store
    ├── retrieval.py              # hybrid search + relevance grading
    ├── groundedness.py           # verify a statement against evidence
    ├── rag_pipeline.py            # question-answering pipeline
    ├── session_ingest.py         # on-the-fly ingestion for uploaded PDFs
    ├── extract_claims.py         # pull factual claims out of pasted text
    ├── verify_paper.py            # check claims + suggest corrections
    ├── llm_provider.py            # routes to Groq or local Ollama
    ├── evaluate.py                # retrieval precision / faithfulness scoring
    └── app.py                     # Streamlit UI (3 modes)
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/standards-compliance-assistant.git
cd standards-compliance-assistant
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Sign up at [console.groq.com](https://console.groq.com)
- Create an API key
- Create a `.env` file in the project root:
  ```
  GROQ_API_KEY=your_key_here
  ```

### 4. Set up local embeddings (Ollama)
```bash
# install Ollama from https://ollama.com, then:
ollama pull nomic-embed-text
```

### 5. Build the vector store
```bash
python src/ingest.py
```

### 6. Launch the app
```bash
streamlit run src/app.py
```

---

## How Accuracy Is Enforced

| Step | Purpose | Failure Mode Prevented |
|---|---|---|
| Hybrid search | Combines meaning-based and exact keyword search | Missing exact terms/numbers pure semantic search overlooks |
| Relevance grading | LLM checks each retrieved chunk is actually relevant | Answering confidently from an unrelated section |
| Groundedness check | Verifies every claim against retrieved source text | Fabricated or unsupported claims (hallucination) |

An answer is only shown as "Verified" once it passes the groundedness
check. If it cannot be verified, the system says so explicitly rather than
presenting an unsupported answer as fact.

---

## Evaluation

```bash
python src/evaluate.py
```

Evaluated on a 5-question test set covering the built-in reference document
(*Attention Is All You Need*), including one deliberately unanswerable
question used to test hallucination avoidance.

| Metric | Result |
|---|---|
| Retrieval precision | **4/5 (80%)** |
| Answers marked Verified | **5/5 (100%)** |

All five answers — including the correct refusal on the unanswerable
question — were independently confirmed as grounded in the retrieved
source text.

---

## Development Notes — Bugs Found and Fixed

**1. False negatives from inline page citations** — the answer prompt
originally told the model to cite page numbers inline (e.g., "on page 4"),
but that literal phrase never appears in the source prose (page numbers
are metadata), so the groundedness checker flagged correct answers as
unsupported. *Fix: moved page citations out of the answer text into the
UI's separate source list.*

**2. Retrieval miss on a rephrased query** — "How many attention heads
does the base model use?" failed to retrieve the exact sentence stating
"8 parallel attention layers." *Fix: increased `RETRIEVE_K` from 5 to 8.*

**3. False negatives from valid multi-chunk synthesis** — an answer
correctly combining two true facts from two different chunks was flagged
as unsupported because no single chunk contained both facts together.
*Fix: updated the groundedness prompt to explicitly allow synthesis of
true facts across multiple provided chunks.*

---

## Known Limitations

- Claim extraction and verification process each claim sequentially, so
  checking a long paper takes proportionally longer.
- Retrieval precision (80%) indicates room to miss content on certain
  query phrasings; `RETRIEVE_K` and hybrid search weighting are the main
  tuning levers.
- The groundedness check relies on LLM judgment, which required iterative
  prompt tuning (see Development Notes) and may be sensitive to prompt
  wording on failure modes not yet observed.

---

## License

MIT
