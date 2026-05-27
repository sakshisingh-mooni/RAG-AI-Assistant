---
title: RAG AI Assistant
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 📄 RAG AI Assistant — Multilingual Voice-Enabled PDF Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** system that lets users upload any PDF and ask questions about it — in their native Indian language, via voice or text — and receive answers grounded strictly in the document.

🚀 **Live Demo**: [Try it here](https://huggingface.co/spaces/Sakshisingh2710/RAG-AI-Assistant)

🎥 **Demo Video**: [Watch full walkthrough](https://drive.google.com/file/d/17D2pkOJh5Qg0Q4BPW-9kk7QHkm64_rG2/view?usp=drive_link)

> **Note:** Voice input (microphone) is disabled in the cloud demo — HuggingFace Spaces has no microphone access. Voice output (text-to-speech) and all multilingual text modes work fully in the live demo. Run locally to test voice input.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────┐
│  Document Ingestion Pipeline    │
│  PyMuPDFLoader (layout-aware)   │
│  → Text Splitter                │
│  → Noise chunk filtering        │
│  (config via config.py)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Embedding + Vector Storage     │
│  HuggingFace all-MiniLM-L6-v2  │
│  → Pinecone (namespace = MD5)   │
│    Same PDF never re-indexed    │
└────────────────┬────────────────┘
                 │
         User Question
         (Voice / Text)
                 │
    [Multilingual] Translate → EN
                 │
                 ▼
┌─────────────────────────────────┐
│  MultiQueryRetriever (MMR)      │
│  Generates 3–5 query variants   │
│  → Union of retrieved chunks    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Groq LLaMA 3.3-70B             │
│  + Custom Prompt Template       │
│  + "Not in document" guard      │
└────────────────┬────────────────┘
                 │
    [Multilingual] Translate → User Language
                 │
                 ▼
         Answer + gTTS Voice Output
```

---

## Key Features

| Feature | Implementation |
|---|---|
| **RAG Pipeline** | LangChain LCEL + Pinecone + HuggingFace embeddings |
| **Anti-hallucination** | Prompt instructs LLM to say "not in document" if answer absent |
| **MultiQueryRetriever** | Generates multiple query reformulations to improve retrieval recall |
| **Namespace deduplication** | Same PDF never re-indexed across sessions — saves cost and latency |
| **Noise chunk filtering** | Drops tokenizer artifact chunks and figure-text noise before indexing |
| **Voice Input** | SpeechRecognition + PyAudio (local only; disabled on HF Spaces) |
| **Voice Output** | gTTS text-to-speech in user's language |
| **10 Indian Languages** | Hindi, Telugu, Tamil, Kannada, Marathi, Gujarati, Bengali, Punjabi, Malayalam, English |
| **Translation pipeline** | deep-translator: user lang → EN for retrieval → user lang for answer |
| **Local Embeddings** | HuggingFace runs locally — no API key, no quota |
| **RAGAS Evaluation** | Faithfulness, AnswerRelevancy, ContextRecall with synthetic ground truth |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq LLaMA 3.3-70B Versatile (free tier: 14,400 req/day) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 — runs locally |
| Vector DB | Pinecone |
| Framework | LangChain v1 (LCEL) |
| PDF Loader | PyMuPDF — layout-aware, handles multi-column research papers |
| Translation | deep-translator (GoogleTranslator) |
| Voice I/O | SpeechRecognition, PyAudio, gTTS |
| Evaluation | RAGAS 0.4.x |
| Frontend | Streamlit |

**Requires Python >= 3.10** (langchain v1 requirement).

---

## Project Structure

```
├── config.py               # Centralised config + env validation (fail-fast)
├── pipeline.py             # RAG pipeline — no Streamlit coupling
├── app.py                  # Streamlit frontend
├── evaluate_pipeline.py    # Offline RAGAS evaluation script
├── delete_namespace.py     # Utility to clear Pinecone namespaces for re-indexing
├── requirements.txt        # Pinned dependencies (verified on PyPI May 2026)
├── packages.txt            # System packages for HuggingFace Spaces (portaudio19-dev)
├── .env.example            # Environment variable template
└── assets/
    ├── demo1.png
    └── demo2.png
```

---

## Setup & Run

### Prerequisites
- Python >= 3.10
- portaudio19-dev (for voice input on Linux/Mac)

### 1. Clone
```bash
git clone https://github.com/sakshisingh-mooni/RAG-AI-Assistant
cd RAG-AI-Assistant
```

### 2. Install system dependency (voice input only)
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

`.env` contents:
```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_index_name_here
```

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Run evaluation (optional)
```bash
python evaluate_pipeline.py --pdf your_document.pdf
```

### 7. Re-index a document (if needed)
```bash
# List current Pinecone namespaces
python delete_namespace.py --list

# Delete namespace for a specific PDF to force re-indexing
python delete_namespace.py --pdf your_document.pdf
```

---

## API Keys (All Free)

| Service | Where to get it | Free tier |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | 14,400 requests/day |
| **Pinecone** | [pinecone.io](https://pinecone.io) | 1 free index, 100k vectors |
| **HuggingFace embeddings** | No key needed | Runs locally |

---

## Pipeline Evaluation (RAGAS)

Evaluated using RAGAS 0.4.x with LLM-as-judge synthetic ground truth — reference answers generated by a separate evaluator LLM call (temperature=0) grounded in retrieved chunks, not derived from the pipeline's own output. This makes ContextRecall a meaningful signal rather than self-referential scoring.

> Evaluated on WHO_guidelines.pdf (WHO Guidelines on second- and third-line medicines and type of insulin for diabetes mellitus, 2018). Full per-question breakdown: [`results/ragas_scores.csv`](results/ragas_scores.csv)

### Retrieval Experiments

**Phase 1 — Retrieval parameter tuning** (5 mixed-quality questions, chunk_size=1000):

| Experiment | chunk_size | k | fetch_k | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|---|---|---|
| Baseline | 1000 | 5 | 20 | 0.90 | 0.88 | 0.60 |
| Wider retrieval | 1000 | 8 | 30 | 0.95 | 0.77 | 0.50 |
| Finer chunks | 500 | 5 | 20 | 0.73 | 0.76 | 0.60 |

Finding: wider retrieval hurt recall; smaller chunks degraded faithfulness without improving recall. Recall plateau at 0.60 traced to 2 of 5 questions being unanswerable from document prose — one asked for dosing numbers absent from a policy guideline, one was too broad to retrieve against.

**Phase 2 — Evaluation question redesign** (3 well-formed questions, baseline config):

After diagnosing that poor question design was masking pipeline quality, questions were rewritten to map to specific, self-contained prose sections of the document.

| Question | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| When to consider long-acting insulin analogues | 1.00 | 0.66 | 1.00 |
| GRADE methodology used for evidence assessment | 1.00 | 0.99 | 1.00 |
| Resource and cost considerations for insulin in low-resource settings | 1.00 | 0.95 | 1.00 |
| **Mean** | **1.00** | **0.87** | **1.00** |

**Current config** ✅: `chunk_size=1000`, `chunk_overlap=100`, `k=5`, `fetch_k=20`.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 1.00 | > 0.80 | ✅ Pass |
| Answer Relevancy | 0.87 | > 0.80 | ✅ Pass |
| Context Recall | 1.00 | > 0.70 | ✅ Pass |

---

## Technical Decisions

| Problem | Solution |
|---|---|
| Gemini embedding quota exhausted | Switched to HuggingFace all-MiniLM-L6-v2 — runs locally, zero quota |
| Gemini LLM quota exhausted | Switched to Groq LLaMA 3.3-70B — 14,400 req/day free |
| pinecone-client vs pinecone conflict | Uninstalled pinecone-client, installed pinecone>=5.0.0 |
| Deprecated langchain imports | Migrated to langchain v1; MultiQueryRetriever now in langchain-classic |
| Multi-user upload race condition | Each upload written to a per-session tempfile, deleted after indexing |
| Arbitrary `time.sleep(20)` for Pinecone | Replaced with exponential-backoff polling loop with configurable timeout |
| `reference = chain_answer` in RAGAS | Fixed: reference generated by separate evaluator LLM call at temperature=0 |
| `<EOS>/<pad>` tokens in chunks | Regex cleaning + noise-ratio filter drops artifact chunks before indexing |
| Figure text noise in research PDFs | `_is_noisy_chunk()` drops chunks with >60% short tokens or >50% repeated sentences |
| Voice button crashing on HF Spaces | Cloud detection via `SPACE_ID` env var; mic button hidden automatically |
| Stale vectors after loader change | `delete_namespace.py` utility with deletion confirmation polling |
