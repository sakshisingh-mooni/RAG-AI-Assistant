# 📄 RAG AI Assistant — Multilingual Voice-Enabled PDF Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** system that lets users upload any PDF and ask questions about it — in their native Indian language, via voice or text — and receive spoken answers grounded strictly in the document.

---

## 📸 Demo

### Basic RAG — Upload & Ask Questions
![Demo 1](assets/demo.png)

### Multilingual Voice — Ask in Hindi, Get Answer in Hindi
![Demo 2](assets/demo2.png)

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────┐
│  Document Ingestion Pipeline    │
│  PyPDFLoader → Text Splitter    │
│  (chunk_size=1000, overlap=100) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Embedding + Vector Storage     │
│  HuggingFace all-MiniLM-L6-v2  │
│  → Pinecone Vector Database     │
└────────────────┬────────────────┘
                 │
         User Question
         (Voice / Text)
                 │
    [Multilingual] Translate → EN
                 │
                 ▼
┌─────────────────────────────────┐
│  MultiQueryRetriever            │
│  Generates 3-5 query variations │
│  → Union of retrieved chunks    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Groq LLaMA 3.3-70B (LLM)      │
│  + Custom Prompt Template       │
│  + Hallucination Guard          │
└────────────────┬────────────────┘
                 │
    [Multilingual] Translate → User Language
                 │
                 ▼
         Answer + gTTS Voice Output
```

---

## ✅ Key Features

| Feature | Implementation |
|---|---|
| **RAG Pipeline** | LangChain + Pinecone vector store + HuggingFace embeddings |
| **Anti-hallucination** | Custom prompt instructs LLM to say "not in document" if answer absent |
| **MultiQueryRetriever** | Generates multiple query reformulations to improve retrieval recall |
| **Voice Input** | SpeechRecognition + PyAudio mic capture |
| **Voice Output** | gTTS text-to-speech in user's language |
| **10 Indian Languages** | Hindi, Telugu, Tamil, Kannada, Marathi, Gujarati, Bengali, Punjabi, Malayalam, English |
| **Translation pipeline** | deep-translator: user lang → EN for retrieval → user lang for answer |
| **No quota limits** | HuggingFace embeddings run locally — no API key, no rate limits |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (LLaMA 3.3-70B Versatile) — free tier |
| Embeddings | HuggingFace all-MiniLM-L6-v2 — runs locally |
| Vector DB | Pinecone |
| Framework | LangChain (langchain-community, langchain-core) |
| Translation | deep-translator (GoogleTranslator) |
| Voice I/O | SpeechRecognition, PyAudio, gTTS |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
├── streamlit_app.py              # Version 1 — Basic text RAG chatbot
├── streamlit_app_voice.py        # Version 2 — RAG + voice input/output
├── streamlit_app_translator.py   # Version 3 — RAG + voice + 10 languages
├── RAG_GEMINI_PINECONE_PDF.ipynb # Core RAG pipeline notebook
├── pyaudio_deep_translator.ipynb # Voice + translation experiments
├── requirements.txt
├── .gitignore
└── assets/
    ├── demo.png                  # Basic chatbot screenshot
    └── demo2.png                 # Multilingual voice screenshot
```

---

## 🚀 Setup & Run

### 1. Clone
```bash
git clone https://github.com/sakshisingh-mooni/RAG-AI-Assistant
cd RAG-AI-Assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up `.env`
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 4. Run

**Basic chatbot:**
```bash
streamlit run streamlit_app.py
```

**With voice:**
```bash
streamlit run streamlit_app_voice.py
```

**With voice + multilingual:**
```bash
streamlit run streamlit_app_translator.py
```

---

## 🔑 API Keys (Both Free)

- **Groq**: [console.groq.com](https://console.groq.com) — 14,400 requests/day free
- **Pinecone**: [pinecone.io](https://pinecone.io) — free starter tier
- **HuggingFace embeddings** — no API key needed, runs locally

---

## 💡 How RAG Prevents Hallucination

Standard LLMs answer from training data — they cannot access your document and will confabulate. RAG solves this by:

1. Converting document chunks to vectors (embeddings)
2. At query time, retrieving only the most semantically relevant chunks
3. Passing those chunks as context to the LLM with an explicit instruction: *"Answer only from this context. If the answer isn't here, say so."*

This grounds every response in the actual document content.

---

## 🔄 Tech Decisions Made During Development

| Problem | Solution |
|---|---|
| Gemini embedding quota exhausted (100 req/min free tier) | Switched to HuggingFace all-MiniLM-L6-v2 — runs locally, zero quota |
| Gemini LLM quota exhausted | Switched to Groq LLaMA 3.3-70B — 14,400 req/day free |
| pinecone-client vs pinecone package conflict | Uninstalled pinecone-client, installed pinecone>=5.0.0 |
| Deprecated langchain imports | Migrated to langchain-community, langchain-core, langchain-classic |
| Safety settings API change in langchain-google-genai 4.x | Replaced enum objects with plain string keys |
