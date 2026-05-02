# 📄 RAG AI Assistant — Multilingual Voice-Enabled PDF Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** system that lets users upload any PDF and ask questions about it — in their native Indian language, via voice or text — and receive spoken answers grounded strictly in the document.

---

## 📸 Demo

> Upload a PDF → Ask in Hindi/Telugu/Tamil/English → Get a spoken, translated answer

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
│  Google Generative AI Embeddings│
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
│  Gemini 2.0 Flash (LLM)         │
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
| **RAG Pipeline** | LangChain + Pinecone vector store + Gemini embeddings |
| **Anti-hallucination** | Custom prompt instructs LLM to say "not in document" if answer absent |
| **MultiQueryRetriever** | Generates multiple query reformulations to improve retrieval recall |
| **Voice Input** | SpeechRecognition + PyAudio mic capture |
| **Voice Output** | gTTS text-to-speech in user's language |
| **10 Indian Languages** | Hindi, Telugu, Tamil, Kannada, Marathi, Gujarati, Bengali, Punjabi, Malayalam, English |
| **Translation pipeline** | deep-translator: user lang → EN for retrieval → user lang for answer |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google Generative AI Embeddings |
| Vector DB | Pinecone |
| Framework | LangChain (langchain-community, langchain-core) |
| Translation | deep-translator (GoogleTranslator) |
| Voice I/O | SpeechRecognition, PyAudio, gTTS |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
├── streamlit_app.py              # Version 1 — Basic text RAG chatbot
├── streamlit_app_voice.py       # Version 2 — RAG + voice input/output
├── streamlit_app_translator.py  # Version 3 — RAG + voice + 10 languages
├── RAG_GEMINI_PINECONE_PDF.ipynb # Core RAG pipeline notebook
├── pyaudio_deep_translator.ipynb # Voice + translation experiments
├── requirements.txt
└── .env                         # API keys (not committed)
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
GEMINI_API_KEY=your_gemini_api_key
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

## 🔑 API Keys

- **Gemini**: [aistudio.google.com]
- **Pinecone**: [pinecone.io]

---

## 💡 How RAG Prevents Hallucination

Standard LLMs answer from training data — they cannot access your document and will confabulate. RAG solves this by:

1. Converting document chunks to vectors (embeddings)
2. At query time, retrieving only the most semantically relevant chunks
3. Passing those chunks as context to the LLM with an explicit instruction: *"Answer only from this context. If the answer isn't here, say so."*

This grounds every response in the actual document content.
