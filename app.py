"""
app.py
------
Streamlit frontend for the RAG AI Assistant.

Modes
-----
  Text only                   — text input → answer
  Text + Voice                — text input + mic → answer + audio playback
  Text + Voice + Multilingual — same as above with 10 Indian languages

Voice input uses PyAudio + SpeechRecognition. These require portaudio19-dev
on Linux (add to packages.txt for HuggingFace Spaces). On cloud environments
where no microphone is available, the voice button is automatically hidden.

Run
---
    streamlit run app.py
"""

import base64
import logging
import logging.config
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import cfg  # noqa: E402
from pipeline import build_pipeline, get_doc_hash, get_embeddings

# ---------------------------------------------------------------------------
# Logging — configured here so pipeline log messages are visible in terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# HuggingFace Spaces sets SPACE_ID — used to detect no-microphone environment.
_IS_CLOUD = os.environ.get("SPACE_ID") is not None

# Sentinel the LLM returns when it cannot find an answer in context.
_NOT_FOUND_SENTINEL = "The answer is not available in the provided document."

LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Malayalam": "ml",
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model… (first run only)")
def _load_embeddings():
    return get_embeddings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate(text: str, tgt: str) -> str:
    """
    Translate text into tgt language using Google Translate auto-detection.

    source is always "auto" — Google detects the actual input language.
    This correctly handles the common case where a user types in English
    while a non-English language is selected in the sidebar.

    Returns original text unchanged on any translation error.
    """
    if tgt == "en":
        # Auto-detecting then translating to English is redundant if the
        # text is already English, but GoogleTranslator handles it correctly.
        # We skip the call entirely for the tgt=="en" + likely-English case
        # by checking if the text is ASCII (covers most English queries).
        if text.isascii():
            return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=tgt).translate(text)
    except Exception:
        logger.exception("Translation failed (auto → %s) — returning original.", tgt)
        return text


def _text_to_speech(text: str, lang_code: str) -> str | None:
    try:
        from gtts import gTTS
        tts = gTTS(text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            tmp_path = fp.name
        with open(tmp_path, "rb") as af:
            b64 = base64.b64encode(af.read()).decode()
        os.unlink(tmp_path)
        return b64
    except Exception:
        logger.exception("TTS failed for lang '%s'.", lang_code)
        return None


def _display_answer(response: dict, lang_code: str, mode: str) -> None:
    answer_en: str = response["result"]

    # Check sentinel BEFORE translating — sentinel is always in English.
    if _NOT_FOUND_SENTINEL.lower() in answer_en.lower():
        st.warning(
            "⚠️ The answer could not be found in the document. "
            "Try rephrasing your question, or check that the document "
            "was indexed correctly (see chunk count above)."
        )
    else:
        answer = _translate(answer_en, tgt=lang_code) if lang_code != "en" else answer_en
        st.markdown("### 📌 Answer")
        st.write(answer)

        if mode in ("Text + Voice", "Text + Voice + Multilingual"):
            b64 = _text_to_speech(answer, lang_code)
            if b64:
                st.markdown(
                    f'<audio autoplay controls src="data:audio/mp3;base64,{b64}"></audio>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Audio playback unavailable — see logs.")

    with st.expander("📚 Source Snippets"):
        docs = response.get("source_documents", [])
        if not docs:
            st.write("No source documents returned.")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content[:500])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG AI Assistant", layout="centered")
st.title("📄 RAG AI Assistant")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")
    mode = st.radio(
        "Mode",
        ["Text only", "Text + Voice", "Text + Voice + Multilingual"],
    )
    lang_code = "en"
    if mode == "Text + Voice + Multilingual":
        selected_language = st.selectbox("Your language:", list(LANGUAGES.keys()))
        lang_code = LANGUAGES[selected_language]

    if _IS_CLOUD and mode != "Text only":
        st.info(
            "ℹ️ Voice input is unavailable in the cloud demo "
            "(no microphone access). Text input works normally."
        )
    if st.sidebar.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not uploaded_file:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded_file.read()
doc_hash = get_doc_hash(file_bytes)

if (
    "rag_chain" not in st.session_state
    or st.session_state.get("doc_hash") != doc_hash
):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("Indexing document… (skipped if already indexed)"):
            embeddings = _load_embeddings()
            rag_chain, chunk_count = build_pipeline(tmp_path, doc_hash, embeddings)
            st.session_state.rag_chain = rag_chain
            st.session_state.doc_hash = doc_hash
            st.session_state.chunk_count = chunk_count

        st.success(f"Document ready. {chunk_count} chunks indexed.")

    except TimeoutError as e:
        st.error(f"Pinecone readiness timeout: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Document indexing failed: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to build pipeline: {e}")
        logger.exception("Pipeline build failed.")
        st.stop()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
else:
    st.success(
        f"Document ready. "
        f"{st.session_state.get('chunk_count', '?')} chunks indexed."
    )

rag_chain = st.session_state.rag_chain

# ── Voice input ───────────────────────────────────────────────────────────────
if not _IS_CLOUD and mode in ("Text + Voice", "Text + Voice + Multilingual"):
    if st.button("🎤 Speak Your Question"):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening… speak clearly.")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)

            lang_hint = (lang_code + "-IN") if lang_code != "en" else "en-US"
            spoken = recognizer.recognize_google(audio, language=lang_hint)
            st.success(f"You said: {spoken}")

            # Translate spoken input to English for retrieval.
            query_en = _translate(spoken, tgt="en")
            with st.spinner("Retrieving answer…"):
                response = rag_chain.invoke({"query": query_en})
            _display_answer(response, lang_code, mode)

        except Exception as e:
            st.error(f"Voice error: {type(e).__name__}: {e}")

# ── Text input ────────────────────────────────────────────────────────────────
user_query = st.text_input("Ask a question about your document:")
if user_query:
    # Strip surrounding quotes, then translate to English for retrieval.
    # _translate uses auto-detect: English text under a Tamil/Hindi setting
    # is passed through unchanged (ASCII fast-path skips the API call).
    clean_query = user_query.strip().strip('"\'')
    query_en = _translate(clean_query, tgt="en")
    logger.info("User query → retriever: %s", query_en)
    with st.spinner("Retrieving answer…"):
        try:
            response = rag_chain.invoke({"query": query_en})
            _display_answer(response, lang_code, mode)
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
            logger.exception("Chain invocation failed.")
