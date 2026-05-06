# app.py
import streamlit as st
import os
import tempfile
import base64
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from pipeline import build_pipeline, get_doc_hash

load_dotenv()

LANGUAGES = {
    "English": "en", "Hindi": "hi", "Telugu": "te",
    "Tamil": "ta", "Kannada": "kn", "Marathi": "mr",
    "Gujarati": "gu", "Bengali": "bn", "Punjabi": "pa",
    "Malayalam": "ml"
}

st.set_page_config(page_title="RAG AI Assistant", layout="centered")
st.title("📄 RAG AI Assistant")

# ── Sidebar: mode + language ──
with st.sidebar:
    st.subheader("⚙️ Settings")
    mode = st.radio(
        "Mode",
        ["Text only", "Text + Voice", "Text + Voice + Multilingual"]
    )
    if mode == "Text + Voice + Multilingual":
        selected_language = st.selectbox(
            "Your language:", list(LANGUAGES.keys())
        )
        lang_code = LANGUAGES[selected_language]
    else:
        lang_code = "en"

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    doc_hash = get_doc_hash(file_bytes)

    # ── Build pipeline once per document ──
    if ("qa_chain" not in st.session_state or
            st.session_state.get("doc_hash") != doc_hash):

        with open("uploaded_doc.pdf", "wb") as f:
            f.write(file_bytes)

        with st.spinner("Indexing document... (once per file)"):
            st.session_state.qa_chain = build_pipeline(
                "uploaded_doc.pdf", doc_hash
            )
            st.session_state.doc_hash = doc_hash
        st.success("Document indexed.")

    qa_chain = st.session_state.qa_chain

    # ── Answer + optional TTS ──
    def answer_and_display(query_en: str):
        response = qa_chain.invoke({"query": query_en})
        answer_en = response["result"]

        if lang_code != "en":
            answer = GoogleTranslator(
                source="en", target=lang_code
            ).translate(answer_en)
        else:
            answer = answer_en

        st.markdown("### 📌 Answer:")
        st.write(answer)

        if mode in ["Text + Voice", "Text + Voice + Multilingual"]:
            tts = gTTS(answer, lang=lang_code)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3"
            ) as fp:
                tts.save(fp.name)
                tmp_path = fp.name
            with open(tmp_path, "rb") as af:
                b64 = base64.b64encode(af.read()).decode()
            os.unlink(tmp_path)  # cleanup
            st.markdown(
                f'<audio autoplay controls '
                f'src="data:audio/mp3;base64,{b64}"></audio>',
                unsafe_allow_html=True
            )

        with st.expander("📚 Source Snippets"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:500])

    # ── Voice input (only in voice modes) ──
    if mode in ["Text + Voice", "Text + Voice + Multilingual"]:
        if st.button("🎤 Speak Your Question"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... speak clearly.")
                audio = recognizer.listen(source)
            try:
                spoken = recognizer.recognize_google(
                    audio,
                    language=lang_code + "-IN" if lang_code != "en" else "en-US"
                )
                st.success(f"You said: {spoken}")
                query_en = GoogleTranslator(
                    source=lang_code, target="en"
                ).translate(spoken) if lang_code != "en" else spoken
                answer_and_display(query_en)
            except sr.UnknownValueError:
                st.error("Could not understand. Please try again.")
            except sr.RequestError as e:
                st.error(f"Voice error: {e}")

    # ── Text input (always available) ──
    user_query = st.text_input("Ask a question about your document:")
    if user_query:
        query_en = GoogleTranslator(
            source=lang_code, target="en"
        ).translate(user_query) if lang_code != "en" else user_query
        with st.spinner("Retrieving answer..."):
            answer_and_display(query_en)

else:
    st.info("Upload a PDF to begin.")