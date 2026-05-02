import streamlit as st
import os
from dotenv import load_dotenv


from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Voice
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64

# Load keys
load_dotenv()
Groq_key = os.getenv("GROQ_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = "rag-pinecone-index"

st.set_page_config(page_title="🗣️ Voice RAG Chatbot", layout="centered")
st.title("🗣️ Voice RAG Chatbot")
st.markdown("Upload a PDF, speak or type your question, and get a spoken answer using Groq + Pinecone.")

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("uploaded_doc.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


    os.environ["PINECONE_API_KEY"] = pinecone_key
    vectordb = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_name
    )

    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.3,
        max_tokens=1000
    )

    prompt_template = """
You are an intelligent assistant that answers questions based strictly on the provided document context.

## Instructions:
Use ONLY the context below to answer the question. Do not use outside knowledge or make assumptions.
If the answer is not found in the context, respond with:
"The answer is not available in the provided document."

## Guidelines:
- Be factual and precise
- Reference specific figures, dates, or names when relevant
- Do not infer or hallucinate information

---

### Context:
{context}

### Question:
{question}

### Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    # ── Voice Input ──
    if st.button("🎙️ Ask by Voice"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak your question clearly.")
            audio = r.listen(source)

        try:
            voice_query = r.recognize_google(audio)
            st.success(f"You asked: {voice_query}")

            response = qa_chain.invoke({"query": voice_query})
            answer = response["result"]
            st.markdown("### 📌 Answer:")
            st.write(answer)

            # Text-to-Speech output
            tts = gTTS(answer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
            with open(fp.name, "rb") as af:
                b64 = base64.b64encode(af.read()).decode()
                st.markdown(
                    f'<audio autoplay controls src="data:audio/mp3;base64,{b64}"></audio>',
                    unsafe_allow_html=True
                )

            with st.expander("📚 Source Snippets"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])

        except sr.UnknownValueError:
            st.error("Could not understand your voice. Please try again.")
        except sr.RequestError as e:
            st.error(f"Voice processing error: {e}")

    # ── Text Input ──
    user_query = st.text_input("Or type your question:")
    if user_query:
        with st.spinner("Fetching answer..."):
            response = qa_chain.invoke({"query": user_query})
            st.markdown("### 📌 Answer:")
            st.write(response["result"])

            with st.expander("📚 Source Snippets"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])
else:
    st.info("Please upload a PDF to begin.")
