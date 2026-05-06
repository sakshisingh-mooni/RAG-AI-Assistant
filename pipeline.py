import os
import hashlib
import time
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain_groq import ChatGroq

PROMPT_TEMPLATE = """
You are an intelligent assistant that answers questions based strictly on the provided document context.

## Instructions:
Use ONLY the context below to answer the question. Do not use outside knowledge or make assumptions.
If the answer is not found in the context, respond with:
"The answer is not available in the provided document."

## Guidelines:
- Be factual and precise
- Reference specific figures, dates, or names from the document when relevant
- Do not infer or hallucinate information

---

### Context:
{context}

### Question:
{question}

### Answer:
"""

def get_doc_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:8]

@st.cache_resource(show_spinner="Loading embedding model... (once only)")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_pipeline(pdf_path: str, doc_hash: str) -> RetrievalQA:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    texts = splitter.split_documents(documents)

    embeddings = get_embeddings()

    for attempt in range(3):
        try:
            vectordb = PineconeVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                index_name=os.environ["PINECONE_INDEX"],
                namespace=doc_hash
            )
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(20)
            else:
                raise e

    chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.3,
        max_tokens=1000
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        ),
        llm=chat_model
    )

    return RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )