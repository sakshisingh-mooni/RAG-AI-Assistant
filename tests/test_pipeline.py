"""
tests/test_pipeline.py
----------------------
Unit tests for pure-logic functions in pipeline.py.
No API keys, no Pinecone, no LLM calls.

The three functions under test (get_doc_hash, _clean_text, _is_noisy_chunk)
have zero external dependencies. We extract them here directly from the source
file using importlib + a stub module, so the tests run without installing the
full langchain/pinecone stack.

Run:
    pytest tests/test_pipeline.py -v
"""

import hashlib
import re
import sys
import types
import importlib
from unittest import mock


# ---------------------------------------------------------------------------
# Bootstrap: create stub modules for all pipeline.py external imports so we
# can import only the pure-logic functions without the full dependency stack.
# ---------------------------------------------------------------------------

def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

for _mod in [
    "langchain", "langchain.retrievers", "langchain.retrievers.multi_query",
    "langchain_community", "langchain_community.document_loaders",
    "langchain_core", "langchain_core.output_parsers",
    "langchain_core.prompts", "langchain_core.runnables",
    "langchain_groq", "langchain_huggingface",
    "langchain_pinecone", "langchain_text_splitters",
    "pinecone",
]:
    _stub(_mod)

# config stub
_cfg_stub = _stub("config")
_cfg_stub.cfg = types.SimpleNamespace(
    groq_api_key="x", pinecone_api_key="x", pinecone_index="x",
    chunk_size=1000, chunk_overlap=100, retriever_k=5, retriever_fetch_k=20,
    llm_temperature=0.3, llm_max_tokens=1000,
    embedding_model="all-MiniLM-L6-v2", llm_model="llama-3.3-70b-versatile",
)

# Set required attributes on stubs so pipeline module-level code doesn't crash
sys.modules["langchain.retrievers.multi_query"].MultiQueryRetriever = object
sys.modules["langchain_community.document_loaders"].PyMuPDFLoader = object
sys.modules["langchain_core.output_parsers"].StrOutputParser = object
sys.modules["langchain_core.prompts"].PromptTemplate = mock.MagicMock(return_value=object())
sys.modules["langchain_core.runnables"].RunnableLambda = object
sys.modules["langchain_groq"].ChatGroq = object
sys.modules["langchain_huggingface"].HuggingFaceEmbeddings = object
sys.modules["langchain_pinecone"].PineconeVectorStore = object
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = object
sys.modules["pinecone"].Pinecone = object

import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pipeline
from pipeline import get_doc_hash, _clean_text, _is_noisy_chunk


# ── get_doc_hash ──────────────────────────────────────────────────────────────

class TestGetDocHash:
    def test_returns_8_chars(self):
        assert len(get_doc_hash(b"some pdf bytes")) == 8

    def test_same_bytes_same_hash(self):
        data = b"identical content"
        assert get_doc_hash(data) == get_doc_hash(data)

    def test_different_bytes_different_hash(self):
        assert get_doc_hash(b"pdf_a") != get_doc_hash(b"pdf_b")

    def test_empty_bytes_does_not_crash(self):
        assert len(get_doc_hash(b"")) == 8

    def test_hash_is_hex(self):
        h = get_doc_hash(b"test content")
        assert all(c in "0123456789abcdef" for c in h)


# ── _clean_text ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_eos_token(self):
        assert "<EOS>" not in _clean_text("The model output <EOS> and stopped.")

    def test_removes_pad_token(self):
        assert "<pad>" not in _clean_text("Padding: <pad><pad><pad>").lower()

    def test_removes_case_insensitive_artifacts(self):
        result = _clean_text("<Pad> <eos> <UNK>")
        for token in ["<Pad>", "<eos>", "<UNK>"]:
            assert token not in result

    def test_collapses_excessive_whitespace(self):
        assert "      " not in _clean_text("word      another")

    def test_collapses_excessive_newlines(self):
        assert "\n\n\n" not in _clean_text("para1\n\n\n\n\npara2")

    def test_strips_leading_trailing_whitespace(self):
        result = _clean_text("   hello world   ")
        assert result == result.strip()

    def test_clean_text_passthrough(self):
        clean = "This is a perfectly normal sentence about machine learning."
        assert _clean_text(clean) == clean


# ── _is_noisy_chunk ───────────────────────────────────────────────────────────

class TestIsNoisyChunk:
    def test_empty_string_is_noisy(self):
        assert _is_noisy_chunk("") is True

    def test_clean_paragraph_is_not_noisy(self):
        text = (
            "Retrieval-Augmented Generation combines a retrieval system with a "
            "generative language model. The retriever fetches relevant document "
            "chunks, which are then passed as context to the LLM. This grounds "
            "the model response in the provided document content."
        )
        assert _is_noisy_chunk(text) is False

    def test_toc_lines_flagged(self):
        toc = "\n".join([
            "Introduction ............ 1",
            "Background .............. 5",
            "Methods ................. 9",
            "Results ................ 14",
            "Conclusion ............. 20",
        ])
        assert _is_noisy_chunk(toc) is True

    def test_high_short_token_ratio_flagged(self):
        # >60% tokens are 1-2 chars
        noisy = " ".join(["a", "b", "c", "d", "1", "2", "3", "4", "ok", "go"] * 5)
        assert _is_noisy_chunk(noisy) is True

    def test_repeated_sentences_flagged(self):
        sentence = "Figure 1 shows the attention weights for the encoder."
        repeated = (sentence + " ") * 6 + "This is unique. This is also unique."
        assert _is_noisy_chunk(repeated) is True

    def test_single_toc_line_not_flagged(self):
        # Only 1 of 5 lines has dots — below 40% threshold
        mixed = "\n".join([
            "The Transformer architecture was introduced in 2017.",
            "Introduction ............ 1",
            "It uses multi-head self-attention instead of recurrence.",
            "The encoder has six identical layers.",
            "Each layer contains two sub-layers.",
        ])
        assert _is_noisy_chunk(mixed) is False
