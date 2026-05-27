"""
pipeline.py
-----------
Core RAG pipeline: embed → store → retrieve → answer.

Public API
----------
    get_embeddings() -> HuggingFaceEmbeddings
        Load the embedding model. No Streamlit coupling — caller handles caching.

    build_pipeline(pdf_path, doc_hash, embeddings) -> tuple[Runnable, int]
        Returns (chain, chunk_count).
        chain accepts {"query": str} and returns:
        {"result": str, "source_documents": list[Document]}
        chunk_count: vectors indexed (or existing count if namespace already existed).

    get_doc_hash(file_bytes) -> str
        MD5 8-char namespace key for Pinecone deduplication.
        Not used for security — collision probability is negligible for this
        deduplication-only use case. SHA-256 would be stronger but overkill here.

Design notes
------------
- MultiQueryRetriever generates 3–5 sub-queries per user query.
  At cfg.retriever_k=5 that means up to 25 LLM calls/user query at scale.
  On Groq free tier (14,400 req/day) this is fine. In production, cache
  or gate behind a latency budget.

- Cross-encoder reranking was evaluated and removed: it silently dropped
  relevant chunks when all scores fell below its internal threshold, producing
  empty context and "not found" answers. Removal documented here so future
  engineers don't re-add it without understanding the tradeoff.

Import note — langchain v1
--------------------------
MultiQueryRetriever moved to langchain-classic in langchain v1.
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
            Requires: pip install langchain-classic>=1.0.7
"""

import hashlib
import logging
import re
import time

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using the context provided below.

Rules:
1. Use ONLY information from the context. Do not use outside knowledge.
2. The context may be partial or mid-sentence — answer from what is available.
3. Be concise and factual.
4. Only say "The answer is not available in the provided document." if the context contains absolutely no information related to the question.

Context:
{context}

Question:
{question}

Answer:""".strip()

_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ---------------------------------------------------------------------------
# Chunk cleaning — applied at INDEX TIME only
# ---------------------------------------------------------------------------
# Cleaning is intentionally NOT repeated at query time. Running regex and
# sentence-dedup logic on every retrieval call adds unnecessary latency for
# chunks that were already cleaned when indexed. If you update the cleaning
# logic, delete the Pinecone namespace and re-index.

_ARTIFACT_PATTERNS = re.compile(
    r"<EOS>|<pad>|<unk>|<s>|</s>|<mask>|<sep>|<cls>",
    flags=re.IGNORECASE,
)
_WHITESPACE_RUNS = re.compile(r"[ \t]{3,}")
_NEWLINE_RUNS = re.compile(r"\n{3,}")
_TOC_PATTERN = re.compile(r"\.{4,}")   # 4+ consecutive dots = table-of-contents line

_MIN_CHUNK_CHARS = 80
_MAX_NOISE_RATIO = 0.60    # drop if >60% of tokens are 1-2 chars
_MAX_REPEAT_RATIO = 0.50   # drop if >50% of sentences are exact duplicates


def _is_noisy_chunk(text: str) -> bool:
    """
    Return True if chunk should be dropped before indexing.

    Three checks (in order of cheapness):
      1. TOC dot-leader check — lines with 4+ consecutive dots = contents page filler
      2. Short-token noise ratio — >60% tokens of length ≤2 signals figure/artifact text
      3. Sentence repetition — >50% duplicate sentences signals caption repeated in PDF layer
    """
    tokens = text.split()
    if not tokens:
        return True
    # TOC dot-leader check
    lines = text.splitlines()
    if lines:
        toc_lines = sum(1 for line in lines if _TOC_PATTERN.search(line))
        if toc_lines / len(lines) > 0.4:
            return True
    # Short-token noise ratio
    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    if short_tokens / len(tokens) > _MAX_NOISE_RATIO:
        return True
    # Sentence repetition
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    if len(sentences) > 2:
        unique = set(sentences)
        if 1 - (len(unique) / len(sentences)) > _MAX_REPEAT_RATIO:
            return True
    return False


def _clean_text(text: str) -> str:
    text = _ARTIFACT_PATTERNS.sub(" ", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = _NEWLINE_RUNS.sub("\n\n", text)
    return text.strip()


def _clean_documents(docs):
    """Clean and filter documents. Returns (cleaned_list, dropped_count)."""
    cleaned = []
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
        if len(doc.page_content) < _MIN_CHUNK_CHARS:
            continue
        if _is_noisy_chunk(doc.page_content):
            logger.debug("Dropped noisy chunk: %s…", doc.page_content[:80])
            continue
        cleaned.append(doc)
    dropped = len(docs) - len(cleaned)
    if dropped:
        logger.info("Dropped %d/%d chunks (too short or noisy).", dropped, len(docs))
    return cleaned, dropped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_doc_hash(file_bytes: bytes) -> str:
    """
    MD5-based 8-char namespace key.

    Used for Pinecone namespace deduplication only — not for security.
    MD5 chosen for speed; collision probability is negligible for this use case.
    Same bytes → same hash → same namespace → skip re-indexing.
    Different PDF, same filename → different bytes → different hash → new namespace.
    """
    return hashlib.md5(file_bytes).hexdigest()[:8]


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model (all-MiniLM-L6-v2, 384 dimensions).

    Not decorated with @st.cache_resource — caller (app.py) handles caching.
    evaluate_pipeline.py passes embeddings explicitly to avoid re-loading.

    Why all-MiniLM-L6-v2:
    - Runs locally, no API key, no quota.
    - 384 dims: fast similarity search, small Pinecone storage.
    - OpenAI ada-002 produces 1536 dims — more expressive but costs money.
    """
    logger.info("Loading embedding model: %s", cfg.embedding_model)
    return HuggingFaceEmbeddings(model_name=cfg.embedding_model)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _get_pinecone_index():
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    return pc.Index(cfg.pinecone_index)


def _namespace_has_vectors(namespace: str) -> bool:
    """
    Return True if namespace exists and has at least one vector.
    Uses pinecone >= 5 attribute API: stats.namespaces[ns].vector_count.
    """
    try:
        stats = _get_pinecone_index().describe_index_stats()
        ns_map = stats.namespaces
        return namespace in ns_map and ns_map[namespace].vector_count > 0
    except Exception:
        logger.exception("Failed to check Pinecone namespace '%s'", namespace)
        return False


def _wait_for_namespace(namespace: str, max_wait: int = 60) -> None:
    """
    Poll with exponential backoff until namespace is ready.

    max_wait=60s — beyond that, something is genuinely wrong and a clear
    error is more useful than a silent freeze. Replaced naive time.sleep(20)
    which gave no feedback and had no upper bound on wait time.
    """
    start = time.monotonic()
    delay = 2
    while time.monotonic() - start < max_wait:
        if _namespace_has_vectors(namespace):
            logger.info("Namespace '%s' is ready.", namespace)
            return
        logger.info(
            "Namespace '%s' not ready — retrying in %ds (%.0fs elapsed)",
            namespace, delay, time.monotonic() - start,
        )
        time.sleep(delay)
        delay = min(delay * 2, 15)
    raise TimeoutError(
        f"Pinecone namespace '{namespace}' not ready after {max_wait}s. "
        "Check your index name and API key, or try re-uploading the document."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pipeline(
    pdf_path: str,
    doc_hash: str,
    embeddings: HuggingFaceEmbeddings = None,
) -> tuple:
    """
    Build and return (rag_chain, chunk_count).

    Parameters
    ----------
    pdf_path   : path to PDF on disk.
    doc_hash   : 8-char hash used as Pinecone namespace.
    embeddings : pre-loaded model; if None, get_embeddings() is called.
                 Always pass this from app.py (cached) or evaluate_pipeline.py
                 (loaded once) to avoid re-downloading the model.

    Returns
    -------
    (Runnable, int)
        Runnable accepts {"query": str}.
        Returns {"result": str, "source_documents": list[Document]}.
        int is the number of chunks available (indexed or pre-existing).

    Notes on retrieval cost
    -----------------------
    MultiQueryRetriever fires 3–5 LLM sub-queries per user query.
    At k=5 that's up to 25 vector lookups. On Groq free tier this is fine.
    At scale, gate behind latency budget or cache sub-query results.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    chunk_count = 0

    # ── Step 1: Index only if namespace is empty ──────────────────────────────
    if _namespace_has_vectors(doc_hash):
        logger.info("Namespace '%s' exists — skipping re-indexing.", doc_hash)
        vectordb = PineconeVectorStore(
            index_name=cfg.pinecone_index,
            embedding=embeddings,
            namespace=doc_hash,
        )
        try:
            stats = _get_pinecone_index().describe_index_stats()
            ns = stats.namespaces.get(doc_hash)
            chunk_count = ns.vector_count if ns else 0
        except Exception:
            logger.warning("Could not read vector count for namespace '%s'.", doc_hash)
            chunk_count = 0
    else:
        logger.info("Namespace '%s' not found — indexing document.", doc_hash)
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        avg_text_len = sum(len(d.page_content) for d in documents) / max(len(documents), 1)
        if avg_text_len < 50:
            raise ValueError(
                "PDF appears to be scanned or image-only "
                f"(average {avg_text_len:.0f} chars/page, threshold 50). "
                "Pre-process with an OCR tool before uploading."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        texts = splitter.split_documents(documents)
        logger.info("Split into %d chunks before cleaning.", len(texts))

        texts, dropped = _clean_documents(texts)
        chunk_count = len(texts)
        logger.info("%d clean chunks will be indexed (%d dropped).", chunk_count, dropped)

        if not texts:
            raise ValueError(
                "All chunks were dropped during cleaning — the PDF may be "
                "scanned/image-only or otherwise unreadable as text."
            )

        vectordb = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=cfg.pinecone_index,
            namespace=doc_hash,
        )
        _wait_for_namespace(doc_hash)

    # ── Step 2: LLM ──────────────────────────────────────────────────────────
    chat_model = ChatGroq(
        model=cfg.llm_model,
        api_key=cfg.groq_api_key,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )

    # ── Step 3: Retriever ─────────────────────────────────────────────────────
    # MMR alone retrieved topically similar but contextually redundant chunks.
    # MultiQueryRetriever wraps it: generates 3–5 query reformulations and
    # takes the union — improving recall without sacrificing MMR diversity.
    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg.retriever_k, "fetch_k": cfg.retriever_fetch_k},
    )
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=chat_model,
    )

    # ── Step 4: LCEL chain ────────────────────────────────────────────────────
    answer_chain = _PROMPT | chat_model | StrOutputParser()

    def retrieve_and_answer(inputs: dict) -> dict:
        query = inputs["query"]
        logger.info("Retriever query: %s", query)
        docs = multi_retriever.invoke(query)
        answer = answer_chain.invoke({
            "context": _format_docs(docs),
            "question": query,
        })
        return {"result": answer, "source_documents": docs}

    return RunnableLambda(retrieve_and_answer), chunk_count
