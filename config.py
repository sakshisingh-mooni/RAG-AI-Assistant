"""
config.py
---------
Centralised configuration loaded once at startup.
Fails immediately with a clear message if any required env var is missing —
rather than throwing a cryptic KeyError deep inside the pipeline.

Usage:
    from config import cfg

    cfg.groq_api_key
    cfg.pinecone_api_key
    cfg.pinecone_index
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index: str

    # RAG tuning knobs — change here, measured via evaluate_pipeline.py
    # Three configurations evaluated (see README for full results table).
    # chunk_size=1000/k=5/fetch_k=20 is the best performing configuration:
    #   Faithfulness 0.90, Answer Relevancy 0.88, Context Recall 0.60.
    # chunk_size=500 was tested and degraded faithfulness (0.73) without
    # improving recall (0.60) — smaller chunks produced fragmented context
    # that hurt answer quality. See README for full experiment analysis.
    chunk_size: int = 1000
    chunk_overlap: int = 100
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"

    @classmethod
    def from_env(cls) -> "Config":
        required = ("GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX")
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}\n"
                "Create a .env file with these keys. See README for details."
            )
        return cls(
            groq_api_key=os.environ["GROQ_API_KEY"],
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            pinecone_index=os.environ["PINECONE_INDEX"],
        )


# Module-level singleton — imported everywhere, validated once.
cfg = Config.from_env()
