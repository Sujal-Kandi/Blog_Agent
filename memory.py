import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger("blog_agent")

# Real local embeddings — no API key needed, runs fully offline.
# Falls back to FakeEmbeddings only if the package isn't installed,
# so the agent still starts rather than crashing on import.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("[MEMORY] Using HuggingFaceEmbeddings (all-MiniLM-L6-v2)")
except Exception as e:
    logger.warning(f"[MEMORY] HuggingFaceEmbeddings unavailable ({e}) — falling back to FakeEmbeddings. Memory retrieval will be random.")
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=384)

vectorstore = Chroma(
    collection_name="blog_memory",
    embedding_function=embeddings,
    persist_directory="./blog_memory_db"
)


def store_memory(text: str):
    vectorstore.add_texts([text])


def retrieve_memory(query: str) -> str:
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs]) if docs else ""
    except Exception:
        return ""
