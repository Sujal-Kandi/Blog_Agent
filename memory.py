import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger("blog_agent")

# Use real local embeddings when enough memory is available (local dev).
# On memory-constrained environments (e.g. Render free tier / 512MB),
# set DISABLE_MEMORY=true in env vars to skip embeddings entirely.
_memory_enabled = os.getenv("DISABLE_MEMORY", "false").lower() != "true"

embeddings = None
vectorstore = None

if _memory_enabled:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            collection_name="blog_memory",
            embedding_function=embeddings,
            persist_directory="./blog_memory_db"
        )
        logger.info("[MEMORY] Using HuggingFaceEmbeddings (all-MiniLM-L6-v2)")
    except Exception as e:
        logger.warning(f"[MEMORY] Failed to initialise embeddings ({e}) — memory disabled.")
        _memory_enabled = False
else:
    logger.info("[MEMORY] Disabled via DISABLE_MEMORY env var — skipping embeddings load.")


def store_memory(text: str):
    if not _memory_enabled or vectorstore is None:
        return
    try:
        vectorstore.add_texts([text])
    except Exception as e:
        logger.warning(f"[MEMORY] store_memory failed: {e}")


def retrieve_memory(query: str) -> str:
    if not _memory_enabled or vectorstore is None:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs]) if docs else ""
    except Exception:
        return ""
