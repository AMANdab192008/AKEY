"""
VECTOR STORE SERVICE MODULE
===========================

This service builds and queries the FAISS vector index used for context retrieval.
Learning data (database/learning_data/*.txt) and past chats (database/chats_data/*.json)
are loaded at startup, split into chunks, embedded with HuggingFace, and stored in FAISS.
When the user asks a question we embed it and retrieve the k most similar chunks; only
those chunks are sent to the LLM, so token usage is bounded.

LIFECYCLE:
  - create_vector_store(): Load all .txt and .json, chunk, embed, build FAISS, save to disk.
    Called once at startup. Restart the server after adding new .txt files so they are included.
  - get_retriever(k): Return a retriever that fetches k nearest chunks for a query string.
  - save_vector_store(): Write the current FAISS index to database/vector_store/ (called after create).

Embeddings run locally (sentence-transformers); no extra API key. Groq and Realtime services
call get_retriever() for every request to get context.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = logging.getLogger("A.K.E.Y")


# ===================================================================================
# VECTOR STORE SERVICE CLASS
# ===================================================================================

class VectorStoreService:
    """
    Builds a FAISS index from learning_data .txt files and chats_data .json files,
    and provides a retriever to fetch the k most relevant chunks for a query.
    """

    def __init__(self):
        """Create the embedding model (local) and text splitter; vectore_store is set in create_vector_store()."""
        # Embeddings run locally (no API key); used to convert text into vectors for similarity search.
        self.embeddings = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL,
            model_kwargs = {"device": "cpu"},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.vectore_store: Optional[FAISS] = None
        self._retriever_cache: dict = {}

# =====================================================================================
# LOAD DOCUMENTS FROM DISK
# =====================================================================================

    def load_learning_data(self) -> List[Document]:
        """Read all .txt files in database/learning_data/ and return one Document per file (content + source name). """
        documents = []
        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": str(file_path.name)}))
                        logger.info("[VECTOR] Loaded learning data: %s (%d chars)", file_path.name, len(content))
            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)
        logger.info("[VECTOR] Total learning data files loaded: %d", len(documents))
        return documents

    def load_chat_history(self) -> List[Document]:
        """Load all .json files in database/chats_data/; turn each into one Document (User:/Assistant: lines). """
        documents = []
        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                messages = chat_data.get("messages", [])
                # Format as "User: ..." / "Asisstant: ..." so the retriever can match past conversations.
                chat_content = "\n".join([
                    f"User: {msg.get('content', '')}" if msg.get('role') == 'user'
                    else f"Assistant: {msg.get('content', '')}"
                    for msg in messages
                ])
                if chat_content.strip():
                    documents.append(Document(page_content=chat_content, metadata={"source": f"chat_{file_path.stem}"}))
                    logger.info("[VECTOR] Loaded chat history: %s (%d messages)", file_path.name, len(messages))
            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)
        logger.info("[VECTOR] Total chat history files loaded: %d", len(documents))
        return documents

    # ----------------------------------------------------------------------------------------------------
    # BUILD AND SAVE FAISS INDEX
    # ----------------------------------------------------------------------------------------------------

    def create_vector_store(self) -> FAISS:
        """
        Load learning_data + chats_data, chunk, embed, build FAISS index, save to disk.
        Called once at startup. If there are no documents we create a tiny placeholder index.
        """
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()
        all_documents = learning_docs + chat_docs
        logger.info("[VECTOR] Total documents to index: %d (learning: %d, chat: %d)",
                    len(all_documents), len(learning_docs), len(chat_docs))

        if not all_documents:
            # Placeholder so get_retriever() never fails; returns this single chunk for any query.
            self.vectore_store = FAISS.from_texts(["No data available yet."], self.embeddings)
            logger.info("[VECTORE] No documents found, created placeholder index")
        else:
            chunks = self.text_splitter.split_documents(all_documents)
            logger.info("[VECTOR] Split into %d chunks (chunk_size=%d, overlap=%d)",
                         len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
            self.vectore_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("[VECTOR] FAISS index built successfully with %d vectors", len(chunks))

        self._retriever_cache.clear()
        self.save_vector_store()
        return self.vectore_store

    def save_vector_store(self):
        """Write the current FAISS index to database/vectore_store/. On error we only log."""
        if self.vectore_store:
            try:
                self.vectore_store.save_local(str(VECTOR_STORE_DIR))
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    # --------------------------------------------------------------------------------------
    # RETRIEVER FOR CONTEXT
    # --------------------------------------------------------------------------------------

    def get_retriever(self, k: int = 10):
        """Return a retriever that returns the k most similar chunks for a query string."""
        if not self.vectore_store:
            raise RuntimeError("Vectore store not initialized. This should not happen.")
        if k not in self._retriever_cache:
            self._retriever_cache[k] = self.vectore_store.as_retriever(search_kwargs={"k": k})
        return self._retriever_cache[k]
    