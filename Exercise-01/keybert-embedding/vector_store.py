import os
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from keybert import KeyBERT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import Config


class KeywordEmbeddings:
    """
    Extracts KeyBERT phrases and embeds only those phrases.

    The complete document text is kept separately in Document.page_content.
    Chroma receives the generated keyword text through this embedding wrapper.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        top_n: int = 15,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
        diversity: float = 0.5,
        use_mmr: bool = True,
        stop_words: Optional[str] = "english",
    ):
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.diversity = diversity
        self.use_mmr = use_mmr
        self.stop_words = stop_words

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Reuse the same sentence-transformer model inside KeyBERT.
        self.keyword_model = KeyBERT(model=self.embedding_model.client)

    def _extract_keyword_text(self, text: str) -> str:
        """Return a compact, semantically representative keyword string."""
        text = text.strip()

        if not text:
            return ""

        keywords = self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=self.keyphrase_ngram_range,
            stop_words=self.stop_words,
            top_n=self.top_n,
            use_mmr=self.use_mmr,
            diversity=self.diversity,
        )

        # Keep only keyphrases, not KeyBERT scores.
        return " | ".join(keyword for keyword, _score in keywords)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Extract keywords from each text and embed only the keywords.

        This method is called by Chroma when documents are inserted.
        """
        keyword_texts = [
            self._extract_keyword_text(text)
            for text in texts
        ]

        # Fallback prevents empty chunks from causing embedding errors.
        keyword_texts = [
            keyword_text if keyword_text else text[:500]
            for keyword_text, text in zip(keyword_texts, texts)
        ]

        return self.embedding_model.embed_documents(keyword_texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Extract keywords from the query and embed only those keywords.
        """
        keyword_text = self._extract_keyword_text(text)

        if not keyword_text:
            keyword_text = text[:500]

        return self.embedding_model.embed_query(keyword_text)


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(self):
        self.config = Config()
        self.embeddings = self._initialize_embeddings()
        self.text_splitter = self._initialize_text_splitter()
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()

    def _initialize_embeddings(self) -> KeywordEmbeddings:
        """Initialize KeyBERT-based keyword embeddings."""
        return KeywordEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            device=getattr(self.config, "EMBEDDING_DEVICE", "cpu"),
            top_n=getattr(self.config, "KEYBERT_TOP_N", 15),
            keyphrase_ngram_range=getattr(
                self.config,
                "KEYBERT_NGRAM_RANGE",
                (1, 3),
            ),
            diversity=getattr(self.config, "KEYBERT_DIVERSITY", 0.5),
            use_mmr=getattr(self.config, "KEYBERT_USE_MMR", True),
            stop_words=getattr(
                self.config,
                "KEYBERT_STOP_WORDS",
                "english",
            ),
        )

    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter for chunking."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _initialize_vector_store(self):
        """Initialize or load the existing ChromaDB vector store."""
        os.makedirs(self.config.CHROMA_PERSIST_DIR, exist_ok=True)

        # Kept for compatibility with existing Chroma configuration.
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.config.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        )

        self.vector_store = Chroma(
            collection_name=self.config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.config.CHROMA_PERSIST_DIR,
        )

    def load_txt_file(self, file_path: str) -> List[Document]:
        """Load a single TXT file."""
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return [
            Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                },
            )
        ]

    def load_txt_from_content(
        self,
        content: str,
        filename: str,
    ) -> List[Document]:
        """Load TXT content directly."""
        return [
            Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "file_path": f"uploaded/{filename}",
                },
            )
        ]

    def process_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into chunks and add metadata."""
        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["chunk_total"] = len(chunks)

        return chunks

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to Chroma.

        Chroma embeds each chunk using extracted KeyBERT phrases.
        The complete chunk remains available as page_content.
        """
        chunks = self.process_documents(documents)

        if chunks:
            self.vector_store.add_documents(chunks)

            # Supported in older LangChain Chroma integrations.
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Perform keyword-based semantic similarity search."""
        k = k or self.config.TOP_K_RESULTS
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[tuple]:
        """Perform keyword-based similarity search with scores."""
        k = k or self.config.TOP_K_RESULTS
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: Optional[int] = None):
        """Get a retriever for a RAG chain."""
        k = k or self.config.TOP_K_RESULTS

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def clear_collection(self):
        """Delete and recreate the Chroma collection."""
        self.vector_store.delete_collection()
        self._initialize_vector_store()

    def get_collection_stats(self) -> dict:
        """Return collection statistics."""
        try:
            collection = self.vector_store._collection

            return {
                "total_documents": collection.count(),
                "collection_name": self.config.COLLECTION_NAME,
            }
        except Exception as error:
            return {
                "total_documents": 0,
                "collection_name": self.config.COLLECTION_NAME,
                "error": str(error),
            }

    def load_directory(self, directory: str) -> int:
        """Load all TXT files from a directory."""
        total_chunks = 0

        if not os.path.exists(directory):
            os.makedirs(directory)
            return 0

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                documents = self.load_txt_file(file_path)
                total_chunks += self.add_documents(documents)

        return total_chunks