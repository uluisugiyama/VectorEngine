import os
from typing import Optional
from groq import Groq
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class VectorEngine:
    """
    Minimal VectorEngine skeleton for RAG (Phase 1).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        # Groq Client Initialization
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please provide it via the 'api_key' parameter "
                "or set the 'GROQ_API_KEY' environment variable (e.g., in a .env file)."
            )
            
        self.client = Groq(api_key=self.api_key)

        # Chroma EphemeralClient (In-memory) Initialization
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_collection"
        )

        # Sentence Transformer for Embeddings
        # NOTE: This can be a heavy operation; consider lazy loading in production.
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


if __name__ == "__main__":
    # Internal check for initialization
    try:
        # Note: This will try to download the model if not present.
        # It also requires GROQ_API_KEY to be set in environment or passed.
        engine = VectorEngine()
        print("VectorEngine initialized successfully.")
    except Exception as e:
        print(f"Initialization check finished (status: {e})")