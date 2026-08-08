import os
import uuid
import logging
import warnings
from typing import Optional

# B2 fix: suppress tokenizers parallelism fork warning (must be set before SentenceTransformer import)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# B2 fix: suppress huggingface_hub resume_download deprecation (cosmetic, not a functional issue)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
import groq
from groq import Groq
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Suppress chromadb telemetry warnings caused by posthog version mismatch
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

logger = logging.getLogger("vectorengine")


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
        persist_directory: Optional[str] = None,
    ):
        # Groq Client Initialization
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please provide it via the 'api_key' parameter "
                "or set the 'GROQ_API_KEY' environment variable (e.g., in a .env file)."
            )
            
        self.client = Groq(api_key=self.api_key)

        # Chroma Ephemeral or Persistent Client Initialization
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=chroma_settings
            )
        else:
            self.chroma_client = chromadb.EphemeralClient(settings=chroma_settings)
            
        # TODO (multi-tenant): The collection name "rag_collection" is currently shared across all engine
        # instances using the same persist_directory. Before any real deployment, decide whether to
        # support per-user/per-session isolated collections (multi-tenant) or accept shared state.
        # This is explicitly out of scope until deployment planning.
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

    def store_text(self, text: str, source: str = "manual") -> list[str]:
        """
        Store text into the vector database, chunking it if necessary.

        Parameters:
        -----------
        text : str
            The input text to store.
        source : str
            The source identifier of the document.

        Returns:
        --------
        list[str]
            A list of generated UUIDs for each chunk.

        Raises:
        -------
        ValueError
            If the input text is empty or whitespace-only, or if no chunks are produced.
        """
        # 1. Validation: Ensure text is not empty/whitespace-only
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Empty text cannot be stored.")

        logger.info("store_text called: source='%s', input_length=%d chars", source, len(text))

        # 2. Splitting text into chunks
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            raise ValueError("Empty text cannot be stored.")

        logger.info("Chunked into %d chunk(s) for source='%s'", len(chunks), source)

        # 3. Create lists for batch insert
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        # Generate embeddings for all chunks in a single call to save time/compute
        encoded_embeddings = self.embedding_model.encode(chunks).tolist()

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(encoded_embeddings[i])
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "chunk_count": len(chunks)
            })

        # 4. Dimensionality Guard
        expected_dim = self.embedding_model.get_sentence_embedding_dimension()
        for vec in embeddings:
            if len(vec) != expected_dim:
                logger.error("Embedding dimension mismatch: expected %d, got %d (source='%s')", expected_dim, len(vec), source)
                raise RuntimeError(
                    f"Embedding dimension mismatch during ingestion: expected {expected_dim}, "
                    f"got {len(vec)}. This indicates a shape bug in embedding generation."
                )

        # 5. Insert into ChromaDB collection in batch
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info("Ingestion complete: %d chunk(s) stored, ids=%s", len(ids), ids)
        return ids

    def search(self, query: str, top_k: int = 3) -> dict:
        """
        Perform a semantic search for the query in the vector database.

        Parameters:
        -----------
        query : str
            The search query.
        top_k : int
            The number of nearest-neighbor results to return.

        Returns:
        --------
        dict
            A flat dictionary with keys:
            - "documents": list[str]
            - "metadatas": list[dict]
            - "distances": list[float]
            - "ids": list[str]

        Raises:
        -------
        ValueError
            If query is empty, top_k is less than 1, or collection is empty.
        """
        # 1. Validation
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.debug("search called: query_length=%d chars, top_k=%d", len(query), top_k)

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        count = self.collection.count()
        if count == 0:
            raise ValueError("Collection is empty: cannot perform search.")

        # 2. Clamping top_k
        effective_k = min(top_k, count)

        # 3. Query embedding & collection search
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k
        )

        # 4. Unwrap Chroma nested format & return flat dic
        flat = {
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadatas": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else [],
            "ids": results.get("ids", [[]])[0] if results.get("ids") else []
        }
        logger.debug("search returned %d result(s), top distance=%.4f", len(flat["documents"]), flat["distances"][0] if flat["distances"] else -1)
        return flat

    def ask(self, question: str, top_k: int = 3, return_sources: bool = False) -> str:
        """
        Answer a question using the retrieved context from stored documents.

        Parameters:
        -----------
        question : str
            The user question.
        top_k : int
            The number of nearest-neighbor context documents to retrieve.
        return_sources : bool
            Whether to return retrieved sources along with the answer.

        Returns:
        --------
        str or tuple[str, list[dict]]
            The LLM generated answer, or a tuple containing the answer and list of sources.

        Raises:
        -------
        ValueError
            If the question is empty or whitespace-only.
        RuntimeError
            If there is a Groq API error or general communication error.
        """
        # 1. Validation
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info("ask called: question_length=%d chars, top_k=%d", len(question), top_k)

        # 2. Retrieval
        try:
            search_results = self.search(question, top_k=top_k)
            documents = search_results.get("documents", [])
        except ValueError as ve:
            # If the search method raised ValueError due to empty collection,
            # intercept it and treat it as the zero-context case.
            if "Collection is empty:" in str(ve):
                if return_sources:
                    return "I don't have any relevant information to answer that.", []
                return "I don't have any relevant information to answer that."
            raise ve

        if not documents:
            if return_sources:
                return "I don't have any relevant information to answer that.", []
            return "I don't have any relevant information to answer that."

        # 3. Build sources list
        sources = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(documents, search_results.get("metadatas", []))
        ]

        # 4. Context Assembly
        context = "\n\n---\n\n".join(documents)

        # 5. Prompt construction
        prompt = f"""Answer the question using only the information in the context below. If the context does not contain enough information to answer, say so explicitly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""

        # 6. Groq SDK Call
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content
            logger.info("Groq completion received: answer_length=%d chars", len(answer))
            if return_sources:
                return answer, sources
            return answer
        except groq.APIError as e:
            logger.error("Groq API call failed: %s", str(e))
            raise RuntimeError(f"Groq API error occurred: {e}") from e
        except Exception as e:
            logger.error("Groq API call failed: %s", str(e))
            raise RuntimeError(f"An unexpected error occurred while communicating with Groq: {e}") from e




if __name__ == "__main__":
    # Internal check for initialization and storage
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        engine = VectorEngine()
        print("✅ VectorEngine initialized successfully.")
        
        # Test Case 1: Store simple text
        result = engine.store_text("This is a test document for Stage 1 validation.", source="test_suite")
        if result["success"]:
            print(f"✅ store_text() successful! ID: {result['id']}, Count: {result['collection_count']}")
        else:
            print(f"❌ store_text() failed: {result.get('error')}")

        # Test Case 2: Validate empty text
        empty_result = engine.store_text("  ")
        if not empty_result["success"]:
            print(f"✅ store_text() correctly rejected empty text: {empty_result['error']}")
        else:
            print("❌ store_text() failed to reject empty text.")

    except Exception as e:
        print(f"❌ Verification finished (status: {e})")