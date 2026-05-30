import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the backend engine
try:
    from rag_engine import VectorEngine
except ImportError as e:
    print(f"FAIL: Could not import rag_engine. Error: {e}")
    sys.exit(1)


def run_phase1_tests():
    print("Starting VectorEngine Phase 1 Validation\n")
    print("-" * 50)
    
    results = {}

    # 1. Environment & API Key Check
    # Why: Without a Groq API key, the LLM inference will fail later.
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print("PASS: GROQ_API_KEY found in environment.")
        results["Environment"] = True
    else:
        print("WARN: GROQ_API_KEY not found in environment. (Will check constructor next)")
        results["Environment"] = False

    # 2. VectorEngine Initialization
    # Why: This verifies that all libraries (Groq, Chroma, SentenceTransformers) are installed and reachable.
    engine: Optional[VectorEngine] = None
    try:
        engine = VectorEngine()
        print("PASS: VectorEngine initialized successfully.")
        results["Initialization"] = True
    except Exception as e:
        print(f"FAIL: VectorEngine failed to initialize. Error: {e}")
        results["Initialization"] = False
        # If initialization fails, most other tests can't proceed.
        print_summary(results)
        return

    # 3. Groq Client Check
    # Why: Ensuring the client is available for future inference tasks.
    if hasattr(engine, "client") and engine.client:
        print("PASS: Groq client is active.")
        results["Groq Client"] = True
    else:
        print("FAIL: Groq client not found on engine instance.")
        results["Groq Client"] = False

    # 4. Chroma Client & Collection Check
    # Why: RAG depends on a functioning vector database to store and retrieve document chunks.
    try:
        if engine.chroma_client and engine.collection:
            # Check if collection name is correct (optional check)
            print(f"PASS: Chroma EphemeralClient active. Collection '{engine.collection.name}' is ready.")
            results["Chroma/Collection"] = True
        else:
            print("FAIL: Chroma client or collection missing.")
            results["Chroma/Collection"] = False
    except Exception as e:
        print(f"FAIL: Error accessing Chroma collection. Error: {e}")
        results["Chroma/Collection"] = False

    # 5. Embedding Model Check
    # Why: SentenceTransformers turn text into vectors. This is the 'math engine' of your RAG.
    if hasattr(engine, "embedding_model") and engine.embedding_model:
        model_name = getattr(engine.embedding_model, "model_card_data", "Unknown") # Slight hack to get info
        print(f"PASS: Embedding model loaded.")
        results["Embedding Model"] = True
    else:
        print("FAIL: Embedding model not loaded.")
        results["Embedding Model"] = False

    # 6. Text Splitter Check
    # Why: Large documents must be broken into chunks to fit LLM context windows and for precise retrieval.
    if hasattr(engine, "text_splitter") and engine.text_splitter:
        print("PASS: RecursiveCharacterTextSplitter configured.")
        results["Text Splitter"] = True
    else:
        print("FAIL: Text splitter not initialized.")
        results["Text Splitter"] = False

    print("-" * 50)
    print_summary(results)


def print_summary(results):
    print("\nPHASE 1 STATUS SUMMARY:")
    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed: all_passed = False
        print(f"  {test.ljust(20)}: {status}")
    
    if all_passed:
        print("\n ALL SYSTEMS GO: Your backend infrastructure is solid!")
    else:
        print("\n ACTION REQUIRED: Fix the failures above before moving to Phase 2.")


if __name__ == "__main__":
    run_phase1_tests()
