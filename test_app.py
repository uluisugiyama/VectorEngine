"""
Streamlit AppTest smoke test for app.py.

Mocks the two external boundaries (embedding model + Groq API) so this test
verifies UI wiring without downloading the real model or making real API calls.
test_engine.py Phases 3-4 already validate real embedding/LLM correctness.
"""
import os
import sys
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv()

# Ensure GROQ_API_KEY is available (mocked Groq won't use it, but app.py checks for it)
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "test-mock-key"

# Clean up any leftover test chroma_db directory before running
TEST_CHROMA_DIR = "./chroma_db"


@patch("rag_engine.chromadb.PersistentClient")
@patch("rag_engine.Groq")
@patch("rag_engine.SentenceTransformer")
def test_streamlit_app(mock_st_cls, mock_groq_cls, mock_persist_client):
    """
    Smoke test verifying UI wiring: file upload → sidebar update → Q&A → answer display.
    
    Mock decorator order (bottom-up):
    - @patch("rag_engine.SentenceTransformer")  → mock_st_cls  (innermost = first arg)
    - @patch("rag_engine.Groq")                 → mock_groq_cls (middle = second arg)
    - @patch("rag_engine.chromadb.PersistentClient") → mock_persist_client (outermost = third arg)
    """
    print("Initializing AppTest simulation with mocked externals...")

    from chromadb import EphemeralClient
    mock_persist_client.side_effect = lambda path=None, settings=None: EphemeralClient(settings=settings)

    # --- Configure SentenceTransformer mock ---
    def fake_encode(text_or_list, *args, **kwargs):
        if isinstance(text_or_list, str):
            return np.random.rand(384).astype(np.float32)
        return np.random.rand(len(text_or_list), 384).astype(np.float32)

    mock_model_instance = MagicMock()
    mock_model_instance.encode.side_effect = fake_encode
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st_cls.return_value = mock_model_instance

    # --- Configure Groq mock ---
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="The capital of Japan is Tokyo."))
    ]
    mock_groq_cls.return_value = mock_groq_instance

    # --- 1. Start AppTest ---
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run(timeout=30)

    if at.exception:
        print(f"FAIL: Exception during startup: {at.exception}")
        sys.exit(1)

    print("PASS: App initialized without errors.")

    # --- 2. Check page title ---
    titles = [t.value for t in at.title]
    assert "🤖 VectorEngine RAG Chatbot" in titles, f"Title not found. Got: {titles}"
    print("PASS: Title verified: '🤖 VectorEngine RAG Chatbot'")

    # --- 3. Simulate file upload ---
    print("Simulating document ingestion via sidebar file_uploader...")
    text_data = b"The capital of Japan is Tokyo. Tokyo is a bustling, high-tech metropolis."

    uploader = at.sidebar.file_uploader[0]
    uploader.upload("japan_info.txt", text_data)
    at.run(timeout=30)

    if at.exception:
        print(f"FAIL: Exception after file upload: {at.exception}")
        sys.exit(1)

    # Verify file appears in session_state tracking
    assert "ingested_docs" in at.session_state, "ingested_docs not in session_state"
    ingested = at.session_state["ingested_docs"]
    assert "japan_info.txt" in ingested, f"File not tracked in ingested_docs. Got: {ingested}"
    print(f"PASS: 'japan_info.txt' tracked in ingested_docs ({ingested['japan_info.txt']} chunks).")

    # Verify processed_files prevents re-ingestion
    assert "processed_files" in at.session_state, "processed_files not in session_state"
    processed = at.session_state["processed_files"]
    assert "japan_info.txt" in processed, "File not in processed_files set."
    print("PASS: Duplicate-ingestion guard is active (processed_files tracking).")

    # Verify chat input is now enabled
    chat_input = at.chat_input[0]
    assert not chat_input.disabled, "Chat input remains disabled after file upload."
    print("PASS: Chat input is enabled after ingestion.")

    # --- 4. Submit a question ---
    print("Submitting question: 'What is the capital of Japan?'...")
    chat_input.set_value("What is the capital of Japan?")
    at.run(timeout=30)

    if at.exception:
        print(f"FAIL: Exception during answer generation: {at.exception}")
        sys.exit(1)

    # --- 5. Verify answer in session state ---
    assert "messages" in at.session_state, "messages not in session_state"
    messages = at.session_state["messages"]
    assert len(messages) >= 2, f"Expected ≥2 messages, got {len(messages)}"

    assistant_msg = messages[-1]
    assert assistant_msg["role"] == "assistant", f"Last message role: {assistant_msg['role']}"
    print(f"PASS: Assistant response stored. Content: '{assistant_msg['content'][:80]}...'")

    assert "tokyo" in assistant_msg["content"].lower(), \
        f"Answer does not mention 'Tokyo'. Got: {assistant_msg['content']}"
    print("PASS: Mocked answer correctly wired through to session state.")

    # --- 6. Verify sources are attached ---
    sources = assistant_msg.get("sources")
    if sources:
        print(f"PASS: {len(sources)} source(s) attached to answer.")
    else:
        print("INFO: No sources attached (may be expected with mock embeddings).")

    print("-" * 50)
    print("SUCCESS: All Streamlit AppTest smoke tests passed!")


if __name__ == "__main__":
    test_streamlit_app()
