import os
import streamlit as st
from pypdf import PdfReader
import groq

from rag_engine import VectorEngine

# Page Config
st.set_page_config(
    page_title="VectorEngine RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# 1. Environment & Secrets Bridge
api_key = os.environ.get("GROQ_API_KEY")
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.error("🔑 **GROQ_API_KEY is not configured!** Please set it in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

# 2. Session State Caching for VectorEngine
# Cache VectorEngine instance to prevent loading embedding models repeatedly
if "engine" not in st.session_state:
    try:
        # Save persistence data to ./chroma_db
        st.session_state.engine = VectorEngine(api_key=api_key, persist_directory="./chroma_db")
    except Exception as e:
        st.error(f"Failed to initialize VectorEngine: {e}")
        st.stop()

engine = st.session_state.engine

# 3. Session State Initializations
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sync UI State with persisted Chroma DB on startup
if not st.session_state.ingested_docs:
    try:
        db_data = engine.collection.get()
        if db_data and db_data.get("metadatas"):
            metadatas = db_data["metadatas"]
            for meta in metadatas:
                if meta:
                    source = meta.get("source", "unknown")
                    chunk_count = meta.get("chunk_count", 1)
                    st.session_state.ingested_docs[source] = chunk_count
                    st.session_state.processed_files.add(source)
    except Exception as e:
        st.warning(f"Could not restore database state from storage: {e}")

# --- Sidebar (Ingestion Panel) ---
st.sidebar.title("📁 Document Ingestion")
st.sidebar.markdown("Upload files to index them inside `VectorEngine` (ChromaDB).")

uploaded_files = st.sidebar.file_uploader(
    "Select files to upload:",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        # Skip if already ingested
        if filename in st.session_state.processed_files:
            continue
            
        with st.sidebar.spinner(f"Ingesting {filename}..."):
            try:
                # Text Extraction
                if filename.endswith(".pdf"):
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page_num, page in enumerate(reader.pages):
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text += extracted_text + "\n"
                else:
                    text = uploaded_file.read().decode("utf-8")
                
                # Validation
                if not text.strip():
                    st.sidebar.error(f"⚠️ {filename} is empty or has no extractable text.")
                    continue
                
                # Store text (chunks automatically)
                chunk_ids = engine.store_text(text, source=filename)
                
                # Record to Session State
                st.session_state.processed_files.add(filename)
                st.session_state.ingested_docs[filename] = len(chunk_ids)
                st.sidebar.success(f"✓ Ingested: {filename} ({len(chunk_ids)} chunks)")
            except Exception as e:
                st.sidebar.error(f"❌ Error processing {filename}: {str(e)}")

# Display Ingested Sources
if st.session_state.ingested_docs:
    st.sidebar.divider()
    st.sidebar.subheader("📚 Ingested Knowledge Base")
    for fname, chunks in st.session_state.ingested_docs.items():
        st.sidebar.text(f"📄 {fname} ({chunks} chunks)")
        
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Knowledge Base", type="primary", use_container_width=True):
        try:
            # Delete collection
            engine.chroma_client.delete_collection("rag_collection")
            # Recreate collection
            engine.collection = engine.chroma_client.get_or_create_collection("rag_collection")
            # Clear UI states
            st.session_state.ingested_docs = {}
            st.session_state.processed_files = set()
            st.session_state.messages = []
            st.success("Knowledge base cleared successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear knowledge base: {e}")
else:
    st.sidebar.divider()
    st.sidebar.info("Upload text or PDF files to see them listed here.")


# --- Main Area (Q&A Interface) ---
st.title("🤖 VectorEngine RAG Chatbot")
st.markdown("### Context-grounded Q&A with Groq completion engine")

# Clear chat history button
if st.session_state.messages:
    if st.button("Clear Chat Conversation"):
        st.session_state.messages = []
        st.rerun()

# Check context
has_docs = len(st.session_state.ingested_docs) > 0

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔍 Show Retrieved Sources"):
                for i, src in enumerate(message["sources"]):
                    doc_preview = src.get("document", "")
                    # Clean up long text preview
                    if len(doc_preview) > 300:
                        doc_preview = doc_preview[:300] + "..."
                    meta = src.get("metadata", {})
                    st.markdown(f"**Chunk {i+1}** | Source: `{meta.get('source')}` (Index: {meta.get('chunk_index') + 1} of {meta.get('chunk_count')})")
                    st.info(doc_preview)

# Chat Input UI
if not has_docs:
    st.info("👋 Hello! Please upload your documents in the sidebar to start asking questions content-grounded by your files.")
    
question = st.chat_input(
    placeholder="Ask a question about your documents..." if has_docs else "Waiting for documents...",
    disabled=not has_docs
)

if question:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        
    # 2. Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching files and generating answer..."):
            try:
                # Call ask method with return_sources=True
                # Design Decision: Using ask(return_sources=True) returns a tuple (answer, sources)
                # keeping backward compatibility when parameter is False.
                answer, sources = engine.ask(question, top_k=3, return_sources=True)
                
                # Update placeholder
                message_placeholder.markdown(answer)
                
                # Display retrieved sources
                if sources:
                    with st.expander("🔍 Show Retrieved Sources"):
                        for i, src in enumerate(sources):
                            doc_preview = src.get("document", "")
                            if len(doc_preview) > 300:
                                doc_preview = doc_preview[:300] + "..."
                            meta = src.get("metadata", {})
                            st.markdown(f"**Chunk {i+1}** | Source: `{meta.get('source')}` (Index: {meta.get('chunk_index') + 1} of {meta.get('chunk_count')})")
                            st.info(doc_preview)
                            
                # Save assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                err_msg = f"Error generating answer: {str(e)}"
                message_placeholder.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}"
                })
