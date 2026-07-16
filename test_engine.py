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
        print_summary(results, phase_name="PHASE 1")
        return engine, results

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
    print_summary(results, phase_name="PHASE 1")
    return engine, results


def run_store_text_tests(engine):
    """
    Phase 2 Testing: Validates the core store_text() functionality with chunking.
    """
    print("\nStarting VectorEngine Phase 2: store_text() Validation\n")
    print("-" * 50)
    
    results = {}
    test_text = "The solar system consists of eight planets and their moons."
    test_source = "science_manual"

    # 1. Text Insertion Check
    current_count = engine.collection.count()
    try:
        ids = engine.store_text(test_text, source=test_source)
        if isinstance(ids, list) and len(ids) > 0:
            print("PASS: store_text() call succeeded, returned list of IDs.")
            results["Insertion Success"] = True
            
            # 2. Document ID Check
            if all(isinstance(i, str) for i in ids):
                print(f"PASS: All Document IDs returned are strings. Chunks: {len(ids)}")
                results["ID Generation"] = True
            else:
                print("FAIL: Non-string ID returned in list.")
                results["ID Generation"] = False

            # 3. Collection Count Check
            new_count = engine.collection.count()
            if new_count == current_count + len(ids):
                print(f"PASS: Collection count updated correctly (new: {new_count}).")
                results["Collection Count"] = True
            else:
                print(f"FAIL: Collection count mismatch. Expected {current_count + len(ids)}, got {new_count}.")
                results["Collection Count"] = False

            # 4. Data Persistence & Metadata Verification
            # Fetch the document back from Chroma
            stored_data = engine.collection.get(ids=[ids[0]])
            if stored_data["ids"] and stored_data["documents"][0] == test_text:
                print("PASS: Document content verified in database.")
                results["Content Persistence"] = True
            else:
                print("FAIL: Stored content mismatch or not found.")
                results["Content Persistence"] = False

            meta = stored_data["metadatas"][0]
            if (meta.get("source") == test_source and 
                meta.get("chunk_index") == 0 and 
                meta.get("chunk_count") == 1):
                print(f"PASS: Single chunk metadata verified (source: '{test_source}', index: 0, count: 1).")
                results["Metadata Persistence"] = True
            else:
                print(f"FAIL: Metadata mismatch or not found: {meta}")
                results["Metadata Persistence"] = False

        else:
            print(f"FAIL: store_text() did not return a valid list of IDs.")
            results["Insertion Success"] = False
            results["ID Generation"] = False
            results["Collection Count"] = False
            results["Content Persistence"] = False
            results["Metadata Persistence"] = False

    except Exception as e:
        print(f"FAIL: Unexpected error during Phase 2 basic tests: {e}")
        results["Insertion Success"] = False

    # 5. Long Text Chunking Check and Sequential Index Verification
    try:
        long_text = (
            "The Pyramids of Giza are some of the most famous structures in human history, located on the outskirts of Cairo, Egypt. "
            "Built during the Fourth Dynasty of the Old Kingdom, these monumental tombs were constructed for the pharaohs Khufu, Khafre, and Menkaure. "
            "The Great Pyramid of Giza, built for Khufu, is the largest of the three, originally standing at 146.6 meters tall and composed of "
            "approximately 2.3 million stone blocks, each weighing between 2.5 and fifteen tons. For over 3,800 years, it stood as the tallest man-made "
            "structure in the world, marveling travelers and historians alike. The construction of the pyramids required a massive, highly organized "
            "workforce of skilled laborers, builders, and administrators. Archaeologists believe that these builders lived in nearby temporary cities "
            "and were well-fed and housed, dispelling the historical myth of slave labor. The precision alignment of the pyramids, oriented almost "
            "perfectly to true north, is a testament to the advanced astronomical and mathematical knowledge possessed by ancient Egyptian civil engineers. "
            "Even today, researchers use modern scanning technologies to probe the interiors of the pyramids, discovering hidden voids and passageways "
            "that continue to challenge our understanding of how they were built and what secrets they may still hold."
        )
        before_count = engine.collection.count()
        long_ids = engine.store_text(long_text, source="history_manual")
        after_count = engine.collection.count()
        
        if len(long_ids) > 1 and (after_count == before_count + len(long_ids)):
            print(f"PASS: Long document chunked successfully (produced {len(long_ids)} chunks).")
            # Verify sequential indexing and counting metadata
            stored_chunks = engine.collection.get(ids=long_ids)
            # Reorder them to match our indices to see sequence
            chunk_metas = stored_chunks["metadatas"]
            # ID alignment confirmation
            sequential = True
            for i, chunk_id in enumerate(long_ids):
                # find metadata for this chunk_id
                idx_in_result = stored_chunks["ids"].index(chunk_id)
                meta = chunk_metas[idx_in_result]
                if meta.get("chunk_index") != i or meta.get("chunk_count") != len(long_ids):
                    sequential = False
                    print(f"FAIL: Metadata mismatch on chunk {i}: {meta}")
                    break
            
            if sequential:
                print("PASS: Chunk index values are sequential and chunk count metadata is correct.")
                results["Long Document Ingestion"] = True
            else:
                results["Long Document Ingestion"] = False
        else:
            print(f"FAIL: Chunking did not split text. Chunks count: {len(long_ids)}")
            results["Long Document Ingestion"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during chunking tests: {e}")
        results["Long Document Ingestion"] = False

    # 6. Validate empty text triggers ValueError
    try:
        try:
            engine.store_text("   ")
            empty_stored_fail = True
        except ValueError as ve:
            empty_stored_fail = False
            print(f"PASS: Empty text rejected with expected ValueError: {ve}")
            
        if not empty_stored_fail:
            results["Zero Length Storage Prevention"] = True
        else:
            print("FAIL: Managed to store empty text without raising ValueError.")
            results["Zero Length Storage Prevention"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during empty text storage verification: {e}")
        results["Zero Length Storage Prevention"] = False


    print("-" * 50)
    print_summary(results, phase_name="PHASE 2")
    return results


def run_phase3_search_tests():
    """
    Phase 3 Testing: Validates search functionality (semantic search, validation, clamping, edge cases).
    """
    print("\nStarting VectorEngine Phase 3: Search Tests\n")
    print("-" * 50)
    
    results = {}
    
    # Check if Groq API key is present
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("FAIL: GROQ_API_KEY not found in environment.")
        results["Search Setup"] = False
        print_summary(results, phase_name="PHASE 3")
        return results
        
    try:
        # Create a fresh engine instance
        engine = VectorEngine()
        results["Engine Creation"] = True
        print("PASS: Fresh VectorEngine created for Phase 3 Search Tests.")
        
        # Clear any existing documents in the collection to start fresh
        all_ids = engine.collection.get()["ids"]
        if all_ids:
            engine.collection.delete(ids=all_ids)
            
        # Seed the collection with at least 4 clearly distinct facts:
        facts = [
            {"text": "The capital of Japan is Tokyo.", "source": "geography"},
            {"text": "The capital of France is Paris.", "source": "geography"},
            {"text": "Photosynthesis converts sunlight into chemical energy in plants.", "source": "science"},
            {"text": "A classic French recipe for coq au vin involves braising chicken in red wine.", "source": "cooking"}
        ]
        
        # Store facts
        seeded_ids = []
        for fact in facts:
            ids = engine.store_text(fact["text"], source=fact["source"])
            seeded_ids.extend(ids)
                
        if len(seeded_ids) == len(facts):
            results["Database Seeding"] = True
            print(f"PASS: Seeding completed (stored {len(facts)} documents).")
        else:
            results["Database Seeding"] = False
            print("FAIL: Seeding could not store all documents.")
            
    except Exception as e:
        print(f"FAIL: Unexpected error during setup or seeding: {e}")
        results["Engine Creation"] = False
        results["Database Seeding"] = False
        print_summary(results, phase_name="PHASE 3")
        return results

    # 1. Basic query works, keys exist
    try:
        res = engine.search("What is the capital of Japan?")
        expected_keys = {"documents", "metadatas", "distances", "ids"}
        if isinstance(res, dict) and expected_keys.issubset(res.keys()):
            print("PASS: Basic search call succeeded and returned correct keys.")
            results["Basic Query"] = True
        else:
            print(f"FAIL: Basic search call returned malformed result: {res}")
            results["Basic Query"] = False
    except Exception as e:
        print(f"FAIL: Basic search call failed with exception: {e}")
        results["Basic Query"] = False

    # 2. Paraphrase semantic check
    try:
        res = engine.search("What's Japan's capital city?", top_k=3)
        if res.get("documents") and len(res["documents"]) > 0 and "Tokyo" in res["documents"][0]:
            print("PASS: Paraphrase test resolved 'Tokyo' as the top result.")
            results["Paraphrase Semantic Check"] = True
        else:
            top_doc = res["documents"][0] if res.get("documents") else "None"
            print(f"FAIL: Paraphrase test failed to rank Tokyo first. Top doc: '{top_doc}'")
            results["Paraphrase Semantic Check"] = False
    except Exception as e:
        print(f"FAIL: Paraphrase search failed: {e}")
        results["Paraphrase Semantic Check"] = False

    # 3. Cross-topic discrimination check
    try:
        res = engine.search("How do you braise chicken?", top_k=4)
        docs = res.get("documents", [])
        if docs and "coq au vin" in docs[0]:
            print("PASS: Cross-topic discrimination placed cooking result first.")
            results["Cross-Topic Discrimination"] = True
        else:
            top_doc = docs[0] if docs else "None"
            print(f"FAIL: Cross-topic discrimination failed. Top doc: '{top_doc}'")
            results["Cross-Topic Discrimination"] = False
    except Exception as e:
        print(f"FAIL: Cross-topic search failed: {e}")
        results["Cross-Topic Discrimination"] = False

    # 4. Ranking sort order sanity
    try:
        res = engine.search("What is photosynthesis?", top_k=4)
        distances = res.get("distances", [])
        if len(distances) >= 2:
            is_sorted = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
            is_closer = distances[0] < distances[-1]
            if is_sorted and is_closer:
                print("PASS: Distances are sorted ascending and top result is closer than the last.")
                results["Ranking Sanity"] = True
            else:
                print(f"FAIL: Distances sanity check failed. Sorted: {is_sorted}, distances: {distances}")
                results["Ranking Sanity"] = False
        else:
            print(f"FAIL: Insufficient distances returned: {distances}")
            results["Ranking Sanity"] = False
    except Exception as e:
        print(f"FAIL: Ranking sanity check failed with exception: {e}")
        results["Ranking Sanity"] = False

    # 5. Index/keys alignment validation
    try:
        res = engine.search("Tell me about French cooking", top_k=3)
        docs = res.get("documents", [])
        meta = res.get("metadatas", [])
        dist = res.get("distances", [])
        ids = res.get("ids", [])
        
        if len(docs) == len(meta) == len(dist) == len(ids):
            spot_id = ids[0]
            spot_retrieved = engine.collection.get(ids=[spot_id])
            if spot_retrieved.get("documents") and spot_retrieved["documents"][0] == docs[0]:
                print("PASS: Results lists are aligned and spot-checked correctly.")
                results["Result Alignment"] = True
            else:
                print("FAIL: Spot check failed. Retrieved document doesn't match search result.")
                results["Result Alignment"] = False
        else:
            print(f"FAIL: Length mismatch. docs: {len(docs)}, meta: {len(meta)}, dist: {len(dist)}, ids: {len(ids)}")
            results["Result Alignment"] = False
    except Exception as e:
        print(f"FAIL: Alignment validation failed with exception: {e}")
        results["Result Alignment"] = False

    # 6. Empty query query input validation
    try:
        try:
            engine.search("")
            empty_ok = False
        except ValueError as ve:
            empty_ok = "Query cannot be empty." in str(ve)
            
        try:
            engine.search("   ")
            space_ok = False
        except ValueError as ve:
            space_ok = "Query cannot be empty." in str(ve)
            
        if empty_ok and space_ok:
            print("PASS: Correctly rejected empty and whitespace-only queries.")
            results["Empty Query Validation"] = True
        else:
            print(f"FAIL: Empty query validation failed. Empty ok: {empty_ok}, Space ok: {space_ok}")
            results["Empty Query Validation"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during empty query validation: {e}")
        results["Empty Query Validation"] = False

    # 7. Invalid top_k query input validation
    try:
        try:
            engine.search("What is France?", top_k=0)
            zero_ok = False
        except ValueError as ve:
            zero_ok = "top_k must be a positive integer." in str(ve)
            
        try:
            engine.search("What is France?", top_k=-2)
            neg_ok = False
        except ValueError as ve:
            neg_ok = "top_k must be a positive integer." in str(ve)
            
        if zero_ok and neg_ok:
            print("PASS: Correctly rejected invalid top_k values (0 and negative).")
            results["Invalid top_k Validation"] = True
        else:
            print(f"FAIL: Invalid top_k validation failed. Zero ok: {zero_ok}, Neg ok: {neg_ok}")
            results["Invalid top_k Validation"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during top_k validation: {e}")
        results["Invalid top_k Validation"] = False

    # 8. Clamped top_k values
    try:
        res = engine.search("What is France?", top_k=10)
        count_docs = len(res.get("documents", []))
        if count_docs == 4:
            print("PASS: top_k values clamped correctly to size of collection.")
            results["Clamped top_k"] = True
        else:
            print(f"FAIL: Clamped top_k check failed. Expected at most 4 results, got {count_docs}.")
            results["Clamped top_k"] = False
    except Exception as e:
        print(f"FAIL: Clamped top_k validation failed with exception: {e}")
        results["Clamped top_k"] = False

    # 9. Empty database search error validation
    try:
        empty_engine = VectorEngine()
        # Reset the collection specifically for this instance of the engine
        all_ids = empty_engine.collection.get()["ids"]
        if all_ids:
            empty_engine.collection.delete(ids=all_ids)
            
        try:
            empty_engine.search("What is France?", top_k=3)
            empty_coll_ok = False
        except ValueError as ve:
            empty_coll_ok = "Collection is empty:" in str(ve)
            
        if empty_coll_ok:
            print("PASS: Correctly raised clean ValueError on empty collection query.")
            results["Empty Collection Edge Case"] = True
        else:
            print(f"FAIL: Did not raise expected clean ValueError on empty collection.")
            results["Empty Collection Edge Case"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during empty collection test: {e}")
        results["Empty Collection Edge Case"] = False

    print("-" * 50)
    print_summary(results, phase_name="PHASE 3")
    return results


def run_phase5_persistence_tests():
    """
    Phase 5 Testing: Validates configurable Chroma persistence.
    """
    print("\nStarting VectorEngine Phase 5: Persistence Tests\n")
    print("-" * 50)
    
    results = {}
    persist_dir = "./test_chroma_data"
    
    # 1. Clean up directory before we start
    if os.path.exists(persist_dir):
        import shutil
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Cleanup error before test: {e}")
            
    try:
        # Create first engine instance with persist_directory
        engine1 = VectorEngine(persist_directory=persist_dir)
        
        # Test document
        test_txt = "Persistent storage validation document: VectorEngine persistence works."
        test_source = "persist_test"
        
        # Store text
        ids = engine1.store_text(test_txt, source=test_source)
        if ids:
            print("PASS: Stored document in persistent VectorEngine instance.")
            results["Persistent Storage Ingestion"] = True
        else:
            print("FAIL: Failed to store document in persistent VectorEngine instance.")
            results["Persistent Storage Ingestion"] = False
            
        # Re-instantiate second engine instance pointing to same directory
        # (This simulates engine restart)
        engine2 = VectorEngine(persist_directory=persist_dir)
        
        # Search for the document using query in engine2
        search_res = engine2.search("VectorEngine persistence", top_k=1)
        documents = search_res.get("documents", [])
        
        if documents and test_txt in documents[0]:
            print("PASS: Successfully retrieved persisted document from second engine instance.")
            results["Persistence Retrieval Check"] = True
        else:
            print(f"FAIL: Retrieve persisted check failed. Got documents: {documents}")
            results["Persistence Retrieval Check"] = False
            
    except Exception as e:
        print(f"FAIL: Unexpected error during Phase 5 persistence tests: {e}")
        results["Persistent Storage Ingestion"] = False
        results["Persistence Retrieval Check"] = False
        
    finally:
        # Clean up directory after we finish
        if os.path.exists(persist_dir):
            import shutil
            try:
                shutil.rmtree(persist_dir)
                print("PASS: Cleaned up test database directory.")
                results["Clean Up Storage"] = True
            except Exception as e:
                print(f"FAIL: Failed to clean up database directory: {e}")
                results["Clean Up Storage"] = False
                
    print("-" * 50)
    print_summary(results, phase_name="PHASE 5")
    return results


def run_phase4_generation_tests():

    """
    Phase 4 Testing: Validates the RAG ask() generation method.
    """
    print("\nStarting VectorEngine Phase 4: Generation/Ask Tests\n")
    print("-" * 50)
    
    results = {}
    
    # Check if Groq API key is present
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("FAIL: GROQ_API_KEY not found in environment.")
        results["Ask Setup"] = False
        print_summary(results, phase_name="PHASE 4")
        return results
        
    try:
        # Create a fresh engine instance
        engine = VectorEngine()
        results["Engine Creation"] = True
        print("PASS: Fresh VectorEngine created for Phase 4 Generation Tests.")
        
        # Clear collection
        all_ids = engine.collection.get()["ids"]
        if all_ids:
            engine.collection.delete(ids=all_ids)
            
        # Seed facts
        facts = [
            {"text": "The capital of Japan is Tokyo.", "source": "geography"},
            {"text": "The capital of France is Paris.", "source": "geography"},
            {"text": "Photosynthesis converts sunlight into chemical energy in plants.", "source": "science"},
            {"text": "A classic French recipe for coq au vin involves braising chicken in red wine.", "source": "cooking"}
        ]
        seeded_ids = []
        for fact in facts:
            ids = engine.store_text(fact["text"], source=fact["source"])
            seeded_ids.extend(ids)
            
        if len(seeded_ids) == len(facts):
            results["Database Seeding"] = True
            print("PASS: Seeding completed successfully.")
        else:
            results["Database Seeding"] = False
            print("FAIL: Seeding did not store all documents.")
            
    except Exception as e:
        print(f"FAIL: Unexpected error during setup or seeding: {e}")
        results["Engine Creation"] = False
        results["Database Seeding"] = False
        print_summary(results, phase_name="PHASE 4")
        return results

    # 1. Grounded answer test
    try:
        res = engine.ask("What is the capital of Japan?")
        if "tokyo" in res.lower():
            print("PASS: Grounded answer test correctly answered 'Tokyo'.")
            results["Grounded Answer Test"] = True
        else:
            print(f"FAIL: Grounded answer test gave wrong/no answer: {res}")
            results["Grounded Answer Test"] = False
    except Exception as e:
        print(f"FAIL: Grounded answer test failed with exception: {e}")
        results["Grounded Answer Test"] = False

    # 2. No-hallucination test
    try:
        res = engine.ask("What is the population of Mars?")
        # Expect phrases stating insufficient information
        lower_res = res.lower()
        hallucination_phrases = ["don't have", "does not contain", "cannot answer", "no information", "insufficient", "not mention"]
        has_flag = any(phrase in lower_res for phrase in hallucination_phrases)
        if has_flag:
            print("PASS: No-hallucination test correctly refused to answer.")
            results["No-Hallucination Test"] = True
        else:
            print(f"FAIL: No-hallucination test failed to refuse to answer. Response: {res}")
            results["No-Hallucination Test"] = False
    except Exception as e:
        print(f"FAIL: No-hallucination test failed with exception: {e}")
        results["No-Hallucination Test"] = False

    # 3. Empty question validation
    try:
        try:
            engine.ask("")
            empty_ok = False
        except ValueError as ve:
            empty_ok = "Question cannot be empty." in str(ve)
            
        try:
            engine.ask("   ")
            space_ok = False
        except ValueError as ve:
            space_ok = "Question cannot be empty." in str(ve)
            
        if empty_ok and space_ok:
            print("PASS: Correctly rejected empty and whitespace-only questions.")
            results["Empty Question Validation"] = True
        else:
            print(f"FAIL: Empty question validation failed. Empty ok: {empty_ok}, Space ok: {space_ok}")
            results["Empty Question Validation"] = False
    except Exception as e:
        print(f"FAIL: Unexpected error during empty question validation: {e}")
        results["Empty Question Validation"] = False

    # 4. Empty collection behavior
    try:
        empty_engine = VectorEngine()
        # Reset collection specifically for this instance of the engine
        all_ids = empty_engine.collection.get()["ids"]
        if all_ids:
            empty_engine.collection.delete(ids=all_ids)
            
        res = empty_engine.ask("What is the capital of France?")
        if res == "I don't have any relevant information to answer that.":
            print("PASS: Correctly returned explicit no-context message on empty collection.")
            results["Empty Collection Behavior"] = True
        else:
            print(f"FAIL: Empty collection behavior failed. Response: {res}")
            results["Empty Collection Behavior"] = False
    except Exception as e:
        print(f"FAIL: Empty collection behavior failed with exception: {e}")
        results["Empty Collection Behavior"] = False

    # 5. Multi-chunk context test
    try:
        long_history = (
            "During the late Middle Ages, the city of Venice rose to prominence as a major maritime power and financial hub. "
            "Its strategic location at the head of the Adriatic Sea made it the primary gateway for trade between Europe and the Byzantine Empire, "
            "as well as the wider Levant. Venice's merchant fleet dominated the trade of luxury goods, spices, and silk. "
            "To protect these trade routes, the Republic of Venice constructed the Arsenal, a vast state-owned shipyard. "
            "The Arsenal was a marvel of pre-industrial manufacturing, capable of producing a fully equipped war galley in a single day. "
            "This industrial efficiency allowed Venice to maintain naval supremacy in the Mediterranean for centuries. "
            "Beyond its military and commercial achievements, Venice was also a center of culture and printing. "
            "The famous Venetian publisher Aldus Manutius revolutionized book production by developing italic typeface and printing smaller, "
            "pocket-sized editions of classical literature. This made books more accessible and affordable, greatly contributing to the spread of "
            "Renaissance humanism across Europe."
        )
        # Store on the main engine
        long_ids = engine.store_text(long_history, source="history_manual")
        
        # Venice facts should now be stored and split
        # Ask question that is only in the Aldus Manutius chunk
        res = engine.ask("Who developed italic typeface in Venice?")
        if "aldus" in res.lower() and "manutius" in res.lower():
            print("PASS: Multi-chunk context test correctly answered 'Aldus Manutius'.")
            results["Multi-Chunk Context Test"] = True
        else:
            print(f"FAIL: Multi-chunk context test gave wrong/no answer: {res}")
            results["Multi-Chunk Context Test"] = False
    except Exception as e:
        print(f"FAIL: Multi-chunk context test failed with exception: {e}")
        results["Multi-Chunk Context Test"] = False

    print("-" * 50)
    print_summary(results, phase_name="PHASE 4")
    return results


def print_summary(results, phase_name="PHASE 1"):
    print(f"\n{phase_name} STATUS SUMMARY:")
    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed: all_passed = False
        print(f"  {test.ljust(30)}: {status}")
    
    if all_passed:
        if phase_name == "PHASE 1":
            print("\n ALL SYSTEMS GO: Your backend infrastructure is solid!")
        else:
            print(f"\n SUCCESS: {phase_name} validation complete!")
    else:
        print(f"\n ACTION REQUIRED: Fix the {phase_name} failures above.")


if __name__ == "__main__":
    # Run Phase 1
    engine, p1_results = run_phase1_tests()
    
    # Run Phase 2 ONLY if initialization was successful
    p2_results = {}
    if p1_results.get("Initialization") and engine:
        p2_results = run_store_text_tests(engine)
    else:
        print("\nSKIPPING Phase 2: Phase 1 must pass first.")

    # Run Phase 3 ONLY if storage baseline passed
    p3_results = {}
    if p2_results.get("Insertion Success"):
        p3_results = run_phase3_search_tests()
    else:
        print("\nSKIPPING Phase 3: Phase 2 Must pass first.")

    # Run Phase 4 ONLY if search baseline passed
    p4_results = {}
    if p3_results.get("Basic Query"):
        p4_results = run_phase4_generation_tests()
    else:
        print("\nSKIPPING Phase 4: Phase 3 Must pass first.")

    # Run Phase 5 ONLY if generation baseline passed
    if p4_results.get("Grounded Answer Test"):
        run_phase5_persistence_tests()
    else:
        print("\nSKIPPING Phase 5: Phase 4 Must pass first.")

