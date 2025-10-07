# afiyapal_multi_tool_agent/merck_tool.py

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings 
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from typing import Dict, Any
import os 
from chromadb import errors as chroma_errors 


# =================================================================
# 1. CONFIGURATION AND INITIALIZATION
# =================================================================

# --- LlamaIndex Settings ---
try:
    # Set the global embedding model to a powerful, local, open-source model.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("SUCCESS: Configured local embedding model (BAAI/bge-small-en-v1.5).")
except ImportError:
    print("WARNING: HuggingFaceEmbedding not installed. Using default OpenAI embedding (requires API key).")
    # Fallback will likely cause an error if no key is set, but this path is necessary.

# --- RAG Paths ---
DATA_DIR = "./data_merck"
CHROMA_PATH = "./chroma_db_merck"
INDEX_NAME = "merck_manual_search"

# Global variable to hold the initialized retriever
MERCK_MANUAL_RETRIEVER = None


# =================================================================
# 2. INDEX BUILDING AND LOADING FUNCTIONS
# =================================================================

def build_merck_manual_index_search():
    """Loads documents, indexes them, and stores the index in ChromaDB."""
    if not os.path.exists(DATA_DIR) or not any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)):
        print(f"ERROR: No PDF files found in {DATA_DIR}. Cannot build index.")
        return None

    print(f"Loading documents from {DATA_DIR}...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        db.get_collection(name=INDEX_NAME)
        db.delete_collection(name=INDEX_NAME)
        print("Existing collection deleted for rebuild.")
    except chroma_errors.NotFoundError:
        pass
        
    chroma_collection = db.get_or_create_collection(INDEX_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building index (this may take a few minutes)...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    print(f"Index built successfully and stored in {CHROMA_PATH}")
    return index.as_retriever(similarity_top_k=5) 


def get_merck_manual_query_engine():
    """Retrieves the existing index and creates a query engine (retriever)."""
    print(f"Loading existing index from {CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # This line raises NotFoundError if the collection doesn't exist
    chroma_collection = db.get_collection(INDEX_NAME) 
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    return index.as_retriever(similarity_top_k=5)


def load_or_build_index():
    """Checks for the index and builds it if not found."""
    try:
        return get_merck_manual_query_engine()
    except chroma_errors.NotFoundError:
        print("!!! Merck_Manual_Index not found. Building index now... !!!")
        retriever = build_merck_manual_index_search()
        if retriever:
             print("Merck_Manual_Index build successful. Proceeding with the retriever.")
        return retriever
    except Exception as e:
        print(f"An unexpected error occurred during Merck Manual index check: {e}")
        return None

# --- CRITICAL: INITIALIZE THE GLOBAL RETRIEVER ON IMPORT ---
# This runs immediately when agent.py imports this file.
MERCK_MANUAL_RETRIEVER = load_or_build_index()


# =================================================================
# 3. RAG Search Function (The Tool Exported to Agent)
# =================================================================

def merck_manual_rag_search(query: str) -> str:
    """
    SEARCHES the First Aid Knowledge Base for relevant text chunks.
    This function is exported and used as a tool by the Agent.
    """
    global MERCK_MANUAL_RETRIEVER # Reference the global variable initialized above

    if MERCK_MANUAL_RETRIEVER is None:
        return "RAG TOOL INIT ERROR: First Aid Knowledge Base failed to initialize. Cannot retrieve context."
    
    try:
        nodes = MERCK_MANUAL_RETRIEVER.retrieve(query)
        
        if not nodes:
            # RAG ran, but found no matches. Instruct the LLM to use general knowledge.
            return f"RETRIEVAL FAILED: No specific first aid context found for query: '{query}'. Respond using general medical knowledge."

        retrieved_context = "\n---\n".join([n.get_text() for n in nodes])
        
        return f"RETRIEVED KNOWLEDGE FROM FIRST AID MANUALS:\n{retrieved_context}"
    
    except Exception as e:
        return f"RAG SEARCH ERROR: Could not retrieve knowledge. Error: {e}"


# --- Test Execution ---
if __name__ == "__main__":
    if MERCK_MANUAL_RETRIEVER:
        print("\n--- TEST SEARCH ---")
        test_query = "What are the indications that a patient may require nutritional evaluation?"
        result = merck_manual_rag_search(test_query)
        print(f"Query: {test_query}\nResult:\n{result[:500]}...")
    else:
        print("\nCould not test search because MERCK_MANUAL_RETRIEVER failed to initialize.")