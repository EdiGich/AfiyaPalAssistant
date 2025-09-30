# rag_tool.py

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings # <--- NEW: Import Settings
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore 

import chromadb
from typing import Dict, Any
import os 
from chromadb import errors as chroma_errors # <--- NEW: Import Chroma errors for better exception handling


# =================================================================
# 1. CONFIGURE LOCAL EMBEDDING MODEL (The fix for your error!)
# =================================================================

# Setting the global embedding model to a powerful, local, open-source model.
# This prevents the need for an OpenAI API key.
# Requires: pip install sentence-transformers
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("SUCCESS: Configured local embedding model (BAAI/bge-small-en-v1.5).")
except ImportError:
    print("WARNING: HuggingFaceEmbedding not installed. Using default OpenAI embedding (requires API key).")
    # If the import fails, it will fall back to the default (which will still error if no key is set)

# --- Configuration ---
DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"
INDEX_NAME = "first_aid_index"

# --- RAG Indexing/Loading Functions ---

def build_first_aid_index():
    """
    Loads documents, indexes them, and stores the index in ChromaDB.
    """
    if not os.path.exists(DATA_DIR) or not any(f.endswith('.pdf') for f in os.listdir(DATA_DIR)):
        print(f"ERROR: No PDF files found in {DATA_DIR}. Cannot build index.")
        return

    print(f"Loading documents from {DATA_DIR}...")
    # 1. Load Documents
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # 2. Setup ChromaDB client and collection
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Check if the collection already exists and delete it if you want to rebuild
    try:
        # Use a specific way to check for existence before attempting delete
        db.get_collection(name=INDEX_NAME)
        db.delete_collection(name=INDEX_NAME)
        print("Existing collection deleted for rebuild.")
    except chroma_errors.NotFoundError:
        pass # Ignore if collection doesn't exist
        
    chroma_collection = db.get_or_create_collection(INDEX_NAME)

    # 3. Create Vector Store and Index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Building the index automatically handles chunking and embedding (using Settings.embed_model)
    print("Building index (this may take a few minutes)...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    
    print(f"Index built successfully and stored in {CHROMA_PATH}")
    # Return the retriever immediately after building
    return index.as_retriever(similarity_top_k=5) 


def get_first_aid_query_engine():
    """
    Retrieves the existing index and creates a query engine (retriever).
    Raises chromadb.errors.NotFoundError if the collection does not exist.
    """
    print(f"Loading existing index from {CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # This line will raise the NotFoundError if the collection doesn't exist
    chroma_collection = db.get_collection(INDEX_NAME) 
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index - it will automatically use the Settings.embed_model for comparison
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    
    # Create a retriever (not a query engine) to pass the raw context to the LLM agent
    return index.as_retriever(similarity_top_k=5)


# --- Dynamic Index Loading/Building (Correction for NotFoundError) ---

def load_or_build_index():
    """Checks for the index and builds it if not found."""
    try:
        # 1. Attempt to load the existing index
        return get_first_aid_query_engine()
    except chroma_errors.NotFoundError:
        # 2. If NotFoundError occurs, build the index
        print("!!! First-Aid Index not found. Building index now... !!!")
        
        # NOTE: The build function now returns the retriever
        retriever = build_first_aid_index()
        
        if retriever:
             print("Index build successful. Proceeding with the retriever.")
        return retriever
    except Exception as e:
        print(f"An unexpected error occurred during index check: {e}")
        return None

# Load the retriever globally. It will build the index if necessary on the first run.
FIRST_AID_RETRIEVER = load_or_build_index()


# --- The Custom Tool for the ADK Agent ---

def first_aid_rag_search(query: str) -> str:
    """
    SEARCHES the First Aid Knowledge Base (indexed books) for detailed, 
    professional first aid procedures. Returns the raw, relevant text chunks.

    Args:
        query (str): The specific injury or condition (e.g., "how to treat a second-degree burn").

    Returns:
        str: A concatenation of relevant, retrieved First Aid text chunks from the books.
    """
    # Check if the retriever was successfully loaded or built
    if FIRST_AID_RETRIEVER is None:
        return "ERROR: First Aid Knowledge Base is not yet indexed or loaded. Please check RAG setup and ensure PDFs are in './data'."
    
    try:
        # Use the retriever to find relevant document nodes/chunks
        nodes = FIRST_AID_RETRIEVER.retrieve(query)
        
        # Combine the text from the top N chunks into one string
        retrieved_context = "\n---\n".join([n.get_text() for n in nodes])
        
        # We add a clear header so the LLM knows what this is
        return f"RETRIEVED KNOWLEDGE FROM FIRST AID MANUALS:\n{retrieved_context}"
    
    except Exception as e:
        return f"RAG SEARCH ERROR: Could not retrieve knowledge. Error: {e}"

# If you run this file directly, it will perform the load/build check and test the search
if __name__ == "__main__":
    if FIRST_AID_RETRIEVER:
        print("\n--- TEST SEARCH ---")
        test_query = "What is the procedure for a compound fracture of the forearm?"
        result = first_aid_rag_search(test_query)
        print(f"Query: {test_query}\nResult:\n{result[:500]}...")
    else:
        print("\nCould not test search because FIRST_AID_RETRIEVER failed to initialize.")