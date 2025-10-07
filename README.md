# AfiyaPal Multi-Tool RAG Agent

AfiyaPal is a specialized multi-tool AI assistant designed to provide professional health guidance, including empathetic mental health support and grounded, detailed first aid instructions derived from authoritative medical manuals.

This document details the architecture, tools, and the challenges encountered during the setup and successful execution of the Retrieval-Augmented Generation (RAG) system.

---

## 1. Project Architecture

The AfiyaPal assistant uses a hierarchical multi-agent structure built with the Google Agent Development Kit (ADK).

| Agent Name                          | Role                            | Core Functionality                                                                                                                                              |
| :---------------------------------- | :------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HealthCoordinatorAgent (Root)**   | Triage and Coordination         | Primary interface. Determines if the query is a **Mental Health** issue or a **First Aid/Injury** issue. **Delegates** First Aid tasks to the expert sub-agent. |
| **FirstAidExpertAgent (Sub-Agent)** | Knowledge Retrieval & Synthesis | Provides grounded, step-by-step first aid procedures by utilizing the specialized RAG knowledge base.                                                           |

## 2. Tools and Functionality

The core intelligence of the agent is powered by two main types of tools: an **Agent Tool** for delegation and a **Custom RAG Tool** for knowledge retrieval.

| Tool Name                    | Type            | Underlying Technology                        | Functionality                                                                                                                                                  |
| :--------------------------- | :-------------- | :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FirstAidExpertAgent Tool** | ADK Agent Tool  | ADK / Gemini 2.5 Flash                       | The root agent (Coordinator) uses this to **delegate** all injury or first aid queries to the specialized sub-agent.                                           |
| **`first_aid_rag_search`**   | Custom ADK Tool | LlamaIndex, ChromaDB, HuggingFace Embeddings | **Retrieves** authoritative, context-specific text from the indexed PDF manuals. This grounds the sub-agent's response, ensuring accuracy and professionalism. |

### Detailed RAG Tool Functionality

The RAG system (`first_aid_rag_search`) is critical for ensuring the first aid advice is accurate and sourced from the medical manuals provided.

| RAG Component         | Technology                          | Function                                                                                                                                    |
| :-------------------- | :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **Document Loader**   | `llama_index.SimpleDirectoryReader` | Loads the raw PDF files (Red Cross, St. John, NOLS) from the `./data` directory.                                                            |
| **Chunking/Indexing** | `llama_index.VectorStoreIndex`      | Splits the documents into small, manageable pieces (chunks) and converts each chunk into a vector (embedding).                              |
| **Embedding Model**   | `BAAI/bge-small-en-v1.5` (Local)    | The neural network used to create the numerical representations (vectors) of the text chunks. **Crucial for search accuracy.**              |
| **Vector Database**   | `ChromaDB` (Persistent)             | Stores the generated vectors and the corresponding text chunks, making the knowledge searchable.                                            |
| **Retriever**         | `index.as_retriever()`              | Takes the user's query, converts it to a vector, searches the ChromaDB for the top 5 most similar vectors, and returns the raw text chunks. |

---

## 3. Challenges and Solutions

Setting up the RAG pipeline required addressing critical issues related to library dependency, configuration, and module resolution within the ADK environment.

| Problem Encountered                | Traceback Error                                                               | Solution Taken                                                                                                                                                                                                    | Outcome                                                                      |
| :--------------------------------- | :---------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **LlamaIndex Import Error**        | `ModuleNotFoundError: No module named 'llama_index.vector_stores'`            | The LlamaIndex integration for ChromaDB was installed via a specific package: `pip install llama-index-vector-stores-chroma`.                                                                                     | **Resolved**                                                                 |
| **Index Not Found on Startup**     | `chromadb.errors.NotFoundError: Collection [first_aid_index] does not exists` | Modified the `rag_tool.py` logic to include a `load_or_build_index()` function, which automatically triggers a build if the index is missing.                                                                     | **Resolved**                                                                 |
| **Missing API Key for Embeddings** | `ValueError: No API key found for OpenAI.`                                    | Configured LlamaIndex to use a local, open-source embedding model (`BAAI/bge-small-en-v1.5`) via `Settings.embed_model`.                                                                                          | **Resolved**                                                                 |
| **Agent Module Resolution**        | `Error: Invalid value for '[AGENTS_DIR]': Directory '...' does not exist.`    | Corrected the ADK startup command to pass the valid module directory: `adk web afiyapal_multi_tool_agent`.                                                                                                        | **Resolved**                                                                 |
| **RAG Tool Failure on Call**       | Agent returns general knowledge and apology (did not retrieve context).       | **1. Fixed Relative Import:** Changed `from rag_tool import...` to `from .rag_tool import...` in `agent.py`. **2. Enforced Index Rebuild:** Deleted the corrupted `chroma_db` directory and re-ran `rag_tool.py`. | **Resolved.** Agent successfully retrieves and synthesizes grounded answers. |

---

## 4. How to Run the Agent

### Prerequisites

- Ensure all authoritative PDF manuals are placed in the `./data` directory.
- Ensure you are operating within an activated Python virtual environment (`.venv`).

### Installation and Setup

1.  **Install Dependencies:**

    ```bash
    pip install google-adk pypdf chromadb sentence-transformers llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface
    ```

2.  **Build RAG Index (Crucial Step):**
    Run the RAG tool script. This will automatically download the local embedding model and create the persistent vector database (`./chroma_db`).
    ```bash
    python afiyapal_multi_tool_agent/rag_tool.py
    ```

### Starting the Agent

3.  **Start the Agent Web UI:**
    To ensure the ADK only loads the correct agent module and avoids loading the `chromadb` or `data` folders as separate agents, specify the correct directory path.
    ```bash
    adk web afiyapal_multi_tool_agent
    ```

### Testing

- **Test Delegation:** Use a First Aid query (e.g., "What is the procedure for a compound fracture of the forearm?") to observe the agent successfully calling the RAG tool.
- **Test Triage:** Use a Mental Health query (e.g., "I feel overwhelmed and stressed") to observe the agent responding with general, empathetic guidance without delegating.
