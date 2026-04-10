Project: Local RAG (Retrieval-Augmented Generation) Engine
Overview
A fully local, privacy-focused RAG system designed for document-based Q&A. This implementation bypasses cloud dependencies by orchestrating on-device LLMs and vector embeddings using Ollama and LangChain.

Technical Stack
Large Language Model (LLM): llama3 (via Ollama)

Embedding Model: llama3 - Llama 3 is a high-parameter Large Language Model designed for complex reasoning and 
text generation. While highly capable, using it for embeddings is resource-intensive on local hardware like the
 M3, as it requires the full model to remain active for simple vector calculations. It is best suited for the
final stage of the RAG pipeline: synthesizing the retrieved data into a natural response.

Nomic-Embed-Text - Nomic is a dedicated "encoder" model purpose-built for semantic search and retrieval.
At a fraction of the size (137M parameters vs. 8B), it is significantly faster and more memory-efficient.
It features a specialized 8,192-token context window, allowing it to index larger blocks of text with higher
precision than general-purpose LLMs.

Vector Database: 
Data Storage & Retrieval Architectures
This project demonstrates two distinct methods for managing vector data on Apple Silicon, each optimized for 
different stages of the development lifecycle.

1. Volatile Memory (InMemoryVectorStore)
The initial implementation utilizes an in-memory orchestration strategy where document embeddings are stored 
directly in active RAM.

Mechanism: Vectors are generated and held in a temporary heap, allowing for high-speed similarity searches
 without disk I/O overhead.

Use Case: Ideal for rapid prototyping and iterative testing where data persistence is not required and the document set is small enough to fit within the system's unified memory.

2. Persistent Storage (ChromaDB)
The advanced implementation incorporates a disk-backed vector database using ChromaDB to provide stateful persistence.

Mechanism: Document embeddings are serialized and stored in a local SQLite-backed directory (./chroma_db). The system performs a check upon initialization: if an existing database is detected, it bypasses the computationally expensive embedding phase and loads the vectors directly from disk.

Use Case: Designed for production-like environments where "cold-start" latency must be minimized. This approach is highly efficient on M3 hardware as it offloads the need to re-process documents during every session, preserving both CPU cycles and battery life.
Hardware Acceleration: Metal/GPU-accelerated inference on MacBook Air M3

Implementation Details
Data Processing: Utilizes RecursiveCharacterTextSplitter with a chunk_size of 500 and chunk_overlap of 50 to maintain semantic coherence of specialized terms (e.g., specific hut features).

Architecture: Implements a RetrievalQA "stuff" chain, retrieving the top k=2 most relevant document chunks to inject into the LLM prompt.

Memory Management: Utilizes an in-memory vector store to eliminate I/O overhead and potential SQLite file-locking issues common in containerized or local macOS environments.

How to Run
Start Ollama: Ensure the Ollama service is running (ollama serve).

Download Weights: ollama pull llama3

Setup Environment:

Bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain-ollama langchain-core
Execute: Run the Python notebook or script to initialize the engine and query data.txt.