# RAG Chatbot – High-Level Architecture & Documentation

## Overview

The RAG Chatbot is a Retrieval-Augmented Generation (RAG) system designed to answer questions based on the contents of uploaded documents (PDF or Excel). It combines a vector database for semantic search with a large language model (LLM) to generate contextual and accurate responses grounded in the user-provided data.

## High Level Flow
![alt text](image-1.png)

## Components

| Component               | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| **Streamlit UI**        | Frontend interface for file upload, chat, and displaying answers       |
| **Document Parsers**    | Extract text from PDFs and Excel files using `PyPDF2` and `pandas`     |
| **Text Splitter**       | Splits long documents into smaller overlapping chunks                  |
| **Embedding Generator** | Uses OpenAI Embeddings API to convert text into vector representations |
| **Vector Store**        | FAISS stores and indexes vectors for fast semantic similarity search   |
| **Retriever**           | Finds top-k relevant text chunks given a query                         |
| **Prompt Constructor**  | Combines user query + retrieved context to build an LLM prompt         |
| **LLM (OpenAI)**        | Answers the query based on the prompt using GPT-3.5 / GPT-4            |
| **Chat Memory**         | Stores previous conversation turns (optional)                          |
---


## How It Works

1. User uploads one or more PDF/Excel documents.
2. The system extracts text from those files and splits them into manageable text chunks.
3. Each chunk is embedded into a high-dimensional vector using a language model.
4. Vectors are stored in FAISS, a fast vector similarity search database.
5. When a user enters a query:
    - It is embedded similarly.
    - FAISS returns the top similar text chunks.
6. The query + relevant chunks are fed into an LLM to generate an accurate answer.
7. The response is displayed in the chat interface.

## Future Improvements

- Add support for DOCX, TXT, and web URLs
- Integrate a local embedding model (e.g., SentenceTransformers)
- Support multiple LLM backends (LLamaCpp, Ollama, Claude)
- Add citation / chunk source highlighting
- Optimize chunking strategy for different document types
- Persist vector store between sessions