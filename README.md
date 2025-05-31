# RAG Chatbot (PDF/Excel)

This project is a Python-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF or Excel files, extracts their content, and answers questions based on the uploaded documents using OpenAI GPT and vector search.

## Features
- Upload PDF and Excel files
- Extract and chunk content from files
- Vector search for relevant context
- GPT-powered answers based on document content
- Streamlit web interface

## Setup
1. Ensure you have Python 3.8+
2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies (to be listed in `requirements.txt`)
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## To Do
- Implement PDF/Excel parsing
- Add vector search (e.g., FAISS, ChromaDB)
- Integrate OpenAI GPT
- Build Streamlit UI
