
# RAG Chatbot (PDF/Excel)

This project is a Python-based Retrieval-Augmented Generation (RAG) chatbot. It allows users to upload PDF or Excel files, and then chat with an AI agent that answers questions based on the content of those documents.

## What is RAG?
RAG (Retrieval-Augmented Generation) is an AI approach that combines information retrieval with generative models. It retrieves relevant chunks of data from a knowledge base (here, your uploaded files) and uses a language model to generate answers grounded in that data.

## How this tool works
- You upload a PDF or Excel file.
- The tool extracts and splits the content into chunks.
- Each chunk is embedded and stored for fast vector search (using FAISS).
- When you ask a question, the tool retrieves the most relevant chunks and sends them as context to a language model (OpenAI GPT or a local fallback) to generate an answer.
- The UI is minimalist: you only see the chat window and file upload option.

## Benefits
- Answers are grounded in your own documents, not just general internet knowledge.
- Works with both OpenAI and local models (for cost savings and privacy).
- Handles both PDF and Excel files.
- Simple, chat-like interface for easy use.

## Learning from this project
- How to build a RAG pipeline: chunking, embedding, vector search, and generative answering.
- How to use LangChain, FAISS, HuggingFace, and OpenAI APIs together.
- How to design a fallback system for LLMs and embeddings.
- How to build a clean, user-friendly Streamlit UI for document-based chat.

## This project is a Python-based Retrieval-Augmented Generation (RAG) chatbot. It should:

- Accept PDF and Excel file uploads.
- Extract and chunk content from these files.
- Use vector search to retrieve relevant context.
- Use OpenAI GPT (or similar) to answer user questions based on the retrieved context.
- Use Streamlit for the user interface.

## Setup
1. Ensure you have Python 3.8+
2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key (if using OpenAI):
   ```bash
   export OPENAI_API_KEY=your-openai-key
   ```
5. (Optional) For local fallback, install transformers and required models.
6. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
