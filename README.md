

# RAG Chatbot (PDF/Excel)

RAG Chatbot is a Python-based Retrieval-Augmented Generation (RAG) application that enables users to upload PDF or Excel files and interact with an AI assistant to answer questions based on the content of those documents.

---

## 🚀 Features

- **Document Upload:** Supports PDF and Excel file uploads.
- **Content Extraction:** Extracts and chunks content for efficient retrieval.
- **Vector Search:** Uses FAISS for fast, relevant context retrieval.
- **AI-Powered Answers:** Utilizes OpenAI GPT (or local models) to generate answers grounded in your documents.
- **Streamlit UI:** Clean, minimalist chat interface for seamless user experience.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with generative AI. It retrieves relevant data chunks from your uploaded files and uses a language model to generate answers based on that data, ensuring responses are accurate and context-aware.

---

## 🛠️ How It Works

1. **Upload** a PDF or Excel file.
2. **Extract & Chunk:** The app extracts text and splits it into manageable chunks.
3. **Embed & Store:** Each chunk is embedded and stored for vector search.
4. **Ask Questions:** When you ask a question, the app retrieves the most relevant chunks and sends them to a language model to generate an answer.
5. **Chat UI:** Interact with the AI in a simple chat window.

---

## 🏆 Benefits

- Answers are grounded in your own documents.
- Supports both OpenAI and local models (for privacy and cost savings).
- Handles both PDF and Excel formats.
- User-friendly, chat-like interface.

---

## 📚 Technologies Used

- **Python 3.8+**
- **Streamlit** (UI)
- **FAISS** (Vector Search)
- **LangChain** (RAG pipeline)
- **OpenAI API** (LLM)
- **HuggingFace Transformers** (Local fallback)
- **PyPDF2, pandas** (File parsing)

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhiram-Mangde/RAG-Chatbot.git
   cd RAG-Chatbot
   ```
2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key (if using OpenAI):**
   ```bash
   export OPENAI_API_KEY=your-openai-key
   ```
5. **(Optional) For local fallback, install transformers and required models.**
6. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Testing the CI Pipeline

This line is a test change to trigger the CI pipeline: **CI Pipeline Test - 2025-10-06**

---

## 📄 License

This project is licensed under the MIT License.
