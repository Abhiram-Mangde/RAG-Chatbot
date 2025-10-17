import os
import tempfile

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from transformers import pipeline  # For local fallback QA

# Set up Streamlit app UI
st.set_page_config(page_title="RAG Chatbot (PDF/Excel)")
st.title("RAG Chatbot: PDF/Excel Q&A")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# File upload widget - only accepts PDF or Excel
uploaded_file = st.file_uploader(
    "Upload a PDF or Excel file", type=["pdf", "xlsx", "xls"]
)

# Initialize session state to store the vector database and chat history
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to extract text from a PDF file
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Function to extract text from an Excel file
def extract_text_from_excel(path):
    df = pd.read_excel(path)
    return df.to_string(index=False)

# Function to build vector store (FAISS) from input text
def build_vectorstore(text, api_key):
    # Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    docs = splitter.create_documents([text])

    try:
        # Try using OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.from_documents(docs, embeddings)

    except Exception as e:
        # If OpenAI fails (quota/rate limits), fallback to HuggingFace embeddings
        if (
            'insufficient_quota' in str(e)
            or 'RateLimitError' in str(e)
            or 'quota' in str(e).lower()
        ):
            st.warning(
                "OpenAI quota exceeded or invalid key. "
                "Falling back to HuggingFace embeddings (all-MiniLM-L6-v2). "
                "This may be slower but does not require an API key."
            )
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.from_documents(docs, embeddings)
        else:
            raise

# Handle file upload
if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text depending on file type
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(tmp_path)
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]:
        text = extract_text_from_excel(tmp_path)
    else:
        text = ""

    # Clean up temp file
    os.remove(tmp_path)

    # If text was extracted, build the vector store
    if text:
        st.session_state.vectorstore = build_vectorstore(text, OPENAI_API_KEY)
        st.success("Document indexed for retrieval!")

# Display chat header and instruction
st.markdown(
    "<h2 style='text-align: center;'>RAG Chatbot</h2>", unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center;'>Upload a PDF or Excel file, then chat below.</div>",
    unsafe_allow_html=True,
)

# If vector store is ready, display chat interface
if st.session_state.vectorstore:
    chat_container = st.container()

    # Display previous chat history (most recent at top)
    with chat_container:
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(
                f"<div style='background:#f1f1f1;padding:8px;border-radius:8px;"
                f"margin-bottom:4px;'><b>You:</b> {q}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#e8f5e9;padding:8px;border-radius:8px;"
                f"margin-bottom:12px;'><b>Bot:</b> {a}</div>",
                unsafe_allow_html=True,
            )

    # Input field for new user question
    user_input = st.text_input("Type your question and press Enter:")

    if user_input:
        # Use the FAISS retriever to get relevant chunks
        retriever = st.session_state.vectorstore.as_retriever()

        try:
            # Use OpenAI LLM to answer question
            llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(user_input)

        except Exception as e:
            # If OpenAI fails, fallback to HuggingFace local QA model
            if (
                'insufficient_quota' in str(e)
                or 'RateLimitError' in str(e)
                or 'quota' in str(e).lower()
            ):
                st.warning(
                    "OpenAI LLM quota exceeded or invalid key. "
                    "Falling back to local HuggingFace pipeline (distilbert-base-uncased). "
                    "This does not require an API key or token, but answers may be less accurate."
                )
                try:
                    # Retrieve top relevant documents
                    docs = retriever.get_relevant_documents(user_input)
                    if not docs:
                        answer = "Sorry, I couldn't find relevant information in the document."
                    else:
                        # Build context from retrieved docs
                        context = "\n".join(doc.page_content for doc in docs[:3])
                        # Use local QA pipeline
                        qa_pipe = pipeline(
                            "question-answering",
                            model="distilbert-base-cased-distilled-squad",
                        )
                        result = qa_pipe(question=user_input, context=context)
                        answer = result["answer"]
                except Exception as e2:
                    st.error(f"Local HuggingFace QA pipeline failed: {e2}")
                    answer = "No answer: Local HuggingFace QA pipeline failed."
            else:
                raise

        # Save user question and bot answer to chat history
        st.session_state.chat_history.append((user_input, answer))

# If no file is uploaded yet, show message
else:
    st.info("Please upload a PDF or Excel file to get started.")
