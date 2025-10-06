
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
from transformers import pipeline

st.set_page_config(page_title="RAG Chatbot (PDF/Excel)")
st.title("RAG Chatbot: PDF/Excel Q&A")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

uploaded_file = st.file_uploader(
    "Upload a PDF or Excel file", type=["pdf", "xlsx", "xls"]
)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_excel(path):
    df = pd.read_excel(path)
    return df.to_string(index=False)


def build_vectorstore(text, api_key):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    docs = splitter.create_documents([text])
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
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
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            return FAISS.from_documents(docs, embeddings)
        else:
            raise


if uploaded_file:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(tmp_path)
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]:
        text = extract_text_from_excel(tmp_path)
    else:
        text = ""
    os.remove(tmp_path)
    if text:
        st.session_state.vectorstore = build_vectorstore(text, OPENAI_API_KEY)
        st.success("Document indexed for retrieval!")


st.markdown(
    "<h2 style='text-align: center;'>RAG Chatbot</h2>", unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center;'>Upload a PDF or Excel file, then chat below.</div>",
    unsafe_allow_html=True,
)

if st.session_state.vectorstore:
    chat_container = st.container()
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
    user_input = st.text_input("Type your question and press Enter:")
    if user_input:
        retriever = st.session_state.vectorstore.as_retriever()
        try:
            llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(user_input)
        except Exception as e:
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
                    docs = retriever.get_relevant_documents(user_input)
                    if not docs:
                        answer = (
                            "Sorry, I couldn't find relevant information in the document."
                        )
                    else:
                        context = "\n".join(doc.page_content for doc in docs[:3])
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
        st.session_state.chat_history.append((user_input, answer))
else:
    st.info("Please upload a PDF or Excel file to get started.")
