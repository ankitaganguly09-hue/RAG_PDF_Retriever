import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="RAG PDF Retriever",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Retriever")
st.markdown("Upload a PDF and ask questions about its content.")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()


embeddings = load_embeddings()


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Creating embeddings and vector database..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)

        st.success("PDF processed successfully!")

if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.7
    )

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Get Answer") and question:

        with st.spinner("Searching and generating answer..."):

            relevant_chunks = retriever.invoke(question)

            context = "\n".join(
                [chunk.page_content for chunk in relevant_chunks]
            )

            formatted_prompt = prompt_template.format(
                context=context,
                question=question
            )

            response = llm.invoke(formatted_prompt)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": response.content
                }
            )

    if st.session_state.chat_history:
        st.subheader("Conversation")

        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f"**🧑 Question:** {chat['question']}"
            )
            st.markdown(
                f"**🤖 Answer:** {chat['answer']}"
            )
            st.divider()

