import streamlit as st
from pathlib import Path
import tempfile

from rag_pipeline import build_index, answer_question, index_exists

st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document Intelligence RAG Assistant")
st.caption("Ask questions about your technical PDFs and get grounded answers with source references.")

DATA_DIR = Path("data")
INDEX_DIR = Path("vectorstore")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("1. Upload documents")
    uploads = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one PDF first.")
        else:
            for old_pdf in DATA_DIR.glob("*.pdf"):
                old_pdf.unlink()

            for uploaded in uploads:
                (DATA_DIR / uploaded.name).write_bytes(uploaded.getbuffer())

            with st.spinner("Extracting, chunking, embedding and indexing documents..."):
                count = build_index(DATA_DIR, INDEX_DIR)

            st.success(f"Knowledge base created with {count} chunks.")

    st.divider()
    st.header("2. Retrieval settings")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

    if index_exists(INDEX_DIR):
        st.success("Vector index: Ready")
    else:
        st.info("Vector index: Not built")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['file']}**, page {source['page']} "
                        f"(similarity: {source['score']:.3f})"
                    )

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if not index_exists(INDEX_DIR):
        st.error("Please upload PDFs and build the knowledge base first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = answer_question(question, INDEX_DIR, top_k=top_k)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.markdown(
                        f"- **{source['file']}**, page {source['page']} "
                        f"(similarity: {source['score']:.3f})"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
