import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a technical document assistant.

Answer the user's question using ONLY the supplied document context.
Do not invent facts that are not supported by the context.
If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded documents."

Give a concise, useful answer.
When useful, use bullet points or numbered steps.

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)


def get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the environment.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the environment.")
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.1,
    )


def load_pdfs(data_dir: Path):
    documents = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["page_number"] = int(doc.metadata.get("page", 0)) + 1

        documents.extend(docs)

    if not documents:
        raise ValueError("No PDF documents found.")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(data_dir: Path, index_dir: Path) -> int:
    documents = load_pdfs(data_dir)
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))

    return len(chunks)


def index_exists(index_dir: Path) -> bool:
    return (
        (index_dir / "index.faiss").exists()
        and (index_dir / "index.pkl").exists()
    )


def retrieve(question: str, index_dir: Path, top_k: int = 4):
    embeddings = get_embeddings()

    # The index is generated locally by this application, so it is trusted.
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.similarity_search_with_score(question, k=top_k)


def answer_question(question: str, index_dir: Path, top_k: int = 4) -> Dict:
    retrieved = retrieve(question, index_dir, top_k)

    context_blocks: List[str] = []
    sources = []

    for doc, score in retrieved:
        context_blocks.append(
            f"[Source: {doc.metadata.get('source_file', 'unknown')}, "
            f"page {doc.metadata.get('page_number', '?')}]\n"
            f"{doc.page_content}"
        )

        sources.append(
            {
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page_number", "?"),
                "score": float(score),
            }
        )

    context = "\n\n---\n\n".join(context_blocks)

    llm = get_llm()
    chain = PROMPT | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": response.content,
        "sources": sources,
    }
