# vector_store.py
import os
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
SAVE_PATH = "faiss_index"


def _load_pdfs_from_path(pdf_path: Optional[str]) -> List[Document]:
    docs = []
    if pdf_path:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        loader = PyPDFLoader(pdf_path, mode="single")
        docs.extend(loader.load())
    else:
        if not os.path.isdir(DATA_DIR):
            return []
        for fname in os.listdir(DATA_DIR):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(DATA_DIR, fname)
                loader = PyPDFLoader(path, mode="single")
                docs.extend(loader.load())
    return docs


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def build_vector_store(pdf_path: Optional[str] = None, save_path: str = SAVE_PATH) -> str:
    """
    Build FAISS index from a single PDF or from all PDFs in DATA_DIR.
    Returns the path where the index was saved.
    """
    documents = _load_pdfs_from_path(pdf_path)
    if not documents:
        raise RuntimeError("No documents found to index.")

    chunks = split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    return save_path


def load_vector_store(save_path: str = SAVE_PATH):
    """
    Load and return a FAISS vector store from save_path.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Vector store not found at {save_path}")
    vector_store = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store
