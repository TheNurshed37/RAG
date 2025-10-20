# vector_store.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
SAVE_PATH = "faiss_index"

def load_all_pdfs(data_dir: str):
    """Load all PDF files from a directory."""
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            print(f"Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path, mode="single")
            docs = loader.load()
            all_docs.extend(docs)
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def split_documents(documents: list[Document]):
    """Split documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def build_vector_store():
    """Build FAISS index from all PDFs in the data folder."""
    documents = load_all_pdfs(DATA_DIR)
    print("Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("Generating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    vector_store.save_local(SAVE_PATH)
    print(f"Vector store saved successfully at: {SAVE_PATH}")

def load_vector_store():
    """Load an existing FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Loading existing FAISS vector store...")
    vector_store = FAISS.load_local(
        SAVE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully.")
    return vector_store
