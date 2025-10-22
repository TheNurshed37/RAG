import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_hash_index(hash_index_file):
    """Load the stored file hashes (if exists)."""
    if os.path.exists(hash_index_file):
        with open(hash_index_file, "r") as f:
            return json.load(f)
    return {}


def save_hash_index(hash_index, hash_index_file):
    """Save the updated file hashes."""
    with open(hash_index_file, "w") as f:
        json.dump(hash_index, f, indent=4)


def add_to_faiss_index(pdf_path, file_hash, faiss_dir, hash_index_file):
    """Append new PDF embeddings to FAISS (if not already processed)."""
    hash_index = load_hash_index(hash_index_file)
    if file_hash in hash_index:
        return f"Skipped: '{os.path.basename(pdf_path)}' already exists in vector store."

    # Load and split document
    loader = PyPDFLoader(pdf_path, mode="single")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If FAISS exists -> append
    if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
        vector_store = FAISS.load_local(faiss_dir, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embedding_model)

    vector_store.save_local(faiss_dir)

    # Update hash index
    hash_index[file_hash] = os.path.basename(pdf_path)
    save_hash_index(hash_index, hash_index_file)

    return f"Embeddings added successfully for '{os.path.basename(pdf_path)}'."
