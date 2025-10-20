# vector_store.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
SAVE_PATH = "faiss_index"

def build_faiss_index(data_dir=DATA_DIR, save_path=SAVE_PATH):
    """Loads PDFs, splits them, embeds them, and saves FAISS index."""
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            print(f" Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path, mode="single")
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(save_path)

    print(f" Vector store saved successfully at: {save_path}")
