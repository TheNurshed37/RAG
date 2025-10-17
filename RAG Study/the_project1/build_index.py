# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
#Config
# -------------------------------
DATA_DIR = "data"
SAVE_PATH = "faiss_index"

# -------------------------------
#Load all PDFs
# -------------------------------
def load_all_pdfs(data_dir):
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            print(f" Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path, mode="single")
            docs = loader.load()
            all_docs.extend(docs)
    print(f" Total documents loaded: {len(all_docs)}")
    return all_docs

documents = load_all_pdfs(DATA_DIR)

# -------------------------------
# Split into chunks
# -------------------------------
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

print(" Splitting into chunks...")
chunks = split_documents(documents)
print(f" Total chunks created: {len(chunks)}")

# -------------------------------
# Generate embeddings
# -------------------------------
print(" Generating embeddings using all-MiniLM-L6-v2...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Create FAISS store
# -------------------------------
print(" Creating FAISS vector store...")
vector_store = FAISS.from_documents(chunks, embedding_model)

# -------------------------------
# Save FAISS index
# -------------------------------
vector_store.save_local(SAVE_PATH)
print(f" Vector store saved successfully at: {SAVE_PATH}")
