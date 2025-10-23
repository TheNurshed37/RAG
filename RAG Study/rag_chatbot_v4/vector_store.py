import os
import json
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.datamodel.settings import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


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

def initialize_docling_converter():
    """Initialize and return the Docling DocumentConverter with minimal options for maximum speed."""
    
    pipeline_options = PdfPipelineOptions(
        do_table_structure=False,
        do_ocr=False, 
        generate_page_images=False,
        generate_picture_images=False,
    )

    doc_converter = DocumentConverter()
    
    return doc_converter

def convert_pdf_with_docling(pdf_path):
    """Convert PDF to text using Docling and return LangChain Documents."""
    
    # Initialize Docling converter
    doc_converter = initialize_docling_converter()
    
    # Convert the document
    result = doc_converter.convert(pdf_path)
    
    # Extract text content
    text_content = result.document.export_to_markdown()
    
    print(f"📊 Docling extracted {len(text_content)} characters from {pdf_path}")
    
    # Create a single LangChain Document with the full text
    doc = Document(
        page_content=text_content,
        metadata={
            "source": pdf_path,
            "page": 1
        }
    )
    
    print(f"✅ Created 1 document with {len(text_content)} characters")
    
    return [doc]

def add_to_faiss_index(pdf_path, file_hash, faiss_dir, hash_index_file):
    """Append new PDF embeddings to FAISS (if not already processed) using Docling for parsing."""
    hash_index = load_hash_index(hash_index_file)
    if file_hash in hash_index:
        return f"Skipped: '{os.path.basename(pdf_path)}' already exists in vector store."

    # Load and process document using Docling instead of PyPDFLoader
    docs = convert_pdf_with_docling(pdf_path)

    # The rest of the logic remains exactly the same as your base code
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
