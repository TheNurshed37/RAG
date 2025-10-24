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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


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

def validate_and_format_cv_structure(raw_text):
    """
    Uses LLM to validate and format CV structure with proper headers.
    Includes comprehensive error handling with fallbacks.
    """
    STANDARD_CV_HEADERS = [
        "Personal Information",
        "Professional Summary",
        "Work Experience", 
        "Education",
        "Skills",
        "Projects",
        "Certifications",
        "Languages",
        "References"
    ]
    
    prompt_template = """
    You are a CV/Resume formatting expert. Your task is to analyze the extracted CV text and ensure it is properly structured with appropriate headers.

    STANDARD CV HEADERS:
    {standard_headers}

    EXTRACTED CV TEXT:
    {raw_text}

    INSTRUCTIONS:
    1. Analyze the existing structure and headers in the text
    2. If the text is already well-structured with clear headers that match the standard, return it as-is
    3. If headers are missing, poorly formatted, or non-standard:
       - Reorganize the content under appropriate standard headers
       - Preserve ALL original information
       - Add missing headers if content exists for them
       - Use markdown formatting with ## for headers
    4. Ensure the structured document flows logically

    Return ONLY the properly formatted CV text with appropriate headers.
    """
    
    try:
        print("🔧 Validating CV structure with LLM...")
        
        # Initialize LLM for structuring
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for consistent formatting
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["raw_text", "standard_headers"]
        )
        
        formatted_prompt = prompt.invoke({
            "raw_text": raw_text,
            "standard_headers": ", ".join(STANDARD_CV_HEADERS)
        })
        
        response = llm.invoke(formatted_prompt)
        print("✅ CV structure validation completed")
        return response.content
        
    except Exception as e:
        print(f"❌ LLM structuring failed: {e}")
        print("🔄 Using basic header formatting as fallback...")
        return add_basic_headers_fallback(raw_text)

def add_basic_headers_fallback(raw_text):
    """
    Basic fallback method that adds simple headers without LLM.
    This ensures the system never breaks completely.
    """
    # Simple keyword-based header detection
    lines = raw_text.split('\n')
    structured_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Basic header detection based on common patterns
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in ['experience', 'work', 'job', 'employment']):
            if not line.startswith('## '):
                structured_lines.append('## Work Experience')
        elif any(keyword in lower_line for keyword in ['education', 'degree', 'university', 'college']):
            if not line.startswith('## '):
                structured_lines.append('## Education')
        elif any(keyword in lower_line for keyword in ['skill', 'technical', 'programming']):
            if not line.startswith('## '):
                structured_lines.append('## Skills')
        elif any(keyword in lower_line for keyword in ['project', 'portfolio']):
            if not line.startswith('## '):
                structured_lines.append('## Projects')
        elif any(keyword in lower_line for keyword in ['certificat', 'license']):
            if not line.startswith('## '):
                structured_lines.append('## Certifications')
        
        structured_lines.append(line)
    
    return '\n'.join(structured_lines)

def convert_pdf_with_docling(pdf_path):
    """Convert PDF to text using Docling and return LangChain Documents."""
    
    # Initialize Docling converter
    doc_converter = initialize_docling_converter()
    
    # Convert the document
    result = doc_converter.convert(pdf_path)
    
    # Extract text content
    raw_text = result.document.export_to_markdown()
    
    print(f"📊 Docling extracted {len(raw_text)} characters from {pdf_path}")
    
    # Enhanced: Validate and structure the CV content
    structured_text = validate_and_format_cv_structure(raw_text)
    
    print(f"📝 After structuring: {len(structured_text)} characters")
    
    # Create a single LangChain Document with the structured text
    doc = Document(
        page_content=structured_text,
        metadata={
            "source": pdf_path,
            "page": 1,
            "processed": "structured"
        }
    )
    
    print(f"✅ Created 1 structured document")
    
    return [doc]

def add_to_faiss_index(pdf_path, file_hash, faiss_dir, hash_index_file):
    """Append new PDF embeddings to FAISS (if not already processed) using enhanced CV processing."""
    hash_index = load_hash_index(hash_index_file)
    if file_hash in hash_index:
        return f"Skipped: '{os.path.basename(pdf_path)}' already exists in vector store."

    # Load and process document using enhanced CV processing
    print(f"🔄 Processing CV with enhanced pipeline: {pdf_path}")
    docs = convert_pdf_with_docling(pdf_path)

    # Use optimized chunking for structured CVs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Slightly larger to preserve header context
        chunk_overlap=80,
        separators=["\n## ", "\n# ", "\n\n", "\n", " "]  # Better for header-based splitting
    )
    chunks = splitter.split_documents(docs)
    
    print(f"📊 Split into {len(chunks)} chunks")

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

    return f"Embeddings added successfully for '{os.path.basename(pdf_path)}' (with enhanced structuring)."