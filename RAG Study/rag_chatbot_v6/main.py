from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import os
import shutil
import hashlib

from vector_store import add_to_faiss_index
from rag import answer_question
#from rag import answer_question_stream

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Chatbot API", version="4.0")

#Enable CORS properly for FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
FAISS_DIR = "faiss_index"
HASH_INDEX_FILE = "hash_index.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)


def compute_pdf_hash(file_path: str) -> str:
    """Compute a SHA256 hash of the PDF content."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@app.post("/upload-pdf")
async def upload_only_pdf(file: UploadFile = File(...)):
    """Upload a PDF, append embeddings, and remove file after indexing."""
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save temporarily
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Compute file hash
        file_hash = compute_pdf_hash(file_path)

        # Build/append embeddings
        msg = add_to_faiss_index(file_path, file_hash, FAISS_DIR, HASH_INDEX_FILE)

        # Remove the uploaded file after processing
        os.remove(file_path)

        return {"message": msg}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question based on uploaded documents."""
    try:
        if not os.path.exists(FAISS_DIR) or not os.listdir(FAISS_DIR):
            return {"question": question, "answer": "I don’t know (No document found)."}

        answer = answer_question(question)

        if not answer or answer.strip() == "":
            return {"question": question, "answer": "I don’t know."}

        return {"question": question, "answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/reset")
async def reset_vector_store():
    """Completely reset FAISS and hash store."""
    try:
        # Delete FAISS and hash index
        if os.path.exists(FAISS_DIR):
            shutil.rmtree(FAISS_DIR)
            os.makedirs(FAISS_DIR, exist_ok=True)

        if os.path.exists(HASH_INDEX_FILE):
            os.remove(HASH_INDEX_FILE)

        return {"message": "Vector store and hash index reset successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Reset failed: {str(e)}"})