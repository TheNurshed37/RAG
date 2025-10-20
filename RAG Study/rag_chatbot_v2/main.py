# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os

from vector_store import build_faiss_index
from rag import answer_question

app = FastAPI(title="RAG Chatbot API", version="1.0")

# Directory for uploads
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and build a FAISS vector index."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        build_faiss_index(UPLOAD_DIR)  # Process all PDFs into FAISS
        return {"message": f"Index built successfully from {file.filename}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question based on the indexed documents."""
    try:
        answer = answer_question(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# @app.get("/")
# def root():
#     return {"message": "RAG System API is running. Use /upload-pdf or /ask."}
