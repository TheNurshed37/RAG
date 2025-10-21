# main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from vector_store import build_vector_store, load_vector_store, SAVE_PATH
from rag import stream_answer_generator

app = FastAPI(title="RAG System API")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # build vector store from the uploaded file (or all files in data/)
        build_vector_store(pdf_path=file_path, save_path=SAVE_PATH)
        return {"message": f"Index built successfully from {file.filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """
    Returns a streaming response of tokens as the model generates them.
    Consumers can stream and print tokens in real time.
    """
    try:
        generator = stream_answer_generator(question, save_path=SAVE_PATH)
        return StreamingResponse(generator, media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# @app.get("/")
# def root():
#     return {"message": "RAG System API running. Use /upload-pdf and /ask endpoints."}
