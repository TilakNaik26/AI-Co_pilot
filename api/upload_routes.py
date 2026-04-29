from fastapi import APIRouter, UploadFile, File, HTTPException
from services.embeddings import add_document, clear_documents
import shutil
import os

router = APIRouter()

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SUPPORTED_TYPES = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "text/markdown": "md",
}


def extract_text(file_path: str, content_type: str) -> str:
    """Extract plain text from uploaded file."""
    if content_type == "application/pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")
    else:
        with open(file_path, "r", errors="ignore") as f:
            return f.read()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text and embed
    try:
        text = extract_text(file_path, file.content_type or "text/plain")
        num_chunks = add_document(text, source=file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "chunks": num_chunks,
    }


@router.delete("/documents")
def delete_documents():
    clear_documents()
    return {"message": "All documents cleared"}
