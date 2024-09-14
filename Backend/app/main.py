import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ProcessResponse, UploadResponse, OccupationSuggestionResponse, SkillSuggestionResponse
from .utils import process_cv, get_similar_documents_faiss
from .config import NB_SUGGESTED_SKILLS, NB_SUGGESTED_OCCUPATIONS
from typing import List
import aiofiles

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename: str) -> str:
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4()}.{ext}"
    return unique_name

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        return UploadResponse(filename=unique_filename, path=file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessResponse)
async def process_cv_endpoint(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        return await process_cv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        return {"message": f"File {filename} successfully deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggestions/skills", response_model=SkillSuggestionResponse)
async def suggest_similar_skills(selected_skill_ids: List[str]):
    try:
        if not selected_skill_ids:
            raise HTTPException(status_code=400, detail="No skill IDs provided")

        similar_skills = get_similar_documents_faiss(selected_skill_ids, 'skill/competence', NB_SUGGESTED_SKILLS)
        similar_skills_ids = [doc['id'] for doc in similar_skills]

        return SkillSuggestionResponse(suggested_skills_ids=similar_skills_ids)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggestions/occupations", response_model=OccupationSuggestionResponse)
async def suggest_similar_occupations(selected_occupation_ids: List[str]):
    try:
        if not selected_occupation_ids:
            raise HTTPException(status_code=400, detail="No occupation IDs provided")

        similar_occupations = get_similar_documents_faiss(selected_occupation_ids, 'occupation', NB_SUGGESTED_OCCUPATIONS)
        similar_occupations_ids = [doc['id'] for doc in similar_occupations]

        return OccupationSuggestionResponse(suggested_occupations_ids=similar_occupations_ids)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)