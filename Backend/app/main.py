import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ProcessResponse, UploadResponse, SuggestionResponse
from .utils import process_cv, get_similar_documents
from .database import chroma_collection, sentence_model
import aiofiles
from typing import List

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
        return await process_cv(file_path, chroma_collection, sentence_model)
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


@app.post("/suggestions/skills", response_model=SuggestionResponse)
async def suggest_similar_skills(selected_skill_ids: List[str]):
    try:
        # Récupérer les documents correspondants aux IDs fournis
        retrieved_docs = {
            'documents': [],
            'embeddings': [],
            'metadatas': [],
            'ids': []
        }

        for skill_id in selected_skill_ids:
            doc = chroma_collection.get(ids=skill_id, include=['documents', 'embeddings', 'metadatas'])
            if doc:
                retrieved_docs['documents'].append(doc['documents'])
                retrieved_docs['embeddings'].append(doc['embeddings'])
                retrieved_docs['metadatas'].append(doc['metadatas'])
                retrieved_docs['ids'].append(doc['ids'])
            else:
                raise HTTPException(status_code=404, detail=f"Skill ID {skill_id} not found")

        # Si aucun embedding n'est trouvé
        if not retrieved_docs['embeddings']:
            raise HTTPException(status_code=400, detail="No valid skill embeddings found")

        # Rechercher des documents similaires en utilisant get_similar_documents
        similar_docs = get_similar_documents(chroma_collection, retrieved_docs, 'skill/competence', n_suggest=15)

        # Extraire les IDs des documents similaires et aplatir la liste
        similar_ids = set([item for sublist in similar_docs['ids'] for item in sublist])

        # Filtrer les IDs pour exclure ceux déjà sélectionnés
        filtered_similar_ids = similar_ids - set(selected_skill_ids)

        return SuggestionResponse(suggested_ids=list(filtered_similar_ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggestions/occupations", response_model=SuggestionResponse)
async def suggest_similar_occupations(selected_occupation_ids: List[str]):
    try:
        retrieved_docs = {
            'documents': [],
            'embeddings': [],
            'metadatas': [],
            'ids': []
        }

        for occupation_id in selected_occupation_ids:
            doc = chroma_collection.get(ids=occupation_id, include=['documents', 'embeddings', 'metadatas'])
            if doc:
                retrieved_docs['documents'].append(doc['documents'])
                retrieved_docs['embeddings'].append(doc['embeddings'])
                retrieved_docs['metadatas'].append(doc['metadatas'])
                retrieved_docs['ids'].append(doc['ids'])
            else:
                raise HTTPException(status_code=404, detail=f"Occupation ID {occupation_id} not found")

        if not retrieved_docs['embeddings']:
            raise HTTPException(status_code=400, detail="No valid occupation embeddings found")

        similar_docs = get_similar_documents(chroma_collection, retrieved_docs, 'occupation', n_suggest=10)

        similar_ids = set([item for sublist in similar_docs['ids'] for item in sublist])

        filtered_similar_ids = similar_ids - set(selected_occupation_ids)

        return SuggestionResponse(suggested_ids=list(filtered_similar_ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
