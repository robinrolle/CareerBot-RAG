from pydantic import BaseModel
from typing import List

class UploadResponse(BaseModel):
    filename: str
    path: str

class ProcessResponse(BaseModel):
    selected_skills_ids: List[str]
    selected_occupations_ids: List[str]
    suggested_skills_ids : List[str]
    suggested_occupations_ids : List[str]