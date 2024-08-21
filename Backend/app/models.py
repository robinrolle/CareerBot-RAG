from pydantic import BaseModel
from typing import List

class UploadResponse(BaseModel):
    filename: str
    path: str

class ProcessResponse(BaseModel):
    skills_ids: List[str]
    occupations_ids: List[str]