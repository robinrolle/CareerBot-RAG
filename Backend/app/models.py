from pydantic import BaseModel
from typing import List
from langchain_core.pydantic_v1 import Field

class UploadResponse(BaseModel):
    filename: str
    path: str

class GradedItem(BaseModel):
    id: str
    item: str
    relevance: str

class SkillSuggestionResponse(BaseModel):
    suggested_skills_ids: List[str]

class OccupationSuggestionResponse(BaseModel):
    suggested_occupations_ids: List[str]

class SuggestionResponse(BaseModel):
    suggested_skills_ids: List[str]
    suggested_occupations_ids: List[str]

class ProcessResponse(BaseModel):
    graded_skills: List[GradedItem]
    graded_occupations: List[GradedItem]
    suggestions: SuggestionResponse

class ExtractedItem(BaseModel):
    name: str = Field(description="The name of the extracted item")

class ExtractedItems(BaseModel):
    items: list[ExtractedItem] = Field(description="A list of extracted items")