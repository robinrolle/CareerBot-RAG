import os
import faiss
import json
from .config import DATABASE_DIR, COLLECTION_NAME

def load_faiss_index_and_metadata(index_type: str):
    """Load FAISS index and metadata for a given index type ('skills' or 'occupations')"""
    faiss_index_path = os.path.join(DATABASE_DIR, f"{COLLECTION_NAME}_{index_type}_faiss.index")
    metadata_path = os.path.join(DATABASE_DIR, f"{COLLECTION_NAME}_{index_type}_metadata.json")

    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index not found for {index_type}: {COLLECTION_NAME}")

    index = faiss.read_index(faiss_index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

# Load both FAISS indices and their metadata
faiss_index_skills, faiss_metadata_skills = load_faiss_index_and_metadata("skills")
faiss_index_occupations, faiss_metadata_occupations = load_faiss_index_and_metadata("occupations")
