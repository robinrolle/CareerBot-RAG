import os
import faiss
import json
from .config import EMBEDDING_MODEL_NAME, DATABASE_DIR, COLLECTION_NAME

def load_faiss_index_and_metadata():
    faiss_index_path = os.path.join(DATABASE_DIR, f"{COLLECTION_NAME}_faiss.index")
    metadata_path = os.path.join(DATABASE_DIR, f"{COLLECTION_NAME}_metadata.json")

    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index not found : {COLLECTION_NAME}")

    index = faiss.read_index(faiss_index_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata

faiss_index, faiss_metadata = load_faiss_index_and_metadata()