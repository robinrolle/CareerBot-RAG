import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME, DATABASE_DIR, COLLECTION_NAME

def init_chroma_db():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print("Database loaded and ready")
    except Exception as e:
        print(f"Failed to load the database: {e}")
        raise

    return collection, model

chroma_collection, sentence_model = init_chroma_db()