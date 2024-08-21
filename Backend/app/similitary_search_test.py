import logging
import re
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Setup constants and initialize logging
MODEL_NAME = "intfloat/e5-large-v2"
DATABASE_DIR = "C:\\Users\\Robin\\Desktop\\final_thesis\\CareerBot-RAG\\data\\processed_data\\ESCO_embeddings"
COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', MODEL_NAME)[:63]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_chroma_db():
    """Initialize the Chroma database and load the collection."""
    model = SentenceTransformer(MODEL_NAME)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logger.info("Database loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load the database: {e}")
        raise

    return collection, model

def similarity_search(query_text: str, collection, model, top_k: int = 5, filter_type: str = None) -> List[str]:
    """Perform a similarity search on the collection using the provided query text."""
    try:
        query_embedding = model.encode(query_text).tolist()
        filter_query = {'type': filter_type} if filter_type else {}
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_query
        )
        documents = results['documents'][0] if 'documents' in results else []
        return documents
    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        return []

def similarity_search_skills(query_text: str, collection, model, top_k: int = 5) -> List[str]:
    """Search for skills in the collection."""
    return similarity_search(query_text, collection, model, top_k, filter_type='skill/competence')

def similarity_search_occupations(query_text: str, collection, model, top_k: int = 5) -> List[str]:
    """Search for occupations in the collection."""
    return similarity_search(query_text, collection, model, top_k, filter_type='occupation')

# Initialize the database and model
chroma_collection, sentence_model = init_chroma_db()

# Define query and retrieve results
query = "Digital marketing, Social media management, Market research, Data analysis, Campaign management"

retrieved_skills = similarity_search_skills(query, chroma_collection, sentence_model, top_k=7)
retrieved_occupations = similarity_search_occupations(query, chroma_collection, sentence_model, top_k=2)

# Print the results
print(f"Retrieved skills: {retrieved_skills}")
print(f"Retrieved occupations: {retrieved_occupations}")
