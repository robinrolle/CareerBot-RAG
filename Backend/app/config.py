import re
import os

EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
GENERATION_MODEL_NAME = "llama-3.1-8b-instant"
OPENAI_API_KEY = "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"
GROQ_API_KEY = "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"
# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data','processed_data', 'ESCO_embeddings'))
COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', EMBEDDING_MODEL_NAME)[:63]

print(DATABASE_DIR)