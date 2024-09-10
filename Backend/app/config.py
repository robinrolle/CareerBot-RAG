import re
import os

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = "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"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', EMBEDDING_MODEL_NAME)[:63]

VECTORSTORE_MAX_RETRIEVED = 1
LLM_MAX_PICKS = [15, 5]  # for [skills, occupations]

NB_SUGGESTED_SKILLS = 20
NB_SUGGESTED_OCCUPATIONS = 10
