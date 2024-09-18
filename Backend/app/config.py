import re
import os

# Define your API key here
OPENAI_API_KEY = "sk-hvoqGf1qkfbzp-mQVWyUBXgrtgMLosbyW69AOWYv-ZT3BlbkFJr0OJnLyCSUpNfxXWcGKZP_WlxN8CQpI35KLJQUtKYA"

# Model's choice Open AI only
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'FAISS_index'))
COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', EMBEDDING_MODEL_NAME)[:63]

NUMBER_DOC_PER_ITEM = 1  # Number of document retrieved for each item from the vectorial database
LLM_MAX_PICKS = [15, 5]  # for [skills, occupations]
NB_SUGGESTED_SKILLS = 20  # Number of skills suggested to the user
NB_SUGGESTED_OCCUPATIONS = 10 # Number of occupations suggested to the user
