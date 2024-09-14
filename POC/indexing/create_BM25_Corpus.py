import os
import pandas as pd
from rank_bm25 import BM25Okapi
import pickle
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'ESCO dataset - v1.1.1 - classification - en - csv'))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_BM25'))
BATCH_SIZE = 5000  # Adjust as needed

def tokenize(text):
    # Simple tokenization by splitting on whitespace
    return text.lower().split()

# Setup DB
os.makedirs(DATABASE_DIR, exist_ok=True)
bm25_path = os.path.join(DATABASE_DIR, 'bm25.pickle')
documents_path = os.path.join(DATABASE_DIR, 'documents.pickle')
metadatas_path = os.path.join(DATABASE_DIR, 'metadatas.pickle')

try:
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    with open(documents_path, 'rb') as f:
        documents = pickle.load(f)
    with open(metadatas_path, 'rb') as f:
        metadatas = pickle.load(f)
    logging.info("Loaded existing BM25 index and documents")
except FileNotFoundError:
    logging.warning("Creating a new BM25 index, this might take a while!")

    # Load and filter skills data
    logging.info("Loading and filtering skills data...")
    skills_data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, 'skills_en.csv'))
    skills_data = skills_data[skills_data.skillType == 'skill/competence']
    skills_data = skills_data[(skills_data.description.str.len() > 30) & (skills_data.preferredLabel.str.len() > 5)]
    assert len(skills_data) > 100, 'Failed to read!'
    skills = [f"{row['preferredLabel']} | {row['description']}" for _, row in skills_data.iterrows()]
    metadatas1 = [{'altLabels': row['altLabels'], 'modifiedDate': row['modifiedDate'], 'type': 'skill/competence', 'url': row['conceptUri']} for _, row in skills_data.iterrows()]

    # Load and filter occupations data
    logging.info("Loading and filtering occupations data...")
    occupations_data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, 'occupations_en.csv'))
    occupations_data = occupations_data[occupations_data.conceptType == 'Occupation']
    occupations_data = occupations_data[(occupations_data.description.str.len() > 30) & (occupations_data.preferredLabel.str.len() > 2)]
    assert len(occupations_data) > 10, 'Failed to read!'
    occupations = [f"{row['preferredLabel']} | {row['description']}" for _, row in occupations_data.iterrows()]
    metadatas2 = [{'modifiedDate': row['modifiedDate'], 'type': 'occupation', 'url': row['conceptUri']} for _, row in occupations_data.iterrows()]

    # Combine skills and occupations
    documents = skills + occupations
    metadatas = metadatas1 + metadatas2
    assert len(documents) == len(metadatas), 'Data size mismatch!'

    logging.info(f"Total number of documents to be indexed: {len(documents)}")

    # Tokenize documents
    tokenized_corpus = [tokenize(doc) for doc in tqdm(documents, desc="Tokenizing documents")]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index and documents
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    with open(documents_path, 'wb') as f:
        pickle.dump(documents, f)
    with open(metadatas_path, 'wb') as f:
        pickle.dump(metadatas, f)

    logging.info('BM25 index created and saved!')

logging.info('All done!')