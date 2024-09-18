import os
import json
import pandas as pd
import re
import logging
import numpy as np
import faiss
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_valid_collection_name(model_name):
    """Automatically create a valid collection name based on the embedding model name"""
    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)

    if len(collection_name) < 3:
        collection_name = collection_name + ('_' * (3 - len(collection_name)))
    elif len(collection_name) > 63:
        collection_name = collection_name[:63]

    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'

    return collection_name

# Define paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'ESCO dataset - v1.1.1 - classification - en - csv'))
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
INDEX_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'FAISS_index'))
COLLECTION_NAME = generate_valid_collection_name(EMBEDDING_MODEL_NAME)

# Try to get the OpenAI API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

# If the environment variable is not set, use the API key declared in the code
if not api_key or api_key == '':
    api_key = "sk-qOMmWrzGL6vM9JAUTkoTDgo6cyLE-Im5syuaAkZ6qZT3BlbkFJo7GMGShu7qhMvJ7cJVeMEWT6D5TF8e1FZuP7qzny8A"

# Setup OpenAI client
client = OpenAI(api_key=api_key)

def get_embedding(text, model=EMBEDDING_MODEL_NAME):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def main():
    # Load and filter skills data
    logging.info("Loading and filtering skills data...")
    skills_data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, 'skills_en.csv'))
    skills_data = skills_data[skills_data.skillType == 'skill/competence']
    skills_data = skills_data[(skills_data.description.str.len() > 30) & (skills_data.preferredLabel.str.len() > 5)]
    assert len(skills_data) > 100, 'Failed to load skills data!'
    skills = [f"{row['preferredLabel']} | {row['description']}" for _, row in skills_data.iterrows()]
    metadatas1 = [{'altLabels': row['altLabels'], 'modifiedDate': row['modifiedDate'], 'type': 'skill/competence', 'url': row['conceptUri']} for _, row in skills_data.iterrows()]

    # Load and filter occupations data
    logging.info("Loading and filtering occupations data...")
    occupations_data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, 'occupations_en.csv'))
    occupations_data = occupations_data[occupations_data.conceptType == 'Occupation']
    occupations_data = occupations_data[(occupations_data.description.str.len() > 30) & (occupations_data.preferredLabel.str.len() > 2)]
    assert len(occupations_data) > 10, 'Failed to load occupations data!'
    occupations = [f"{row['preferredLabel']} | {row['description']}" for _, row in occupations_data.iterrows()]
    metadatas2 = [{'modifiedDate': row['modifiedDate'], 'type': 'occupation', 'url': row['conceptUri']} for _, row in occupations_data.iterrows()]

    # Combine skills and occupations
    documents = skills + occupations
    metadatas = metadatas1 + metadatas2
    ids = [f"id_{i}" for i in range(len(documents))]
    assert len(documents) == len(metadatas) == len(ids), 'Data size mismatch!'

    # Generate embeddings for all documents
    logging.info(f"Generating embeddings for all {len(documents)} documents...")
    embeddings = []
    embedding_dim = None

    for idx, doc in enumerate(documents):
        try:
            embedding = get_embedding(doc)
            embeddings.append(embedding)
            if embedding_dim is None:
                embedding_dim = len(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding for document {idx}: {doc}")
            logging.error(str(e))
            # Append a zero vector as a placeholder
            if embedding_dim is not None:
                embeddings.append([0.0] * embedding_dim)
            else:
                # If embedding_dim is not set yet, skip this document
                continue

    embeddings = np.array(embeddings).astype('float32')

    # Check if embeddings were generated
    if len(embeddings) == 0:
        logging.error("No embeddings were generated. Exiting.")
        return

    if embedding_dim is None:
        embedding_dim = embeddings.shape[1]
        logging.info(f"The embedding dimension is {embedding_dim}")

    # Initialize the FAISS index
    index = faiss.IndexFlatL2(embedding_dim)

    # Add all embeddings to the index
    logging.info("Adding all embeddings to the FAISS index...")
    index.add(embeddings)

    # Save the FAISS index to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_index_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_faiss.index")
    faiss.write_index(index, faiss_index_path)
    logging.info(f"FAISS index saved at {faiss_index_path}")

    # Save the metadata (documents, metadatas, ids) to a JSON file
    metadata_to_save = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids
    }
    metadata_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_to_save, f, indent=4, ensure_ascii=False)
    logging.info(f"Metadata saved at {metadata_path}")

    logging.info('Processing completed successfully!')

if __name__ == "__main__":
    main()
