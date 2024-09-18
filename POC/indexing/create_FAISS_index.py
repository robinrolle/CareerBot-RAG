import os
import json
import pandas as pd
import re
import logging
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

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

# Setup OpenAI client
client = OpenAI()


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
    metadatas_skills = [{
        'altLabels': row['altLabels'],
        'modifiedDate': row['modifiedDate'],
        'type': 'skill/competence',
        'url': row['conceptUri']
    } for _, row in skills_data.iterrows()]

    # Assign unique IDs for skills with a 'skill_' prefix
    ids_skills = [f"skill_{i}" for i in range(len(skills))]

    # Load and filter occupations data
    logging.info("Loading and filtering occupations data...")
    occupations_data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, 'occupations_en.csv'))
    occupations_data = occupations_data[occupations_data.conceptType == 'Occupation']
    occupations_data = occupations_data[
        (occupations_data.description.str.len() > 30) & (occupations_data.preferredLabel.str.len() > 2)]
    assert len(occupations_data) > 10, 'Failed to load occupations data!'

    occupations = [f"{row['preferredLabel']} | {row['description']}" for _, row in occupations_data.iterrows()]
    metadatas_occupations = [{
        'modifiedDate': row['modifiedDate'],
        'type': 'occupation',
        'url': row['conceptUri']
    } for _, row in occupations_data.iterrows()]

    # Assign unique IDs for occupations with an 'occupation_' prefix
    ids_occupations = [f"occupation_{i}" for i in range(len(occupations))]

    # Generate embeddings for skills
    logging.info(f"Generating embeddings for {len(skills)} skills...")
    embeddings_skills = []
    embedding_dim = None  # To be set based on the first successful embedding

    for idx, doc in enumerate(skills):
        try:
            embedding = get_embedding(doc)
            embeddings_skills.append(embedding)
            if embedding_dim is None:
                embedding_dim = len(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding for skill {idx}: {doc}")
            logging.error(str(e))
            # Append a zero vector as a placeholder if possible
            if embedding_dim is not None:
                embeddings_skills.append([0.0] * embedding_dim)
            else:
                continue  # Skip if embedding_dim is not set yet

    embeddings_skills = np.array(embeddings_skills).astype('float32')

    # Generate embeddings for occupations
    logging.info(f"Generating embeddings for {len(occupations)} occupations...")
    embeddings_occupations = []

    for idx, doc in enumerate(occupations):
        try:
            embedding = get_embedding(doc)
            embeddings_occupations.append(embedding)
            if embedding_dim is None:
                embedding_dim = len(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding for occupation {idx}: {doc}")
            logging.error(str(e))
            # Append a zero vector as a placeholder if possible
            if embedding_dim is not None:
                embeddings_occupations.append([0.0] * embedding_dim)
            else:
                continue  # Skip if embedding_dim is not set yet

    embeddings_occupations = np.array(embeddings_occupations).astype('float32')

    # Check if embeddings were generated
    if len(embeddings_skills) == 0:
        logging.error("No embeddings were generated for skills. Exiting.")
        return
    if len(embeddings_occupations) == 0:
        logging.error("No embeddings were generated for occupations. Exiting.")
        return

    if embedding_dim is None:
        embedding_dim = embeddings_skills.shape[1]  # Assuming at least skills have embeddings
        logging.info(f"The embedding dimension is {embedding_dim}")

    # Initialize FAISS index for skills
    logging.info("Initializing FAISS index for skills...")
    index_skills = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to the skills index
    logging.info("Adding skill embeddings to the FAISS index...")
    index_skills.add(embeddings_skills)

    # Initialize FAISS index for occupations
    logging.info("Initializing FAISS index for occupations...")
    index_occupations = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to the occupations index
    logging.info("Adding occupation embeddings to the FAISS index...")
    index_occupations.add(embeddings_occupations)

    # Save the FAISS indices to disk
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss_index_skills_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_skills_faiss.index")
    faiss.write_index(index_skills, faiss_index_skills_path)
    logging.info(f"FAISS index for skills saved at {faiss_index_skills_path}")

    faiss_index_occupations_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_occupations_faiss.index")
    faiss.write_index(index_occupations, faiss_index_occupations_path)
    logging.info(f"FAISS index for occupations saved at {faiss_index_occupations_path}")

    # Save the metadata for skills
    metadata_skills_to_save = {
        "documents": skills,
        "metadatas": metadatas_skills,
        "ids": ids_skills
    }
    metadata_skills_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_skills_metadata.json")
    with open(metadata_skills_path, "w", encoding="utf-8") as f:
        json.dump(metadata_skills_to_save, f, indent=4, ensure_ascii=False)
    logging.info(f"Metadata for skills saved at {metadata_skills_path}")

    # Save the metadata for occupations
    metadata_occupations_to_save = {
        "documents": occupations,
        "metadatas": metadatas_occupations,
        "ids": ids_occupations
    }
    metadata_occupations_path = os.path.join(INDEX_DIR, f"{COLLECTION_NAME}_occupations_metadata.json")
    with open(metadata_occupations_path, "w", encoding="utf-8") as f:
        json.dump(metadata_occupations_to_save, f, indent=4, ensure_ascii=False)
    logging.info(f"Metadata for occupations saved at {metadata_occupations_path}")

    logging.info('Processing completed successfully!')


if __name__ == "__main__":
    main()
