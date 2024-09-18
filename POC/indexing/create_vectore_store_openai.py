import os
import pandas as pd
import chromadb
import re
from tqdm import tqdm
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_valid_collection_name(model_name):
    """Automatically create collection name based the embedding
    model in order to meet chromaDB name restrictions"""

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

# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'ESCO dataset - v1.1.1 - classification - en - csv'))
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data','processed_data', 'ESCO_embeddings'))
COLLECTION_NAME = generate_valid_collection_name(EMBEDDING_MODEL_NAME)
BATCH_SIZE = 5461  # Define the batch size as per the chroma client limit, it's not chunking!

# Setup OpenAI client
client = OpenAI()

def get_embedding(text, model=EMBEDDING_MODEL_NAME):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Setup DB
chroma_client = chromadb.PersistentClient(path=DATABASE_DIR)

try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    assert collection.count() > 10, 'Empty database!'
    logging.info("Old database loaded and ready")
except Exception as e:
    logging.warning("Creating a new database, this might cost you!")

    # Deleting old collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        logging.info("No existing collection to delete")

    # Creating new collection with model_name in metadata
    collection_metadata = {"model_name": EMBEDDING_MODEL_NAME}
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=collection_metadata
    )

    # Load and filter skills data
    logging.info("Loading and filtering skills data...")
    skills_data = pd.read_csv(os.path.join(os.getcwd(), SOURCE_DATA_PATH, 'skills_en.csv'))
    skills_data = skills_data[skills_data.skillType == 'skill/competence']
    skills_data = skills_data[(skills_data.description.str.len() > 30) & (skills_data.preferredLabel.str.len() > 5)]
    assert len(skills_data) > 100, 'Failed to read!'
    skills = [f"{row['preferredLabel']} | {row['description']}" for _, row in skills_data.iterrows()]
    metadatas1 = [{'altLabels': row['altLabels'], 'modifiedDate': row['modifiedDate'], 'type': 'skill/competence', 'url': row['conceptUri']} for _, row in skills_data.iterrows()]

    # Load and filter occupations data
    logging.info("Loading and filtering occupations data...")
    occupations_data = pd.read_csv(os.path.join(os.getcwd(), SOURCE_DATA_PATH, 'occupations_en.csv'))
    occupations_data = occupations_data[occupations_data.conceptType == 'Occupation']
    occupations_data = occupations_data[(occupations_data.description.str.len() > 30) & (occupations_data.preferredLabel.str.len() > 2)]
    assert len(occupations_data) > 10, 'Failed to read!'
    occupations = [f"{row['preferredLabel']} | {row['description']}" for _, row in occupations_data.iterrows()]
    metadatas2 = [{'modifiedDate': row['modifiedDate'], 'type': 'occupation', 'url': row['conceptUri']} for _, row in occupations_data.iterrows()]

    # Combine skills and occupations
    documents = skills + occupations
    metadatas = metadatas1 + metadatas2
    ids = [f"id_{i}" for i in range(len(documents))]
    assert len(documents) == len(metadatas) == len(ids), 'Data size mismatch!'

    # Debugging: Check the type and content of the documents
    for i, doc in enumerate(documents[:5]):  # Show the first 5 documents for inspection
        logging.info(f"Document {i}: {doc} (type: {type(doc)})")

    logging.info(f"Total number of documents to be stored: {len(documents)}")

    #Generate embeddings and add documents to the collection in batches
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Adding batches"):
        batch_documents = documents[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        logging.info(f"Generating embeddings for batch {i // BATCH_SIZE + 1}")
        batch_embeddings = [get_embedding(doc) for doc in batch_documents]

        logging.info(f"Adding batch {i // BATCH_SIZE + 1} to the collection")
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
        except Exception as e:
            logging.error(f"Error adding batch: {str(e)}")
            # Log more details about the error if possible
            if hasattr(e, 'response'):
                logging.error(f"API response: {e.response.text}")
            raise  # Re-raise the exception after logging

    assert collection.count() > 10, 'Empty database!'
    logging.info(f'New database size is {collection.count()} documents!')
    logging.info('Database created and saved!')

logging.info('All done!')
