import numpy as np
import faiss
import chromadb
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))

def get_chroma_collections():
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    collections = chroma_client.list_collections()
    if not collections:
        raise ValueError("No Chroma collections found.")
    return collections

def create_and_save_faiss_index(collection_name):
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    print(f"Processing Chroma collection: {collection_name}")

    collection = chroma_client.get_collection(collection_name)

    results = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = results['embeddings']
    documents = results['documents']
    metadatas = results['metadatas']
    ids = results['ids']

    # Check if embeddings exist
    if not embeddings:
        print(f"No embeddings found for collection: {collection_name}")
        return

    embeddings_array = np.array(embeddings).astype('float32')

    # Check if embeddings have the correct shape
    if len(embeddings_array.shape) == 1 or embeddings_array.shape[1] == 0:
        print(f"Embeddings for collection {collection_name} do not have the expected dimensions.")
        return

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings_array)

    faiss_index_path = os.path.join(DATABASE_DIR, f"{collection_name}_faiss.index")
    faiss.write_index(index, faiss_index_path)

    # Add Chroma collection ID to each metadata
    for metadata in metadatas:
        metadata['chroma_collection_id'] = collection_name

    metadata_to_save = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids
    }
    metadata_path = os.path.join(DATABASE_DIR, f"{collection_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_to_save, f)

    print(f"FAISS index saved to: {faiss_index_path}")
    print(f"Metadata saved to: {metadata_path}")

def process_all_collections():
    collections = get_chroma_collections()
    print(f"Total number of collections to process: {len(collections)}")

    for collection in collections:
        create_and_save_faiss_index(collection.name)
        print(f"Processing completed for collection: {collection.name}")
        print("---")

if __name__ == "__main__":
    process_all_collections()
    print("Processing of all collections completed.")
