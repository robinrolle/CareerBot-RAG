import os
import json
import re
from Backend.app.config import EMBEDDING_MODEL_NAME

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed_data', 'FAISS_index'))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'UI', 'public', 'data'))
SKILLS_OUTPUT_FILE_NAME = 'skills_options.json'
OCCUPATION_OUTPUT_FILE_NAME = 'occupations_options.json'

def generate_valid_collection_name(model_name):
    """Automatically create a collection name based on the embedding
    model to meet ChromaDB name restrictions"""
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

collection_name = generate_valid_collection_name(EMBEDDING_MODEL_NAME)

def create_json_from_metadata(index_type):
    metadata_path = os.path.join(DATABASE_DIR, f"{collection_name}_{index_type}_metadata.json")
    output_file_name = SKILLS_OUTPUT_FILE_NAME if index_type == "skills" else OCCUPATION_OUTPUT_FILE_NAME
    output_path = os.path.join(OUTPUT_DIR, output_file_name)

    # Check if the metadata file exists
    if not os.path.exists(metadata_path):
        print(f"The metadata file for collection '{collection_name}_{index_type}' does not exist.")
        return

    # Load metadata from the JSON file
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    documents = metadata['documents']
    ids = metadata['ids']

    items_list = []

    for doc_id, document in zip(ids, documents):
        # Extract the label by taking the part before the '|' symbol
        if '|' in document:
            label = document.split("|")[0].strip()
        else:
            label = document.strip()  # If no '|', use the entire document

        # Force the first letter of the label to uppercase without changing the rest
        if label:
            label = label[0].upper() + label[1:]

        item = {
            "value": doc_id,
            "label": label
        }

        items_list.append(item)

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write the resulting list to the respective JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items_list, f, indent=4, ensure_ascii=False)

    print(f"The {index_type} JSON file has been saved at: {output_path}")

if __name__ == "__main__":
    print(f"Processing collection: {collection_name}")
    create_json_from_metadata("skills")  # Process the skills index
    create_json_from_metadata("occupations")  # Process the occupations index
    print("JSON files generation completed.")
