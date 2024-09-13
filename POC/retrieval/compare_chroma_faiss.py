import numpy as np
import faiss
import json
import os
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
PDF_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'accountant.pdf'))

# Set the OpenAI API key
os.environ[
    'OPENAI_API_KEY'] = "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"

EMBEDDING_MODEL_NAME = "text-embedding-3-small"


def extract_pdf(pdf_path):
    """Return extracted text from PDF File"""
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    return " ".join(page.page_content for page in data)


def load_faiss_index_and_metadata(collection_name):
    faiss_index_path = os.path.join(DATABASE_DIR, f"{collection_name}_faiss.index")
    metadata_path = os.path.join(DATABASE_DIR, f"{collection_name}_metadata.json")

    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index not found : {collection_name}")

    index = faiss.read_index(faiss_index_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata


def query_faiss_index(index, query_embedding, top_k=5):
    query_embedding = np.array([query_embedding]).astype('float32')

    # Mesuring time
    start_time = time.time()
    distances, indices = index.search(query_embedding, top_k)
    end_time = time.time()

    search_time = end_time - start_time
    return distances[0], indices[0], search_time


def query_chroma_collection(collection, query_embedding, top_k=5):
    # Mesuring time
    start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    end_time = time.time()

    search_time = end_time - start_time
    return results, search_time


def main():
    collection_name = "text-embedding-3-small"
    embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Exemple user input
    input_text = extract_pdf(PDF_PATH)

    # Turn input into embedding
    query_embedding = embedding_ef.embed_query(text=input_text)

    index, metadata = load_faiss_index_and_metadata(collection_name)

    # Query FAISS
    distances, indices, search_time = query_faiss_index(index, query_embedding, top_k=25)

    print(f"Top {len(indices)} results (Time seconds {search_time:.6f}) :")
    for i, idx in enumerate(indices):
        print(f"Rank {i + 1}:")
        print(f"Document: {metadata['documents'][idx]}")
        print(f"Distance: {distances[i]}")
        print("---")

    # Chroma DB
    client = chromadb.PersistentClient(path=DATABASE_DIR)
    chroma_collection = client.get_collection(collection_name)

    # Query Chroma
    results, search_time = query_chroma_collection(chroma_collection, query_embedding, top_k=25)

    print(f"Top {len(results['ids'][0])} results (Time seconds {search_time:.6f}) :")
    for i in range(len(results['ids'][0])):
        print(f"Rank {i + 1}:")
        print(f"Document: {results['documents'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
        print("---")


if __name__ == "__main__":
    main()
