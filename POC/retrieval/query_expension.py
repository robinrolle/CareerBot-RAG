import os
import re
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb

# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
PDF_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'accountant.pdf'))

# Models config
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


# Configuration Constants
VECTORSTORE_MAX_RETRIEVED = 25  # Max number of documents to retrieve from vectorstore.
NUM_CHUNKS = 4  # Number of chunks to split the text into

# Set the OpenAI API key
os.environ[
    'OPENAI_API_KEY'] = "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"


def generate_valid_collection_name(model_name):
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


def init_chroma_db(db_collection_name, embedding_function):
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    try:
        db = Chroma(client=chroma_client, collection_name=db_collection_name, embedding_function=embedding_function)
        print("Database loaded and ready")
    except Exception as e:
        print(f"Failed to load the database: {e}")
        raise
    return db


def query_documents(db, input_embedding, doc_type, max_retrieved):
    retrieved_docs = db._collection.query(
        query_embeddings=input_embedding,
        n_results=max_retrieved,
        include=['embeddings', 'documents', 'metadatas'],
        where={"type": doc_type}
    )
    return retrieved_docs

def extract_pdf(pdf_path):
    """Return extracted text from PDF File"""
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    return " ".join(page.page_content for page in data)


def split_text_into_chunks(text, num_chunks=NUM_CHUNKS):
    """Split the text into a specified number of chunks"""
    total_length = len(text)
    chunk_size = total_length // num_chunks
    chunks = [text[i:i + chunk_size] for i in range(0, total_length, chunk_size)]

    # If there are any remaining characters, add them to the last chunk
    if len(chunks) > num_chunks:
        chunks[-2] += chunks[-1]
        chunks.pop()

    return chunks


def main():
    # Init embedding function
    embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    collection_name = generate_valid_collection_name(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    db = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embedding_ef)

    # Extract text from PDF
    input_text = extract_pdf(PDF_PATH)

    # Split text into 4 chunks
    chunks = split_text_into_chunks(input_text)

    # Calculate the number of documents to retrieve per chunk
    docs_per_chunk = VECTORSTORE_MAX_RETRIEVED // NUM_CHUNKS

    all_skills = []
    all_occupations = []

    for chunk in chunks:
        # Turn a chunk into embedding
        input_embedding = embedding_ef.embed_query(text=chunk)

        # Retrieve documents by querying the DB
        retrieved_skills = query_documents(db, input_embedding, 'skill/competence', docs_per_chunk)
        retrieved_occupations = query_documents(db, input_embedding, 'occupation', docs_per_chunk)

        # Add skills and occupations to the lists
        all_skills.extend(retrieved_skills['documents'][0])
        all_occupations.extend(retrieved_occupations['documents'][0])

    # Count occurrences
    skill_counts = Counter(all_skills)
    occupation_counts = Counter(all_occupations)

    # Print results
    print("Skills found (with counts):")
    for skill, count in skill_counts.items():
        print(f"- {skill}: {count}")

    print("\nOccupations found (with counts):")
    for occupation, count in occupation_counts.items():
        print(f"- {occupation}: {count}")

    # Print summary of duplicates
    print("\nSummary:")
    print(f"Total unique skills: {len(skill_counts)}")
    print(f"Skills returned more than once: {sum(1 for count in skill_counts.values() if count > 1)}")
    print(f"Total unique occupations: {len(occupation_counts)}")
    print(f"Occupations returned more than once: {sum(1 for count in occupation_counts.values() if count > 1)}")


if __name__ == "__main__":
    main()