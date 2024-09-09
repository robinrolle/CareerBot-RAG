import os
import re
import time
import csv
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import textwrap
from typing import List


# Set the OpenAI API key and LangSmith environment variables
os.environ['OPENAI_API_KEY'] = "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"

# Configuration Constants
VECTORSTORE_MAX_RETRIEVED = 25
NUM_SUB_QUERIES = 4

# Define the relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
DATASET = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'text_datasets', 'cv_extracts.csv'))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'POC', 'evaluations', 'results'))


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
        print(f"Database loaded and ready for {db_collection_name}")
    except Exception as e:
        print(f"Failed to load the database for {db_collection_name}: {e}")
        raise
    return db


def query_documents(db, input_embedding, doc_type, max_retrieved):
    start_time = time.time()
    retrieved_docs = db._collection.query(
        query_embeddings=input_embedding,
        n_results=max_retrieved,
        include=['documents'],
        where={"type": doc_type}
    )
    elapsed_time = time.time() - start_time
    return retrieved_docs, elapsed_time


def clean_text(text):
    # Remove any non-printable characters
    return ''.join(char for char in text if char.isprintable())


def split_input_text(input_text: str, num_chunks: int = 4) -> List[str]:
    """
    Split the input text into a specified number of roughly equal chunks.

    Args:
    input_text (str): The text to be split.
    num_chunks (int): The number of chunks to split the text into. Default is 4.

    Returns:
    List[str]: A list of text chunks.
    """
    # Clean the input text
    cleaned_input_text = clean_text(input_text)

    # Calculate the approximate length of each chunk
    chunk_size = len(cleaned_input_text) // num_chunks

    # Use textwrap to split the text into chunks
    chunks = textwrap.wrap(cleaned_input_text, width=chunk_size, break_long_words=False, replace_whitespace=False)

    # Ensure we have exactly num_chunks by combining or splitting as needed
    while len(chunks) > num_chunks:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    while len(chunks) < num_chunks:
        chunks.append("")

    return chunks


def run_retrieval_experiment(embedding_model_name, input_text):
    try:
        if embedding_model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            embedding_ef = OpenAIEmbeddings(model=embedding_model_name)
        else:
            embedding_ef = HuggingFaceEmbeddings(model_name=embedding_model_name)

        collection_name = generate_valid_collection_name(embedding_model_name)
        db = init_chroma_db(collection_name, embedding_ef)

        # Split the input text into 4 chunks
        sub_queries = split_input_text(input_text, NUM_SUB_QUERIES)

        # Calculate doc_per_query
        doc_per_query = max(1, VECTORSTORE_MAX_RETRIEVED // NUM_SUB_QUERIES)

        total_skills_query_time = 0
        total_occupation_query_time = 0
        duplicates_count = 0
        all_skills = set()
        all_occupations = set()

        for query in sub_queries:
            input_embedding = embedding_ef.embed_query(text=query)

            retrieved_skills, skills_query_time = query_documents(db, input_embedding, 'skill/competence',
                                                                  doc_per_query)
            retrieved_occupations, occupations_query_time = query_documents(db, input_embedding, 'occupation',
                                                                            doc_per_query)

            total_skills_query_time += skills_query_time
            total_occupation_query_time += occupations_query_time

            # Check for duplicates
            new_skills = set(clean_text(doc) for sublist in retrieved_skills.get("documents", []) for doc in sublist)
            new_occupations = set(
                clean_text(doc) for sublist in retrieved_occupations.get("documents", []) for doc in sublist)

            duplicates_count += len(all_skills.intersection(new_skills))
            duplicates_count += len(all_occupations.intersection(new_occupations))

            all_skills.update(new_skills)
            all_occupations.update(new_occupations)

        return {
            "skills": list(all_skills),
            "skills_query_time": total_skills_query_time,
            "occupations": list(all_occupations),
            "occupations_query_time": total_occupation_query_time,
            "doc_overlap_count": duplicates_count,
        }
    except Exception as e:
        print(f"Error in run_retrieval_experiment for {embedding_model_name}: {e}")
        return {
            "skills": [],
            "skills_query_time": 0,
            "occupations": [],
            "occupations_query_time": 0,
            "doc_overlap_count": 0
        }


def process_cv(args):
    embedding_model, input_text, text_file_name = args
    result = run_retrieval_experiment(embedding_model, input_text)
    return [
        embedding_model,
        VECTORSTORE_MAX_RETRIEVED,
        text_file_name,
        ','.join(result['skills']),
        result['skills_query_time'],
        ','.join(result['occupations']),
        result['occupations_query_time'],
        input_text,  # text from the CSV,
        result['doc_overlap_count']
    ]


def create_results_csv():
    # Check if the results directory exists, if not create it
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")

    results_file = os.path.join(RESULTS_DIR, 'query_decomp_experiment_results.csv')
    headers = ['embedding_model', 'max_retrieved', 'text_file_name', 'retrieved_skills', 'skills_query_time',
               'retrieved_occupations', 'occupations_query_time', 'original_text', 'duplicates_count']

    with open(results_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    return results_file


def run_experiment(embedding_models, dataset_path, results_file):
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, trying ISO-8859-1")
        df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

    with ProcessPoolExecutor() as executor:
        futures = []
        for _, row in df.iterrows():
            for embedding_model in embedding_models:
                futures.append(executor.submit(process_cv, (embedding_model, row['Extracted Text'], row['Filename'])))

        with open(results_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CV extracts"):
                writer.writerow(future.result())


def main():
    embedding_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "mixedbread-ai/mxbai-embed-large-v1",
        "intfloat/e5-large-v2"
    ]

    results_file = create_results_csv()
    run_experiment(embedding_models, DATASET, results_file)

    print(f"Evaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
