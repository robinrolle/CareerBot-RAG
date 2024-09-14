import json
import os
import re
import csv
import time
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
VECTOR_MAX_RETRIEVED = 25

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Define the relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
DATASET = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'text_datasets', 'cv_extracts.csv'))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'POC', 'evaluations', 'results'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the LLM for evaluation
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# System prompt template
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an expert in analyzing and listing {item_type} in CV texts.
    Your task is to analyze and list {item_type} relevant to the provided CV text. 
    Use only {item_type} names that are in the European Skills, Competences, Qualifications, and Occupations (ESCO) dataset.
    Provide your answer ONLY as a valid JSON object, without any additional text, explanation, or markdown formatting.
    """
)

human_prompt = HumanMessagePromptTemplate.from_template(
    """
    Extract and list each {item_type} detailed in the following CV text. 
    Your response should be ONLY a valid JSON object as shown below, without any additional text:
    {{
        "{item_type}s": [
            {{
                "name": "{item_type} name"
            }},
            {{
                "name": "{item_type} name"
            }},
            ...
        ]
    }}

    CV Text:
    {cv_text}
    """
)

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

def generate_valid_collection_name(model_name: str) -> str:
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

def init_chroma_db(db_collection_name: str, embedding_function) -> Chroma:
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    try:
        db = Chroma(client=chroma_client, collection_name=db_collection_name, embedding_function=embedding_function)
        logging.info("Chroma database loaded and ready")
    except Exception as e:
        logging.error(f"Failed to load the Chroma database: {e}")
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


def generate_hypothetical_docs(input_text: str, item_type: str) -> List[str]:
    formatted_prompt = prompt.format(item_type=item_type, cv_text=input_text)
    response = llm.invoke(formatted_prompt)

    try:
        parsed_response = json.loads(response.content)
        items = parsed_response.get(f"{item_type}s", [])
        return [item["name"] for item in items]
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {response.content}")
        logging.error(f"JSON decode error: {str(e)}")
        return []
    except KeyError as e:
        logging.error(f"Unexpected JSON structure in LLM response: {response.content}")
        logging.error(f"KeyError: {str(e)}")
        return []



def run_hyde_experiment(embedding_model_name: str, input_text: str):
    try:
        if embedding_model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            embedding_ef = OpenAIEmbeddings(model=embedding_model_name)
        else:
            embedding_ef = HuggingFaceEmbeddings(model_name=embedding_model_name)

        collection_name = generate_valid_collection_name(embedding_model_name)
        vectorstore = init_chroma_db(collection_name, embedding_ef)

        hypothetical_skills = generate_hypothetical_docs(input_text, "skill")
        hypothetical_occupations = generate_hypothetical_docs(input_text, "occupation")

        skills_embeddings = embedding_ef.embed_documents(hypothetical_skills)
        occupations_embeddings = embedding_ef.embed_documents(hypothetical_occupations)

        skills_docs, skills_query_time = query_documents(vectorstore, skills_embeddings, 'skill/competence', VECTOR_MAX_RETRIEVED)
        occupation_docs, occupations_query_time = query_documents(vectorstore, occupations_embeddings, 'occupation', VECTOR_MAX_RETRIEVED)

        return {
            "skills": [doc for sublist in skills_docs.get("documents", []) for doc in sublist],
            "skills_query_time": skills_query_time,
            "occupations": [doc for sublist in occupation_docs.get("documents", []) for doc in sublist],
            "occupations_query_time": occupations_query_time
        }

    except Exception as e:
        logging.error(f"Error in run_hyde_experiment for {embedding_model_name}: {e}")
        return {
            "skills": [],
            "skills_query_time": 0,
            "occupations": [],
            "occupations_query_time": 0
        }

def create_results_csv():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        logging.info(f"Created results directory: {RESULTS_DIR}")

    results_file = os.path.join(RESULTS_DIR, 'hyde_experiment_results.csv')
    headers = ['embedding_model', 'max_retrieved', 'text_file_name', 'retrieved_skills', 'skills_query_time',
               'retrieved_occupations', 'occupations_query_time', 'original_text']

    with open(results_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    return results_file

def process_cv(args):
    embedding_model, input_text, text_file_name = args
    result = run_hyde_experiment(embedding_model, input_text)
    return [
        embedding_model,
        VECTOR_MAX_RETRIEVED,
        text_file_name,
        ','.join(result['skills']),
        result['skills_query_time'],
        ','.join(result['occupations']),
        result['occupations_query_time'],
        input_text
    ]

def run_experiment(embedding_models, dataset_path, results_file):
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        logging.info("UTF-8 encoding failed, trying ISO-8859-1")
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

    logging.info(f"HyDE experiment complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()