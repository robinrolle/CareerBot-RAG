import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, \
    SystemMessagePromptTemplate
import os
import re
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb
import logging

# Constants
VECTOR_MAX_RETRIEVED = 25

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Define the relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
PDF_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'accountant.pdf'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the LLM for evaluation
llm = ChatOpenAI(temperature=1, model="gpt-4")

# System prompt template
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an expert in analyzing and listing {item_type} in CV texts.
    Your task is to analyze and list {item_type} relevant to the provided CV text. 
    Use only {item_type} names that are in the European Skills, Competences, Qualifications, and Occupations (ESCO) dataset.
    Provide your answer only in the requested format, without any additional explanation.
    """
)

# Main prompt template
human_prompt = HumanMessagePromptTemplate.from_template(
    """
    Extract and list each {item_type} detailed in the following CV text. 
    Your response should be in JSON format as shown below:
    {{
        "{item_type}s": [
            {{
                "name": "{item_type} name",
            }},
            {{
                "name": "{item_type} name",
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


def extract_pdf(pdf_path: str) -> str:
    """Return extracted text from PDF File"""
    try:
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()
        return " ".join(page.page_content for page in data)
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return ""


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
    retrieved_docs = db._collection.query(
        query_embeddings=input_embedding,
        n_results=max_retrieved,
        include=['embeddings', 'documents', 'metadatas'],
        where={"type": doc_type}
    )
    return retrieved_docs


def generate_hypothetical_docs(input_text: str, item_type: str) -> List[str]:
    formatted_prompt = prompt.format(item_type=item_type, cv_text=input_text)
    response = llm.invoke(formatted_prompt)

    try:
        parsed_response = json.loads(response.content)
        items = parsed_response.get(f"{item_type}s", [])
        return [item["name"] for item in items]
    except json.JSONDecodeError:
        logging.error(f"Failed to parse LLM response as JSON: {response.content}")
        return []
    except KeyError:
        logging.error(f"Unexpected JSON structure in LLM response: {response.content}")
        return []


if __name__ == "__main__":
    embedding_model = "text-embedding-3-small"
    embedding_ef = OpenAIEmbeddings(model=embedding_model)

    # Init Chroma DB
    collection_name = generate_valid_collection_name(embedding_model)
    vectorstore = init_chroma_db(collection_name, embedding_ef)

    # Extract input text
    input_text = extract_pdf(PDF_PATH)

    hypothetical_skills = generate_hypothetical_docs(input_text, "skill")
    hypothetical_occupations = generate_hypothetical_docs(input_text, "occupation")

    # Embed hypothetical documents
    skills_embeddings = embedding_ef.embed_documents(hypothetical_skills)
    occupations_embeddings = embedding_ef.embed_documents(hypothetical_occupations)

    # Query documents
    skills_docs = query_documents(vectorstore, skills_embeddings, 'skill/competence', VECTOR_MAX_RETRIEVED)
    occupation_docs = query_documents(vectorstore, occupations_embeddings, 'occupation', VECTOR_MAX_RETRIEVED)

    # Print results
    print("Retrieved Skills:")
    for doc in skills_docs['documents'][0]:
        print(f"- {doc}")

    print("\nRetrieved Occupations:")
    for doc in occupation_docs['documents'][0]:
        print(f"- {doc}")