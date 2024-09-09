import os
import re
import pickle
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import logging

# Constants
RESULTS_PER_SEARCH = 25

# Set the OpenAI API key
os.environ[
    'OPENAI_API_KEY'] = "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"

# Define the relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_embeddings'))
PDF_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'accountant.pdf'))
BM_25_DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'processed_data', 'ESCO_BM25'))
DOCUMENTS_PATH = os.path.join(BM_25_DATABASE_DIR, 'documents.pickle')
METADATAS_PATH = os.path.join(BM_25_DATABASE_DIR, 'metadatas.pickle')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def extract_pdf(pdf_path: str) -> str:
    """Return extracted text from PDF File"""
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    return " ".join(page.page_content for page in data)


def load_documents() -> List[Document]:
    """Load the documents and metadatas from pickle files."""
    try:
        with open(DOCUMENTS_PATH, 'rb') as f:
            documents = pickle.load(f)
        with open(METADATAS_PATH, 'rb') as f:
            metadatas = pickle.load(f)

        langchain_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadatas)
        ]
        logging.info("Successfully loaded documents and metadatas.")
        return langchain_docs
    except FileNotFoundError as e:
        logging.error(f"Error loading documents: {e}")
        return []


def create_type_specific_retriever(vectorstore: Chroma, documents: List[Document], doc_type: str) -> EnsembleRetriever:
    # Filter documents by type
    type_specific_docs = [doc for doc in documents if doc.metadata['type'] == doc_type]

    # Create type-specific BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(type_specific_docs)
    bm25_retriever.k = RESULTS_PER_SEARCH

    # Create type-specific vectorstore retriever
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={
        "k": RESULTS_PER_SEARCH,
        "filter": {"type": doc_type}
    })

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever


def search_documents(retriever: EnsembleRetriever, query: str) -> List[Dict]:
    results = retriever.invoke(query)

    # Ensure we return exactly RESULTS_PER_SEARCH documents
    final_results = results[:RESULTS_PER_SEARCH]

    return [
        {
            "document": doc.page_content,
            "metadata": doc.metadata,
            "score": 1 / (i + 1)  # Simple ranking score
        }
        for i, doc in enumerate(final_results)
    ]


if __name__ == "__main__":
    embedding_model = "text-embedding-3-small"
    embedding_ef = OpenAIEmbeddings(model=embedding_model)

    # Init Chroma DB
    collection_name = generate_valid_collection_name(embedding_model)
    vectorstore = init_chroma_db(collection_name, embedding_ef)

    # Load documents
    documents = load_documents()

    if not documents:
        logging.error("Failed to load documents. Exiting.")
        exit(1)

    # Create type-specific retrievers
    skills_retriever = create_type_specific_retriever(vectorstore, documents, 'skill/competence')
    occupations_retriever = create_type_specific_retriever(vectorstore, documents, 'occupation')

    # Extract input text
    input_text = extract_pdf(PDF_PATH)

    # Perform searches
    skills_results = search_documents(skills_retriever, input_text)
    occupations_results = search_documents(occupations_retriever, input_text)

    # Print results
    print(f"\nTop {RESULTS_PER_SEARCH} Skills/Competences:")
    for i, result in enumerate(skills_results, 1):
        print(f"{i}. {result['document']} (Score: {result['score']:.4f})")

    print(f"\nTop {RESULTS_PER_SEARCH} Occupations:")
    for i, result in enumerate(occupations_results, 1):
        print(f"{i}. {result['document']} (Score: {result['score']:.4f})")