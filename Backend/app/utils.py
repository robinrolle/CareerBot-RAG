
from .models import ProcessResponse
from langchain_community.document_loaders import PyMuPDFLoader
import logging
from typing import List, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from .templates import full_prompt
import numpy as np
from sklearn.cluster import KMeans


GROQ_API_KEY = "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"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def similarity_search(query_text: str, collection, model, top_k: int, filter_type: str = None):
    try:
        query_embedding = model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={'type': filter_type},
            include=['embeddings', 'documents', 'metadatas']
        )

        # Return results
        return results

    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        return []

def similarity_search_skills(query_text: str, collection, model, top_k: int):
    return similarity_search(query_text, collection, model, top_k, filter_type='skill/competence')

def similarity_search_occupations(query_text: str, collection, model, top_k: int):
    return similarity_search(query_text, collection, model, top_k, filter_type='occupation')

def extract_pdf(file_path: str) -> str:
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    extracted_text_pdf = ""
    for page in data:
        extracted_text_pdf += page.page_content
    return extracted_text_pdf

def summarize_pdf(extracted_text: str) -> List[dict]:

    inputs = {"cv_text": extracted_text}
    model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=GROQ_API_KEY)

    response = model.invoke(full_prompt.format_messages(**inputs))

    parser = JsonOutputParser()
    try:
        llm_response = parser.parse(response.content)
        summaries = llm_response.get('summaries', [])
    except Exception as e:
        logger.error(f"Failed to parse LLM summary response: {e}")
        summaries = []

    return summaries

def get_similar_documents(collection, retrieved_docs, doc_type, n_suggest):
    # Extraire les embeddings des documents récupérés
    embeddings = np.array(retrieved_docs['embeddings'][0])

    # Déterminer le nombre de clusters pour KMeans
    n_clusters = max(1, len(embeddings) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Regrouper les embeddings par clusters
    clusters = kmeans.fit_predict(embeddings)

    all_suggested_docs = {
        'documents': [],
        'embeddings': [],
        'metadatas': [],
        'ids': []
    }

    # Pour chaque cluster, rechercher des documents similaires
    for cluster in range(n_clusters):
        cluster_embeddings = embeddings[clusters == cluster]
        suggested_docs = collection.query(
            query_embeddings=cluster_embeddings.tolist(),
            n_results=n_suggest,
            include=['embeddings', 'documents', 'metadatas'],
            where={"type": doc_type}
        )

        # Ajouter les résultats à l'ensemble global
        all_suggested_docs['documents'].extend(suggested_docs['documents'])
        all_suggested_docs['embeddings'].extend(suggested_docs['embeddings'])
        all_suggested_docs['metadatas'].extend(suggested_docs['metadatas'])
        all_suggested_docs['ids'].extend(suggested_docs['ids'])

    return all_suggested_docs

async def process_cv(file_path: str, collection, model) -> ProcessResponse:
    extracted_text = extract_pdf(file_path)
    summaries = summarize_pdf(extracted_text)
    print(f"summaries : {summaries}")

    unique_skills = set()
    unique_occupations = set()

    all_retrieved_skills = {
        'documents': [],
        'embeddings': [],
        'metadatas': [],
        'ids': []
    }
    all_retrieved_occupations = {
        'documents': [],
        'embeddings': [],
        'metadatas': [],
        'ids': []
    }

    for summary in summaries:
        items = summary['summary']
        print(f"items : {items}")
        query = ', '.join(items)
        print(f"query : {query}")

        # Rechercher les compétences et occupations de base
        retrieved_skills = similarity_search_skills(query, collection, model, top_k=2)
        retrieved_occupations = similarity_search_occupations(query, collection, model, top_k=1)

        # Ajouter les résultats aux ensembles agrégés
        all_retrieved_skills['documents'].extend(retrieved_skills['documents'])
        all_retrieved_skills['embeddings'].extend(retrieved_skills['embeddings'])
        all_retrieved_skills['metadatas'].extend(retrieved_skills['metadatas'])
        all_retrieved_skills['ids'].extend(retrieved_skills['ids'])

        all_retrieved_occupations['documents'].extend(retrieved_occupations['documents'])
        all_retrieved_occupations['embeddings'].extend(retrieved_occupations['embeddings'])
        all_retrieved_occupations['metadatas'].extend(retrieved_occupations['metadatas'])
        all_retrieved_occupations['ids'].extend(retrieved_occupations['ids'])

        unique_skills.update(retrieved_skills['ids'][0] if 'ids' in retrieved_skills else [])
        unique_occupations.update(retrieved_occupations['ids'][0] if 'ids' in retrieved_occupations else [])

    # Rechercher les documents similaires une fois pour toutes les compétences et occupations récupérées
    similar_skills = get_similar_documents(collection, all_retrieved_skills, 'skill/competence', 15)
    similar_occupations = get_similar_documents(collection, all_retrieved_occupations, 'occupation', 5)

    similar_skills_ids = set(similar_skills['ids'][0] if 'ids' in similar_skills else [])
    similar_occupations_ids = set(similar_occupations['ids'][0] if 'ids' in similar_occupations else [])

    # Filtrer les documents similaires pour éliminer ceux déjà récupérés
    filtered_similar_skills_ids = similar_skills_ids - unique_skills
    filtered_similar_occupations_ids = similar_occupations_ids - unique_occupations

    return ProcessResponse(
        selected_skills_ids=list(unique_skills),
        selected_occupations_ids=list(unique_occupations),
        suggested_skills_ids=list(filtered_similar_skills_ids),
        suggested_occupations_ids=list(filtered_similar_occupations_ids)
    )
