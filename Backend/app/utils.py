from .models import ProcessResponse
from langchain_community.document_loaders import PyMuPDFLoader
import logging
from typing import List, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from .templates import full_prompt


GROQ_API_KEY = "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"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def similarity_search(query_text: str, collection, model, top_k: int, filter_type: str = None) -> List[str]:
    try:
        query_embedding = model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={'type': filter_type}
        )

        # Extract doc & ID
        documents = results['documents'][0] if 'documents' in results else []
        ids = results['ids'][0] if 'ids' in results else []

        # Return tuples list
        return list(zip(documents, ids))

    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        return []

def similarity_search_skills(query_text: str, collection, model, top_k: int) -> List[str]:
    return similarity_search(query_text, collection, model, top_k, filter_type='skill/competence')

def similarity_search_occupations(query_text: str, collection, model, top_k: int) -> List[str]:
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


async def process_cv(file_path: str, collection, model) -> ProcessResponse:
    extracted_text = extract_pdf(file_path)
    summaries = summarize_pdf(extracted_text)
    print(f"summaries : {summaries}")

    unique_skills = set()
    unique_occupations = set()

    for summary in summaries:
        items = summary['summary']
        print(f"items : {items}")
        query = ', '.join(items)
        print(f"query : {query}")

        # Rechercher les compétences et occupations de base
        retrieved_skills = similarity_search_skills(query, collection, model, top_k=2)
        retrieved_occupations = similarity_search_occupations(query, collection, model, top_k=1)

        print(f"retrieved_skills {retrieved_skills}")
        print(f"retrieved_occupation {retrieved_occupations}")

        # Mettre à jour les sets d'IDs uniques
        unique_skills.update([skill_id for _, skill_id in retrieved_skills])
        unique_occupations.update([occupation_id for _, occupation_id in retrieved_occupations])

        #Rechercher les documents similaire
        # similar_skills = get_similar_documents(collection, retrieved_skills, 'skill/com)
        print(f"returned skills : {list(unique_skills)}")
        print(f"returned occupations : {list(unique_occupations)}")


    return ProcessResponse(skills_ids=list(unique_skills), occupations_ids=list(unique_occupations))

