from .models import ProcessResponse
from langchain_community.document_loaders import PyMuPDFLoader
import logging
from typing import List, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from .templates import full_prompt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .config import NB_SELECTED_SKILLS, NB_SELECTED_OCCUPATIONS, NB_SUGGESTED_SKILLS, NB_SUGGESTED_OCCUPATIONS


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

def get_similar_documents(db, docs, doc_type, n_suggest):
    used_docs = set(doc for sublist in docs['documents'] for doc in sublist)
    n_old = len(used_docs)
    n_new = 2 * int(np.ceil(n_suggest / n_old))

    # Calculer les distances entre les embeddings
    embeddings = np.array([embedding for sublist in docs['embeddings'] for embedding in sublist])

    if len(embeddings) < 2:
        raise ValueError("Pas assez d'embeddings pour calculer les distances.")

    distances = pdist(embeddings, metric='cosine')
    distance_matrix = squareform(distances)

    # Trier les documents par similarité
    sorted_indices = np.argsort(distance_matrix.sum(axis=1))

    # Aplatir les documents, embeddings et IDs pour un accès facile
    flat_docs = [{'document': doc, 'embedding': emb, 'id': doc_id}
                 for sublist_docs, sublist_embs, sublist_ids in zip(docs['documents'], docs['embeddings'], docs['ids'])
                 for doc, emb, doc_id in zip(sublist_docs, sublist_embs, sublist_ids)]

    sorted_docs = [flat_docs[i] for i in sorted_indices]

    # Définir la taille des batchs
    batch_size = 10  # Ajustez selon vos besoins

    new_docs = []
    for i in range(0, len(sorted_docs), batch_size):
        batch = sorted_docs[i:i + batch_size]
        batch_embeddings = [doc['embedding'] for doc in batch]

        suggested_docs = db.query(
            query_embeddings=batch_embeddings,
            n_results=n_old + n_new,
            include=['embeddings', 'documents', 'metadatas'],
            where={"type": doc_type}
        )

        for docs_list, embs_list, ids_list in zip(suggested_docs['documents'], suggested_docs['embeddings'],
                                                  suggested_docs['ids']):
            cands = [{'documents': doc, 'embeddings': emb, 'ids': doc_id}
                     for doc, emb, doc_id in zip(docs_list, embs_list, ids_list) if doc not in used_docs]
            new_docs.extend(cands)
            used_docs.update(doc['documents'] for doc in cands)

    # Sélection finale
    final_docs = []
    for doc in new_docs:
        if len(final_docs) < n_suggest:
            final_docs.append(doc)
        else:
            break

    return final_docs

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


    for i, summary in enumerate(summaries):
        items = summary['summary']
        print(f"items : {items}")
        query = ', '.join(items)
        print(f"query : {query}")

        # Retrieve docs for each experiences
        retrieved_skills = similarity_search_skills(query, collection, model, top_k=2)
        retrieved_occupations = similarity_search_occupations(query, collection, model, top_k=1)

        # Formating
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

    # Get similar doc based on the retrieved ones
    similar_skills = get_similar_documents(collection, all_retrieved_skills, 'skill/competence', NB_SUGGESTED_SKILLS)
    similar_occupations = get_similar_documents(collection, all_retrieved_occupations, 'occupation', NB_SUGGESTED_OCCUPATIONS)

    similar_skills_ids = set([doc['ids'] for doc in similar_skills])
    similar_occupations_ids = set([doc['ids'] for doc in similar_occupations])


    return ProcessResponse(
        selected_skills_ids=list(unique_skills),
        selected_occupations_ids=list(unique_occupations),
        suggested_skills_ids=list(similar_skills_ids),
        suggested_occupations_ids=list(similar_occupations_ids)
    )
