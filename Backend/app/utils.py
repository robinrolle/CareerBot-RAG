import os
from typing import List, Dict
from .models import ProcessResponse, SuggestionResponse
from langchain_community.document_loaders import PyMuPDFLoader
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .database import faiss_index, faiss_metadata
from .templates import full_prompt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .config import EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, NB_SUGGESTED_SKILLS, NB_SUGGESTED_OCCUPATIONS, OPENAI_API_KEY, VECTORSTORE_MAX_RETRIEVED, LLM_MAX_PICKS
from .models import ExtractedItems, GradedItem
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
import traceback

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

def query_faiss_index(query_embedding, doc_type, max_retrieved):
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, max_retrieved)

    results = []
    for i, idx in enumerate(indices[0]):
        if faiss_metadata['metadatas'][idx]['type'] == doc_type:
            results.append({
                'id': faiss_metadata['ids'][idx],
                'document': faiss_metadata['documents'][idx],
                'metadata': faiss_metadata['metadatas'][idx],
                'distance': distances[0][i]
            })

    return results[:max_retrieved]

def extract_pdf(file_path: str) -> str:
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    extracted_text_pdf = ""
    for page in data:
        extracted_text_pdf += page.page_content
    return extracted_text_pdf

def process_cv_text_chain():
    model = ChatOpenAI(model=GENERATION_MODEL_NAME, temperature=0)
    structured_llm = model.with_structured_output(ExtractedItems)
    chain = RunnableSequence(
        full_prompt |
        structured_llm
    )
    return chain

def grading_chain(doc_type):
    prompt = ChatPromptTemplate.from_template("""
    Given the following CV content and a list of potential {doc_type}, please identify the most relevant {doc_type} for this CV. 

    CV Content: {question}

    Potential {doc_type}:
    {context}

    Please provide your answer in JSON format with a list of the most relevant items, up to a maximum of {max_picks} items.
    For each item, include the 'id', 'item' (the text after the '|'), and a 'relevance' score (HIGH, MEDIUM, or LOW) based on how well it matches the CV.

    Example format:
    {{
        "items": [
            {{"id": "id_123", "item": "Financial reporting", "relevance": "HIGH"}},
            {{"id": "id_456", "item": "Budgeting", "relevance": "MEDIUM"}}
        ]
    }}
    """)

    model = ChatOpenAI(model=GENERATION_MODEL_NAME, temperature=0)
    output_parser = JsonOutputParser()

    chain = RunnableSequence(
        prompt |
        model |
        output_parser
    )

    return chain


def get_similar_documents_faiss(ids: List[str], doc_type: str, n_suggest: int) -> List[Dict]:
    used_ids = set(ids)
    n_old = len(used_ids)
    n_new = 2 * int(np.ceil(n_suggest / n_old))

    # Retrieve the embeddings that correspond to the ids using faiss
    embeddings = []
    for id in ids:
        idx = faiss_metadata['ids'].index(id)
        embedding = faiss_index.reconstruct(idx)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')

    if len(embeddings) < 2:
        raise ValueError("Pas assez d'embeddings pour calculer les distances.")

    distances, indices = faiss_index.search(embeddings, n_old + n_new)

    new_docs = []
    for doc_idx, (distances_row, indices_row) in enumerate(zip(distances, indices)):
        for dist, idx in zip(distances_row, indices_row):
            if idx != -1:  # FAISS returns -1 for invalid results
                doc = faiss_metadata['documents'][idx]
                doc_id = faiss_metadata['ids'][idx]
                if doc_id not in used_ids:
                    new_docs.append({
                        'document': doc,
                        'id': doc_id,
                        'distance': dist
                    })
                    used_ids.add(doc_id)

    new_docs.sort(key=lambda x: x['distance'])

    final_docs = new_docs[:n_suggest]

    return final_docs

async def process_cv(file_path: str) -> ProcessResponse:
    try:
        input_text = extract_pdf(file_path)
        logger.info(f"PDF extracted successfully from {file_path}")

        process_cv_chain = process_cv_text_chain()
        logger.info("CV text processing chain initialized")

        extracted_skills = process_cv_chain.invoke(input={"doc_type": "skills", "cv_text": input_text})
        logger.info(f"Extracted skills: {extracted_skills}")

        extracted_occupations = process_cv_chain.invoke(input={"doc_type": "occupations", "cv_text": input_text})
        logger.info(f"Extracted occupations: {extracted_occupations}")

        all_retrieved_skills = []
        all_retrieved_occupations = []

        for skill in extracted_skills.items:
            try:
                skill_embedding = embedding_ef.embed_query(text=skill.name)
                retrieved_skill = query_faiss_index(skill_embedding, 'skill/competence', VECTORSTORE_MAX_RETRIEVED)
                all_retrieved_skills.extend(retrieved_skill)
            except Exception as e:
                logger.error(f"Error retrieving skill {skill.name}: {str(e)}")
                logger.error(traceback.format_exc())

        for occupation in extracted_occupations.items:
            try:
                occupation_embedding = embedding_ef.embed_query(text=occupation.name)
                retrieved_occupation = query_faiss_index(occupation_embedding, 'occupation', VECTORSTORE_MAX_RETRIEVED)
                all_retrieved_occupations.extend(retrieved_occupation)
            except Exception as e:
                logger.error(f"Error retrieving occupation {occupation.name}: {str(e)}")
                logger.error(traceback.format_exc())

        logger.info(f"Retrieved {len(all_retrieved_skills)} skills and {len(all_retrieved_occupations)} occupations")

        skills_context = "\n".join([f"{item['id']}|{item['document']}" for item in all_retrieved_skills])
        occupations_context = "\n".join([f"{item['id']}|{item['document']}" for item in all_retrieved_occupations])

        skills_chain = grading_chain("skills")
        skills_result = skills_chain.invoke({
            "question": input_text,
            "context": skills_context,
            "max_picks": LLM_MAX_PICKS[0],
            "doc_type": "skills"
        })
        logger.info(f"Skills grading result: {skills_result}")

        occupations_chain = grading_chain("occupations")
        occupations_result = occupations_chain.invoke({
            "question": input_text,
            "context": occupations_context,
            "max_picks": LLM_MAX_PICKS[1],
            "doc_type": "occupations"
        })
        logger.info(f"Occupations grading result: {occupations_result}")
        graded_skills = [GradedItem(id=item['id'], item=item['item'], relevance=item['relevance']) for item in
                         skills_result["items"]]
        graded_occupations = [GradedItem(id=item['id'], item=item['item'], relevance=item['relevance']) for item in
                              occupations_result["items"]]

        # Create a list of graded skills ids based on GradedItem
        graded_skills_ids = [item.id for item in graded_skills]

        # Create a list of graded occupation ids based on GradedItem
        graded_occupations_ids = [item.id for item in graded_occupations]

        similar_skills = get_similar_documents_faiss(graded_skills_ids, 'skill/competence', NB_SUGGESTED_SKILLS)
        logger.info(f"Found {len(similar_skills)} similar skills")

        similar_occupations = get_similar_documents_faiss(graded_occupations_ids, 'occupation',
                                                          NB_SUGGESTED_OCCUPATIONS)
        logger.info(f"Found {len(similar_occupations)} similar occupations")


        suggestion_response = SuggestionResponse(
            suggested_skills_ids=[doc['id'] for doc in similar_skills],
            suggested_occupations_ids=[doc['id'] for doc in similar_occupations]
        )

        return ProcessResponse(
            graded_skills=graded_skills,
            graded_occupations=graded_occupations,
            suggestions=suggestion_response
        )

    except Exception as e:
        logger.error(f"Error in process_cv: {str(e)}")
        logger.error(traceback.format_exc())
        raise