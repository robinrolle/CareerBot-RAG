import os
import re
import numpy as np
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable, RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
import chromadb

from POC.retrieval.prompts.templates import MY_PROMPT_TEMPLATE

# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data','processed_data', 'ESCO_embeddings'))

# Models config
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"

# Configuration Constants
VECTORSTORE_MAX_RETRIEVED = 25 # Max number of documents to retrieve from vectorstore. Don't go over context window
VECTORSTORE_MAX_SUGGESTED = [20, 10] # for [skills, occupations] how many potential items to suggest from vectorstore
LLM_MAX_PICKS = [15, 5] # for [skills, occupations] how many items LLM must pick from retrieved options

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

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

def get_retriever(db, max_retrieved, doc_type):
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": max_retrieved,
            "filter": {"type": doc_type}
        }
    )

def format_skills_for_prompt(skills_docs):
    formatted_skills = []
    for doc in skills_docs:
        item_description = f"{doc.page_content} | {doc.page_content}"
        formatted_skills.append(item_description)
    return "\n".join(formatted_skills)

def format_occupations_for_prompt(occupation_docs):
    formatted_occupations = []
    for doc in occupation_docs:
        item_description = f"{doc.page_content} | {doc.page_content}"
        formatted_occupations.append(item_description)
    return "\n".join(formatted_occupations)

def initialize_chain():
    # Create the prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(MY_PROMPT_TEMPLATE)

    # Initialize the language model
    model = ChatOpenAI(model=GENERATION_MODEL_NAME, temperature=0)

    # Initialize the JSON output parser
    output_parser = JsonOutputParser()

    # Chain the prompt and model together, followed by the output parser
    chain = RunnableSequence(
        prompt |
        model |
        output_parser
    )

    return chain

def create_item_relevance_mapping(answer):
    # Verify and adjust the response format
    assert ('items' in answer) and ('relevances' in answer), 'Response not in correct format!'

    # Adjust the length of relevances to match the number of items
    di = len(answer['relevances']) - len(answer['items'])
    if di > 0:
        answer['relevances'] = answer['relevances'][0:len(answer['items'])]
    elif di < 0:
        answer['relevances'] = answer['relevances'] + ['LOW'] * (-di)

    # Remove duplicates
    df_items = pd.DataFrame(answer['items'], columns=['item'])
    idx = df_items[df_items.duplicated(keep=False)].index.tolist()

    answer['relevances'] = [x.strip() for k, x in enumerate(answer['relevances']) if k not in idx]
    answer['items'] = [x.strip() for k, x in enumerate(answer['items']) if k not in idx]

    # Define the relevance mapping
    relevance_mapping = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

    # Create a mapping of items to their numerical relevances
    item_relevance_map = {answer['items'][i]: relevance_mapping[answer['relevances'][i]] for i in range(len(answer['items']))}

    return item_relevance_map

def structure_terms(retrieved_docs, response_processed, doc_type):
    structured_docs = []

    for i, doc in enumerate(retrieved_docs['documents'][0]):
        for term in response_processed.keys():
            labels = doc.split('|')[0]
            if term in labels:
                relevance = response_processed[term]
                if isinstance(relevance, str):
                    relevance_mapping = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    relevance = relevance_mapping[relevance]

                structured_docs.append({
                    'embedding': retrieved_docs['embeddings'][0][i],
                    'document': doc,
                    'relevance': relevance,
                    'type': doc_type,
                    'label': labels.strip()
                })
                break

    return structured_docs

def get_similar_documents(db, docs, doc_type, n_suggest):
    # Memorise already selected docs
    used_docs = set([x['document'] for x in docs])
    n_old = len(used_docs)
    n_new = 2 * int(np.ceil(n_suggest / n_old))
    new_docs = []

    # Get new similar documents
    for current_doc in docs:
        suggested_docs = db._collection.query(
            query_embeddings=[current_doc['embedding']],
            n_results=n_old + n_new,
            include=['embeddings', 'documents', 'metadatas'],
            where={"type": doc_type}
        )
        z = suggested_docs['documents'][0]
        cands = [{'document': z[k], 'embedding': suggested_docs['embeddings'][0][k]} for k in range(len(z)) if (z[k] not in used_docs)]
        new_docs.append(cands)
        used_docs.update([x['document'] for x in cands])

    # Sélectionner les suggestions finales
    round = -1
    final_docs = []

    while True:
        round += 1
        is_empty = True
        for cands in new_docs:
            if len(cands) > round:
                is_empty = False
                if len(final_docs) < n_suggest:
                    final_docs.append(cands[round])
                else:
                    break
        if is_empty or len(final_docs) == n_suggest:
            break

    # Format final result
    for k, x in enumerate(final_docs):
        final_docs[k]['label'] = x['document'].split('|')[0].strip()

    return final_docs


if __name__ == "__main__":
    # Init embedding functiion
    embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    collection_name = generate_valid_collection_name(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    db = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embedding_ef)

    input_text = """
    I have worked last 5 years as a contractor that sells and installs heat pumps for consumers. 
    I have an engineering degree and know lots of refrigerants, and technology of heating and cooling systems. 
    I have a license to do electrical jobs."""

    # Turn query into embedding
    input_embedding = embedding_ef.embed_query(text=input_text)

    # Retrieve documents by querying the collection
    retrieved_skills = db._collection.query(query_embeddings=input_embedding, n_results=VECTORSTORE_MAX_RETRIEVED,include=['embeddings','documents'],where={"type": "skill/competence"})
    retrieved_occupations = db._collection.query(query_embeddings=input_embedding, n_results=VECTORSTORE_MAX_RETRIEVED, include=['embeddings', 'documents'], where={"type": "occupation"})

    # Output the embeddings
    print("Skills Embeddings:", retrieved_skills['documents'][0])
    print("Occupation Embeddings:", retrieved_occupations['documents'][0])

    # Clean and prepare the contexts for the prompt
    context_skills = [x.replace('\n', ' ').replace('\r', '') for x in retrieved_skills['documents'][0]]
    context_skills = [x for x in context_skills if len(x) > 4]  # Only include documents with more than 4 characters
    context_skills = '\n\n'.join(context_skills)
    context_occupations = [x.replace('\n', ' ').replace('\r', '') for x in retrieved_occupations['documents'][0]]
    context_occupations = [x for x in context_occupations if len(x) > 4]  # Only include documents with more than 4 characters
    context_occupations = '\n\n'.join(context_occupations)

    chain = initialize_chain()

    # LLM call for skills
    skills_response = chain.invoke({
        "question": input_text,
        "context": context_skills,
        "LLM_MAX_PICKS": LLM_MAX_PICKS[0],
    })

    # LLM call for occupations
    occupations_response = chain.invoke({
        "question": input_text,
        "context": context_occupations,
        "LLM_MAX_PICKS": LLM_MAX_PICKS[1],
    })

    print("Generated Response:")
    print(f"Skills: {skills_response}")
    print(f"Occupations: {occupations_response}")

    response_processed_skills = create_item_relevance_mapping(skills_response)
    response_processed_occupation = create_item_relevance_mapping(occupations_response)

    print(f"Skills mapping: {response_processed_skills}")
    print(f"Occupations mapping: {response_processed_occupation}")

    # Structure docs
    structured_skills = structure_terms(retrieved_skills, response_processed_skills, 'skill/competence')
    print("Structured Skills:", structured_skills)
    structured_occupations = structure_terms(retrieved_occupations, response_processed_occupation, 'occupation')
    print("Structured Occupations:", structured_occupations)

    # Similarity search for suggestions
    similar_skills = get_similar_documents(db, structured_skills, 'skill/competence', VECTORSTORE_MAX_SUGGESTED[0])
    similar_occupations = get_similar_documents(db, structured_occupations, 'occupation', VECTORSTORE_MAX_SUGGESTED[1])


    # Parse and display only the labels
    print("Labels for Similar Skills Suggestions:")
    for doc in similar_skills:
        print(doc['label'])

    print("Labels for Similar Occupations Suggestions:")
    for doc in similar_occupations:
        print(doc['label'])

