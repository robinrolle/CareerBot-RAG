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
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb
from POC.retrieval.prompts import templates
from langsmith import traceable
from langsmith import Client

# Set the OpenAI API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CareerBot"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_API_KEY'] = "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"

# Initialize LangSmith
langsmith = Client()

# Define the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data','processed_data', 'ESCO_embeddings'))
PDF_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data','CVs', 'accountant.pdf'))

# Models config
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"

# Configuration Constants
VECTORSTORE_MAX_RETRIEVED = 25 # Max number of documents to retrieve from vectorstore. Don't go over context window
VECTORSTORE_MAX_SUGGESTED = [20, 10] # for [skills, occupations] how many potential items to suggest+ from vectorstore
LLM_MAX_PICKS = [15, 5] # for [skills, occupations] how many items LLM must pick from retrieved options

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

@traceable
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

@traceable
def init_chroma_db(db_collection_name, embedding_function):
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    try:
        db = Chroma(client=chroma_client, collection_name=db_collection_name, embedding_function=embedding_function)
        print("Database loaded and ready")
    except Exception as e:
        print(f"Failed to load the database: {e}")
        raise
    return db

@traceable
def query_documents(db, input_embedding, doc_type, max_retrieved):
    retrieved_docs = db._collection.query(
        query_embeddings=input_embedding,
        n_results=max_retrieved,
        include=['embeddings', 'documents', 'metadatas'],
        where={"type": doc_type}
    )
    return retrieved_docs

@traceable
def format_skills_for_prompt(skills_docs):
    formatted_skills = []
    for doc in skills_docs:
        item_description = f"{doc.page_content} | {doc.page_content}"
        formatted_skills.append(item_description)
    return "\n".join(formatted_skills)

@traceable
def format_occupations_for_prompt(occupation_docs):
    formatted_occupations = []
    for doc in occupation_docs:
        item_description = f"{doc.page_content} | {doc.page_content}"
        formatted_occupations.append(item_description)
    return "\n".join(formatted_occupations)

@traceable
def initialize_chain():
    # Create the prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(templates.MY_PROMPT_TEMPLATE)

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

@traceable
def map_and_structure_terms(retrieved_docs, answer, doc_type):
    # Check response format
    assert ('items' in answer) and ('relevances' in answer), 'Response not in correct format!'

    # Adjust relevance length to meet item length
    di = len(answer['relevances']) - len(answer['items'])
    if di > 0:
        answer['relevances'] = answer['relevances'][:len(answer['items'])]
    elif di < 0:
        answer['relevances'] = answer['relevances'] + ['LOW'] * (-di)

    # Delete doubles
    df_items = pd.DataFrame(answer['items'], columns=['item'])
    idx = df_items[df_items.duplicated(keep=False)].index.tolist()

    answer['relevances'] = [x.strip() for k, x in enumerate(answer['relevances']) if k not in idx]
    answer['items'] = [x.strip() for k, x in enumerate(answer['items']) if k not in idx]

    # Map item with relevance
    relevance_mapping = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    item_relevance_map = {answer['items'][i]: relevance_mapping[answer['relevances'][i]] for i in range(len(answer['items']))}

    # Structure docs
    structured_docs = []
    for i, doc in enumerate(retrieved_docs['documents'][0]):
        for term in item_relevance_map.keys():
            labels = doc.split('|')[0]
            if term in labels:
                structured_docs.append({
                    'embedding': retrieved_docs['embeddings'][0][i],
                    'document': doc,
                    'relevance': item_relevance_map[term],
                    'type': doc_type,
                    'label': labels.strip()
                })
                break

    return structured_docs

@traceable
def clean_and_prepare_context(retrieved_docs):
    context = [x.replace('\n', ' ').replace('\r', '') for x in retrieved_docs['documents'][0]]
    context = [x for x in context if len(x) > 4]  # Only include documents with more than 4 characters
    context = '\n\n'.join(context)
    return context

@traceable
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

    # Select final docs
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

@traceable
def extract_pdf(pdf_path):
    """Return extracted text from PDF File"""
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    return " ".join(page.page_content for page in data)

@traceable(name="Skills LLM Call")
def invoke_skills_chain(chain, input_text, context_skills):
    return chain.invoke({
        "question": input_text,
        "context": context_skills,
        "LLM_MAX_PICKS": LLM_MAX_PICKS[0],
    })

@traceable(name="Occupations LLM Call")
def invoke_occupations_chain(chain, input_text, context_occupations):
    return chain.invoke({
        "question": input_text,
        "context": context_occupations,
        "LLM_MAX_PICKS": LLM_MAX_PICKS[1],
    })


@traceable(name=f"Trace: {GENERATION_MODEL_NAME} + {EMBEDDING_MODEL_NAME}")
def main():
    # Init embedding functiion
    embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    collection_name = generate_valid_collection_name(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    db = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embedding_ef)


    #Exemple user input
    input_text = extract_pdf(PDF_PATH)

    # Turn input into embedding
    input_embedding = embedding_ef.embed_query(text=input_text)

    # Retrieve documents by querying the DB
    retrieved_skills = query_documents(db, input_embedding, 'skill/competence', VECTORSTORE_MAX_RETRIEVED)
    retrieved_occupations = query_documents(db, input_embedding, 'occupation', VECTORSTORE_MAX_RETRIEVED)


    # Clean and prepare the contexts for the prompt
    context_skills = clean_and_prepare_context(retrieved_skills)
    context_occupations = clean_and_prepare_context(retrieved_occupations)

    chain = initialize_chain()

    skills_response = invoke_skills_chain(chain, input_text, context_skills)
    occupations_response = invoke_occupations_chain(chain, input_text, context_occupations)

    # Map and structure the terms for skills and occupations / Formatting data
    structured_skills = map_and_structure_terms(retrieved_skills, skills_response, 'skill/competence')
    structured_occupations = map_and_structure_terms(retrieved_occupations, occupations_response, 'occupation')

    # Find suggestions based on LLM answer
    similar_skills = get_similar_documents(db, structured_skills, 'skill/competence', VECTORSTORE_MAX_SUGGESTED[0])
    similar_occupations = get_similar_documents(db, structured_occupations, 'occupation', VECTORSTORE_MAX_SUGGESTED[1])


if __name__ == "__main__":
    main()
