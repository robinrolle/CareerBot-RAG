import os
import re
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
GENERATION_MODEL_NAME = "gpt-4o"

# Configuration Constants
VECTORSTORE_MAX_RETRIEVED = 25 # Max number of documents to retrieve from vectorstore. Don't go over context window
VECTORSTORE_MAX_SUGGESTED = [20, 10] # for [skills, occupations] how many potential items to suggest from vectorstore
LLM_MAX_PICKS = [15, 5] # for [skills, occupations] how many items LLM must pick from retrieved options

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"

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

if __name__ == "__main__":
    embedding_ef = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    collection_name = generate_valid_collection_name(EMBEDDING_MODEL_NAME)
    chroma_db = init_chroma_db(collection_name, embedding_ef)

    skills_retriever = get_retriever(db=chroma_db, max_retrieved=VECTORSTORE_MAX_SUGGESTED[0], doc_type="skill/competence")
    occupation_retriever = get_retriever(db=chroma_db, max_retrieved=VECTORSTORE_MAX_SUGGESTED[1], doc_type="occupation")

    # Example of user input text
    input_text = """
    I have worked last 5 years as an contractor that sells and installs heat pumps for consumers. 
    I have engineering degree and know lots of refrigerants, and technology of heating and cooling systems. 
    I have licence to do electrical jobs."""

    selected_skills = skills_retriever.invoke(input=input_text)
    selected_occupations = occupation_retriever.invoke(input=input_text)

    context_skills = format_skills_for_prompt(selected_skills)
    context_occupations = format_occupations_for_prompt(selected_occupations)

    # Initialize the LLM chain
    chain = initialize_chain()

    # Generate the final response
    response = chain.invoke({
        "question": input_text,
        "context": context_skills,
        "LLM_MAX_PICKS": LLM_MAX_PICKS[0],
    })

    print("Generated Response:")
    print(response)
