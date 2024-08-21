from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import re

# Setup constants and initialize logging
MODEL_NAME = "intfloat/e5-large-v2"
DATABASE_DIR = "C:\\Users\\Robin\\Desktop\\final_thesis\\CareerBot-RAG\\data\\processed_data\\ESCO_embeddings"
COLLECTION_NAME = re.sub(r'[^a-zA-Z0-9_-]', '_', MODEL_NAME)[:63]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILEPATH = r"C:\Users\Robin\Desktop\final_thesis\CareerBot-RAG\data\CV\CV.pdf"

def init_chroma_db():
    """Initialize the Chroma database and load the collection."""
    model = SentenceTransformer(MODEL_NAME)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logger.info("Database loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load the database: {e}")
        raise

    return collection, model

def similarity_search(query_text: str, collection, model, top_k: int = 5, filter_type: str = None) -> List[str]:
    """Perform a similarity search on the collection using the provided query text."""
    try:
        query_embedding = model.encode(query_text).tolist()
        filter_query = {'type': filter_type} if filter_type else {}
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_query
        )
        documents = results['documents'][0] if 'documents' in results else []
        return documents
    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        return []

def similarity_search_skills(query_text: str, collection, model, top_k: int = 5) -> List[str]:
    """Search for skills in the collection."""
    return similarity_search(query_text, collection, model, top_k, filter_type='skill/competence')

def similarity_search_occupations(query_text: str, collection, model, top_k: int = 5) -> List[str]:
    """Search for occupations in the collection."""
    return similarity_search(query_text, collection, model, top_k, filter_type='occupation')

def extract_pdf(file_path: str) -> str:
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    extracted_text_pdf = ""
    for page in data:
        extracted_text_pdf += page.page_content
    return extracted_text_pdf

def summarize_pdf(extracted_text: str) -> List[dict]:
    examples = [
        {
            "experience": "Software Developer at XYZ Corp. Developed web applications using Python and JavaScript. Improved system performance by 20%.",
            "summary": ["Software Developer", "web development", "Python", "JavaScript", "system performance improvement"]
        },
        {
            "experience": "Data Analyst at ABC Inc. Analyzed large datasets to provide business insights. Utilized SQL and Excel for data manipulation.",
            "summary": ["Data Analyst", "data analysis", "SQL", "Excel", "business insights"]
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("Experience: {experience}"),
        AIMessagePromptTemplate.from_template("Summary: {summary}")
    ])

    few_shot_prompt = []
    for example in examples:
        few_shot_prompt.extend(example_prompt.format_messages(**example))

    main_prompt = HumanMessagePromptTemplate.from_template(
        """
        Extract and separate each experience written in the following CV text.
        Summarize each experience. Focus on describing the skills demonstrated in each experience.
        Be generic and summarize by listing some keywords using the ESCO skills dataset.
        Each summary should be in JSON format as shown below:
        {{
            "summaries": [
                {{
                    "experience": "Title experience 1",
                    "summary": ["list of skills"]
                }},
                {{
                    "experience": "Title experience 2",
                    "summary": ["list of skills"]
                }}
            ]
        }}

        CV Text:
        {cv_text}
        """
    )

    full_prompt = ChatPromptTemplate.from_messages(few_shot_prompt + [main_prompt])
    
    inputs = {"cv_text": extracted_text}
    model = ChatOllama(model="llama3.1", temperature=0.0)

    response = model(full_prompt.format_messages(**inputs))

    parser = JsonOutputParser()
    try:
        llm_response = parser.parse(response.content)
        summaries = llm_response.get('summaries', [])
    except Exception as e:
        logger.error(f"Failed to parse LLM summary response: {e}")
        summaries = []

    return summaries

def extract_labels(documents: List[str]) -> List[str]:
    """Extract labels from the list of documents."""
    return [doc.split('|')[0].strip() for doc in documents]

# Initialize the database and model
chroma_collection, sentence_model = init_chroma_db()

# Extraire le texte du PDF et le résumer
extracted_text = extract_pdf(FILEPATH)
summaries = summarize_pdf(extracted_text)

# Sets pour stocker les compétences et les professions uniques
unique_skills = set()
unique_occupations = set()

# Effectuer des recherches de similarité pour chaque liste de compétences
for summary in summaries:
    experience = summary['experience']
    skills_list = summary['summary']
    skills_query = ', '.join(skills_list)
    
    retrieved_skills = similarity_search_skills(skills_query, chroma_collection, sentence_model, top_k=7)
    retrieved_occupations = similarity_search_occupations(skills_query, chroma_collection, sentence_model, top_k=2)
    
    # Extraire les labels et les ajouter aux ensembles
    unique_skills.update(extract_labels(retrieved_skills))
    unique_occupations.update(extract_labels(retrieved_occupations))
    
    print(f"Experience: {experience}")
    print(f"Skills: {skills_query}")
    print(f"Retrieved skills: {retrieved_skills}")
    print(f"Retrieved occupations: {retrieved_occupations}")
    print("\n")

# Afficher les ensembles de compétences et professions uniques
print("Unique Skills:", unique_skills)
print("Unique Occupations:", unique_occupations)
