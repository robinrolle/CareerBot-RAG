from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILEPATH = r"C:\Users\Robin\Desktop\final_thesis\CareerBot-RAG\data\CV\CV.pdf"

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

extracted_text = extract_pdf(FILEPATH)
summaries = summarize_pdf(extracted_text)
print(summaries)


# Extraire la liste des compétences pour chaque expérience en tant que chaîne unique
skills_per_experience = {item['experience']: ', '.join(item['summary']) for item in summaries}

# Afficher les compétences par expérience
for experience, skills in skills_per_experience.items():
    print(f"Experience: {experience}")
    print(f"Skills: {skills}")
    print("\n")