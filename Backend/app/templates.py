from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


# Step 2: Define the system and human prompt templates
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an expert in analyzing CVs and accurately extracting {doc_type} from them.
    These {doc_type} may be explicitly mentioned or may require careful analysis of the candidate's education, work experience, and other relevant sections of the CV.
    Your task is to analyze the provided CV text and extract the relevant {doc_type}.
    Use only name {doc_type} that are part of the ESCO (European Skills, Competences, Qualifications and Occupations) dataset.
    Provide your response in JSON format, strictly following the structure below:
     {{
            "items": [
                {{
                    "name": "{doc_type} name"
                }},
                ...
            ]
        }}
    Do not include any additional explanations or comments outside of this format.
    """
)

human_prompt = HumanMessagePromptTemplate.from_template(
    """ CV Text: {cv_text} """
)

# Step 3: Combine the system and human prompts into a full chat prompt template
full_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
