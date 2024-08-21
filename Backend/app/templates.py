from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt template
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an assistant tasked with extracting and summarizing professional experiences from a given CV. 
    Your goal is to identify and separate individual experiences from the provided text, focusing specifically on the content and skills demonstrated, 
    while excluding any dates, durations, or company names. Each experience should be summarized clearly and concisely, 
    highlighting the key skills and responsibilities. The final output should be formatted as JSON, with each experience and its corresponding summary 
    presented as asked.
    """
)

# Main prompt template
main_prompt = HumanMessagePromptTemplate.from_template(
    """
    Extract and separate each experience written in the following CV text.
    Focus on the experiences content itself, and exclude any dates, durations, and company names.
    Summarize each experience. Focus on describing the skills demonstrated in each experience.
    Each summary should be in JSON format as shown below:
    {{
        "summaries": [
            {{
                "experience": "Title experience 1",
                "summary": ["summarized text experience 1"]
            }},
            {{
                "experience": "Title experience 2",
                "summary": ["summarized text experience 2"]
            }}
        ]
    }}

    CV Text:
    {cv_text}
    """
)

# Full prompt ready to be used
full_prompt = ChatPromptTemplate.from_messages([system_prompt, main_prompt])

full_prompt_text = """
You are an assistant tasked with extracting and summarizing professional experiences from a given CV. Your goal is to identify and separate individual experiences from the provided text, focusing specifically on the content and skills demonstrated, while excluding any dates, durations, or company names. Each experience should be summarized clearly and concisely, highlighting the key skills and responsibilities. The final output should be formatted as JSON, with each experience and its corresponding summary presented as asked.

Extract and separate each experience written in the following CV text. Focus on the experiences content itself, and exclude any dates, durations, and company names. Summarize each experience. Focus on describing the skills demonstrated in each experience. Each summary should be in JSON format as shown below:

{
    "summaries": [
        {
            "experience": "Title experience 1",
            "summary": ["summarized text experience 1"]
        },
        {
            "experience": "Title experience 2",
            "summary": ["summarized text experience 2"]
        }
    ]
}

"""

