from pydantic import BaseModel
import os
import csv
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import List, Tuple
import tiktoken

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'POC', 'evaluations', 'results'))
INPUT_FILENAME = 'Hybrid Search_big_sample_results.csv' # Change filename to evaluate
OUTPUT_FILENAME = 'Hybrid Search_sample_eval.csv' # Change output evaluation filename
INPUT_CSV = os.path.join(RESULTS_DIR, INPUT_FILENAME)
OUTPUT_CSV = os.path.join(RESULTS_DIR, OUTPUT_FILENAME)

# Set the OpenAI API key and LangSmith environment variables
os.environ['OPENAI_API_KEY'] = "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"

# Initialize OpenAI client
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model(JUDGE_MODEL)

class RelevanceEvaluation(BaseModel):
    relevant_count: int

def count_tokens(messages: List[dict]) -> int:
    """Count the number of tokens in the messages."""
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
        num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def evaluate_relevance(cv_text: str, items: List[str], item_type: str) -> Tuple[RelevanceEvaluation, int]:
    system_message = f"""You are an expert CV evaluator. Your task is to determine how many of the given {item_type}s are relevant to the provided CV text.

    Important instructions:
    1. Only consider the {item_type}s provided in the list. Do not invent or suggest any additional {item_type}s.
    2. Each {item_type} item is represented as "label | description". Treat each item as a single, atomic unit.
    3. Base your evaluation solely on the information given in the CV text and the provided {item_type} list.
    """

    user_message = f"""CV Text: {cv_text}

    {item_type}s: {items}

    Evaluate the relevance and provide only the count of relevant {item_type}s. Do not provide anything else."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Count tokens
    token_count = count_tokens(messages)

    # Use OpenAI API for structured parsing
    completion = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=RelevanceEvaluation,
    )

    return completion.choices[0].message.parsed, token_count

def create_output_csv():
    headers = ['embedding_model', 'filename', 'max_retrieved', 'relevant_skills_count',
               'relevant_occupations_count', 'total_tokens_used']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def process_results(input_csv: str, output_csv: str):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    create_output_csv()
    total_tokens = 0

    with open(output_csv, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating relevance"):
            cv_text = row['original_text']
            skills = row['retrieved_skills'].split(',')
            occupations = row['retrieved_occupations'].split(',')

            skills_evaluation, skills_tokens = evaluate_relevance(cv_text, skills, "skill")
            occupations_evaluation, occupations_tokens = evaluate_relevance(cv_text, occupations, "occupation")

            tokens_used = skills_tokens + occupations_tokens
            total_tokens += tokens_used

            writer.writerow([
                row['embedding_model'],
                row['text_file_name'],
                row['max_retrieved'],
                skills_evaluation.relevant_count,
                occupations_evaluation.relevant_count,
                tokens_used
            ])

    return total_tokens

def main():
    print(f"Starting relevance evaluation. Input file: {INPUT_CSV}")
    total_tokens = process_results(INPUT_CSV, OUTPUT_CSV)
    print(f"Evaluation complete. Results saved to {OUTPUT_CSV}")
    print(f"Total tokens used: {total_tokens}")

if __name__ == "__main__":
    main()
