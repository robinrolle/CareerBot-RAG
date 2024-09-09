import os
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs'))
TEXT_DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CVs', 'text_datasets'))
DATASET = os.path.join(TEXT_DATASET_DIR, 'cv_extracts.csv')

# Check if the PDF directory exists
if not os.path.exists(PDF_DIR):
    print(f"The PDF directory {PDF_DIR} does not exist.")
    os.makedirs(PDF_DIR)
    print(f"The directory {PDF_DIR} has been created.")

# Check if the dataset directory exists, otherwise create it
if not os.path.exists(TEXT_DATASET_DIR):
    os.makedirs(TEXT_DATASET_DIR)
    print(f"The dataset directory {TEXT_DATASET_DIR} has been created.")

def extract_pdf(pdf_path):
    """Returns the extracted text from a PDF file"""
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    return " ".join(page.page_content for page in data)

# List to store the extracted data
extracted_data = []

# Loop through all PDF files in the directory
for filename in os.listdir(PDF_DIR):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(PDF_DIR, filename)
        cv_text = extract_pdf(pdf_path)
        extracted_data.append({'Filename': filename, 'Extracted Text': cv_text})

# Save the extracted data to a CSV file
df = pd.DataFrame(extracted_data)
df.to_csv(DATASET, index=False)

print(f"Extraction completed and saved to {DATASET}!")
