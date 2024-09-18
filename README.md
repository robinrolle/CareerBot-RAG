# CareerBot-RAG

CareerBot-RAG is a project that analyzes CVs and provides skill and occupation suggestions based on the ESCO dataset.

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm 6+

## Installation

### Backend

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Frontend

1. Navigate to the UI directory:
   ```
   cd UI
   ```

2. Install npm dependencies:
   ```
   npm install
   ```

## Configuration

1. OpenAI API key:
   - The project uses a `.env` file in the Backend directory to store the OpenAI API key.
   - Create or edit the `.env` file:
     ```
     OPENAI_API_KEY=your_new_api_key_here
     ```
   - Make sure not to commit this file to version control to keep your API key secure.

2. Constants:
   The project uses several important constants that can be found in the `Backend/app/config.py` file:

   - `EMBEDDING_MODEL_NAME`: The OpenAI model used for generating embeddings (default: "text-embedding-3-small")
   - `GENERATION_MODEL_NAME`: The OpenAI model used for text generation (default: "gpt-4o-mini")
   - `NUMBER_DOC_PER_ITEM`: Number of documents retrieved for each item from the vector database. The higher the more items will be given to the context of the LLM picking step (default: 1). 
   - `LLM_MAX_PICKS`: Maximum number of items identified from the CV shown to the user [skills, occupations] (default: [15, 5])
   - `NB_SUGGESTED_SKILLS`: Number of skills suggested to the user (default: 20)
   - `NB_SUGGESTED_OCCUPATIONS`: Number of occupations suggested to the user (default: 10)

   You can modify these constants in the `config.py` file to adjust the behavior of the application. For example:

   ```python
   EMBEDDING_MODEL_NAME = "text-embedding-3-large"
   NB_SUGGESTED_SKILLS = 15
   ```

   Note: Changing some of these constants (especially `EMBEDDING_MODEL_NAME`) may require regenerating the FAISS index.
   
## FAISS Index

### Creating a New FAISS Index (Optional)


1. Update the `EMBEDDING_MODEL_NAME` in `Backend/app/config.py` to your desired model. ( text-embedding-3-small by default and recommanded )
2. Navigate to the indexing directory:
   ```
   cd POC/indexing
   ```
3. Run the FAISS index creation script:
   ```
   python create_FAISS_index.py
   ```

This will create new FAISS index files in the `data/processed_data/FAISS_index` directory.

4. After creating the new index, you must run the script to update the options list:
   ```
   cd ../../utils
   python creating_options_list.py
   ```

This script generates updated JSON files for skills and occupations options, which are used by the frontend.

Note: Creating a new index can be time-consuming, costs money and may require significant computational resources, especially for larger models.

## Running the Application

### Backend

1. Navigate to the Backend directory:
   ```
   cd Backend
   ```

2. Start the FastAPI server:
   ```
   uvicorn app.main:app --reload
   ```

The backend will be available at `http://localhost:8000`.

### Frontend

1. Navigate to the UI directory:
   ```
   cd UI
   ```

2. Start the Next.js development server:
   ```
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`.

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Upload a CV file (PDF format)
3. Click "Analyze" to process the CV
4. View the suggested skills and occupations based on the CV content

## Additional Information

### File Structure
- `Backend/`: Contains the FastAPI backend
- `UI/`: Contains the Next.js frontend
- `POC/`: Proof of concept scripts and experiments
- `data/`: Data files and processed indexes


### Notes
- This work was done as a project for a Haaga-Helia thesis
- This project uses Open AI's model. each call costs money. Ensure you have sufficient OpenAI API credits
- The application uses the ESCO (European Skills, Competences, Qualifications and Occupations) dataset
- The API key shown have been revoked
