import pandas as pd
import os

# Define BASE_DIR for relative path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define OUTPUT_GRAPH_DIR for saving the tables
OUTPUT_GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, 'tables'))

# Ensure the directory exists
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

def load_and_label_csv(file_path, label):
    """
    Load a CSV file and add a 'source' column with the provided label.
    """
    df = pd.read_csv(file_path)
    df['source'] = label
    return df

def combine_data_from_experiments():
    """
    Combine all experiment CSV files into one DataFrame, adding a 'source' column
    for each file to identify the experiment.
    """
    combined_df = pd.DataFrame()

    # Iterate over each CSV file in the directory
    for file_name in os.listdir(BASE_DIR):
        if file_name.endswith("_eval.csv"):  # Only process CSV files
            file_path = os.path.join(BASE_DIR, file_name)

            # Extract the title (first part of the file name)
            experiment_name = file_name.split('_')[0]

            # Load the CSV and add the 'source' column
            df = load_and_label_csv(file_path, experiment_name)

            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def create_comparative_table_by_filename(combined_df):
    """
    Create comparative tables where each row is an embedding model, and each column
    is an experiment (source) showing the average skills and occupations count, with
    a sub-column for 'Skills' and 'Occupations' under each source. A separate table
    is created for each filename (PDF).
    """
    # Get a list of unique filenames
    filenames = combined_df['filename'].unique()

    tables_by_filename = {}

    # Iterate over each filename
    for filename in filenames:
        # Filter the combined data for this filename
        filtered_df = combined_df[combined_df['filename'] == filename]

        # Group by 'embedding_model' and 'source' to get the mean of 'relevant_skills_count' and 'relevant_occupations_count'
        grouped_df = filtered_df.groupby(['embedding_model', 'source'])[['relevant_skills_count', 'relevant_occupations_count']].mean().reset_index()

        # Create pivot tables for skills and occupations under each source
        pivot_skills = grouped_df.pivot(index='embedding_model', columns='source', values='relevant_skills_count')
        pivot_occupations = grouped_df.pivot(index='embedding_model', columns='source', values='relevant_occupations_count')

        # Combine both pivot tables into one with multi-level columns (Skills and Occupations under each source)
        combined_table = pd.concat([pivot_skills, pivot_occupations], axis=1, keys=['Skills', 'Occupations'])

        # Reorder the columns so they are grouped by source with Skills and Occupations next to each other
        combined_table = combined_table.swaplevel(axis=1).sort_index(axis=1)

        # Store the table for this filename
        tables_by_filename[filename] = combined_table

    return tables_by_filename

def save_comparative_tables(tables_by_filename):
    """
    Save each comparative table to a CSV file in the OUTPUT_GRAPH_DIR directory.
    """
    for filename, table in tables_by_filename.items():
        # Create a valid filename by replacing problematic characters
        safe_filename = filename.replace(".pdf", "").replace(" ", "_") + '_comparative_table.csv'

        # Define the output file path in OUTPUT_GRAPH_DIR
        output_file = os.path.join(OUTPUT_GRAPH_DIR, safe_filename)

        # Save the table to CSV
        table.to_csv(output_file)

        print(f"Comparative table for {filename} saved to {output_file}")

if __name__ == "__main__":
    # Combine data from all experiment files
    combined_data = combine_data_from_experiments()

    # Create comparative tables based on the combined data, one for each filename
    tables_by_filename = create_comparative_table_by_filename(combined_data)

    # Save the comparative tables to CSV files in the OUTPUT_GRAPH_DIR
    save_comparative_tables(tables_by_filename)
