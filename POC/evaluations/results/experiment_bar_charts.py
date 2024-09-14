import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Define BASE_DIR for relative path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, 'bar_graphs'))

def ensure_output_dir_exists():
    if not os.path.exists(OUTPUT_GRAPH_DIR):
        os.makedirs(OUTPUT_GRAPH_DIR)

def save_graph(plt, title):
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(OUTPUT_GRAPH_DIR, filename)
    plt.savefig(filepath)
    print(f"Graph saved as {filepath}")

def show_graph_bar_skills_count(file, title_prefix):
    # Read the CSV file
    df_embeddings_evaluation = pd.read_csv(file)

    # Group by 'embedding_model' and 'filename' to get the mean of 'relevant_skills_count'
    avg_relevant_skills_embeddings = df_embeddings_evaluation.groupby(['embedding_model', 'filename'])[
        'relevant_skills_count'].mean().reset_index()

    # Pivot data for bar chart
    avg_relevant_skills_pivot_embeddings = avg_relevant_skills_embeddings.pivot(index='embedding_model',
                                                                                columns='filename',
                                                                                values='relevant_skills_count')

    # Create the bar plot
    ax1_embeddings = avg_relevant_skills_pivot_embeddings.plot(kind='bar', figsize=(12, 6))

    # Set axis labels and title
    plt.xlabel('Embedding Model')
    plt.ylabel('Relevant Skills Count')
    title = f'{title_prefix} - Relevant Skills Count by Embedding Model per CV'
    plt.title(title)
    plt.xticks(rotation=45)

    # Set y-axis limit dynamically based on the max value in the dataset
    plt.ylim(0, avg_relevant_skills_embeddings['relevant_skills_count'].max() + 3)

    # Move the legend to the right
    ax1_embeddings.legend(title="Filename", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    save_graph(plt, title)

    # Clear the current figure
    plt.clf()

def show_graph_bar_occupations_count(file, title_prefix):
    # Read the CSV file
    df_embeddings_evaluation = pd.read_csv(file)

    # Group by 'embedding_model' and 'filename' to get the mean of 'relevant_occupations_count'
    avg_relevant_occupations_embeddings = df_embeddings_evaluation.groupby(['embedding_model', 'filename'])[
        'relevant_occupations_count'].mean().reset_index()

    # Pivot data for bar chart
    avg_relevant_occupations_pivot_embeddings = avg_relevant_occupations_embeddings.pivot(index='embedding_model',
                                                                                          columns='filename',
                                                                                          values='relevant_occupations_count')

    # Create the bar plot
    ax2_embeddings = avg_relevant_occupations_pivot_embeddings.plot(kind='bar', figsize=(12, 6))

    # Set axis labels and title
    plt.xlabel('Embedding Model')
    plt.ylabel('Count Relevant Occupation')
    title = f'{title_prefix} - Relevant Occupations by Embedding Model per CV'
    plt.title(title)
    plt.xticks(rotation=45)

    # Set y-axis limit dynamically based on the max value in the dataset
    plt.ylim(0, avg_relevant_occupations_embeddings['relevant_occupations_count'].max() + 5)

    # Move the legend to the right
    ax2_embeddings.legend(title="Filename", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    save_graph(plt, title)

    # Clear the current figure
    plt.clf()

def main():
    ensure_output_dir_exists()

    # Iterate over each CSV file in the directory
    for file_name in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, file_name)

        if file_name.endswith("_eval.csv"):  # Only process CSV files
            print(f"Processing file: {file_name}")

            # Extract the first part of the file name before "_eval.csv"
            title_prefix = file_name.split('_')[0]

            # Show bar graphs for skills count and occupations count with title prefix
            show_graph_bar_skills_count(file_path, title_prefix)
            time.sleep(2)  # Wait for 2 seconds between plots
            show_graph_bar_occupations_count(file_path, title_prefix)

if __name__ == "__main__":
    main()