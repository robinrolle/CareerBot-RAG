import pandas as pd
import matplotlib.pyplot as plt
import os

# Define BASE_DIR for relative path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define OUTPUT_GRAPH_DIR for saving the plot
OUTPUT_GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, 'bar_graphs'))

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

def create_comparative_plot(combined_df):
    """
    Create a bar chart comparing the average relevant_skills_count and relevant_occupations_count
    across different experiments (sources).
    """
    # Group by 'source' to calculate the average of 'relevant_skills_count' and 'relevant_occupations_count'
    avg_combined = combined_df.groupby('source')[['relevant_skills_count', 'relevant_occupations_count']].mean().reset_index()

    # Renaming the columns to make them more user-friendly for the plot
    avg_combined = avg_combined.rename(columns={
        'relevant_skills_count': 'Skills',
        'relevant_occupations_count': 'Occupations'
    })

    # Calculate the maximum value across both relevant_skills_count and relevant_occupations_count
    max_value = max(avg_combined['Skills'].max(), avg_combined['Occupations'].max())

    # Plot the data
    ax = avg_combined.plot(kind='bar', x='source', figsize=(10, 6), legend=True)

    plt.xlabel('Experiment')
    plt.ylabel('Average Count')
    plt.title('Average Relevant Skills and Occupations Count per Experiment - all CV - Higher is better')
    plt.xticks(rotation=45)

    # Set y-axis limit to the highest average value + 3
    plt.ylim(0, max_value + 3)

    # Save the plot to the output directory
    output_file = os.path.join(OUTPUT_GRAPH_DIR, 'experiment_comparison.png')
    plt.tight_layout()
    plt.savefig(output_file)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Combine data from all experiment files
    combined_data = combine_data_from_experiments()

    # Create a comparative plot based on the combined data
    create_comparative_plot(combined_data)
