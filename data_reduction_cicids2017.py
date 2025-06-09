import pandas as pd

# Load the dataset (replace 'your_dataset.csv' with your file path)
file_path = r'C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\cleaned-CICIDS2017-02.csv'

# Load only a sample of the data to cut down its size
chunk_size = 100000  # Define the chunk size for processing
chunks = []

# Read the CSV in chunks and append half the rows from each chunk to a list. CHANGE THE FRACTION (or frac) SIZE TO less than 1.0 (or less than 100%)
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    half_chunk = chunk.sample(frac=0.5, random_state=42)
    chunks.append(half_chunk)

# Concatenate the half-size chunks to form the final reduced dataframe
reduced_df = pd.concat(chunks)

# Save the reduced dataframe to a new CSV
reduced_file_path = r'C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\reduced_and_cleaned_CICIDS2017-02.csv'
reduced_df.to_csv(reduced_file_path, index=False)
