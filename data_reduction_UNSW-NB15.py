"""
Script Name: sample_unsw_nb15.py

Description:
    This script takes an existing CSV (preferably the cleaned one),
    and randomly samples 50% of the rows to reduce the dataset size.
    The sample is saved to a new CSV.
"""

import pandas as pd

def sample_dataset(input_file, output_file, fraction=0.5, random_state=42):
    # 1. Load Dataset (could be the cleaned file from step 1)
    df = pd.read_csv(input_file)
    print("Original Data Shape:", df.shape)

    # 2. Randomly sample a fraction of the dataset (in this case 50%)
    df_sample = df.sample(frac=fraction, random_state=random_state)
    print(f"Sampled Data Shape (frac={fraction}):", df_sample.shape)

    # 3. Save the sampled dataset
    df_sample.to_csv(output_file, index=False)
    print(f"Sampled dataset saved to {output_file}")

if __name__ == "__main__":
    input_path = "UNSW-NB15_cleaned.csv"  # or "UNSW-NB15.csv"
    output_path = "UNSW-NB15_half.csv"
    sample_dataset(input_path, output_path, fraction=0.5, random_state=42)
