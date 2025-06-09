import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

def remove_control_chars(x):
    """
    Removes non-printable/control characters from a string.
    If x is not a string, returns it unchanged.
    """
    if isinstance(x, str):
        return "".join(ch for ch in x if ch.isprintable())
    return x

def clean_cicids2017(file_path, output_path, label_columns=None, outlier_threshold=3):
    """
    Cleans the CICIDS2017 dataset without affecting label columns,
    then label-encodes the label column following the thesis standard.
    
    Parameters:
    - file_path: Path to the raw dataset CSV.
    - output_path: Path where the cleaned CSV will be saved.
    - label_columns: List of columns that are labels. These columns will be excluded from cleaning.
    - outlier_threshold: Z-score threshold for outlier detection (default=3).
    
    Returns:
    - The cleaned DataFrame.
    """
    # Load the dataset and strip whitespace from column names
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace
    print("Initial Data Shape:", data.shape)
    print("\nColumn Info:")
    print(data.info())
    
    # If label_columns is not provided, assume none should be excluded
    if label_columns is None:
        label_columns = []
    
    # Identify feature columns (exclude label columns)
    feature_columns = [col for col in data.columns if col not in label_columns]
    
    # 1. Remove control characters from object columns in features only
    obj_cols = data[feature_columns].select_dtypes(include=['object']).columns
    data[obj_cols] = data[obj_cols].applymap(remove_control_chars)
    
    # 2. Convert feature columns that look numeric to numeric
    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')
    
    # 3. Handle infinite values and missing values across the entire DataFrame
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    
    print("\nShape after cleaning missing/infinite values:", data.shape)
    
    # 4. Outlier detection on numeric features only (excluding label columns)
    numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns
    z_scores = data[numeric_features].apply(zscore)
    outliers = (z_scores.abs() > outlier_threshold).any(axis=1)
    print(f"\nNumber of rows identified as outliers (|Z|>{outlier_threshold}):", outliers.sum())
    
    # 5. Remove rows with outliers (features only; label columns remain unchanged)
    data_cleaned = data[~outliers].copy()
    print("Shape after outlier removal:", data_cleaned.shape)
    
    # --- Label Encoding Step ---
    # Look for a column matching 'label' in a case-insensitive manner.
    label_col = None
    for col in data_cleaned.columns:
        if col.lower() == 'label':
            label_col = col
            break

    if label_col is None:
        raise KeyError("No column with name 'label' (case-insensitive) found in the dataset.")
    else:
        le = LabelEncoder()
        data_cleaned[label_col] = le.fit_transform(data_cleaned[label_col].astype(str))
        print("Label encoding mapping (CICIDS2017):", dict(zip(le.classes_, range(len(le.classes_)))))
    
    # Save the cleaned dataset
    data_cleaned.to_csv(output_path, index=False)
    print(f"\nCleaned CICIDS2017 dataset saved as '{output_path}'")
    
    return data_cleaned

# Example usage:
file_path = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\CICIDS2017.csv"
output_path = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\cleaned-CICIDS2017-02.csv"

# Specify the label column(s); here we assume the intended label column is "Label"
label_columns = ['Label']
cleaned_data = clean_cicids2017(file_path, output_path, label_columns=label_columns)
