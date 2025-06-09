import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def remove_control_chars(x):
    """
    Removes non-printable/control characters from a string.
    If x is not a string, returns it unchanged.
    """
    if isinstance(x, str):
        return "".join(ch for ch in x if ch.isprintable())
    return x

def clean_unsw_nb15(input_file, output_file, label_columns=None, fill_value_numeric=0, fill_value_non_numeric=""):
    """
    Cleans the UNSW‑NB15 dataset without affecting label columns, then label-encodes
    the "label" column according to the thesis standard.
    
    Parameters:
    - input_file: Path to the raw UNSW‑NB15 CSV.
    - output_file: Path where the cleaned CSV will be saved.
    - label_columns: List of columns that should not be modified. Defaults to ['attack_cat', 'label'].
    - fill_value_numeric: Value to fill missing numeric entries (default: 0).
    - fill_value_non_numeric: Value to fill missing non‑numeric entries (default: empty string).
    
    Returns:
    - The cleaned DataFrame.
    """
    # 1. Define Column Names (39 columns typical for UNSW‑NB15)
    unsw_col_names = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
        "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
        "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
        "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "attack_cat", "label"
    ]
    
    # 2. Load the dataset with assigned column names
    df = pd.read_csv(input_file, header=None, names=unsw_col_names)
    print("Initial Data Shape:", df.shape)
    print("Columns in the dataset:\n", df.columns)
    
    # 3. Define label columns if not provided
    if label_columns is None:
        label_columns = ['attack_cat', 'label']
    
    # 4. Clean only feature columns (i.e. those not in label_columns)
    for col in df.columns:
        if col not in label_columns:
            # a) If the column is of object type, remove control characters.
            if df[col].dtype == 'object':
                df[col] = df[col].apply(remove_control_chars)
            # b) Replace infinite values with NaN.
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # c) Fill missing values based on data type.
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(fill_value_numeric)
            else:
                df[col] = df[col].fillna(fill_value_non_numeric)
    
    print("Shape after cleaning features:", df.shape)
    
    # 5. (Optional) Additional cleaning such as outlier removal could be added here.
    
    # 6. Label encode the "label" column (following the thesis standard)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'].astype(str))
    print("Label encoding mapping (UNSW-NB15 'label'):", dict(zip(le.classes_, range(len(le.classes_)))))
    
    # 7. Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"Cleaned UNSW-NB15 dataset saved to {output_file}")
    
    return df

# Example usage:
if __name__ == "__main__":
    input_path = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\UNSW-NB15.csv"
    output_path = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\cleaned_UNSW-NB15-02.csv"
    # Specify label columns to protect them from cleaning (adjust if needed)
    label_columns = ['attack_cat', 'label']
    
    cleaned_df = clean_unsw_nb15(input_path, output_path, label_columns=label_columns)
