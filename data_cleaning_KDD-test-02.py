import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

def remove_control_chars(x):
    """
    Removes non-printable/control characters from a string.
    If x is not a string, returns it unchanged.
    """
    if isinstance(x, str):
        return "".join(ch for ch in x if ch.isprintable())
    return x

def clean_kdd_test(input_file, output_file, label_column='label', drop_columns=['difficulty'], k_features=20):
    """
    Cleans KDDTest+ data without affecting the label column.
    
    Steps:
    1. Loads data with predefined column names.
    2. Drops any specified columns (e.g. 'difficulty').
    3. Cleans only the feature columns:
       - Removes control characters from object-type columns.
       - Attempts numeric conversion for object columns.
       - Replaces infinite values with NaN and fills missing values with 0.
       - Converts any remaining categorical (object) features to numeric codes.
    4. Separates features and label.
    5. Applies Chi-Square feature selection (top k features).
    6. Label-encodes the target column.
    7. Saves the cleaned dataset.
    """
    kdd_cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty'
    ]
    
    # Load dataset (no header row; assign columns)
    df = pd.read_csv(input_file, header=None, names=kdd_cols)
    print("Initial shape (KDDTest+):", df.shape)
    
    # Drop any unwanted columns (e.g., 'difficulty')
    for col in drop_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Determine feature columns (all except the label)
    feature_columns = [col for col in df.columns if col != label_column]
    
    # Clean object-type feature columns: remove control characters
    object_cols = df[feature_columns].select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].applymap(remove_control_chars)
    
    # Attempt conversion of object columns to numeric where possible
    for col in feature_columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Replace infinite values with NaN and fill missing values in feature columns
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Convert any remaining categorical (object-type) features to numeric codes
    for col in feature_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    # Separate features and label
    X = df[feature_columns]
    y = df[label_column]
    
    # Apply Chi-Square feature selection (chi2 expects non-negative features)
    k = min(k_features, X.shape[1])
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Retrieve the original names of the selected features
    selected_mask = selector.get_support()
    selected_feature_names = X.columns[selected_mask]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
    
    # Encode the label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    y_encoded_df = pd.DataFrame(y_encoded, columns=[label_column])
    print("Label encoding mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
    
    # Combine features and label, then save
    final_df = pd.concat([X_selected_df, y_encoded_df], axis=1)
    final_df.to_csv(output_file, index=False)
    print(f"Cleaned KDDTest+ saved to {output_file}")
    return final_df

if __name__ == "__main__":
    test_input = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\KDDTest+.txt"
    test_output = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\cleaned_KDDTest-02.csv"
    clean_kdd_test(test_input, test_output)
