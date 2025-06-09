import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Define columns as per KDD datasets
kdd_cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

def remove_control_chars(x):
    """ Removes non-printable/control characters from a string. """
    if isinstance(x, str):
        return "".join(ch for ch in x if ch.isprintable())
    else:
        return x

def clean_and_merge(train_input, test_input, drop_columns, output_file, k_features=20):
    # Load data
    df_train = pd.read_csv(train_input, names=kdd_cols, header=None)
    df_test = pd.read_csv(test_input, names=kdd_cols, header=None)

    # Combine train and test datasets
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Drop unwanted columns
    df_combined.drop(columns=drop_columns, inplace=True)

    # Remove control characters
    df_combined = df_combined.applymap(remove_control_chars)

    # Convert categorical columns to numeric
    categorical_cols = df_combined.select_dtypes(include=['object']).columns.drop('label')
    df_combined = pd.get_dummies(df_combined, columns=categorical_cols)

    # Replace infinite and missing values
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.fillna(0, inplace=True)

    # Separate features and label
    X = df_combined.drop('label', axis=1)
    y = df_combined['label']

    # Chi-square feature selection
    selector = SelectKBest(chi2, k=k_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]

    # Label encode the target column and show mapping
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Label encoding mapping:")
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(label_mapping)

    # Combine features and encoded label
    df_final = pd.DataFrame(X_selected, columns=selected_features)
    df_final['label'] = le.transform(y)

    # Save to file
    df_final.to_csv(output_file, index=False)
    print(f"Combined cleaned dataset saved to {output_file}")

if __name__ == '__main__':
    train_input = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\KDDTrain+.txt"
    test_input = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\KDDTest+.txt"
    drop_columns = ['difficulty']
    output_file = r"C:\Users\wafi\OneDrive - National University of Singapore\2025-2026 FYP\Test Environment Final\combined_cleaned_KDD.csv"

    clean_and_merge(train_input, test_input, drop_columns, output_file, k_features=20)
