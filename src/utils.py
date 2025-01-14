import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data as needed (e.g., cleaning, tokenization)."""
    # Example: Remove any rows with missing values
    data = data.dropna()
    return data