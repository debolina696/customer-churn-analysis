import pandas as pd

def load_raw_data(path):
    """Load raw churn dataset from CSV."""
    return pd.read_csv(path)

def save_processed_data(df, path):
    """Save processed dataset to CSV."""
    df.to_csv(path, index=False)
