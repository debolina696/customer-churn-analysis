import pandas as pd

def clean_data(df):
    """Basic cleaning: drop duplicates, handle missing values, encode categories."""
    df = df.drop_duplicates()
    df = df.fillna(0)  # simple imputation, adjust as needed
    df = pd.get_dummies(df, drop_first=True)
    return df
