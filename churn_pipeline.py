import pandas as pd
from scripts.data_loader import load_raw_data, save_processed_data
from scripts.cleaning_utils import clean_data
from scripts.feature_engineering import create_features
from scripts.train_models import train_log_reg, train_rf
from scripts.evaluation_utils import evaluate_models

def run_pipeline():
    # Step 1: Load raw data
    df_raw = load_raw_data("../data/raw/churn_data.csv")

    # Step 2: Clean data
    df_clean = clean_data(df_raw)
    save_processed_data(df_clean, "../data/processed/churn_clean.csv")

    # Step 3: Feature engineering
    df_features = create_features(df_clean)
    save_processed_data(df_features, "../data/processed/churn_features.csv")

    # Step 4: Train models
    log_reg = train_log_reg(df_features)
    rf = train_rf(df_features)

    # Step 5: Evaluate models
    evaluate_models(log_reg, rf, df_features)

if __name__ == "__main__":
    run_pipeline()
python churn_pipeline.py 