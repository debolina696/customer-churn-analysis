def create_features(df):
    """Create engineered features for churn prediction."""
    df["HighSupportCalls"] = (df["Support Calls"] > 5).astype(int)
    df["DelayRatio"] = df["Payment Delay"] / (df["Tenure"] + 1)
    df["SpendPerMonth"] = df["Total Spend"] / (df["Tenure"] + 1)
    return df
