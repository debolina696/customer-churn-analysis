from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_models(log_reg, rf, df):
    """Evaluate models and print summary table."""
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_pred_lr = log_reg.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_prob_lr = log_reg.predict_proba(X_test)[:,1]
    y_prob_rf = rf.predict_proba(X_test)[:,1]

    results = {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
        "Precision": [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf)],
        "Recall": [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
        "F1 Score": [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)],
        "AUC": [roc_auc_score(y_test, y_prob_lr), roc_auc_score(y_test, y_prob_rf)]
    }

    summary_df = pd.DataFrame(results)
    print(summary_df)
    return summary_df
