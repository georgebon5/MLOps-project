import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def print_evaluation_report(model, X_test, y_test, model_name: str = "Model"):
    y_pred = model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Predicted: No Churn", "Predicted: Churn"],
        )
    )
