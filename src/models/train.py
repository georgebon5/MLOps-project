import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath("."))
from src.models.evaluate import compute_metrics  # noqa: E402


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(processed_dir: str):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — class distribution:\n{pd.Series(y_resampled).value_counts()}")
    return X_resampled, y_resampled


def get_models(params: dict) -> dict:
    rs = params["model"]["random_state"]
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            random_state=rs,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            random_state=rs,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=rs,
        ),
    }


def train_and_log(models: dict, X_train, y_train, X_test, y_test, params: dict):
    mlflow.set_experiment("churn-prediction")

    best_model = None
    best_f1 = 0.0
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")

            # Log params
            mlflow.log_params(
                {
                    "model_type": name,
                    "n_estimators": params["model"].get("n_estimators", "N/A"),
                    "max_depth": params["model"].get("max_depth", "N/A"),
                    "test_size": params["data"]["test_size"],
                    "smote": True,
                }
            )

            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  Accuracy : {metrics['accuracy']:.4f}")
            print(f"  F1 Score : {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall   : {metrics['recall']:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_run_id = mlflow.active_run().info.run_id
                best_name = name

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    return best_model, best_run_id


if __name__ == "__main__":
    params = load_params()
    processed_dir = os.path.join("data", "processed")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train, params["data"]["random_state"])

    models = get_models(params)
    best_model, best_run_id = train_and_log(
        models, X_train_bal, y_train_bal, X_test, y_test, params
    )

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, os.path.join("models", "best_model.pkl"))

    # Export best model metrics for DVC tracking

    best_metrics = compute_metrics(best_model, X_test, y_test)
    with open("metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print("\nBest model saved to models/best_model.pkl")
    print(f"MLflow run ID: {best_run_id}")
    print(f"Metrics: {best_metrics}")
