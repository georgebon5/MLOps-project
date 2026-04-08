"""
Data Drift Monitoring with Evidently AI.

Compares the training data distribution (reference) against
a simulated production batch (current) to detect data drift.

Usage:
    python monitoring/evidently_report.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    reference = X_train.copy()
    reference["target"] = y_train.values

    current = X_test.copy()
    current["target"] = y_test.values

    return reference, current


def simulate_production_drift(current: pd.DataFrame) -> pd.DataFrame:
    """Simulate data drift by shifting key feature distributions."""
    drifted = current.copy()
    rng = np.random.default_rng(seed=99)

    # Simulate aging customer base
    drifted["Age"] = drifted["Age"] + rng.normal(5, 2, len(drifted))

    # Simulate economic downturn — lower balances
    drifted["Balance"] = drifted["Balance"] * rng.uniform(0.7, 0.9, len(drifted))

    # Simulate lower activity
    drifted["IsActiveMember"] = rng.choice([0, 1], size=len(drifted), p=[0.65, 0.35])

    return drifted


def get_column_mapping():
    return ColumnMapping(
        target="target",
        numerical_features=[
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary",
        ],
        categorical_features=[
            "Gender",
            "HasCrCard",
            "IsActiveMember",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain",
        ],
    )


def run_report(reference, current, output_path, metrics):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=get_column_mapping(),
    )
    report.save_html(output_path)
    print(f"Report saved to {output_path}")
    return report


if __name__ == "__main__":
    print("Loading data...")
    reference, current = load_data()
    print(f"Reference shape: {reference.shape} | Current shape: {current.shape}")

    output_dir = "monitoring/reports"

    print("\n--- Running baseline report (no drift) ---")
    run_report(
        reference,
        current,
        os.path.join(output_dir, "baseline_report.html"),
        metrics=[DataDriftPreset(), DatasetDriftMetric()],
    )

    print("\n--- Simulating production drift ---")
    drifted_current = simulate_production_drift(current)

    print("\n--- Running drift detection report ---")
    run_report(
        reference,
        drifted_current,
        os.path.join(output_dir, "data_drift_report.html"),
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="Age"),
            ColumnDriftMetric(column_name="Balance"),
            ColumnDriftMetric(column_name="IsActiveMember"),
        ],
    )

    print("\nDone! Open the HTML reports in your browser:")
    print(f"  open {output_dir}/baseline_report.html")
    print(f"  open {output_dir}/data_drift_report.html")
