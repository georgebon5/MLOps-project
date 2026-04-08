import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.data.preprocess import clean_data, encode_features


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "RowNumber": [1, 2, 3],
        "CustomerId": [111, 222, 333],
        "Surname": ["A", "B", "C"],
        "CreditScore": [650, 700, 580],
        "Geography": ["France", "Germany", "Spain"],
        "Gender": ["Male", "Female", "Male"],
        "Age": [35, 45, 28],
        "Tenure": [5, 3, 7],
        "Balance": [75000.0, 0.0, 120000.0],
        "NumOfProducts": [2, 1, 3],
        "HasCrCard": [1, 0, 1],
        "IsActiveMember": [1, 1, 0],
        "EstimatedSalary": [90000.0, 60000.0, 110000.0],
        "Exited": [0, 1, 0],
    })


def test_clean_data_drops_irrelevant_columns(raw_df):
    cleaned = clean_data(raw_df)
    assert "RowNumber" not in cleaned.columns
    assert "CustomerId" not in cleaned.columns
    assert "Surname" not in cleaned.columns


def test_clean_data_drops_duplicates():
    df = pd.DataFrame({
        "CreditScore": [650, 650],
        "Geography": ["France", "France"],
        "Gender": ["Male", "Male"],
        "Age": [35, 35],
        "Tenure": [5, 5],
        "Balance": [75000.0, 75000.0],
        "NumOfProducts": [2, 2],
        "HasCrCard": [1, 1],
        "IsActiveMember": [1, 1],
        "EstimatedSalary": [90000.0, 90000.0],
        "Exited": [0, 0],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_encode_features_gender(raw_df):
    cleaned = clean_data(raw_df)
    encoded = encode_features(cleaned)
    assert encoded["Gender"].dtype in [np.int32, np.int64, int]


def test_encode_features_geography_dummies(raw_df):
    cleaned = clean_data(raw_df)
    encoded = encode_features(cleaned)
    assert "Geography" not in encoded.columns
    assert any(col.startswith("Geography_") for col in encoded.columns)


def test_no_nulls_after_clean(raw_df):
    raw_df.loc[0, "Age"] = np.nan
    cleaned = clean_data(raw_df)
    assert cleaned.isnull().sum().sum() == 0
