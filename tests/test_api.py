import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))


SAMPLE_CUSTOMER = {
    "CreditScore": 650,
    "Gender": 1,
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 90000.0,
    "Geography_France": 1,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
}


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 12))

    with patch("src.api.main.model", mock_model), patch("src.api.main.scaler", mock_scaler):
        from src.api.main import app

        with TestClient(app) as c:
            yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_no_churn(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert data["label"] in ["Churn", "No Churn"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_invalid_credit_score(client):
    bad_customer = SAMPLE_CUSTOMER.copy()
    bad_customer["CreditScore"] = 100  # below minimum 300
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_missing_field(client):
    bad_customer = SAMPLE_CUSTOMER.copy()
    del bad_customer["Age"]
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422
