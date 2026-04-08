import os
from contextlib import asynccontextmanager

import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# If USE_REGISTRY=true, load from MLflow Model Registry (@production alias)
# Otherwise fall back to local pickle file (useful in CI/Docker without MLflow server)
USE_REGISTRY = os.getenv("USE_REGISTRY", "false").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    if USE_REGISTRY:
        print("Loading model from MLflow Registry (@production)...")
        model = mlflow.pyfunc.load_model("models:/churn-predictor@production")
        print("Model loaded from MLflow Registry")
    else:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    yield
    model = None
    scaler = None


app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade ML API for Bank Customer Churn Prediction",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

    CreditScore: int = Field(..., ge=300, le=900)
    Gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Geography_France: int = Field(..., ge=0, le=1)
    Geography_Germany: int = Field(..., ge=0, le=1)
    Geography_Spain: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    label: str


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Churn Prediction API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_order = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain",
    ]

    input_df = pd.DataFrame([customer.model_dump()])[feature_order]
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        label="Churn" if prediction == 1 else "No Churn",
    )
