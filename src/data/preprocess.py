import os

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop irrelevant columns
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop nulls
    df = df.dropna()

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()

    if "Gender" in df.columns:
        df["Gender"] = le.fit_transform(df["Gender"])

    if "Geography" in df.columns:
        df = pd.get_dummies(df, columns=["Geography"], drop_first=False)

    return df


def split_and_scale(df: pd.DataFrame, params: dict):
    target = params["training"]["target_column"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler,
    )


def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print(f"Saved processed data to {output_dir}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Class distribution (train):\n{y_train.value_counts()}")


if __name__ == "__main__":
    params = load_params()

    raw_path = os.path.join("data", "raw", "churn.csv")
    processed_dir = os.path.join("data", "processed")

    df = load_raw_data(raw_path)
    print(f"Raw data shape: {df.shape}")

    df = clean_data(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, params)

    save_processed_data(X_train, X_test, y_train, y_test, scaler, processed_dir)
