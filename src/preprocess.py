"""Feature definitions and preprocessing for the triage model.

The preprocessor is kept separate from the data split so it can live inside
a single sklearn/imblearn pipeline. That pipeline is what gets saved and
served, which means the web app sends raw patient values and never has to
re-implement scaling or one-hot encoding itself.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/triage_records.csv")
TARGET = "triage_tier"

NUMERIC_FEATURES = [
    "age",
    "heart_rate",
    "resp_rate",
    "systolic_bp",
    "spo2",
    "temp_c",
    "blood_glucose",
    "gcs_score",
    "pain_score",
]
CATEGORICAL_FEATURES = ["gender", "comorbidity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_NAMES = ["Low", "Medium", "High"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def load_dataset(csv_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    # keep_default_na=False: the comorbidity value "None" is a real category
    # ("no comorbidity"), but pandas would otherwise read it as missing.
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    return df[FEATURES], df[TARGET]


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Stratified hold-out split. Done before any fitting to avoid leakage."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
