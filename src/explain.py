"""Build a TreeSHAP explainer artifact for the trained triage model."""

from pathlib import Path
from typing import Any

import joblib
import shap
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path("models/base_rf_model.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")
EXPLAINER_PATH = Path("models/explainer.pkl")


def generate_explainer(
    model_path: Path = MODEL_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    output_path: Path = EXPLAINER_PATH,
) -> Any:
    """Create and save TreeSHAP for the trained Random Forest model."""
    model_path = Path(model_path)
    feature_names_path = Path(feature_names_path)
    output_path = Path(output_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run `python -m src.train` first."
        )
    if not feature_names_path.is_file():
        raise FileNotFoundError(
            f"Feature names not found: {feature_names_path}. Run `python -m src.train` first."
        )

    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            f"TreeSHAP expects a RandomForestClassifier, but {model_path} contains "
            f"{type(model).__name__}. The training script saves its best-scoring "
            "candidate under this filename; verify Random Forest was selected."
        )

    feature_names = joblib.load(feature_names_path)
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and len(feature_names) != expected_count:
        raise ValueError(
            f"Feature mismatch: artifact has {len(feature_names)} names, but the "
            f"model expects {expected_count}. Retrain and regenerate the artifacts."
        )

    print("Generating SHAP TreeExplainer for the Random Forest triage model...")
    explainer = shap.TreeExplainer(model, feature_names=list(feature_names))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, output_path)
    print(f"SUCCESS: SHAP TreeExplainer saved to {output_path}")
    return explainer


if __name__ == "__main__":
    generate_explainer()
