"""Train, compare and calibrate the triage risk model.

Run from the project root:

    python -m src.train

What happens:
  1. Hold out 20% of the data as a test set that is not touched until the end.
  2. Compare four candidate models with 5-fold stratified cross-validation on the
     training set. SMOTE sits inside each pipeline, so oversampling only ever
     sees the training folds and validation scores are not inflated.
  3. Wrap the best candidate in sigmoid (Platt) calibration. Calibration is fitted
     on real, non-oversampled folds so the probabilities match real class rates.
  4. Pick a lower decision threshold for the High tier from out-of-fold
     predictions, because under-triage is the costly mistake in an ED.
  5. Evaluate once on the hold-out set and write a model card with the results.

Outputs:
  models/triage_pipeline.joblib  full pipeline (raw features in, probabilities out)
  models/model_card.json         metrics, settings and versions for the admin page
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.policy import apply_policy, choose_high_threshold
from src.preprocess import (
    CLASS_NAMES,
    DATA_PATH,
    FEATURES,
    build_preprocessor,
    load_dataset,
    split,
)

MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "triage_pipeline.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
SEED = 42
TARGET_HIGH_RECALL = 0.85

# Reference values for a clinically unremarkable adult. Explanations measure how
# far each of the patient's values moves the risk away from this baseline.
REFERENCE_PATIENT = {
    "age": 40,
    "gender": "Female",
    "comorbidity": "None",
    "heart_rate": 75.0,
    "resp_rate": 16.0,
    "systolic_bp": 120.0,
    "spo2": 98.0,
    "temp_c": 37.0,
    "blood_glucose": 100.0,
    "gcs_score": 15,
    "pain_score": 0,
}


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        # SVC has no native probabilities; an inner calibration supplies them.
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=SEED), ensemble=False, cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
    }


def make_pipeline(estimator) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=SEED)),
            ("model", estimator),
        ]
    )


def compare_candidates(X_train, y_train) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for name, estimator in candidate_models().items():
        started = time.perf_counter()
        scores = cross_validate(
            make_pipeline(estimator),
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc_ovr", "f1_macro": "f1_macro", "accuracy": "accuracy"},
            n_jobs=-1,
        )
        results.append(
            {
                "name": name,
                "cv_roc_auc": float(np.mean(scores["test_roc_auc"])),
                "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "cv_f1_macro": float(np.mean(scores["test_f1_macro"])),
                "cv_accuracy": float(np.mean(scores["test_accuracy"])),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(
            f"  {name:<24} ROC-AUC {results[-1]['cv_roc_auc']:.4f} "
            f"(+/- {results[-1]['cv_roc_auc_std']:.4f})  macro-F1 {results[-1]['cv_f1_macro']:.4f}"
        )
    return sorted(results, key=lambda r: r["cv_roc_auc"], reverse=True)


def evaluate(proba: np.ndarray, y_test, high_threshold: float | None = None) -> dict:
    pred = apply_policy(proba, high_threshold) if high_threshold else proba.argmax(axis=1)
    report = classification_report(
        y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    # Multiclass Brier score: mean squared error between probabilities and one-hot truth.
    onehot = np.eye(len(CLASS_NAMES))[np.asarray(y_test)]
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return {
        "roc_auc_ovr": float(roc_auc_score(y_test, proba, multi_class="ovr")),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "brier": brier,
        "high_risk_recall": float(report["High"]["recall"]),
        "per_class": {
            name: {k: float(report[name][k]) for k in ("precision", "recall", "f1-score", "support")}
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def train(csv_path: Path | str = DATA_PATH, quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else print
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split(X, y)
    log(f"Training rows: {len(X_train)}  hold-out rows: {len(X_test)}")

    log("\nComparing candidates with 5-fold cross-validation:")
    leaderboard = compare_candidates(X_train, y_train)
    best = leaderboard[0]
    log(f"\nSelected: {best['name']}")

    log("Calibrating probabilities (sigmoid, 5 folds)...")
    estimator = candidate_models()[best["name"]]
    model = CalibratedClassifierCV(make_pipeline(estimator), method="sigmoid", cv=5)

    # Missing a High-risk patient is far worse than over-triaging a Medium one, so
    # we lower the bar for calling High. The threshold is picked on out-of-fold
    # predictions from the training set, never on the hold-out set.
    log("Choosing the High-risk threshold from out-of-fold predictions...")
    oof = cross_val_predict(
        model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba", n_jobs=-1,
    )
    high_threshold = choose_high_threshold(oof, y_train, target_recall=TARGET_HIGH_RECALL)
    log(f"  P(High) >= {high_threshold:.2f} is treated as High")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    metrics_argmax = evaluate(proba, y_test)
    metrics = evaluate(proba, y_test, high_threshold)
    for label, m in (("argmax", metrics_argmax), ("with threshold", metrics)):
        log(
            f"Hold-out ({label:<14}) ROC-AUC {m['roc_auc_ovr']:.4f}  macro-F1 {m['f1_macro']:.4f}  "
            f"High recall {m['high_risk_recall']:.4f}"
        )

    card = {
        "model_name": best["name"],
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "path": str(csv_path),
            "rows": int(len(X)),
            "class_balance": {
                CLASS_NAMES[int(k)]: int(v) for k, v in y.value_counts().sort_index().items()
            },
        },
        "features": FEATURES,
        "class_names": CLASS_NAMES,
        "reference_patient": REFERENCE_PATIENT,
        "leaderboard": leaderboard,
        "high_threshold": high_threshold,
        "target_high_recall": TARGET_HIGH_RECALL,
        "test_metrics": metrics,
        "test_metrics_argmax": metrics_argmax,
    }

    joblib.dump(model, PIPELINE_PATH)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    log(f"\nSaved {PIPELINE_PATH} and {MODEL_CARD_PATH}")
    return card


if __name__ == "__main__":
    train()
