import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocess import get_prepared_data

def train_and_calibrate():
    os.makedirs("models", exist_ok=True)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = get_prepared_data()

    # 4 Candidate Models as required by Lab 14
    # multi_class parameter removed to support latest scikit-learn versions
    candidate_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive_Bayes": GaussianNB(),
        "Support_Vector_Machine": SVC(probability=True, kernel='rbf', random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    }

    print("=" * 65)
    print("LAB 14: BENCHMARKING MODELS (MULTICLASS CLINICAL EVALUATION)")
    print("=" * 65)

    best_score = 0.0
    best_clf = None
    best_name = ""

    for name, clf in candidate_models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        
        auc = roc_auc_score(y_test, probs, multi_class="ovr")
        print(f"\n--- Model: {name} (Multiclass ROC-AUC: {auc:.4f}) ---")
        print(classification_report(y_test, preds, target_names=["Low Risk", "Medium Risk", "High Risk"]))

        if auc > best_score:
            best_score = auc
            best_clf = clf
            best_name = name

    print("=" * 65)
    print(f"SELECTED BASE ENSEMBLE: {best_name} (ROC-AUC: {best_score:.4f})")
    print("=" * 65)

    # Probability Calibration using Platt Sigmoid method
    print("Fitting CalibratedClassifierCV for clinically reliable probabilities...")
    calibrated_model = CalibratedClassifierCV(estimator=best_clf, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)

    # Export production artifacts
    joblib.dump(calibrated_model, "models/best_calibrated_model.pkl")
    joblib.dump(best_clf, "models/base_rf_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("\nSUCCESS: All model artifacts successfully saved to /models folder.")

if __name__ == "__main__":
    train_and_calibrate()