import shap
import joblib
import os
from src.preprocess import get_prepared_data

def generate_explainer():
    os.makedirs("models", exist_ok=True)
    
    base_model_path = os.path.join("models", "base_rf_model.pkl")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model file '{base_model_path}' nahi mili. Pehle 'py -m src.train' chalayein.")
        
    base_rf = joblib.load(base_model_path)
    X_train, _, _, _, _, feature_names = get_prepared_data()
    
    print("Generating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(base_rf)
    
    output_path = os.path.join("models", "explainer.pkl")
    joblib.dump(explainer, output_path)
    print(f"SUCCESS: SHAP TreeExplainer saved to {output_path}")

if __name__ == "__main__":
    generate_explainer()