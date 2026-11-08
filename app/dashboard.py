import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

st.set_page_config(
    page_title="End-to-End Medical Triage Assistant",
    page_icon="🏥",
    layout="wide"
)

# Model Artifacts Loader
@st.cache_resource
def load_all_models():
    # Exact paths created by train.py and explain.py
    model_path = os.path.join("models", "best_calibrated_model.pkl")
    prep_path = os.path.join("models", "preprocessor.pkl")
    expl_path = os.path.join("models", "explainer.pkl")
    feat_path = os.path.join("models", "feature_names.pkl")

    missing = [p for p in [model_path, prep_path, expl_path, feat_path] if not os.path.exists(p)]
    if missing:
        st.error(f"⚠️ Yeh files missing hain: {missing}. Terminal par train.py aur explain.py chalayein!")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    explainer = joblib.load(expl_path)
    features = joblib.load(feat_path)
    return model, preprocessor, explainer, features

model, preprocessor, explainer, feature_names = load_all_models()

st.title("🏥 Clinical Emergency Triage Assistant (Explainable AI)")
st.caption("Lab #14: Complex Computing Activity • Multi-Model Risk Stratification with Uncertainty & SHAP")
st.markdown("---")

col_vitals, col_decision = st.columns([1.1, 1.4])

with col_vitals:
    st.subheader("📋 Patient EMR Intake & Vitals")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (Years)", 18, 95, 62)
        gender = st.selectbox("Gender", ["Male", "Female"])
        comorbidity = st.selectbox("Comorbidity Profile", ["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"])
        heart_rate = st.slider("Heart Rate (BPM)", 40, 185, 112)
        resp_rate = st.slider("Respiratory Rate", 8, 45, 26)
    with c2:
        systolic_bp = st.slider("Systolic BP (mmHg)", 65, 230, 88)
        spo2 = st.slider("Blood Oxygen SpO2 (%)", 78, 100, 89)
        temp_c = st.number_input("Temp (°C)", 34.5, 41.5, 38.8, step=0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", 50, 480, 210)
        gcs_score = st.selectbox("GCS Mental Score", [15, 14, 13, 11, 9, 6, 3], index=2)
        pain_score = st.slider("Pain Scale (0-10)", 0, 10, 8)

    patient_raw = pd.DataFrame([{
        'age': age, 'gender': gender, 'comorbidity': comorbidity,
        'heart_rate': heart_rate, 'resp_rate': resp_rate,
        'systolic_bp': systolic_bp, 'spo2': spo2, 'temp_c': temp_c,
        'blood_glucose': glucose, 'gcs_score': gcs_score, 'pain_score': pain_score
    }])

with col_decision:
    st.subheader("⚡ Triage Stratification & Risk Assessment")
    
    transformed_data = preprocessor.transform(patient_raw)
    probs = model.predict_proba(transformed_data)[0]  # [P(Low), P(Medium), P(High)]
    pred_class = int(np.argmax(probs))
    
    # Normalized Shannon Entropy
    uncertainty = float(entropy(probs) / np.log(3))
    
    tiers = ["LOW RISK (Standard Queue)", "MEDIUM RISK (Urgent / Bed Needed)", "HIGH RISK (Immediate Resuscitation)"]
    colors = ["#22c55e", "#eab308", "#ef4444"]
    
    st.markdown(f"""
    <div style='background:rgba(15,23,42,0.8); border-left: 8px solid {colors[pred_class]}; padding:14px 18px; border-radius:10px; margin-bottom:12px;'>
        <h3 style='margin:0; color:{colors[pred_class]};'>{tiers[pred_class]}</h3>
        <p style='margin:4px 0 0 0; color:#94a3b8;'>Calibrated Confidence: <b>{probs[pred_class]*100:.1f}%</b> | Algorithmic Uncertainty: <b>{uncertainty:.2f} / 1.00</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Low Risk Probability", f"{probs[0]*100:.1f}%")
    m2.metric("Medium Risk Probability", f"{probs[1]*100:.1f}%")
    m3.metric("High Risk Probability", f"{probs[2]*100:.1f}%")
    
    st.markdown("---")
    st.subheader("🔍 Clinician Decision Rules & SHAP Attribution")
    
    shap_vals = explainer(transformed_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_vals[0, :, pred_class], max_display=7, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Red bars increase priority for the predicted tier; blue bars represent stabilizing vitals.")