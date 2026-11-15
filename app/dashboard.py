import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import entropy

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="End-to-End Medical Triage Assistant | AI-CDSS",
    page_icon="🏥",
    layout="wide"
)

# --- CUSTOM CSS STYLING FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    h1, h2, h3 { color: #f8fafc; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL ARTIFACTS LOADER ---
@st.cache_resource
def load_all_models():
    model_path = os.path.join("models", "best_calibrated_model.pkl")
    prep_path = os.path.join("models", "preprocessor.pkl")
    feat_path = os.path.join("models", "feature_names.pkl")

    missing = [p for p in [model_path, prep_path, feat_path] if not os.path.exists(p)]
    if missing:
        st.error(f"⚠️ Missing files: {missing}. Pehle terminal mein training scripts chalayein!")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    features = joblib.load(feat_path)
    return model, preprocessor, features

model, preprocessor, feature_names = load_all_models()

# --- HEADER SECTION ---
st.title("🏥 Clinical Emergency Triage Assistant (AI-CDSS)")
st.caption("Lab #14 Complex Computing Activity • Multi-Model Risk Stratification with Platt Calibration & Shannon Entropy Uncertainty")
st.markdown("---")

# --- SIDEBAR: LIVE DEMO PRESETS ---
st.sidebar.markdown("## ⚡ Live Presentation Controls")
preset = st.sidebar.selectbox("Load Patient Scenario:", [
    "Custom Manual Input", 
    "🚨 Critical Shock Patient (Code Red)", 
    "🟡 Urgent Monitored Patient (Code Yellow)",
    "🟢 Routine Stable Patient (Code Green)"
])

d_age, d_gender, d_comorb = 62, "Male", "None"
d_hr, d_rr, d_sbp, d_spo2, d_temp, d_gluc, d_gcs, d_pain = 112, 26, 88, 89, 38.8, 210, 13, 8

if "Critical" in preset:
    d_age, d_gender, d_comorb = 68, "Male", "Cardiovascular"
    d_hr, d_rr, d_sbp, d_spo2, d_temp, d_gluc, d_gcs, d_pain = 125, 28, 82, 87, 39.2, 290, 11, 9
elif "Urgent" in preset:
    d_age, d_gender, d_comorb = 52, "Female", "Diabetes"
    d_hr, d_rr, d_sbp, d_spo2, d_temp, d_gluc, d_gcs, d_pain = 98, 22, 110, 94, 38.1, 190, 14, 5
elif "Routine" in preset:
    d_age, d_gender, d_comorb = 28, "Female", "None"
    d_hr, d_rr, d_sbp, d_spo2, d_temp, d_gluc, d_gcs, d_pain = 72, 16, 120, 98, 36.8, 95, 15, 2

# --- MAIN LAYOUT ---
col_vitals, col_decision = st.columns([1.1, 1.4])

with col_vitals:
    st.subheader("📋 Patient EMR Intake & Vitals")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (Years)", 18, 95, d_age)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if d_gender=="Male" else 1)
        comorbidity = st.selectbox("Comorbidity Profile", ["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"], index=["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"].index(d_comorb))
        heart_rate = st.slider("Heart Rate (BPM)", 40, 185, d_hr)
        resp_rate = st.slider("Respiratory Rate", 8, 45, d_rr)
    with c2:
        systolic_bp = st.slider("Systolic BP (mmHg)", 65, 230, d_sbp)
        spo2 = st.slider("Blood Oxygen SpO2 (%)", 78, 100, d_spo2)
        temp_c = st.number_input("Temp (°C)", 34.5, 41.5, d_temp, step=0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", 50, 480, d_gluc)
        gcs_score = st.selectbox("GCS Mental Score", [15, 14, 13, 11, 9, 6, 3], index=[15, 14, 13, 11, 9, 6, 3].index(d_gcs))
        pain_score = st.slider("Pain Scale (0-10)", 0, 10, d_pain)

    patient_raw = pd.DataFrame([{
        'age': age, 'gender': gender, 'comorbidity': comorbidity,
        'heart_rate': heart_rate, 'resp_rate': resp_rate,
        'systolic_bp': systolic_bp, 'spo2': spo2, 'temp_c': temp_c,
        'blood_glucose': glucose, 'gcs_score': gcs_score, 'pain_score': pain_score
    }])

with col_decision:
    st.subheader("⚡ Triage Stratification & Risk Assessment")
    
    transformed_data = preprocessor.transform(patient_raw)
    probs = model.predict_proba(transformed_data)[0]
    pred_class = int(np.argmax(probs))
    
    uncertainty = float(entropy(probs) / np.log(3))
    
    tiers = [
        "LOW RISK (Standard Outpatient Queue)", 
        "MEDIUM RISK (Urgent / Bed Needed)", 
        "HIGH RISK (Immediate Resuscitation Bay)"
    ]
    colors = ["#22c55e", "#eab308", "#ef4444"]
    
    st.markdown(f"""
    <div style='background:rgba(15,23,42,0.9); border-left: 8px solid {colors[pred_class]}; padding:16px 20px; border-radius:10px; margin-bottom:15px;'>
        <h3 style='margin:0; color:{colors[pred_class]}; font-size: 21px;'>{tiers[pred_class]}</h3>
        <p style='margin:6px 0 0 0; color:#cbd5e1; font-size: 14px;'>Calibrated Confidence: <b>{probs[pred_class]*100:.1f}%</b> | Algorithmic Uncertainty (Entropy): <b>{uncertainty:.2f} / 1.00</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Low Risk P(Low)", f"{probs[0]*100:.1f}%")
    m2.metric("Medium Risk P(Med)", f"{probs[1]*100:.1f}%")
    m3.metric("High Risk P(High)", f"{probs[2]*100:.1f}%")
    
    st.markdown("---")
    st.subheader("📊 Clinical Decision Metrics & Summary")
    st.info("💡 **Clinical Note:** The model successfully utilizes calibrated random forest ensemble probabilities combined with Shannon entropy uncertainty quantification to stratify emergency intake risk without runtime C-extension bottlenecks.")
