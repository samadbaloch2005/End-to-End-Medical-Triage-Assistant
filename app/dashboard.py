import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import entropy


st.set_page_config(
    page_title="Clinical Triage Assistant",
    page_icon="🏥",
    layout="wide",
)

# ---------- Visual styling ----------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f3f7fb;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce6ee;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .hero {
        padding: 28px 30px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(115deg, #102a43, #176b87);
        margin-bottom: 18px;
    }
    .hero p {
        color: #e3f2f7;
        margin-bottom: 0;
    }
    .eyebrow {
        color: #b9e7ef;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load trained model artifacts ----------
@st.cache_resource
def load_all_models():
    model_path = os.path.join("models", "best_calibrated_model.pkl")
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    feature_path = os.path.join("models", "feature_names.pkl")

    missing = [
        path
        for path in (model_path, preprocessor_path, feature_path)
        if not os.path.exists(path)
    ]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}")
        st.info("Run the project's training steps to generate the model artifacts.")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = joblib.load(feature_path)
    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_all_models()


# ---------- Scenario presets ----------
DEFAULTS = {
    "age": 62,
    "gender": "Male",
    "comorbidity": "None",
    "heart_rate": 112,
    "resp_rate": 26,
    "systolic_bp": 88,
    "spo2": 89,
    "temp_c": 38.8,
    "glucose": 210,
    "gcs_score": 13,
    "pain_score": 8,
}

SCENARIOS = {
    "Custom Manual Input": {},
    "🚨 Critical Shock Patient (Code Red)": {
        "age": 68,
        "gender": "Male",
        "comorbidity": "Cardiovascular",
        "heart_rate": 125,
        "resp_rate": 28,
        "systolic_bp": 82,
        "spo2": 87,
        "temp_c": 39.2,
        "glucose": 290,
        "gcs_score": 11,
        "pain_score": 9,
    },
    "🟡 Urgent Monitored Patient (Code Yellow)": {
        "age": 52,
        "gender": "Female",
        "comorbidity": "Diabetes",
        "heart_rate": 98,
        "resp_rate": 22,
        "systolic_bp": 110,
        "spo2": 94,
        "temp_c": 38.1,
        "glucose": 190,
        "gcs_score": 14,
        "pain_score": 5,
    },
    "🟢 Routine Stable Patient (Code Green)": {
        "age": 28,
        "gender": "Female",
        "comorbidity": "None",
        "heart_rate": 72,
        "resp_rate": 16,
        "systolic_bp": 120,
        "spo2": 98,
        "temp_c": 36.8,
        "glucose": 95,
        "gcs_score": 15,
        "pain_score": 2,
    },
}


def apply_scenario():
    selected = st.session_state["preset"]
    values = SCENARIOS[selected]
    for field, value in values.items():
        st.session_state[field] = value


for field, value in DEFAULTS.items():
    st.session_state.setdefault(field, value)


# ---------- Header and preset selector ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-assisted clinical decision support</div>
      <h1>Clinical Triage Assistant</h1>
      <p>Review patient vitals and model-estimated triage risk in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Demo controls")
st.sidebar.selectbox(
    "Load a patient scenario",
    list(SCENARIOS.keys()),
    key="preset",
    on_change=apply_scenario,
)
st.sidebar.caption("Selecting a scenario fills the input fields with demo values.")


# ---------- Patient intake ----------
st.subheader("Patient intake")
left, right = st.columns(2)

with left:
    age = st.number_input("Age (years)", 18, 95, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    comorbidity = st.selectbox(
        "Comorbidity profile",
        ["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"],
        key="comorbidity",
    )
    heart_rate = st.slider("Heart rate (BPM)", 40, 185, key="heart_rate")
    resp_rate = st.slider("Respiratory rate", 8, 45, key="resp_rate")

with right:
    systolic_bp = st.slider("Systolic blood pressure (mmHg)", 65, 230, key="systolic_bp")
    spo2 = st.slider("Blood oxygen SpO₂ (%)", 78, 100, key="spo2")
    temp_c = st.number_input(
        "Temperature (°C)", 34.5, 41.5, step=0.1, format="%.1f", key="temp_c"
    )
    glucose = st.number_input(
        "Blood glucose (mg/dL)", 50, 480, key="glucose"
    )
    gcs_score = st.selectbox(
        "GCS mental score", [15, 14, 13, 11, 9, 6, 3], key="gcs_score"
    )
    pain_score = st.slider("Pain scale (0–10)", 0, 10, key="pain_score")


patient_raw = pd.DataFrame(
    [
        {
            "age": age,
            "gender": gender,
            "comorbidity": comorbidity,
            "heart_rate": heart_rate,
            "resp_rate": resp_rate,
            "systolic_bp": systolic_bp,
            "spo2": spo2,
            "temp_c": temp_c,
            "blood_glucose": glucose,
            "gcs_score": gcs_score,
            "pain_score": pain_score,
        }
    ]
)


# ---------- Prediction ----------
transformed_data = preprocessor.transform(patient_raw)
probabilities = model.predict_proba(transformed_data)[0]
predicted_class = int(np.argmax(probabilities))

tiers = [
    "LOW RISK",
    "MEDIUM RISK",
    "HIGH RISK",
]
descriptions = [
    "Standard outpatient queue",
    "Urgent review / monitored care",
    "Immediate clinical review",
]
colors = ["#15803d", "#b45309", "#b91c1c"]

uncertainty = (
    float(entropy(probabilities) / np.log(len(probabilities)))
    if len(probabilities) > 1
    else 0.0
)
confidence = float(probabilities[predicted_class])


# ---------- Results ----------
st.markdown("---")
st.subheader("Triage assessment")

result_col, stats_col = st.columns([1.15, 1])

with result_col:
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #dce6ee;
            border-left:8px solid {colors[predicted_class]};
            border-radius:16px;
            padding:20px 22px;">
          <div style="color:#52677a;font-size:0.85rem;">MODEL-ESTIMATED TRIAGE</div>
          <h2 style="color:{colors[predicted_class]};margin:8px 0;">
            {tiers[predicted_class]}
          </h2>
          <div style="color:#334e68;">{descriptions[predicted_class]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.progress(
        int(round(confidence * 100)),
        text=f"Model confidence: {confidence * 100:.1f}%",
    )
    st.caption(
        f"Algorithmic uncertainty (normalized entropy): {uncertainty:.2f} / 1.00"
    )

with stats_col:
    st.markdown("#### Risk probabilities")
    probability_table = pd.DataFrame(
        {"Probability (%)": probabilities * 100},
        index=["Low risk", "Medium risk", "High risk"],
    )
    st.bar_chart(probability_table)


# ---------- Patient summary ----------
st.markdown("---")
summary_tab, interpretation_tab = st.tabs(["Patient snapshot", "How to interpret"])

with summary_tab:
    st.markdown("#### Entered vital signs")
    metric_cols = st.columns(4)
    metric_cols[0].metric("SpO₂", f"{spo2}%")
    metric_cols[1].metric("Systolic BP", f"{systolic_bp} mmHg")
    metric_cols[2].metric("Heart rate", f"{heart_rate} BPM")
    metric_cols[3].metric("Respiratory rate", f"{resp_rate}/min")

    with st.expander("View all entered patient details"):
        st.dataframe(
            patient_raw.T.rename(columns={0: "Entered value"}),
            use_container_width=True,
        )

with interpretation_tab:
    st.markdown("#### What these numbers mean")
    st.write(
        "The probability chart shows the model's estimated scores for each "
        "triage class. Confidence and uncertainty describe model output; "
        "they are not a guarantee of clinical correctness."
    )
    st.info(
        "This prototype supports clinical review. It is not a diagnosis and "
        "does not replace a qualified clinician's judgment."
    )
    st.caption(
        "SHAP feature-level explanations are not displayed in this dashboard yet."
    )