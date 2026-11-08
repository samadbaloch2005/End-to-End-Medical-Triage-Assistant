import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 3000

# Continuous Vital Signs
age = np.random.randint(18, 92, size=n_samples)
heart_rate = np.random.normal(loc=82, scale=19, size=n_samples).clip(45, 185)
resp_rate = np.random.normal(loc=18, scale=5, size=n_samples).clip(8, 45)
systolic_bp = np.random.normal(loc=122, scale=24, size=n_samples).clip(65, 230)
spo2 = np.random.uniform(82, 100, size=n_samples).clip(78, 100)
temp_c = np.random.normal(loc=37.1, scale=0.9, size=n_samples).clip(34.8, 41.5)
blood_glucose = np.random.normal(loc=120, scale=48, size=n_samples).clip(50, 480)
gcs_score = np.random.choice([15, 14, 13, 11, 9, 6, 3], size=n_samples, p=[0.70, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])
pain_score = np.random.randint(0, 11, size=n_samples)

# Categorical Features
gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
comorbidity = np.random.choice(["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"], size=n_samples, p=[0.45, 0.22, 0.15, 0.10, 0.08])

# 3-Tier Clinical Risk Logic
raw_risk = (
    (age > 65) * 1.5 +
    (heart_rate > 115) * 2.2 +
    (resp_rate > 24) * 2.5 +
    (systolic_bp < 90) * 3.8 +
    (systolic_bp > 175) * 2.0 +
    (spo2 < 91) * 4.2 +
    (temp_c > 38.7) * 2.0 +
    (blood_glucose > 250) * 1.8 +
    (gcs_score < 13) * 3.5 +
    (pain_score >= 8) * 1.2 +
    (comorbidity == "Cardiovascular") * 2.0 +
    (comorbidity == "Diabetes") * 1.0 +
    np.random.normal(0, 1.2, size=n_samples)
)

# 0 = Low Risk (Routine), 1 = Medium Risk (Urgent), 2 = High Risk (Emergency Resuscitation)
triage_tier = np.zeros(n_samples, dtype=int)
triage_tier[(raw_risk >= 3.8) & (raw_risk < 7.0)] = 1
triage_tier[raw_risk >= 7.0] = 2

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'comorbidity': comorbidity,
    'heart_rate': heart_rate.round(1),
    'resp_rate': resp_rate.round(1),
    'systolic_bp': systolic_bp.round(1),
    'spo2': spo2.round(1),
    'temp_c': temp_c.round(1),
    'blood_glucose': blood_glucose.round(1),
    'gcs_score': gcs_score,
    'pain_score': pain_score,
    'triage_tier': triage_tier
})

# Missing values injection (to satisfy Lab 14 Imputation requirement)
mask_missing_glucose = np.random.rand(n_samples) < 0.03
df.loc[mask_missing_glucose, 'blood_glucose'] = np.nan
mask_missing_bp = np.random.rand(n_samples) < 0.02
df.loc[mask_missing_bp, 'systolic_bp'] = np.nan

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "triage_records.csv")
df.to_csv(csv_path, index=False)

print(f"SUCCESS: Fresh dataset created with 'triage_tier' at: {csv_path}")
print("Class breakdown:\n", df['triage_tier'].value_counts(normalize=True) * 100)