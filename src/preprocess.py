import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def get_prepared_data(csv_path="data/triage_records.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['triage_tier'])
    y = df['triage_tier']

    numeric_features = [
        'age', 'heart_rate', 'resp_rate', 'systolic_bp', 
        'spo2', 'temp_c', 'blood_glucose', 'gcs_score', 'pain_score'
    ]
    categorical_features = ['gender', 'comorbidity']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Train / Test split strictly before preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract feature names after One-Hot Encoding
    cat_cols = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_cols)

    # Multiclass SMOTE strictly on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_transformed, y_train)

    return X_train_res, X_test_transformed, y_train_res, y_test, preprocessor, all_feature_names