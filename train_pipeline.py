import os
import json
import joblib # type: ignore
import pandas as pd

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from xgboost import XGBClassifier # type: ignore

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "Bank Customer Churn Prediction.csv")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

TARGET = "churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =====================================================
# INPUT SCHEMA
# =====================================================
INPUT_SCHEMA = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "gender",
    "country",
    "active_member",
    "credit_card",
    "products_number",
    "estimated_salary"
]

schema_path = os.path.join(ARTIFACTS_DIR, "input_schema.json")
with open(schema_path, "w") as f:
    json.dump(INPUT_SCHEMA, f, indent=4)

print("Input schema saved at:", schema_path)

# =====================================================
# FEATURE GROUPS
# =====================================================
numeric_features = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary"
]

binary_features = [
    "credit_card",
    "active_member"
]

categorical_features = [
    "gender",
    "country"
]

# =====================================================
# PREPROCESSOR
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =====================================================
# TRAIN / TEST SPLIT (ONCE)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# MODELS TO EVALUATE
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# =====================================================
# TRAIN, EVALUATE, SELECT BEST
# =====================================================
metrics = {
    "Model": [],
    "Accuracy (%)": [],
    "Precision (%)": [],
    "Recall (%)": [],
    "F1-Score (%)": []
}

best_pipeline = None
best_f1 = 0.0

for model_name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    metrics["Model"].append(model_name)
    metrics["Accuracy (%)"].append(round(acc, 2))
    metrics["Precision (%)"].append(round(prec, 2))
    metrics["Recall (%)"].append(round(rec, 2))
    metrics["F1-Score (%)"].append(round(f1, 2))

    print(f"{model_name} | F1-Score: {f1:.2f}")

    if f1 > best_f1:
        best_f1 = f1
        best_pipeline = pipeline

# =====================================================
# SAVE BEST PIPELINE
# =====================================================
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "churn_pipeline.pkl")
joblib.dump(best_pipeline, PIPELINE_PATH)

print("Best model pipeline saved at:", PIPELINE_PATH)

# =====================================================
# SAVE MODEL METRICS
# =====================================================
metrics_path = os.path.join(ARTIFACTS_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Model metrics saved at:", metrics_path)
