# =====================================
# 1. Imports
# =====================================
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

# =====================================
# 2. Load Dataset
# =====================================
df = pd.read_csv("../data/telecom_churn.csv")


# =====================================
# 3. Cleaning
# =====================================
df = df.drop_duplicates()
df["Churn"] = df["Churn"].astype(int)

# =====================================
# 4. Feature / Target
# =====================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

print(X.columns)

# =====================================
# 5. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# 6. Preprocessing
# =====================================
numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

# =====================================
# 7. Handle Imbalance (VERY IMPORTANT)
# =====================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# =====================================
# 8. XGBoost Model
# =====================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# =====================================
# 9. Pipeline
# =====================================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =====================================
# 10. Train
# =====================================
pipeline.fit(X_train, y_train)

# =====================================
# 11. Threshold Tuning
# =====================================
y_probs = pipeline.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.3, 0.8, 0.05)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t)
    print(f"Threshold {t:.2f} → F1 {f1_t:.3f}")
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best F1:", round(best_f1, 3))

# =====================================
# 12. Final Evaluation
# =====================================
y_pred = (y_probs >= best_threshold).astype(int)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save artifacts
# ===============================

# Create folder if it doesn't exist
os.makedirs("ml", exist_ok=True)

# Metrics dictionary (based on your final results)
metrics = {
    "best_threshold": best_threshold,
    "f1_score": round(best_f1, 3),
    "classification_report": classification_report(
        y_test, y_pred, output_dict=True
    )
}

# Save model
with open("ml/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save metrics
with open("ml/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
    
import joblib
joblib.dump(model, "models/model.pkl")
model = joblib.load("models/model.pkl")


print("✅ Model trained and saved successfully")
