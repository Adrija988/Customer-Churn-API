import joblib
import numpy as np

model = joblib.load("artifacts/model.pkl")
features = joblib.load("artifacts/features.pkl")

THRESHOLD = 0.35

def predict_churn(data_dict):
    input_data = np.array([[data_dict[feature] for feature in features]])
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob >= THRESHOLD)
    return prediction, float(prob)