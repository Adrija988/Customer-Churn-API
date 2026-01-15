from fastapi import FastAPI
import pickle
import pandas as pd

from app.schema import ChurnInput   # ✅ IMPORTANT FIX

app = FastAPI(title="Customer Churn Prediction API")

# Load trained PIPELINE (not raw model)
with open("ML/churn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

def get_risk_label(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"

@app.post("/predict")
def predict_churn(data: ChurnInput):

    input_df = pd.DataFrame([{
        "AccountWeeks": data.AccountWeeks,
        "ContractRenewal": data.ContractRenewal,
        "DataPlan": data.DataPlan,
        "DataUsage": data.DataUsage,
        "CustServCalls": data.CustServCalls,
        "DayMins": data.DayMins,
        "DayCalls": data.DayCalls,
        "MonthlyCharge": data.MonthlyCharge,
        "OverageFee": data.OverageFee,
        "RoamMins": data.RoamMins
    }])

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(prob), 3),
        "risk_level": get_risk_label(prob)
    }
