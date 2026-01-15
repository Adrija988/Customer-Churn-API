# Customer Churn Prediction (XGBoost + FastAPI)
1. Problem Statement
Predict whether a telecom customer will churn using usage and billing data.

2.Dataset
File: telecom_churn.csv
10 numerical and binary input features

3. Features Used
AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins

4. Model
XGBClassifier (XGBoost)
Pipeline: StandardScaler + XGBClassifier
Binary classification (Churn / No Churn)

5. Output
churn_prediction (0 / 1)
churn_probability
risk_label (Low / Medium / High)

6. Risk Label Logic
Probability < 0.30 → Low
0.30 – 0.60 → Medium
0.60 → High

7. API
Framework: FastAPI
Endpoint: POST /predict

8. Input: JSON customer data

9. Output: Churn prediction with probability
    
10. How to run: uvicorn app.main:app --reload
 
