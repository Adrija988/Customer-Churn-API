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

6. Output
churn_prediction (0 / 1)
churn_probability
risk_label (Low / Medium / High)

7. Risk Label Logic
Probability < 0.30 → Low
0.30 – 0.60 → Medium
0.60 → High

8. Other Features
Implemented input schema validation, structured logging, and exception handling to improve production robustness and API reliability.

9. Evaluation result
Achieved a ROC-AUC score of 0.91 through feature selection and hyperparameter tuning.

10. API
Framework: FastAPI
Endpoint: POST /predict

11. Input: JSON customer data

12. Output: Churn prediction with probability
    
13. How to run: uvicorn app.main:app --reload
 
