# import pickle
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd

# # Load existing pipeline
# with open("model.pkl", "rb") as f:
#     pipeline = pickle.load(f)

# print(pipeline.named_steps)


# # Separate model & scaler for API
# model = pipeline.named_steps["model"]
# scaler = pipeline.named_steps["scaler"]

# # Save separately
# with open("model_only.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# print("Model and scaler saved separately for API")


import pickle

# Load trained pipeline
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Debug: show pipeline structure
print("Pipeline steps:", pipeline.named_steps)

# Save full pipeline (preprocessing + model together)
with open("churn_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Full churn pipeline saved successfully")

