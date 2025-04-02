# Data_Crunch_0078
Data_Crunch_0078 HackSquad


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load datasets
train_df = pd.read_excel(r"C:\Users\User\Downloads\processed_train.xlsx")
test_df = pd.read_excel(r"C:\Users\User\Downloads\data-crunch-round-1\test.xlsx")
sample_submission = pd.read_csv(r"C:\Users\User\Downloads\data-crunch-round-1\sample_submission.csv")

# Encode categorical variables
le = LabelEncoder()
train_df["kingdom_encoded"] = le.fit_transform(train_df["kingdom"])

if "kingdom" in test_df.columns:
    test_df["kingdom_encoded"] = test_df["kingdom"].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
else:
    raise ValueError("The test dataset is missing the 'kingdom' column.")

# Define target variables and features
target_vars = ["Avg_Temperature", "Radiation", "Rain_Amount", "Wind_Speed", "Wind_Direction"]
common_features = ["Year", "Month", "Day", "kingdom_encoded"]
optional_features = ["latitude", "longitude", "Temperature_Range", "Rain_Duration"]

# Keep only features that exist in both datasets
features_to_keep = common_features + [col for col in optional_features if col in train_df.columns and col in test_df.columns]

# Select features and target variables
X = train_df[features_to_keep]
y = train_df[target_vars]
test_features = test_df[features_to_keep]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train models with optimized parameters
optimized_params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# Train and predict
predictions = {}
models = {}

for target in target_vars:
    model = XGBRegressor(**optimized_params)
    model.fit(X_train, y_train[target])
    models[target] = model
    predictions[target] = model.predict(test_features)

# Prepare submission file
if "ID" in test_df.columns:
    submission = pd.DataFrame({"ID": test_df["ID"]})
else:
    raise KeyError("The test dataset is missing the 'ID' column.")

for target in target_vars:
    submission[target] = predictions[target]

# Save submission file
submission.to_csv("submission1.csv", index=False)
print("Submission file saved as submission.csv")

