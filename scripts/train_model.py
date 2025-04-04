import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 📂 Load Dataset
df = pd.read_csv("../data/raw/churn.csv")  # Ensure this file exists!

# 🔹 Convert categorical columns
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()  # Drop rows with missing values

# 🏷️ Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 🎯 Define Features & Target
X = df.drop(columns=["Churn_Yes"])  # Keep all columns except target
y = df["Churn_Yes"]

# 🔀 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔬 Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 💾 Save Model & Scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and Scaler saved successfully!")
