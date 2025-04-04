import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 💾 Load Model & Scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# 🔹 User Input Form
st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details to predict if they will churn.")

# Define input fields based on dataset features
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=10)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, value=50)
total_charges = st.number_input("Total Charges ($)", min_value=0, value=500)

# Convert categorical values into one-hot encoding format
input_data = pd.DataFrame(
    {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender_Male": [1 if gender == "Male" else 0],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner_Yes": [1 if partner == "Yes" else 0],
        "Dependents_Yes": [1 if dependents == "Yes" else 0],
        "PhoneService_Yes": [1 if phone_service == "Yes" else 0],
        "MultipleLines_No phone service": [1 if multiple_lines == "No phone service" else 0],
        "MultipleLines_Yes": [1 if multiple_lines == "Yes" else 0],
        "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
        "InternetService_No": [1 if internet_service == "No" else 0],
        "OnlineSecurity_No internet service": [1 if online_security == "No internet service" else 0],
        "OnlineSecurity_Yes": [1 if online_security == "Yes" else 0],
        "OnlineBackup_No internet service": [1 if online_backup == "No internet service" else 0],
        "OnlineBackup_Yes": [1 if online_backup == "Yes" else 0],
        "DeviceProtection_No internet service": [1 if device_protection == "No internet service" else 0],
        "DeviceProtection_Yes": [1 if device_protection == "Yes" else 0],
        "TechSupport_No internet service": [1 if tech_support == "No internet service" else 0],
        "TechSupport_Yes": [1 if tech_support == "Yes" else 0],
        "StreamingTV_No internet service": [1 if streaming_tv == "No internet service" else 0],
        "StreamingTV_Yes": [1 if streaming_tv == "Yes" else 0],
        "StreamingMovies_No internet service": [1 if streaming_movies == "No internet service" else 0],
        "StreamingMovies_Yes": [1 if streaming_movies == "Yes" else 0],
        "Contract_One year": [1 if contract == "One year" else 0],
        "Contract_Two year": [1 if contract == "Two year" else 0],
        "PaperlessBilling_Yes": [1 if paperless_billing == "Yes" else 0],
        "PaymentMethod_Credit card": [1 if payment_method == "Credit card" else 0],
        "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
        "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],
    }
)

# Ensure columns match model's expected features
expected_features = scaler.feature_names_in_
missing_features = set(expected_features) - set(input_data.columns)
for feature in missing_features:
    input_data[feature] = 0  # Add missing columns with default value 0

# Reorder columns to match training data
input_data = input_data[expected_features]

# Scale input data
input_scaled = scaler.transform(input_data)

# 🔍 Predict Churn
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn**. (Risk: {probability:.2%})")
    else:
        st.success(f"✅ The customer is **not likely to churn**. (Risk: {probability:.2%})")

st.write("🔹 This app predicts whether a telecom customer will churn based on their profile.")
