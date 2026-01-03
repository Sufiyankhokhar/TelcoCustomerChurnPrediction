import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model & feature names
# ===============================
model = joblib.load("rf_model.pkl")

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
feature_names = model.feature_names_in_


st.markdown("Enter customer details to predict churn")

# ===============================
# User Inputs
# ===============================
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 8000.0, 500.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# ===============================
# Manual Encoding (as per training)
# ===============================
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# ===============================
# Prediction
# ===============================
if st.button("🔮 Predict Churn"):

    # basic input dictionary
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_One year': contract_one_year,
        'Contract_Two year': contract_two_year
    }

    # DataFrame banana
    input_data = pd.DataFrame([input_dict])

    # ⭐ MOST IMPORTANT LINE ⭐
    # Missing columns add + order same as training
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 1:
        st.error("❌ High Risk Of Churn")
    else:
        st.success("✅ Low Risk Of Churn")

    # Optional: debug info
    with st.expander("🔍 Debug Info"):
        st.write("Input shape:", input_data.shape)
        st.write(input_data.head())
