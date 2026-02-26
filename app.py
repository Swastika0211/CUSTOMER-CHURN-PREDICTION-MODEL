import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor")

st.title("📊 Customer Churn Prediction")

# ==============================
# Load Model Safely
# ==============================
try:
    model_path = os.path.join(os.path.dirname(__file__), "best_churn_pipeline.pkl")
    model = joblib.load(model_path)
except Exception as e:
    st.error("❌ Model failed to load. Check requirements.txt or model file.")
    st.stop()

st.subheader("Enter Customer Details")

# Adjust these fields to match your training features
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (Years)", 0, 20, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):

    # IMPORTANT: Feature order must match training
    input_data = np.array([[credit_score,
                            age,
                            tenure,
                            balance,
                            num_products,
                            estimated_salary]])

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

    except Exception:
        st.error("Prediction failed. Feature order may not match training data.")
