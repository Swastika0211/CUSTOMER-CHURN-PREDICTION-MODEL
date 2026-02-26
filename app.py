import streamlit as st
import numpy as np
import os

st.set_page_config(page_title="Customer Churn Predictor")

st.title("Customer Churn Prediction App")

model = None

# Try loading model
try:
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), "best_churn_pipeline.pkl")
    model = joblib.load(model_path)
except Exception as e:
    st.warning("Model not loaded. Please check requirements.txt and model file.")

st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 20, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):

    if model is None:
        st.error("Model is not available. Fix deployment dependencies.")
    else:
        input_data = np.array([[credit_score, age, tenure,
                                balance, products, salary]])
        try:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("Customer is likely to CHURN")
            else:
                st.success("Customer is likely to STAY")

        except Exception:
            st.error("Prediction failed. Check feature order.")
