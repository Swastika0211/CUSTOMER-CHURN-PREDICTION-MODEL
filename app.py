import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# Inputs (match your dataset columns)
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (years)", 0, 20, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products_number = st.number_input("Number of Products", 1, 4, 1)
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    input_data = np.array([[credit_score, age, tenure, balance,
                            products_number, estimated_salary]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
