import streamlit as st
import pandas as pd
import pickle

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "gradient_boosting_regressor_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)


st.title("💊 Medical Expenses Prediction App")

st.write("Enter customer details to predict annual medical expenses.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=64, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create input DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Prediction
if st.button("Predict Expenses"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Annual Medical Expenses: ${prediction:,.2f}")
