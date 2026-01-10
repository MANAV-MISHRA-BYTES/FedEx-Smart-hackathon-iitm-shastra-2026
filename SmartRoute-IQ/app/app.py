
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/delivery_time_model.pkl")

st.set_page_config(page_title="SmartRoute IQ", layout="centered")

st.title("SmartRoute IQ")
st.subheader("Delivery Time Predictor")

distance = st.number_input("Distance (km)", 1.0, 2000.0, 120.0)
weight = st.number_input("Package Weight (kg)", 0.1, 100.0, 5.0)

traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])
delivery_type = st.selectbox("Delivery Type", ["Normal", "Express"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Evening", "Night"])

mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Clear": 0,
    "Rain": 1,
    "Fog": 2,
    "Normal": 0,
    "Express": 1,
    "Morning": 0,
    "Evening": 1,
    "Night": 2
}

input_df = pd.DataFrame([{
    "distance": distance,
    "weight": weight,
    "traffic": mapping[traffic],
    "weather": mapping[weather],
    "delivery_type": mapping[delivery_type],
    "time_of_day": mapping[time_of_day]
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} hours")
