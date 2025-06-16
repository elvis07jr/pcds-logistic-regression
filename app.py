import streamlit as st
import pandas as pd
import joblib

model = joblib.load("pcds_model.pkl")

st.title("PCDS Risk Prediction")

cap_refill = st.slider("Capillary Refill Time (seconds)", 1.0, 5.0, 2.5, step=0.1)
oxygen_sat = st.slider("Oxygen Saturation (%)", 85.0, 100.0, 97.0, step=0.1)
heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 80)
age = st.slider("Age (years)", 18, 85, 45)

input_df = pd.DataFrame([{
    "capillary_refill_time": cap_refill,
    "oxygen_saturation": oxygen_sat,
    "heart_rate": heart_rate,
    "age": age
}])

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.write("Has PCDS:" if pred == 1 else "No PCDS")
st.write(f"Risk Probability: {prob:.2f}")
