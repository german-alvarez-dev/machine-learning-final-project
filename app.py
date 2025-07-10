import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and threshold
model_bundle = joblib.load("outputs/models/final_logistic_model_balanced.joblib")
model, threshold = model_bundle

st.title("🎢 Incident Risk Predictor")
st.markdown("Estimate the risk of an incident based on visitor and ride context.")

# Input fields
ride_type = st.selectbox("Ride type", [
    "flat_ride", "simulator", "water_ride", "transport", "dark_ride",
    "drop_tower", "roller_coaster", "carousel", "theater", "interactive"
])
duration_min = st.slider("Ride duration (minutes)", 1, 15, 5)
age = st.slider("Visitor age", 1, 90, 30)
gender = st.selectbox("Gender", ["F", "M"])
age_group = pd.cut([age], bins=[0,9,19,29,39,49,59,69,120], labels=["<10","10–19","20–29","30–39","40–49","50–59","60–69","70+"])[0]
first_time = st.checkbox("First-time visitor")
season = st.selectbox("Season", ["winter", "spring", "summer", "fall"])
is_weekend = st.checkbox("Weekend")
temp_max = st.slider("Max temperature (°C)", 0, 45, 30)
precip = st.slider("Precipitation (mm)", 0.0, 50.0, 5.0)

# Build input dataframe
input_dict = {
    "ride_type_simplified": ride_type,
    "duration_min": duration_min,
    "age": age,
    "gender": gender,
    "age_group": age_group,
    "first_time_visitor": first_time,
    "season": season,
    "is_weekend": is_weekend,
    "temperature_max": temp_max,
    "precipitation_sum": precip
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
prob = model.predict_proba(input_encoded)[0][1]
label = prob >= threshold

st.markdown(f"### 🧮 Estimated risk: `{prob:.2%}`")
if label:
    st.error("⚠️ High risk of incident predicted.")
else:
    st.success("✅ Low risk predicted.")
