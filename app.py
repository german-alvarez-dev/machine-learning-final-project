import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Wide layout
st.set_page_config(
    page_title="Incident Risk Predictor",
    layout="wide"
)

# Load model
model_bundle = joblib.load("outputs/models/final_logistic_model_balanced.joblib")
model, threshold, feature_names = model_bundle

st.title("🎢 Incident Risk Predictor")
st.markdown("Estimate the risk of an incident based on visitor and ride context.")

# Layout with extra space between columns
left, spacer, right = st.columns([1.3, 0.3, 1.7])

with left:
    st.header("Inputs")
    ride_type = st.selectbox("Ride type", [
        "flat_ride", "simulator", "water_ride", "transport", "dark_ride",
        "drop_tower", "roller_coaster", "carousel", "theater", "interactive"
    ])
    duration_min = st.slider("Ride duration (minutes)", 1, 15, 5)
    age = st.slider("Visitor age", 1, 90, 30)
    gender = st.selectbox("Gender", ["F", "M"])
    first_time = st.checkbox("First-time visitor")
    season = st.selectbox("Season", ["winter", "spring", "summer", "fall"])
    is_weekend = st.checkbox("Weekend")
    temp_max = st.slider("Max temperature (°C)", 0, 45, 30)
    precip = st.slider("Precipitation (mm)", 0.0, 50.0, 5.0)

# Derived feature
age_group = pd.cut([age], bins=[0,9,19,29,39,49,59,69,120],
                   labels=["<10","10–19","20–29","30–39","40–49","50–59","60–69","70+"])[0]

# Assemble input
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
input_encoded = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)

# Predict
prob = model.predict_proba(input_encoded)[0][1]
visual_threshold = 0.15
label = prob >= visual_threshold

with right:
    st.subheader("🧮 Estimated risk")
    st.markdown(f"### `{prob:.2%}`")
    if label:
        st.error("⚠️ High risk of incident predicted.")
    else:
        st.success("✅ Low risk predicted.")

    # Comparison chart
    st.markdown("### 🔎 User profile vs average")
    user_values = {
        "age": age,
        "temperature_max": temp_max,
        "duration_min": duration_min,
    }
    mean_values = {
        "age": 35,
        "temperature_max": 25,
        "duration_min": 5,
    }
    df_compare = pd.DataFrame({
        "Feature": list(user_values.keys()),
        "User": list(user_values.values()),
        "Average": list(mean_values.values())
    })
    chart = alt.Chart(df_compare.melt("Feature")).mark_bar().encode(
        x=alt.X("Feature:N", title="Feature"),
        y=alt.Y("value:Q", title="Value"),
        color="variable:N"
    ).properties(title="User vs average profile")
    st.altair_chart(chart, use_container_width=True)

    # Risk by age group
    st.markdown("### 📊 Risk by age group")
    risk_by_age_group = pd.DataFrame({
        "Age Group": ["<10","10–19","20–29","30–39","40–49","50–59","60–69","70+"],
        "Incident Rate": [0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    })
    bar = alt.Chart(risk_by_age_group).mark_bar().encode(
        x="Age Group",
        y=alt.Y("Incident Rate", axis=alt.Axis(format="%")),
        tooltip=["Age Group", "Incident Rate"]
    ).properties(title="Historical incident risk by age group")
    st.altair_chart(bar, use_container_width=True)

    # Contributing factors (simulated)
    st.markdown("### 💡 Top contributing factors (simulated)")
    contrib = pd.DataFrame({
        "Feature": ["age_group_70+", "ride_type_simplified_drop_tower", "temperature_max", "first_time_visitor"],
        "Weight": [0.82, 0.61, 0.48, 0.33]
    })
    st.bar_chart(contrib.set_index("Feature"))
