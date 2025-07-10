# test_cases.py

import pandas as pd
from joblib import load

# Load the model, threshold and expected feature names
model, threshold, features = load("outputs/models/final_logistic_model_balanced.joblib")

# Define multiple test cases with various risk profiles
test_cases = [
    # Low risk: adult, calm ride, summer weekday
    {
        "ride_type_simplified": "dark_ride",
        "duration_min": 4,
        "age": 42,
        "age_group": "40–49",
        "gender": "M",
        "first_time_visitor": True,
        "season": "summer",
        "is_weekend": False,
        "temperature_max": 31.5,
        "precipitation_sum": 0.0
    },
    # Moderate risk: senior, simulator, winter weekend
    {
        "ride_type_simplified": "simulator",
        "duration_min": 10,
        "age": 66,
        "age_group": "60–69",
        "gender": "F",
        "first_time_visitor": False,
        "season": "winter",
        "is_weekend": True,
        "temperature_max": 15.2,
        "precipitation_sum": 4.5
    },
    # Low risk: child, short ride, mild weather
    {
        "ride_type_simplified": "other",
        "duration_min": 6,
        "age": 9,
        "age_group": "<10",
        "gender": "M",
        "first_time_visitor": True,
        "season": "spring",
        "is_weekend": True,
        "temperature_max": 22.4,
        "precipitation_sum": 2.1
    },
    # High risk: very elderly, long ride, rainy day
    {
        "ride_type_simplified": "simulator",
        "duration_min": 15,
        "age": 89,
        "age_group": "70+",
        "gender": "F",
        "first_time_visitor": False,
        "season": "fall",
        "is_weekend": False,
        "temperature_max": 14.0,
        "precipitation_sum": 12.0
    },
    # Medium risk: teenager, coaster, summer weekend
    {
        "ride_type_simplified": "roller_coaster",
        "duration_min": 8,
        "age": 17,
        "age_group": "10–19",
        "gender": "M",
        "first_time_visitor": True,
        "season": "summer",
        "is_weekend": True,
        "temperature_max": 35.0,
        "precipitation_sum": 0.0
    },
    # Edge: adult in extreme rain
    {
        "ride_type_simplified": "dark_ride",
        "duration_min": 7,
        "age": 33,
        "age_group": "30–39",
        "gender": "F",
        "first_time_visitor": True,
        "season": "fall",
        "is_weekend": False,
        "temperature_max": 18.5,
        "precipitation_sum": 25.0
    },
    # Edge: perfect storm – senior, long simulator, stormy
    {
        "ride_type_simplified": "simulator",
        "duration_min": 20,
        "age": 79,
        "age_group": "70+",
        "gender": "M",
        "first_time_visitor": True,
        "season": "winter",
        "is_weekend": True,
        "temperature_max": 8.0,
        "precipitation_sum": 30.0
    },
    # Control: adult, calm ride, normal conditions
    {
        "ride_type_simplified": "other",
        "duration_min": 5,
        "age": 35,
        "age_group": "30–39",
        "gender": "F",
        "first_time_visitor": False,
        "season": "spring",
        "is_weekend": False,
        "temperature_max": 24.0,
        "precipitation_sum": 1.0
    }
]

# Create dataframe
input_df = pd.DataFrame(test_cases)

# One-hot encode and align with expected model features
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# Predict
probs = model.predict_proba(input_encoded)[:, 1]
preds = probs > threshold

# Output results
for i, prob in enumerate(probs):
    print(f"\nTest case #{i+1}")
    print("Probability of incident:", round(prob * 100, 2), "%")
    print("Prediction:", "Incident" if preds[i] else "No incident")
