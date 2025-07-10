import pandas as pd
import numpy as np

def generate_negative_cases(df_positives: pd.DataFrame, n_negatives: int = 3410, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic negative cases (no incident) based on the structure of df_positives.
    Includes realistic values sampled from the real dataset.
    """
    np.random.seed(seed)
    df = df_positives.copy()

    # Base pools for sampling
    rides = df[["ride_name", "theme_park", "ride_type", "ride_type_simplified", "ride_incident_count", "duration_min"]].drop_duplicates()
    dates = df["incident_date_parsed"].dropna().unique()
    ages = df["age"].dropna()
    genders = df["gender"].dropna().unique()

    use_medical = "simulated_medical_condition" in df.columns
    conditions = df["simulated_medical_condition"].dropna().unique() if use_medical else ["none"]

    weather = df[["incident_date_parsed", "theme_park", "temperature_max", "temperature_min", "precipitation_sum"]].dropna().drop_duplicates()

    synthetic = []
    for _ in range(n_negatives):
        ride = rides.sample(1).iloc[0]
        date = pd.to_datetime(np.random.choice(dates))
        age = np.random.choice(ages)
        gender = np.random.choice(genders)
        first_time = np.random.rand() < 0.4
        condition = np.random.choice(conditions)

        age_group = pd.cut([age], bins=[0,9,19,29,39,49,59,69,120], labels=["<10","10–19","20–29","30–39","40–49","50–59","60–69","70+"])[0]
        is_minor = age < 18
        is_senior = age >= 65

        month = date.month
        day_of_week = date.day_name()
        is_weekend = day_of_week in ["Saturday", "Sunday"]
        is_summer = month in [6, 7, 8]
        season = ("winter" if month in [12, 1, 2] else
                  "spring" if month in [3, 4, 5] else
                  "summer" if month in [6, 7, 8] else
                  "fall")

        w = weather[
            (weather["incident_date_parsed"] == date) &
            (weather["theme_park"] == ride["theme_park"])
        ]
        if not w.empty:
            wrow = w.iloc[0]
            temp_max = wrow["temperature_max"]
            temp_min = wrow["temperature_min"]
            precip = wrow["precipitation_sum"]
        else:
            temp_max = temp_min = precip = np.nan

        row = {
            "incident_occurred": False,
            "incident_date_parsed": date,
            "ride_name": ride["ride_name"],
            "theme_park": ride["theme_park"],
            "ride_type": ride["ride_type"],
            "ride_type_simplified": ride["ride_type_simplified"],
            "ride_incident_count": ride["ride_incident_count"],
            "duration_min": ride["duration_min"],
            "age": age,
            "gender": gender,
            "first_time_visitor": first_time,
            "age_group": age_group,
            "is_minor": is_minor,
            "is_senior": is_senior,
            "day_of_week": day_of_week,
            "month": month,
            "season": season,
            "is_weekend": is_weekend,
            "is_summer": is_summer,
            "temperature_max": temp_max,
            "temperature_min": temp_min,
            "precipitation_sum": precip
        }
        if use_medical:
            row["simulated_medical_condition"] = condition

        synthetic.append(row)

    df_negatives = pd.DataFrame(synthetic)
    return df_negatives
