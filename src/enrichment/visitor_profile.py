import pandas as pd
import numpy as np

def enrich_visitor_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich visitor features:
    - age_group
    - is_minor
    - is_senior
    - simulated_medical_condition
    - first_time_visitor
    """
    df = df.copy()

    # Age bins
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 120]
    labels = ["<10", "10–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    # Booleans
    df["is_minor"] = df["age"] < 18
    df["is_senior"] = df["age"] >= 65

    # Simulated medical conditions (categorical)
    np.random.seed(42)
    conditions = ["none", "asthma", "cardiac", "diabetes", "epilepsy"]
    probs = [0.70, 0.10, 0.08, 0.07, 0.05]
    df["simulated_medical_condition"] = np.random.choice(conditions, size=len(df), p=probs)

    # Simulated first time visitor (boolean)
    df["first_time_visitor"] = np.random.rand(len(df)) < 0.4  # 40% first-timers

    return df
