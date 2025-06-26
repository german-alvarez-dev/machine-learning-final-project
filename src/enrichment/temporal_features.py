import pandas as pd

def enrich_temporal_features(df: pd.DataFrame, date_col: str = "incident_date_parsed") -> pd.DataFrame:
    """
    Add temporal features based on the parsed incident date:
    - day_of_week
    - month
    - season
    - is_weekend
    - is_summer
    """
    df = df.copy()

    df["day_of_week"] = df[date_col].dt.day_name()
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    df["is_summer"] = df["month"].isin([6, 7, 8])

    def map_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    df["season"] = df["month"].apply(map_season)

    return df
