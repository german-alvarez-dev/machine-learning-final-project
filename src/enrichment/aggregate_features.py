import pandas as pd

def enrich_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aggregate features:
    - ride_incident_count: number of incidents per ride
    - park_incident_count: number of incidents per park
    - ride_type_simplified: grouped ride type into coarse categories
    """
    df = df.copy()

    # Total incidents per ride
    ride_counts = df["ride_name"].value_counts()
    df["ride_incident_count"] = df["ride_name"].map(ride_counts)

    # Total incidents per park
    park_counts = df["theme_park"].value_counts()
    df["park_incident_count"] = df["theme_park"].map(park_counts)

    # Ride type simplified
    def simplify_ride_type(rt):
        if not isinstance(rt, str):
            return "unknown"
        rt = rt.lower()
        if "coaster" in rt or "steel" in rt:
            return "coaster"
        if "simulator" in rt:
            return "simulator"
        if "dark" in rt:
            return "dark_ride"
        if "water" in rt or "log flume" in rt or "rapids" in rt:
            return "water_ride"
        if "carousel" in rt or "flat" in rt:
            return "flat_ride"
        return "other"

    df["ride_type_simplified"] = df["ride_type"].apply(simplify_ride_type)

    return df
