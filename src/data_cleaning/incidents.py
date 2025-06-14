# src/data_cleaning/incidents.py

import pandas as pd


def parse_incident_dates_dual_format(df: pd.DataFrame, col: str = "Incident_date") -> pd.DataFrame:
    """
    Parse a date column with mixed 2-digit and 4-digit year formats.
    Adds a new column: 'Incident_date_parsed'.
    
    Parameters:
        df: Raw input DataFrame.
        col: Name of the column containing date strings.
        
    Returns:
        df: DataFrame with a new column 'Incident_date_parsed' (as datetime).
    """
    df = df.copy()

    # Normalize separators and remove invisible characters
    df[col] = df[col].astype(str).str.strip().str.replace(r"[–—−]", "/", regex=True)

    # Try parsing with 4-digit year
    parsed_4 = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    # Try parsing with 2-digit year where parsing failed
    parsed_2 = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

    # Combine both results
    df["Incident_date_parsed"] = parsed_4.combine_first(parsed_2)

    return df
