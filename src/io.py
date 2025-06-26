import pandas as pd

def load_incident_data(path: str) -> pd.DataFrame:
    """
    Load incident data from CSV.
    """
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save processed DataFrame to CSV or Parquet.
    """
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file format")