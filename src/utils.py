import pandas as pd
import re

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all column names to snake_case and lowercase.
    """
    df = df.copy()
    df.columns = [
        re.sub(r"\W+", "_", col).strip().lower() for col in df.columns
    ]
    return df


def normalize_text_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Standardize text in the 'Company' column: strip, uppercase, and unify known variants.
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.upper()

    # Optional: unify known aliases
    df[col] = df[col].replace({
        "SIXFLAGS": "SIX FLAGS",
        "SIX-FLAGS": "SIX FLAGS"
    })

    return df