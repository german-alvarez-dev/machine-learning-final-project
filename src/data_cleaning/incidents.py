import pandas as pd


def parse_incident_date_column(df: pd.DataFrame, col: str = "incident_date") -> pd.DataFrame:
    """
    Parse a column containing mixed date formats (MM/DD/YY and MM/DD/YYYY).
    Creates a new column 'incident_date_parsed' as datetime.
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.replace(r"[–—−]", "/", regex=True)

    # Try parsing with 4-digit year
    parsed_4 = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    # Then try 2-digit year where previous attempt failed
    parsed_2 = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

    # Combine both results
    df["incident_date_parsed"] = parsed_4.combine_first(parsed_2)

    return df

import re

def split_age_gender_column(df: pd.DataFrame, col: str = "age_gender") -> pd.DataFrame:
    """
    Split the combined 'age_gender' column into two new columns: 'age' (int) and 'gender' (M/F).
    """
    df = df.copy()

    # Extract numeric age
    df["age"] = df[col].str.extract(r"(\d{1,3})")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Extract gender from suffix
    df["gender"] = df[col].str.extract(r"(yo[f|m])", flags=re.IGNORECASE)
    df["gender"] = df["gender"].str[-1].str.upper()  # take last letter: 'f' or 'm'

    # Optional: clean up unexpected values
    df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]))

    return df


def classify_incident_type(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    """
    Classify incidents into high-level categories based on keywords in the description column.
    Adds a new column called 'incident_type'.
    """
    df = df.copy()

    def map_type(text: str) -> str:
        text = str(text).lower()

        if any(x in text for x in [
            "fracture", "fractured", "broken", "sprain", "dislocation",
            "injury", "wound", "laceration", "lacerated", "cut", "bleed", "amputation"
        ]):
            return "trauma"

        if any(x in text for x in [
            "fall", "fell", "trip", "slip"
        ]):
            return "fall"

        if any(x in text for x in [
            "motion sickness", "dizzy", "dizziness", "nausea", "nauseous",
            "vomit", "ill", "disoriented", "didn't feel well", "didn’t feel well"
        ]):
            return "motion"

        if any(x in text for x in [
            "chest pain", "chest discomfort", "heart", "heartburn", "unconscious",
            "loss of consciousness", "seizure", "numbness", "collapsed", "stroke",
            "head pain", "pain", "spots", "hermatemesis", "stomach", "fainting",
            "fainted", "syncope", "confusion", "confused", "weakness", "lightheaded",
            "altered mental status", "lowered level of consciousness", "headache",
            "passed away", "dead", "unresponsive"
        ]):
            return "medical"

        if any(x in text for x in [
            "pre-existing", "preexisting"
        ]):
            return "pre_existing"

        return "other"

    df["incident_type"] = df[col].apply(map_type)
    return df
