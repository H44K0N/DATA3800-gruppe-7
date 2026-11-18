
import pandas as pd

# The conditions around the crash, which the first models used alone
CONDITION_FEATURES = [
    "POSTED_SPEED_LIMIT",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "CRASH_HOUR",
]

# The layout of the road where the crash happened
ROAD_FEATURES = [
    "TRAFFIC_CONTROL_DEVICE",
    "TRAFFICWAY_TYPE",
]

# How the crash happened: what was hit, and how many vehicles and people took part
CRASH_FEATURES = [
    "FIRST_CRASH_TYPE",
    "NUM_UNITS",
]

FEATURES = CONDITION_FEATURES + ROAD_FEATURES + CRASH_FEATURES

# Columns that clean_crashes needs in the raw data
REQUIRED_COLUMNS = ["CRASH_DATE", "INJURIES_TOTAL"] + FEATURES


def clean_crashes(df):
    df = df.copy()

    # Check that the necessary columns exist
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Keep crashes from 2025
    df["CRASH_DATE"] = pd.to_datetime(
        df["CRASH_DATE"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    df = df[df["CRASH_DATE"].dt.year == 2025].copy()

    # Remove records without a valid injury total
    df["INJURIES_TOTAL"] = pd.to_numeric(
        df["INJURIES_TOTAL"],
        errors="coerce"
    )

    df = df[
        df["INJURIES_TOTAL"].notna()
        & (df["INJURIES_TOTAL"] >= 0)
    ].copy()

    # Create the target: 0 = no injury, 1 = injury
    df["injury"] = (df["INJURIES_TOTAL"] > 0).astype(int)

    # Convert numerical columns
    df["POSTED_SPEED_LIMIT"] = pd.to_numeric(
        df["POSTED_SPEED_LIMIT"],
        errors="coerce"
    )

    df["CRASH_HOUR"] = pd.to_numeric(
        df["CRASH_HOUR"],
        errors="coerce"
    )

    df["NUM_UNITS"] = pd.to_numeric(
        df["NUM_UNITS"],
        errors="coerce"
    )

    # Treat zero and negative speed limits as missing
    df.loc[
        df["POSTED_SPEED_LIMIT"] <= 0,
        "POSTED_SPEED_LIMIT"
    ] = float("nan")

    # Crash hour must be between 0 and 23
    df.loc[
        ~df["CRASH_HOUR"].between(0, 23),
        "CRASH_HOUR"
    ] = float("nan")

    return df


def get_model_data(df):
    X = df[FEATURES].copy()
    y = df["injury"].copy()

    return X, y
