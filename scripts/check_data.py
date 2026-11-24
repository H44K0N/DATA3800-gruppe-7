import pandas as pd

df = pd.read_csv(
    "data/raw_dataset.csv",
    low_memory=False
)

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nSelected columns:")
columns = [
    "CRASH_RECORD_ID",
    "CRASH_DATE",
    "POSTED_SPEED_LIMIT",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "CRASH_HOUR",
    "INJURIES_TOTAL",
]

print(df[columns].head())

print("\nMissing injury values:")
print(df["INJURIES_TOTAL"].isna().sum())

print("\nDuplicate crash IDs:")
print(df["CRASH_RECORD_ID"].duplicated().sum())

# Convert crash dates to datetime
df["CRASH_DATE"] = pd.to_datetime(
    df["CRASH_DATE"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce"
)

print("\nInvalid dates:")
print(df["CRASH_DATE"].isna().sum())

# Check crash counts by year
print("\nCrash counts by year:")
print(
    df["CRASH_DATE"]
    .dt.year
    .value_counts()
    .sort_index()
)

# Select only crashes from 2025
df_2025 = df[
    df["CRASH_DATE"].dt.year == 2025
].copy()

print("\n2025 total crashes:")
print(len(df_2025))

print("\n2025 missing injury values:")
print(df_2025["INJURIES_TOTAL"].isna().sum())

print("\n2025 duplicate crash IDs:")
print(
    df_2025["CRASH_RECORD_ID"]
    .duplicated()
    .sum()
)