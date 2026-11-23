import pandas as pd

df = pd.read_csv(
    "data/cleaned_crashes.csv",
    low_memory=False
)

features = [
    "POSTED_SPEED_LIMIT",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "CRASH_HOUR",
]

print("\n=== MISSING VALUES ===")
print(df[features].isna().sum())

print("\n=== SPEED LIMIT ===")
print(df["POSTED_SPEED_LIMIT"].value_counts().sort_index())

print("\n=== ZERO SPEED LIMIT ===")
print((df["POSTED_SPEED_LIMIT"] == 0).sum())

print("\n=== WEATHER CONDITION ===")
print(df["WEATHER_CONDITION"].value_counts(dropna=False))

print("\n=== LIGHTING CONDITION ===")
print(df["LIGHTING_CONDITION"].value_counts(dropna=False))

print("\n=== ROAD SURFACE ===")
print(df["ROADWAY_SURFACE_COND"].value_counts(dropna=False))

print("\n=== CRASH HOUR ===")
print(df["CRASH_HOUR"].value_counts(dropna=False).sort_index())