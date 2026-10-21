
import pandas as pd

from data3800_gruppe_7.cleaning import (
    clean_crashes,
    get_model_data,
)

# Load the original dataset
df = pd.read_csv(
    "data/raw_dataset.csv",
    low_memory=False
)

# Clean the data
clean_df = clean_crashes(df)

# Get the features and target
X, y = get_model_data(clean_df)

# Show the results
print("Original records:", len(df))
print("Cleaned records:", len(clean_df))
print("Injury cases:", y.sum())
print("No injury cases:", (y == 0).sum())
print("Model features:", X.columns.tolist())

# Save the cleaned dataset
clean_df.to_csv(
    "data/cleaned_crashes.csv",
    index=False
)

print("Cleaned data saved!")