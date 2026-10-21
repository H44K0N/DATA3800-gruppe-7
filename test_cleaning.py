
import pandas as pd

from data3800_gruppe_7.cleaning import (
    FEATURES,
    clean_crashes,
    get_model_data,
)

# Load and clean the dataset
df = pd.read_csv(
    "data/raw_dataset.csv",
    low_memory=False
)

clean_df = clean_crashes(df)
X, y = get_model_data(clean_df)

# Check the number of records
assert len(clean_df) == 108903
assert y.sum() == 17902

# Check the model features
assert X.shape[1] == 5
assert X.columns.tolist() == FEATURES

# Check that injury information is not in X
assert "INJURIES_TOTAL" not in X.columns
assert "MOST_SEVERE_INJURY" not in X.columns
assert "injury" not in X.columns

# Check the target
assert y.notna().all()
assert y.isin([0, 1]).all()

# Check the speed limits
assert (X["POSTED_SPEED_LIMIT"] == 0).sum() == 0
assert X["POSTED_SPEED_LIMIT"].isna().sum() == 198

# Check crash hours
assert X["CRASH_HOUR"].dropna().between(0, 23).all()

# Check that X and y match
assert X.index.equals(y.index)

print("ALL TESTS PASSED!")
print("Cleaned records:", len(clean_df))
print("Injury cases:", y.sum())
print("No injury cases:", (y == 0).sum())