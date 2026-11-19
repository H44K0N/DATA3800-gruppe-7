import pandas as pd

from data3800_gruppe_7.modeling import (
    load_crashes,
    select_best,
    split_by_period,
)
from data3800_gruppe_7.risk import NUMBER_LIMITS

# Load the cleaned crashes and split them by period
crashes = load_crashes()
train, validation, test = split_by_period(crashes)

# Check that every crash is in exactly one period
assert len(train) + len(validation) + len(test) == len(crashes)
all_ids = pd.concat([train, validation, test])["CRASH_RECORD_ID"]
assert all_ids.is_unique

# Check that the periods follow each other in time
assert train["CRASH_DATE"].max() < validation["CRASH_DATE"].min()
assert validation["CRASH_DATE"].max() < test["CRASH_DATE"].min()

# Check that the limits of the crash scorer are the range of the cleaned crashes
for column, limits in NUMBER_LIMITS.items():
    assert (crashes[column].min(), crashes[column].max()) == limits

# Check that the best setting is the one with the highest validation score
scores = pd.DataFrame(
    {
        "model": ["tree", "tree", "forest"],
        "setting": ["shallow", "deep", "default"],
        "validation_pr_auc": [0.30, 0.25, 0.40],
    }
)
candidates = {
    ("tree", "shallow"): "shallow tree",
    ("tree", "deep"): "deep tree",
    ("forest", "default"): "forest",
}
assert select_best(candidates, scores) == {"tree": "shallow tree", "forest": "forest"}

print("ALL TESTS PASSED!")
print("Training crashes:", len(train))
print("Validation crashes:", len(validation))
print("Test crashes:", len(test))
