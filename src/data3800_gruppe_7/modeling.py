"""Train the injury models and save them for the evaluation.

Run from the project root with:

    uv run python -m data3800_gruppe_7.modeling

The script cleans data/raw_dataset.csv with cleaning.py, splits it into a
training set and a test set, and trains four classifiers. The fitted models
and their predictions on the test set are saved in output/models/.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data3800_gruppe_7.cleaning import (
    FEATURES,
    REQUIRED_COLUMNS,
    clean_crashes,
    get_model_data,
)

# This file is two folders below the project root. Building the paths from it
# makes them work from the notebooks folder too.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
PREDICTIONS_PATH = MODELS_DIR / "predictions.csv"

# Reading only these columns is much faster than reading all 49. The cleaning
# needs REQUIRED_COLUMNS, and the predictions need the crash IDs.
RAW_COLUMNS = ["CRASH_RECORD_ID"] + REQUIRED_COLUMNS

NUMERIC_FEATURES = ["POSTED_SPEED_LIMIT"]
# Every other feature is a category. That includes the hour, because injury
# risk does not rise steadily with it.
CATEGORICAL_FEATURES = [column for column in FEATURES if column not in NUMERIC_FEATURES]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Starting values for the models. We have not tuned them yet.
TREE_MAX_DEPTH = 8
FOREST_TREES = 200
FOREST_MIN_LEAF_SIZE = 50
ALL_CPU_CORES = -1


def main():
    X, y, crash_ids = load_model_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipelines = train_models(X_train, y_train)
    predictions = predict_test_set(pipelines, X_test, y_test, crash_ids)
    save_models(pipelines)
    save_predictions(predictions)
    print(f"Saved {len(pipelines)} models and their predictions in {MODELS_DIR}")


def load_model_data():
    """Return the features X, the target y and the ID of each crash."""
    raw = pd.read_csv(RAW_DATA_PATH, usecols=RAW_COLUMNS)
    crashes = clean_crashes(raw)
    X, y = get_model_data(crashes)
    return X, y, crashes["CRASH_RECORD_ID"]


def split_train_test(X, y):
    """Hold out 20 % as a test set with the same injury share as the training set."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def train_models(X_train, y_train):
    """Fit each model in a pipeline, so the preprocessing only sees training data."""
    pipelines = {}
    for name, model in build_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
    return pipelines


def build_models():
    """Return the four models we compare, by name."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=FOREST_TREES,
            min_samples_leaf=FOREST_MIN_LEAF_SIZE,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=ALL_CPU_CORES,
        ),
    }


def build_preprocessor():
    """One-hot encode the categories, and fill in and scale the speed limit."""
    numeric = Pipeline(
        [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def predict_test_set(pipelines, X_test, y_test, crash_ids):
    """Return one row per crash and model with the true and predicted injury."""
    test_ids = crash_ids.loc[X_test.index].to_numpy()
    frames = []
    for name, pipeline in pipelines.items():
        # Column 1 of predict_proba is the probability of class 1, injury.
        injury_probability = pipeline.predict_proba(X_test)[:, 1]
        frame = pd.DataFrame(
            {
                "crash_record_id": test_ids,
                "model": name,
                "y_true": y_test.to_numpy(),
                "y_pred": pipeline.predict(X_test),
                "y_proba": injury_probability,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_models(pipelines):
    """Save each fitted pipeline as output/models/<name>.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")


def save_predictions(predictions):
    """Save the test set predictions in PREDICTIONS_PATH."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    main()
