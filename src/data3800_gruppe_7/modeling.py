"""Train the injury models and save them for the evaluation.

Run from the project root with:

    uv run python -m data3800_gruppe_7.modeling

The script cleans data/raw_dataset.csv with cleaning.py and splits the year
into a training, a validation and a test period. It fits every model setting
on the training period and keeps the setting of each model that scores best
on the validation period. Those models and their predictions on the test
period are saved in output/models/.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data3800_gruppe_7.cleaning import (
    CONDITION_FEATURES,
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

NUMERIC_FEATURES = ["POSTED_SPEED_LIMIT", "NUM_UNITS"]
# Every other feature is a category. That includes the hour, because injury
# risk does not rise steadily with it.
CATEGORICAL_FEATURES = [column for column in FEATURES if column not in NUMERIC_FEATURES]

# The models learn from January to August. We choose their settings on
# September and October, and test them once on November and December.
LAST_TRAINING_MONTH = 8
LAST_VALIDATION_MONTH = 10

RANDOM_STATE = 42

# A logistic regression that only gets the five conditions. It shows how far
# the first version of the models came, and what the new features add.
CONDITIONS_ONLY = "conditions_only"

# The settings we try for the trees. A deeper tree, and a forest with smaller
# leaves, can follow the training set more closely.
TREE_DEPTHS = [4, 6, 8, 12, 16]
FOREST_LEAF_SIZES = [1, 5, 10, 25, 50]
FOREST_TREES = 200
ALL_CPU_CORES = -1


def main():
    train, validation, test = split_by_period(load_crashes())
    candidates = fit_candidates(train)
    scores = score_candidates(candidates, train, validation)
    pipelines = select_best(candidates, scores)
    predictions = predict_test_set(pipelines, test)
    save_models(pipelines)
    save_predictions(predictions)
    print(scores.round(3).to_string(index=False))
    print(f"Saved {len(pipelines)} models and their predictions in {MODELS_DIR}")


def load_crashes():
    """Return the cleaned crashes, one row per crash."""
    raw = pd.read_csv(RAW_DATA_PATH, usecols=RAW_COLUMNS)
    return clean_crashes(raw)


def split_by_period(crashes):
    """Return the crashes of the training, validation and test periods.

    The periods follow each other in time, so the models are tested on crashes
    that happened after the ones they learned from.
    """
    month = crashes["CRASH_DATE"].dt.month
    train = crashes[month <= LAST_TRAINING_MONTH]
    validation = crashes[month.between(LAST_TRAINING_MONTH + 1, LAST_VALIDATION_MONTH)]
    test = crashes[month > LAST_VALIDATION_MONTH]
    return train, validation, test


def fit_candidates(train):
    """Fit every model setting on the training set, by (model name, setting)."""
    X_train, y_train = get_model_data(train)
    return {
        key: pipeline.fit(X_train, y_train)
        for key, pipeline in build_candidates().items()
    }


def build_candidates():
    """Return an unfitted pipeline for every setting, by (model name, setting).

    Only the decision tree and the random forest have more than one setting.
    """
    candidates = {
        ("baseline", "most frequent class"): build_pipeline(
            DummyClassifier(strategy="most_frequent"), FEATURES
        ),
        (CONDITIONS_ONLY, "five conditions"): build_pipeline(
            LogisticRegression(class_weight="balanced"), CONDITION_FEATURES
        ),
        ("logistic_regression", "all features"): build_pipeline(
            LogisticRegression(class_weight="balanced"), FEATURES
        ),
    }
    for depth in TREE_DEPTHS:
        candidates["decision_tree", f"max_depth={depth}"] = build_pipeline(
            DecisionTreeClassifier(
                max_depth=depth, class_weight="balanced", random_state=RANDOM_STATE
            ),
            FEATURES,
        )
    for leaf_size in FOREST_LEAF_SIZES:
        candidates["random_forest", f"min_samples_leaf={leaf_size}"] = build_pipeline(
            RandomForestClassifier(
                n_estimators=FOREST_TREES,
                min_samples_leaf=leaf_size,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=ALL_CPU_CORES,
            ),
            FEATURES,
        )
    return candidates


def build_pipeline(model, features):
    """Return a pipeline that prepares the features and then fits the model.

    The preprocessing is a step of the pipeline, so it only learns from the
    crashes the pipeline is fitted on.
    """
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])


def build_preprocessor(features):
    """One-hot encode the categories, and fill in and scale the numbers."""
    numeric = Pipeline(
        [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                [column for column in features if column in NUMERIC_FEATURES],
            ),
            (
                "categorical",
                categorical,
                [column for column in features if column not in NUMERIC_FEATURES],
            ),
        ]
    )


def score_candidates(candidates, train, validation):
    """Return the PR-AUC of every candidate on the training and validation sets.

    A candidate that scores much better on the training set than on the
    validation set has learned the training crashes by heart (overfitting).
    """
    rows = [
        {
            "model": name,
            "setting": setting,
            "train_pr_auc": pr_auc(pipeline, train),
            "validation_pr_auc": pr_auc(pipeline, validation),
        }
        for (name, setting), pipeline in candidates.items()
    ]
    return pd.DataFrame(rows)


def pr_auc(pipeline, crashes):
    """Return the PR-AUC of a fitted pipeline on a set of crashes."""
    X, y = get_model_data(crashes)
    # Column 1 of predict_proba is the probability of class 1, injury.
    return average_precision_score(y, pipeline.predict_proba(X)[:, 1])


def select_best(candidates, scores):
    """Return the setting of each model with the best PR-AUC on the validation set."""
    best = scores.loc[scores.groupby("model", sort=False)["validation_pr_auc"].idxmax()]
    return {row.model: candidates[row.model, row.setting] for row in best.itertuples()}


def predict_test_set(pipelines, test):
    """Return one row per crash and model with the true and predicted injury."""
    X_test, y_test = get_model_data(test)
    frames = []
    for name, pipeline in pipelines.items():
        frame = pd.DataFrame(
            {
                "crash_record_id": test["CRASH_RECORD_ID"].to_numpy(),
                "model": name,
                "y_true": y_test.to_numpy(),
                "y_pred": pipeline.predict(X_test),
                "y_proba": pipeline.predict_proba(X_test)[:, 1],
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
