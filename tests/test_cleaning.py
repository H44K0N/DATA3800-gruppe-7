from pathlib import Path

import pandas as pd
import pytest

from data3800_gruppe_7.cleaning import (
    FEATURES,
    clean_crashes,
    get_model_data,
)

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_dataset.csv"

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data missing")


@pytest.fixture(scope="module")
def cleaned():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    clean_df = clean_crashes(df)
    X, y = get_model_data(clean_df)
    return clean_df, X, y


def test_record_counts(cleaned):
    clean_df, _, y = cleaned
    assert len(clean_df) == 108903
    assert y.sum() == 17902


def test_model_features(cleaned):
    _, X, _ = cleaned
    assert X.shape[1] == 9
    assert X.columns.tolist() == FEATURES


def test_no_injury_leakage(cleaned):
    _, X, _ = cleaned
    assert "INJURIES_TOTAL" not in X.columns
    assert "MOST_SEVERE_INJURY" not in X.columns
    assert "injury" not in X.columns


def test_target(cleaned):
    _, _, y = cleaned
    assert y.notna().all()
    assert y.isin([0, 1]).all()


def test_speed_limits(cleaned):
    _, X, _ = cleaned
    assert (X["POSTED_SPEED_LIMIT"] == 0).sum() == 0
    assert X["POSTED_SPEED_LIMIT"].isna().sum() == 198


def test_crash_hours(cleaned):
    _, X, _ = cleaned
    assert X["CRASH_HOUR"].dropna().between(0, 23).all()


def test_x_and_y_match(cleaned):
    _, X, y = cleaned
    assert X.index.equals(y.index)
