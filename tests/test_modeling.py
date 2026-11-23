import pandas as pd
import pytest

from data3800_gruppe_7.modeling import (
    RAW_DATA_PATH,
    load_crashes,
    select_best,
    split_by_period,
)
from data3800_gruppe_7.risk import NUMBER_LIMITS

needs_raw_data = pytest.mark.skipif(
    not RAW_DATA_PATH.exists(), reason="raw data missing"
)


@pytest.fixture(scope="module")
def crashes():
    return load_crashes()


@pytest.fixture(scope="module")
def periods(crashes):
    return split_by_period(crashes)


@needs_raw_data
def test_every_crash_is_in_one_period(crashes, periods):
    train, validation, test = periods
    assert len(train) + len(validation) + len(test) == len(crashes)
    all_ids = pd.concat([train, validation, test])["CRASH_RECORD_ID"]
    assert all_ids.is_unique


@needs_raw_data
def test_periods_follow_each_other(periods):
    train, validation, test = periods
    assert train["CRASH_DATE"].max() < validation["CRASH_DATE"].min()
    assert validation["CRASH_DATE"].max() < test["CRASH_DATE"].min()


@needs_raw_data
def test_scorer_limits_match_data(crashes):
    # The limits of the crash scorer are the range of the cleaned crashes
    for column, limits in NUMBER_LIMITS.items():
        assert (crashes[column].min(), crashes[column].max()) == limits


def test_select_best():
    # The best setting is the one with the highest validation score
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
    assert select_best(candidates, scores) == {
        "tree": "shallow tree",
        "forest": "forest",
    }
