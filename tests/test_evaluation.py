import numpy as np
import pandas as pd
import pytest

from data3800_gruppe_7.evaluation import (
    EVALUATION_DIR,
    PREFERRED_MODEL,
    THRESHOLD,
    gains_table,
    load_models,
    load_typical_crash,
    threshold_metrics,
)
from data3800_gruppe_7.modeling import MODELS_DIR

needs_outputs = pytest.mark.skipif(
    not (EVALUATION_DIR / "odds_ratios.csv").exists(),
    reason="evaluation output missing",
)
needs_models = pytest.mark.skipif(
    not (MODELS_DIR / f"{PREFERRED_MODEL}.joblib").exists(),
    reason="trained models missing",
)


def test_threshold_metrics():
    # Four crashes, two with an injury, with scores we can check by hand
    injury = np.array([0, 1, 0, 1])
    risk_score = np.array([0.10, 0.42, 0.58, 0.90])
    by_threshold = threshold_metrics(injury, risk_score).set_index("threshold")

    # At 0.30 three crashes are flagged, and both injuries are among them
    assert by_threshold.loc[0.30, "flagged_share"] == 0.75
    assert np.isclose(by_threshold.loc[0.30, "precision"], 2 / 3)
    assert by_threshold.loc[0.30, "recall"] == 1.0

    # At 0.50 two crashes are flagged, and one of them has an injury
    assert by_threshold.loc[0.50, "flagged_share"] == 0.5
    assert by_threshold.loc[0.50, "precision"] == 0.5
    assert by_threshold.loc[0.50, "recall"] == 0.5
    assert by_threshold.loc[0.50, "f1"] == 0.5

    # At 0.80 only the crash with the highest score is flagged
    assert by_threshold.loc[0.80, "flagged_share"] == 0.25
    assert by_threshold.loc[0.80, "precision"] == 1.0
    assert by_threshold.loc[0.80, "recall"] == 0.5


def test_gains_table():
    # Twenty crashes with the scores 0.05, 0.10 and so on up to 1.00. The three
    # with the highest scores and the one with the lowest score have an injury.
    risk_score = np.arange(1, 21) / 20
    injury = np.zeros(20, dtype=int)
    injury[[0, 17, 18, 19]] = 1
    gains = gains_table(injury, risk_score).set_index("tenth")

    # Every tenth has two crashes, and tenth 1 has the two highest scores
    assert gains["crashes"].tolist() == [2] * 10
    assert gains.loc[1, "lowest_score"] == 0.95
    assert gains.loc[1, "injuries"] == 2
    assert gains.loc[1, "injury_share"] == 1.0
    assert gains.loc[1, "share_of_crashes"] == 0.1
    assert gains.loc[1, "share_of_injuries_found"] == 0.5

    # The third injury is in tenth 2, and the last one is found in tenth 10
    assert gains.loc[2, "share_of_injuries_found"] == 0.75
    assert gains.loc[9, "share_of_injuries_found"] == 0.75
    assert gains.loc[10, "share_of_injuries_found"] == 1.0


@needs_outputs
def test_saved_gains_find_every_injury():
    saved_gains = pd.read_csv(EVALUATION_DIR / "gains.csv")
    assert saved_gains["share_of_crashes"].iloc[-1] == 1.0
    assert saved_gains["share_of_injuries_found"].iloc[-1] == 1.0


@needs_outputs
def test_odds_ratio_references():
    # Each of the seven categorical columns has one reference, odds ratio 1
    odds_ratios = pd.read_csv(EVALUATION_DIR / "odds_ratios.csv", dtype=str)
    references = odds_ratios[odds_ratios["category"] == odds_ratios["reference"]]
    assert len(references) == 7
    assert references["column"].is_unique
    assert (references["odds_ratio"].astype(float) == 1.0).all()


@needs_outputs
@needs_models
def test_odds_ratios_match_model():
    # An odds ratio in the table equals the ratio of the odds the model itself
    # gives the typical crash before and after one value is changed
    odds_ratios = pd.read_csv(EVALUATION_DIR / "odds_ratios.csv", dtype=str)
    pipeline = load_models([PREFERRED_MODEL])[PREFERRED_MODEL]
    typical_crash = load_typical_crash()

    def odds_of_injury(crash):
        risk_score = pipeline.predict_proba(pd.DataFrame([crash]))[0, 1]
        return risk_score / (1 - risk_score)

    saved_odds_ratio = odds_ratios.set_index("category")["odds_ratio"].astype(float)
    changed_crashes = {
        "PEDESTRIAN": {"FIRST_CRASH_TYPE": "PEDESTRIAN"},
        "+10 mph": {"POSTED_SPEED_LIMIT": typical_crash["POSTED_SPEED_LIMIT"] + 10},
        "+1 unit": {"NUM_UNITS": typical_crash["NUM_UNITS"] + 1},
    }
    for category, change in changed_crashes.items():
        from_model = odds_of_injury(typical_crash | change) / odds_of_injury(
            typical_crash
        )
        assert np.isclose(from_model, saved_odds_ratio[category])


@needs_outputs
def test_threshold_table_matches_metrics():
    thresholds = pd.read_csv(EVALUATION_DIR / "thresholds.csv")
    metrics = pd.read_csv(EVALUATION_DIR / "metrics.csv", index_col=0)
    at_threshold = thresholds[
        (thresholds["period"] == "test") & (thresholds["threshold"] == THRESHOLD)
    ].iloc[0]
    for metric in ["precision", "recall", "f1"]:
        assert np.isclose(at_threshold[metric], metrics.loc[PREFERRED_MODEL, metric])
