import pytest

from data3800_gruppe_7.modeling import MODELS_DIR, PREDICTIONS_PATH, TYPICAL_CRASH_PATH
from data3800_gruppe_7.risk import MODEL_NAME, CrashScorer

pytestmark = pytest.mark.skipif(
    not (
        (MODELS_DIR / f"{MODEL_NAME}.joblib").exists()
        and PREDICTIONS_PATH.exists()
        and TYPICAL_CRASH_PATH.exists()
    ),
    reason="trained model output missing, run modeling first",
)


@pytest.fixture(scope="module")
def scorer():
    return CrashScorer()


@pytest.fixture(scope="module")
def typical(scorer):
    return scorer.score({})


@pytest.fixture(scope="module")
def pedestrian(scorer):
    return scorer.score(
        {
            "FIRST_CRASH_TYPE": "pedestrian",
            "LIGHTING_CONDITION": "DARKNESS, LIGHTED ROAD",
            "CRASH_HOUR": 23,
        }
    )


def test_empty_crash_is_typical(scorer, typical):
    # A crash with no details given is scored as the typical crash
    assert typical["crash"] == scorer.typical_crash
    assert len(typical["assumed_typical_values"]) == 9
    assert typical["what_if"] == []


def test_pedestrian_scores_higher(typical, pedestrian):
    # A pedestrian hit at night scores far higher than the typical crash
    assert pedestrian["crash"]["FIRST_CRASH_TYPE"] == "PEDESTRIAN"
    assert pedestrian["risk_score"] > typical["risk_score"]
    assert len(pedestrian["assumed_typical_values"]) == 6


def test_similar_score_injury_share(pedestrian):
    # Crashes with a similar score had an injury more often than crashes overall
    assert pedestrian["test_crashes_with_similar_score"] > 0
    assert (
        pedestrian["injury_share_of_test_crashes_with_similar_score"]
        > pedestrian["injury_share_of_all_test_crashes"]
    )


def test_what_if_crash_type(pedestrian):
    # The crash type changes the score most, and a parked car would lower it
    biggest_change = pedestrian["what_if"][0]
    assert biggest_change["feature"] == "FIRST_CRASH_TYPE"
    assert biggest_change["risk_score_with_typical_value"] < pedestrian["risk_score"]
    assert biggest_change["raises_risk_score_by"] > 0


@pytest.mark.parametrize(
    "described",
    [
        {"WEATHER_CONDITION": "SUNNY"},
        {"POSTED_SPEED_LIMIT": 500},
        {"POSTED_SPEED_LIMIT": 71},
        {"NUM_UNITS": 15},
        {"CRASH_HOUR": 24},
        {"COLOR": "RED"},
    ],
)
def test_unscorable_values_are_rejected(scorer, described):
    with pytest.raises(ValueError):
        scorer.score(described)
