from data3800_gruppe_7.risk import CrashScorer

scorer = CrashScorer()

# A crash with no details given is scored as the typical crash
typical = scorer.score({})
assert typical["crash"] == scorer.typical_crash
assert len(typical["assumed_typical_values"]) == 9
assert typical["what_if"] == []

# A pedestrian hit at night scores far higher than the typical crash
pedestrian = scorer.score(
    {
        "FIRST_CRASH_TYPE": "pedestrian",
        "LIGHTING_CONDITION": "DARKNESS, LIGHTED ROAD",
        "CRASH_HOUR": 23,
    }
)
assert pedestrian["crash"]["FIRST_CRASH_TYPE"] == "PEDESTRIAN"
assert pedestrian["risk_score"] > typical["risk_score"]
assert len(pedestrian["assumed_typical_values"]) == 6

# Crashes with a similar score had an injury more often than crashes overall
assert pedestrian["test_crashes_with_similar_score"] > 0
assert (
    pedestrian["injury_share_of_test_crashes_with_similar_score"]
    > pedestrian["injury_share_of_all_test_crashes"]
)

# The crash type changes the score most, and a parked car would lower it
biggest_change = pedestrian["what_if"][0]
assert biggest_change["feature"] == "FIRST_CRASH_TYPE"
assert biggest_change["risk_score_with_typical_value"] < pedestrian["risk_score"]
assert biggest_change["raises_risk_score_by"] > 0

# Values the model cannot score are rejected
rejected = [
    {"WEATHER_CONDITION": "SUNNY"},
    {"POSTED_SPEED_LIMIT": 500},
    {"CRASH_HOUR": 24},
    {"COLOR": "RED"},
]
for described in rejected:
    try:
        scorer.score(described)
    except ValueError:
        continue
    raise AssertionError(f"{described} should be rejected")

print("ALL TESTS PASSED!")
print("Typical crash score:", typical["risk_score"])
print("Pedestrian at night score:", pedestrian["risk_score"])
