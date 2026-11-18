"""Score one crash with the saved logistic regression and explain the score.

The assistant calls CrashScorer.score when a user describes a crash. Every
number comes from the trained model, the training set or the test set, so the
language model only has to put them into words.
"""

import json

import joblib
import pandas as pd

from data3800_gruppe_7.modeling import MODELS_DIR, PREDICTIONS_PATH, TYPICAL_CRASH_PATH

# The logistic regression is the model we prefer in the evaluation.
MODEL_NAME = "logistic_regression"

# Test crashes count as similar when their risk score is at most this far away.
SIMILAR_SCORE_DISTANCE = 0.05

# The numbers a crash may have. A value outside these limits is rejected.
NUMBER_LIMITS = {"POSTED_SPEED_LIMIT": (1, 100), "NUM_UNITS": (1, 20)}


class CrashScorer:
    """Scores crashes with the saved model, which is loaded once."""

    def __init__(self):
        self.pipeline = joblib.load(MODELS_DIR / f"{MODEL_NAME}.joblib")
        predictions = pd.read_csv(PREDICTIONS_PATH)
        self.test_scores = predictions[predictions["model"] == MODEL_NAME]
        # The most common value of each feature in the training set.
        self.typical_crash = json.loads(TYPICAL_CRASH_PATH.read_text())

    def score(self, described):
        """Return the risk score of a crash and the numbers that explain it.

        described holds the features the user gave. The others get the value
        of the typical crash, and the result lists them as assumed.
        """
        given = self.validate(described)
        crash = {**self.typical_crash, **given}
        risk_score = self.risk_score(crash)
        similar = self.test_scores[
            (self.test_scores["y_proba"] - risk_score).abs() <= SIMILAR_SCORE_DISTANCE
        ]
        return {
            "crash": crash,
            "assumed_typical_values": [
                feature for feature in crash if feature not in given
            ],
            "risk_score": round(risk_score, 3),
            "test_crashes_with_similar_score": len(similar),
            "injury_share_of_test_crashes_with_similar_score": (
                None if similar.empty else round(similar["y_true"].mean(), 3)
            ),
            "injury_share_of_all_test_crashes": round(
                self.test_scores["y_true"].mean(), 3
            ),
            "what_if": self.what_if(crash, given, risk_score),
        }

    def validate(self, described):
        """Return the given features in the form the model expects.

        Raises ValueError for a value the model cannot score. The encoder
        ignores a category it never saw, so such a crash would get a score
        without any warning.
        """
        if not isinstance(described, dict):
            raise TypeError("The crash must be given as named features.")
        allowed = self.allowed_categories()
        given = {}
        for feature, value in described.items():
            if value is None:
                continue
            if feature in NUMBER_LIMITS:
                low, high = NUMBER_LIMITS[feature]
                given[feature] = to_number(feature, value)
                if not low <= given[feature] <= high:
                    raise ValueError(
                        f"{feature} must be between {low} and {high}, not {value}."
                    )
            elif feature in allowed:
                is_text = isinstance(allowed[feature][0], str)
                given[feature] = (
                    str(value).upper() if is_text else to_number(feature, value)
                )
                if given[feature] not in allowed[feature]:
                    raise ValueError(
                        f"{feature} must be one of {allowed[feature]}, not {value}."
                    )
            else:
                raise ValueError(f"{feature} is not a feature of the model.")
        return given

    def allowed_categories(self):
        """Return the categories the model saw in training, by feature."""
        encoder = self.pipeline.named_steps["preprocess"].named_transformers_[
            "categorical"
        ]
        return {
            feature: categories.tolist()
            for feature, categories in zip(
                encoder.feature_names_in_, encoder.categories_
            )
        }

    def risk_score(self, crash):
        """Return the model's score for one crash, between 0 and 1."""
        # Column 1 of predict_proba is the score of class 1, injury.
        return float(self.pipeline.predict_proba(pd.DataFrame([crash]))[0, 1])

    def what_if(self, crash, given, risk_score):
        """Return the score with each given detail swapped for its typical value.

        The list starts with the detail that changes the score most.
        """
        rows = []
        for feature, typical in self.typical_crash.items():
            if feature in given and crash[feature] != typical:
                score_with_typical = self.risk_score({**crash, feature: typical})
                rows.append(
                    {
                        "feature": feature,
                        "value": crash[feature],
                        "typical_value": typical,
                        "risk_score_with_typical_value": round(score_with_typical, 3),
                        # Below 0 when the given value lowers the score.
                        "raises_risk_score_by": round(
                            risk_score - score_with_typical, 3
                        ),
                    }
                )
        return sorted(
            rows, key=lambda row: abs(row["raises_risk_score_by"]), reverse=True
        )


def to_number(feature, value):
    """Return the value as a number, or raise a ValueError that names the feature."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{feature} must be a number, not {value!r}.") from None
