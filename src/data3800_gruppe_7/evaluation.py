"""Evaluate the injury models on the test set.

Run from the project root, after the modeling module, with:

    uv run python -m data3800_gruppe_7.evaluation

The script reads the fitted models and their test set predictions from
output/models/, and saves the tables and figures of the evaluation in
output/evaluation/.
"""

import json
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data3800_gruppe_7.cleaning import get_model_data
from data3800_gruppe_7.modeling import (
    CATEGORICAL_FEATURES,
    CONDITIONS_ONLY,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREDICTIONS_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    TYPICAL_CRASH_PATH,
    load_crashes,
    split_by_period,
)

EVALUATION_DIR = PROJECT_ROOT / "output" / "evaluation"

BASELINE = "baseline"
# The model we prefer. The tables on thresholds, gains, report types, errors
# and odds ratios are made for this model only.
PREFERRED_MODEL = "logistic_regression"
CLASS_LABELS = ["No injury", "Injury"]

# A model flags a crash as injury when its risk score is at least this high.
THRESHOLD = 0.5
# The thresholds we compare it with: 0.30, 0.35 and so on up to 0.80.
THRESHOLDS = [percent / 100 for percent in range(30, 85, 5)]

# The two ways the police take a report. Some crashes have no report type.
REPORT_TYPES = ["ON SCENE", "NOT ON SCENE (DESK REPORT)"]
NOT_RECORDED = "not recorded"

# The odds ratio of a numeric column is given for a step people understand,
# with the label the step gets in the table.
NUMERIC_STEPS = {"POSTED_SPEED_LIMIT": (10, "+10 mph"), "NUM_UNITS": (1, "+1 unit")}

# Number of resamples for the confidence intervals, and for each feature in
# the permutation importance.
BOOTSTRAP_SAMPLES = 1000
PERMUTATION_REPEATS = 5
CONFIDENCE_LEVEL = 0.95

# The baseline and the conditions-only model are grey and dark grey because
# they are only references. The three full models get the first three colors
# of a palette that is checked for color blindness, always in this order.
MODEL_COLORS = {
    "baseline": "#8a8a85",
    CONDITIONS_ONLY: "#33363e",
    "logistic_regression": "#2a78d6",
    "decision_tree": "#eb6834",
    "random_forest": "#1baf7a",
}
TEXT_COLOR = "#33363E"

# The names the figures show instead of the names in the code and the data.
MODEL_LABELS = {
    "baseline": "Baseline",
    CONDITIONS_ONLY: "Conditions only",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
}
FEATURE_LABELS = {
    "POSTED_SPEED_LIMIT": "Posted speed limit",
    "WEATHER_CONDITION": "Weather",
    "LIGHTING_CONDITION": "Lighting",
    "ROADWAY_SURFACE_COND": "Road surface",
    "CRASH_HOUR": "Hour of the crash",
    "TRAFFIC_CONTROL_DEVICE": "Traffic control device",
    "TRAFFICWAY_TYPE": "Trafficway type",
    "FIRST_CRASH_TYPE": "First crash type",
    "NUM_UNITS": "Number of units",
}


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    pipelines = load_models(predictions["model"].unique())
    validation, test = load_validation_and_test(predictions)
    X_test, y_test = get_model_data(test)
    preferred = pipelines[PREFERRED_MODEL]
    scored_validation = add_risk_score(validation, preferred)
    scored_test = add_risk_score(test, preferred)

    metrics = compute_metrics(predictions)
    intervals = bootstrap_intervals(predictions)
    thresholds = compare_thresholds(scored_validation, scored_test)
    gains = gains_table(scored_test["injury"], scored_test["risk_score"])
    report_types = compare_report_types(scored_test)
    crash_types = crash_type_errors(scored_test)
    importance = compute_permutation_importance(pipelines, X_test, y_test)
    odds_ratios = logistic_odds_ratios(preferred, X_test, load_typical_crash())

    metrics.to_csv(EVALUATION_DIR / "metrics.csv")
    intervals.to_csv(EVALUATION_DIR / "metric_intervals.csv")
    thresholds.to_csv(EVALUATION_DIR / "thresholds.csv", index=False)
    gains.to_csv(EVALUATION_DIR / "gains.csv", index=False)
    report_types.to_csv(EVALUATION_DIR / "report_types.csv", index=False)
    crash_types.to_csv(EVALUATION_DIR / "crash_type_errors.csv", index=False)
    importance.to_csv(EVALUATION_DIR / "permutation_importance.csv", index=False)
    odds_ratios.to_csv(EVALUATION_DIR / "odds_ratios.csv", index=False)

    all_models = list(predictions["model"].unique())
    save_figure(
        plot_confusion_matrices(predictions, all_models), "confusion_matrices.png"
    )
    save_figure(
        plot_confusion_matrices(predictions, [PREFERRED_MODEL]), "confusion_matrix.png"
    )
    save_figure(plot_curves(predictions), "roc_pr_curves.png")
    save_figure(plot_gains(predictions), "gains.png")
    save_figure(plot_permutation_importance(importance), "permutation_importance.png")
    print(metrics.round(3).to_string())
    print(f"Saved the evaluation in {EVALUATION_DIR}")


def load_predictions():
    """Return the test set predictions, one row per crash and model."""
    return pd.read_csv(PREDICTIONS_PATH)


def load_models(names):
    """Return the fitted pipeline of each model, by name."""
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in names}


def load_typical_crash():
    """Return the most common value of each feature in the training set."""
    return json.loads(TYPICAL_CRASH_PATH.read_text())


def load_validation_and_test(predictions):
    """Return the validation and test crashes, split the same way as in modeling.py.

    The periods are fixed, so the split gives the same crashes. We check the
    test crashes against the crash IDs in the predictions, in case the
    modeling changes.
    """
    _, validation, test = split_by_period(load_crashes())
    first_model = predictions["model"].iloc[0]
    saved_ids = predictions.loc[predictions["model"] == first_model, "crash_record_id"]
    if not np.array_equal(test["CRASH_RECORD_ID"].to_numpy(), saved_ids.to_numpy()):
        raise ValueError(
            "The test set does not match predictions.csv. Run the modeling again."
        )
    return validation, test


def add_risk_score(crashes, pipeline):
    """Return the crashes with the risk score of a model, and whether it flags them.

    The risk score is the model's output for the injury class, between 0 and 1.
    The models are fitted with balanced class weights, so it ranks the crashes
    by risk but is not the probability of an injury.
    """
    X, _ = get_model_data(crashes)
    risk_score = pipeline.predict_proba(X)[:, 1]
    return crashes.assign(risk_score=risk_score, flagged=risk_score >= THRESHOLD)


def compute_metrics(predictions):
    """Return one row per model with the metrics we compare them on.

    Accuracy, precision, recall and F1 use the predicted class, so they depend
    on the threshold of 0.5. ROC-AUC and PR-AUC use the risk score and look at
    every threshold at once.
    """
    rows = {}
    for name, group in predictions.groupby("model", sort=False):
        y_true, y_pred, y_proba = group["y_true"], group["y_pred"], group["y_proba"]
        rows[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            # The baseline never predicts injury, so its precision is 0 / 0.
            # We count that as 0.
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "predicted_injury_share": y_pred.mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_intervals(predictions):
    """Return percentile bootstrap intervals for the F1 and PR-AUC.

    There is one row for each model, and one for the difference between each
    pair of models. Each resample draws the same crashes for every model, so
    the interval of a difference shows whether one model is better than another
    by more than the noise from which crashes happened to end up in the test set.
    """
    by_model = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("model", sort=False)
        if name != BASELINE
    }
    n_crashes = len(next(iter(by_model.values())))
    rng = np.random.default_rng(RANDOM_STATE)
    samples = {name: {"f1": [], "pr_auc": []} for name in by_model}
    for _ in range(BOOTSTRAP_SAMPLES):
        rows = rng.integers(0, n_crashes, n_crashes)
        for name, group in by_model.items():
            resample = group.iloc[rows]
            samples[name]["f1"].append(f1_score(resample["y_true"], resample["y_pred"]))
            samples[name]["pr_auc"].append(
                average_precision_score(resample["y_true"], resample["y_proba"])
            )
    samples = {
        name: {metric: np.array(values) for metric, values in metrics.items()}
        for name, metrics in samples.items()
    }

    comparisons = {name: samples[name] for name in by_model}
    names = list(by_model)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            comparisons[f"{first} - {second}"] = {
                metric: samples[first][metric] - samples[second][metric]
                for metric in ("f1", "pr_auc")
            }

    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    rows = []
    for comparison, metrics in comparisons.items():
        row = {"comparison": comparison}
        for metric, values in metrics.items():
            row[f"{metric}_low"], row[f"{metric}_high"] = np.percentile(
                values, [tail, 100 - tail]
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("comparison")


def threshold_metrics(y_true, risk_score):
    """Return the share of crashes flagged, precision, recall and F1 per threshold.

    A crash is flagged when its risk score is at least as high as the threshold.
    """
    rows = []
    for threshold in THRESHOLDS:
        flagged = risk_score >= threshold
        rows.append(
            {
                "threshold": threshold,
                "flagged_share": flagged.mean(),
                "precision": precision_score(y_true, flagged, zero_division=0),
                "recall": recall_score(y_true, flagged),
                "f1": f1_score(y_true, flagged),
            }
        )
    return pd.DataFrame(rows)


def compare_thresholds(scored_validation, scored_test):
    """Return the threshold metrics of the validation months, then the test months.

    The validation rows are what the threshold is chosen on. The test rows
    only show whether the choice carried over to crashes the model never saw.
    """
    periods = {"validation": scored_validation, "test": scored_test}
    tables = {
        period: threshold_metrics(crashes["injury"], crashes["risk_score"])
        for period, crashes in periods.items()
    }
    return (
        pd.concat(tables, names=["period", None])
        .reset_index(level="period")
        .reset_index(drop=True)
    )


def gains_table(y_true, risk_score):
    """Return what each tenth of the crashes holds, ranked by risk score.

    Tenth 1 has the highest scores. The cumulative columns show how large a
    share of the injury crashes is found by checking the crashes from the top
    of the ranking. The injury share next to the lowest score of a tenth also
    shows how the score compares with the real share of injuries.
    """
    ranked = pd.DataFrame(
        {"injury": np.asarray(y_true), "risk_score": np.asarray(risk_score)}
    ).sort_values("risk_score", ascending=False, kind="stable")
    # The tenths are cut by rank, so crashes with the same score cannot make
    # one tenth larger than another.
    ranked["tenth"] = np.arange(len(ranked)) * 10 // len(ranked) + 1
    gains = ranked.groupby("tenth").agg(
        crashes=("injury", "size"),
        lowest_score=("risk_score", "min"),
        injuries=("injury", "sum"),
        injury_share=("injury", "mean"),
    )
    gains["share_of_crashes"] = gains["crashes"].cumsum() / len(ranked)
    gains["share_of_injuries_found"] = (
        gains["injuries"].cumsum() / gains["injuries"].sum()
    )
    return gains.reset_index()


def compare_report_types(scored_test):
    """Return the metrics of the test crashes by how the police took the report.

    The crashes reported on scene are the closest group in the data to the
    calls a dispatch centre takes. Desk reports are filed later and seldom
    have an injury, so part of the score on all crashes comes from telling
    the two groups apart.
    """
    report_type = scored_test["REPORT_TYPE"].fillna(NOT_RECORDED)
    groups = {"all test crashes": scored_test}
    for name in REPORT_TYPES + [NOT_RECORDED]:
        groups[name] = scored_test[report_type == name]

    rows = []
    for name, crashes in groups.items():
        injury, flagged = crashes["injury"], crashes["flagged"]
        rows.append(
            {
                "report_type": name,
                "crashes": len(crashes),
                "injury_share": injury.mean(),
                "roc_auc": roc_auc_score(injury, crashes["risk_score"]),
                "pr_auc": average_precision_score(injury, crashes["risk_score"]),
                "flagged_share": flagged.mean(),
                "precision": precision_score(injury, flagged),
                "recall": recall_score(injury, flagged),
            }
        )
    return pd.DataFrame(rows)


def crash_type_errors(scored_test):
    """Return the injuries found and missed for each first crash type.

    The crash types with the most missed injuries come first. The recall is
    empty for a crash type with no injury crash in the test set.
    """
    had_injury = scored_test["injury"] == 1
    errors = (
        scored_test.assign(
            found=had_injury & scored_test["flagged"],
            missed=had_injury & ~scored_test["flagged"],
        )
        .groupby("FIRST_CRASH_TYPE")
        .agg(
            crashes=("injury", "size"),
            injuries=("injury", "sum"),
            injury_share=("injury", "mean"),
            flagged_share=("flagged", "mean"),
            injuries_found=("found", "sum"),
            injuries_missed=("missed", "sum"),
        )
    )
    errors["recall"] = errors["injuries_found"] / errors["injuries"]
    return errors.sort_values(
        ["injuries_missed", "crashes"], ascending=False
    ).reset_index()


def compute_permutation_importance(pipelines, X_test, y_test):
    """Return how much the PR-AUC drops when each input column is shuffled.

    Shuffling a column breaks its link to the target and keeps everything else.
    We shuffle the original columns, not the one-hot columns, so every
    condition gets a single number, and the same method works for every model.
    """
    rows = []
    for name, pipeline in pipelines.items():
        # The two reference models are left out: the baseline uses no column,
        # and the conditions-only model does not get the new ones.
        if name in (BASELINE, CONDITIONS_ONLY):
            continue
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        for feature, mean, std in zip(
            X_test.columns, result.importances_mean, result.importances_std
        ):
            rows.append({"model": name, "feature": feature, "mean": mean, "std": std})
    return pd.DataFrame(rows)


def logistic_odds_ratios(pipeline, X_test, typical_crash):
    """Return the odds ratios of the logistic regression, one row per category.

    A category is compared with the reference of its column, which is the
    value of the typical crash. The odds ratio is the factor the odds of
    injury are multiplied by when a crash has the category instead of the
    reference, holding the other columns fixed, so the reference itself gets 1.
    For the two numeric columns it is the factor for the step in NUMERIC_STEPS.

    The coefficient column holds the model's own coefficient, which is per
    standard deviation for the numeric columns. The model has an L2 penalty,
    so the sizes are approximate. The model is fitted on the training months,
    so the number of test crashes in each category is only a guide to which
    odds ratios rest on few crashes and are unreliable.
    """
    preprocess = pipeline.named_steps["preprocess"]
    scaler = preprocess.named_transformers_["numeric"].named_steps["scale"]
    encoder = preprocess.named_transformers_["categorical"]
    # The column transformer puts the numeric columns first, then the one-hot
    # columns in the order of the encoder's categories.
    coefficients = pipeline.named_steps["model"].coef_[0]
    numeric_coefficients = coefficients[: len(NUMERIC_FEATURES)]
    categorical_coefficients = coefficients[len(NUMERIC_FEATURES) :]

    rows = []
    for column, coefficient, standard_deviation in zip(
        NUMERIC_FEATURES, numeric_coefficients, scaler.scale_
    ):
        step, label = NUMERIC_STEPS[column]
        rows.append(
            {
                "column": column,
                "category": label,
                "reference": "",
                "crashes": X_test[column].notna().sum(),
                "coefficient": coefficient,
                # The column was scaled, so the step is counted in standard
                # deviations of the training set.
                "odds_ratio": np.exp(coefficient * step / standard_deviation),
            }
        )

    first = 0
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        column_coefficients = categorical_coefficients[first : first + len(categories)]
        first += len(categories)
        reference = typical_crash[column]
        reference_coefficient = column_coefficients[list(categories).index(reference)]
        counts = X_test[column].value_counts(dropna=False)
        for category, coefficient in zip(categories, column_coefficients):
            rows.append(
                {
                    "column": column,
                    "category": category,
                    "reference": reference,
                    "crashes": counts.get(category, 0),
                    "coefficient": coefficient,
                    "odds_ratio": np.exp(coefficient - reference_coefficient),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["column", "odds_ratio"], ascending=[True, False], ignore_index=True
    )


def plot_confusion_matrices(predictions, models):
    """Plot the confusion matrix of each of the given models on the test set.

    Each cell shows the number of crashes and their share of the true class.
    The shares in a row sum to 100%, so the diagonal is the recall of each class.
    """
    columns = min(len(models), 3)
    rows = math.ceil(len(models) / columns)
    # A matrix on its own is drawn larger, because the report shows it alone.
    panel_size = 3.1 if len(models) > 1 else 4.5
    fig, axes = plt.subplots(
        rows, columns, figsize=(panel_size * columns, panel_size * rows), squeeze=False
    )
    for ax in axes.flat[len(models) :]:
        ax.axis("off")
    for ax, name in zip(axes.flat, models):
        group = predictions[predictions["model"] == name]
        counts = confusion_matrix(group["y_true"], group["y_pred"])
        shares = counts / counts.sum(axis=1, keepdims=True)
        display = ConfusionMatrixDisplay(shares, display_labels=CLASS_LABELS)
        display.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 12})
        display.im_.set_clim(0, 1)
        # The display has picked a text color that can be read on each cell.
        # We keep the color and only replace the text.
        for text, count, share in zip(display.text_.flat, counts.flat, shares.flat):
            text.set_text(f"{count:,}\n{share:.0%}")
        ax.set_title(MODEL_LABELS[name], color=TEXT_COLOR)
        ax.set(xlabel="Predicted class", ylabel="True class")
    fig.suptitle("Test crashes and their share of each true class")
    fig.tight_layout()
    return fig


def plot_curves(predictions):
    """Plot the ROC curve and the precision-recall curve of each model."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(9, 4.5))
    base_rate = predictions["y_true"].mean()
    for name, group in predictions.groupby("model", sort=False):
        if name == BASELINE:
            continue
        color = MODEL_COLORS[name]
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_proba"])
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_proba"])
        # The random forest lies almost on top of the model we prefer, so we
        # draw the model we prefer in front.
        zorder = 3 if name == PREFERRED_MODEL else 2
        roc_ax.plot(
            fpr, tpr, color=color, linewidth=2, zorder=zorder, label=MODEL_LABELS[name]
        )
        # The models have the same colors in both panels, so only the ROC
        # panel names them. A full legend here would cover the curves.
        pr_ax.plot(recall, precision, color=color, linewidth=2, zorder=zorder)

    # A model that guesses at random lies on these lines, and so does the baseline.
    roc_ax.plot(
        [0, 1],
        [0, 1],
        color=MODEL_COLORS[BASELINE],
        linestyle="--",
        label="Random guess / baseline",
    )
    pr_ax.axhline(
        base_rate,
        color=MODEL_COLORS[BASELINE],
        linestyle="--",
        label=f"Random guess / baseline ({base_rate:.1%})",
    )

    roc_ax.set(
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
    )
    pr_ax.set(
        title="Precision-recall curve", xlabel="Recall", ylabel="Precision", ylim=(0, 1)
    )
    for ax in (roc_ax, pr_ax):
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gains(predictions):
    """Plot the share of the injury crashes found against the share of crashes checked.

    The crashes are checked in the order of their risk score, highest first.
    Checking them in random order follows the diagonal. The marker shows where
    the threshold of 0.5 stops for the model we prefer.
    """
    fig, ax = plt.subplots(figsize=(6, 3.9))
    for name in (PREFERRED_MODEL, CONDITIONS_ONLY):
        group = predictions[predictions["model"] == name]
        ranked = group.sort_values("y_proba", ascending=False, kind="stable")
        checked = np.arange(1, len(ranked) + 1) / len(ranked) * 100
        found = ranked["y_true"].cumsum() / ranked["y_true"].sum() * 100
        ax.plot(
            checked,
            found,
            color=MODEL_COLORS[name],
            linewidth=2,
            label=MODEL_LABELS[name],
        )
    ax.plot(
        [0, 100],
        [0, 100],
        color=MODEL_COLORS[BASELINE],
        linestyle="--",
        label="Random order",
    )

    preferred = predictions[predictions["model"] == PREFERRED_MODEL]
    flagged_share = preferred["y_pred"].mean() * 100
    recall = recall_score(preferred["y_true"], preferred["y_pred"]) * 100
    ax.plot(
        flagged_share,
        recall,
        marker="o",
        markersize=9,
        linestyle="none",
        color=MODEL_COLORS[PREFERRED_MODEL],
        markeredgecolor=TEXT_COLOR,
        # On two lines, so the legend stays below the line for random order.
        label=f"Threshold {THRESHOLD}\n({flagged_share:.0f}% checked, {recall:.0f}% found)",
    )
    ax.set(
        title="Injury crashes found when the riskiest\ntest crashes are checked first",
        xlabel="Share of crashes checked, highest risk score first (%)",
        ylabel="Share of injury crashes found (%)",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_permutation_importance(importance):
    """Plot the drop in PR-AUC per input column, one group of bars per model."""
    features = importance.groupby("feature")["mean"].mean().sort_values().index.tolist()
    models = importance["model"].unique()
    height = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, name in enumerate(models):
        rows = (
            importance[importance["model"] == name].set_index("feature").loc[features]
        )
        # barh draws from the bottom up, so the first model gets the top bar,
        # in the same order as the legend.
        positions = np.arange(len(features)) + (len(models) - 1 - i) * height
        ax.barh(
            positions,
            rows["mean"],
            height=height * 0.9,
            xerr=rows["std"],
            color=MODEL_COLORS[name],
            label=MODEL_LABELS[name],
        )
    ax.set_yticks(np.arange(len(features)) + height * (len(models) - 1) / 2)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in features])
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set(
        title="Permutation importance on the test set",
        xlabel="Drop in PR-AUC when the feature is shuffled",
        ylabel="Feature",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, filename):
    """Save a figure in EVALUATION_DIR and close it."""
    fig.savefig(EVALUATION_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
