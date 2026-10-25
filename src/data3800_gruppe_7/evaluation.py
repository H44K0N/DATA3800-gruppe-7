"""Evaluate the injury models on the test set.

Run from the project root, after the modeling module, with:

    uv run python -m data3800_gruppe_7.evaluation

The script reads the fitted models and their test set predictions from
output/models/, and saves the tables and figures of the evaluation in
output/evaluation/.
"""

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

from data3800_gruppe_7.modeling import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREDICTIONS_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    load_model_data,
    split_train_test,
)

EVALUATION_DIR = PROJECT_ROOT / "output" / "evaluation"

BASELINE = "baseline"
CLASS_LABELS = ["No injury", "Injury"]

# Number of resamples for the confidence intervals, and for each feature in
# the permutation importance.
BOOTSTRAP_SAMPLES = 1000
PERMUTATION_REPEATS = 5
CONFIDENCE_LEVEL = 0.95

# The baseline is grey because it is only a reference. The three models get the
# first three colors of a palette that is checked for color blindness, always
# in this order.
MODEL_COLORS = {
    "baseline": "#8a8a85",
    "logistic_regression": "#2a78d6",
    "decision_tree": "#eb6834",
    "random_forest": "#1baf7a",
}
TEXT_COLOR = "#33363E"


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    pipelines = load_models(predictions["model"].unique())
    X_test, y_test = load_test_set(predictions)

    metrics = compute_metrics(predictions)
    intervals = bootstrap_intervals(predictions)
    importance = compute_permutation_importance(pipelines, X_test, y_test)
    odds_ratios = logistic_odds_ratios(pipelines["logistic_regression"], X_test)

    metrics.to_csv(EVALUATION_DIR / "metrics.csv")
    intervals.to_csv(EVALUATION_DIR / "metric_intervals.csv")
    importance.to_csv(EVALUATION_DIR / "permutation_importance.csv", index=False)
    odds_ratios.to_csv(EVALUATION_DIR / "odds_ratios.csv", index=False)

    save_figure(plot_confusion_matrices(predictions), "confusion_matrices.png")
    save_figure(plot_curves(predictions), "roc_pr_curves.png")
    save_figure(plot_permutation_importance(importance), "permutation_importance.png")
    print(metrics.round(3).to_string())
    print(f"Saved the evaluation in {EVALUATION_DIR}")


def load_predictions():
    """Return the test set predictions, one row per crash and model."""
    return pd.read_csv(PREDICTIONS_PATH)


def load_models(names):
    """Return the fitted pipeline of each model, by name."""
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in names}


def load_test_set(predictions):
    """Rebuild the test set the same way as in modeling.py.

    The split uses a fixed random state, so it gives the same crashes. We check
    that against the crash IDs in the predictions, in case the modeling changes.
    """
    X, y, crash_ids = load_model_data()
    _, X_test, _, y_test = split_train_test(X, y)
    first_model = predictions["model"].iloc[0]
    saved_ids = predictions.loc[predictions["model"] == first_model, "crash_record_id"]
    if not np.array_equal(crash_ids.loc[X_test.index].to_numpy(), saved_ids.to_numpy()):
        raise ValueError(
            "The test set does not match predictions.csv. Run the modeling again."
        )
    return X_test, y_test


def compute_metrics(predictions):
    """Return one row per model with the metrics we compare them on.

    Accuracy, precision, recall and F1 use the predicted class, so they depend
    on the threshold of 0.5. ROC-AUC and PR-AUC use the predicted probability
    and look at every threshold at once.
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


def compute_permutation_importance(pipelines, X_test, y_test):
    """Return how much the PR-AUC drops when each input column is shuffled.

    Shuffling a column breaks its link to the target and keeps everything else.
    We shuffle the original columns, not the one-hot columns, so every
    condition gets a single number, and the same method works for every model.
    """
    rows = []
    for name, pipeline in pipelines.items():
        if name == BASELINE:
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


def logistic_odds_ratios(pipeline, X_test):
    """Return the odds ratio of each input column of the logistic regression.

    An odds ratio above 1 means the odds of injury are higher for that category,
    holding the other columns fixed. For the speed limit it is per standard
    deviation, since the column is scaled. The number of test crashes in each
    category shows which odds ratios rest on very few crashes.
    """
    # The column transformer puts the numeric columns first, then the one-hot
    # columns in the order of the encoder's categories.
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    rows = [
        {"column": column, "category": "", "crashes": X_test[column].notna().sum()}
        for column in NUMERIC_FEATURES
    ]
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        counts = X_test[column].value_counts(dropna=False)
        rows += [
            {"column": column, "category": category, "crashes": counts.get(category, 0)}
            for category in categories
        ]

    odds = pd.DataFrame(rows)
    odds["coefficient"] = pipeline.named_steps["model"].coef_[0]
    odds["odds_ratio"] = np.exp(odds["coefficient"])
    return odds.sort_values("coefficient", ascending=False, ignore_index=True)


def plot_confusion_matrices(predictions):
    """Plot one confusion matrix per model, as shares of the true class.

    Each row sums to 1, so the diagonal is the recall of each class.
    """
    groups = list(predictions.groupby("model", sort=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4))
    for ax, (name, group) in zip(axes, groups):
        matrix = confusion_matrix(group["y_true"], group["y_pred"], normalize="true")
        display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_LABELS)
        display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        display.im_.set_clim(0, 1)
        ax.set_title(name, color=TEXT_COLOR)
    fig.suptitle("Confusion matrices on the test set, as shares of each true class")
    fig.tight_layout()
    return fig


def plot_curves(predictions):
    """Plot the ROC curve and the precision-recall curve of each model."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))
    base_rate = predictions["y_true"].mean()
    for name, group in predictions.groupby("model", sort=False):
        if name == BASELINE:
            continue
        color = MODEL_COLORS[name]
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_proba"])
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_proba"])
        roc_ax.plot(fpr, tpr, color=color, linewidth=2, label=name)
        pr_ax.plot(recall, precision, color=color, linewidth=2, label=name)

    # A model that guesses at random lies on these lines, and so does the baseline.
    roc_ax.plot(
        [0, 1],
        [0, 1],
        color=MODEL_COLORS[BASELINE],
        linestyle="--",
        label="random guess / baseline",
    )
    pr_ax.axhline(
        base_rate,
        color=MODEL_COLORS[BASELINE],
        linestyle="--",
        label=f"random guess / baseline ({base_rate:.1%})",
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


def plot_permutation_importance(importance):
    """Plot the drop in PR-AUC per input column, one group of bars per model."""
    features = importance.groupby("feature")["mean"].mean().sort_values().index.tolist()
    models = importance["model"].unique()
    height = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 5))
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
            label=name,
        )
    ax.set_yticks(np.arange(len(features)) + height * (len(models) - 1) / 2)
    ax.set_yticklabels(features)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set(
        title="Permutation importance on the test set",
        xlabel="Drop in PR-AUC when the column is shuffled",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, filename):
    """Save a figure in EVALUATION_DIR and close it."""
    fig.savefig(EVALUATION_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
