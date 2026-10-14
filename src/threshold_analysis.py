
# src/threshold_analysis.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_thresholds(y_true, probabilities):
    """
    Evaluate different probability thresholds for churn prediction.

    Lower thresholds generally identify more potential churners,
    but can also produce more false positives.
    """

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_true, predictions, zero_division=0
            ),
            "Customers_Flagged": int(predictions.sum()),
            "Actual_Churners_Caught": int(
                ((predictions == 1) & (y_true == 1)).sum()
            ),
        })

    results_df = pd.DataFrame(results)

    print("\n--- Threshold Comparison ---")
    print(results_df.round(3).to_string(index=False))

    return results_df


def plot_threshold_comparison(results_df, output_path):
    """Plot precision, recall and F1-score at different thresholds."""

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["Threshold"],
        results_df["Precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        results_df["Threshold"],
        results_df["Recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        results_df["Threshold"],
        results_df["F1-score"],
        marker="o",
        label="F1-score",
    )

    plt.xlabel("Churn Probability Threshold")
    plt.ylabel("Score")
    plt.title("Churn Threshold Analysis")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nThreshold chart saved to: {output_path}")
