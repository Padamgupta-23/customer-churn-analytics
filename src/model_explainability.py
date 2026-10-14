
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, output_dir):
    """
    Save the confusion matrix and a business-friendly error summary.
    Positive class (1) = Churned.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    # Save confusion matrix image
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Stayed", "Churned"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Customer Churn - Confusion Matrix")
    plt.tight_layout()

    image_path = output_dir / "confusion_matrix.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save business interpretation
    summary = pd.DataFrame([
        {
            "Outcome": "Correctly predicted Stayed",
            "Count": int(tn),
            "Business_Meaning": "Customers correctly identified as staying"
        },
        {
            "Outcome": "Incorrectly flagged as Churned",
            "Count": int(fp),
            "Business_Meaning": "Customers flagged for retention who actually stayed"
        },
        {
            "Outcome": "Missed Churners",
            "Count": int(fn),
            "Business_Meaning": "Customers who churned but were not flagged"
        },
        {
            "Outcome": "Correctly detected Churners",
            "Count": int(tp),
            "Business_Meaning": "Churners the model successfully identified"
        }
    ])

    summary.to_csv(
        output_dir / "confusion_matrix_business_summary.csv",
        index=False
    )

    print("\n--- Confusion Matrix: Business Interpretation ---")
    print(f"True Negatives  (correctly stayed): {tn}")
    print(f"False Positives (unnecessary flags): {fp}")
    print(f"False Negatives (missed churners): {fn}")
    print(f"True Positives  (churners caught): {tp}")
    print(f"\nSaved confusion matrix: {image_path}")

    return summary


def save_logistic_regression_importance(model, output_dir, top_n=20):
    """
    Extract Logistic Regression coefficients from the fitted pipeline.
    Positive coefficients increase predicted churn odds;
    negative coefficients decrease predicted churn odds.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the final estimator from the fitted pipeline
    estimator = model.steps[-1][1]

    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "This function requires a fitted model with coefficients, "
            "such as Logistic Regression."
        )

    # Get feature names after imputation, scaling and one-hot encoding
    feature_names = model[:-1].get_feature_names_out()
    coefficients = estimator.coef_[0]

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature-name count does not match coefficient count."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Importance": abs(coefficients)
    })

    importance_df["Direction"] = importance_df["Coefficient"].apply(
        lambda value: (
            "Higher predicted churn odds"
            if value > 0
            else "Lower predicted churn odds"
            if value < 0
            else "No linear effect"
        )
    )

    importance_df = importance_df.sort_values(
        "Absolute_Importance", ascending=False
    )

    csv_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # Plot the strongest positive and negative coefficients
    plot_df = importance_df.head(top_n).sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_df["Feature"], plot_df["Coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_ylabel("Encoded Feature")
    ax.set_title("Top Churn-Related Features - Logistic Regression")
    plt.tight_layout()

    image_path = output_dir / "feature_importance.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Top Features by Absolute Coefficient ---")
    print(
        importance_df[
            ["Feature", "Coefficient", "Direction"]
        ].head(top_n).round(3).to_string(index=False)
    )
    print(f"\nSaved feature importance CSV: {csv_path}")
    print(f"Saved feature importance chart: {image_path}")

    return importance_df
