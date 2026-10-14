from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "customer_churn_prepared.csv"
)
REPORTING_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "customer_churn_reporting.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"

REPORT_PATH = REPORTS_PATH / "customer_retention_report.csv"
SUMMARY_PATH = REPORTS_PATH / "retention_summary.csv"


def recommend_retention_action(risk_category):
    """Suggest a retention action based on predicted churn risk."""

    if risk_category == "High":
        return (
            "Prioritize personal outreach and review "
            "contract or pricing options."
        )

    if risk_category == "Medium":
        return "Send a personalized offer and monitor engagement."

    return "Maintain regular engagement and monitor customer satisfaction."


def generate_retention_report():
    """Generate customer-level churn scores and a risk summary."""

    print("Loading prepared customer data...")

    for path in [DATA_PATH, REPORTING_DATA_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    data = pd.read_csv(DATA_PATH)
    reporting_data = pd.read_csv(REPORTING_DATA_PATH)
    saved_model = joblib.load(MODEL_PATH)

    # Validate reporting data before generating customer-level results.
    required_reporting_columns = {"Customer_ID", "Churn"}
    missing_reporting_columns = (
        required_reporting_columns - set(reporting_data.columns)
    )

    if missing_reporting_columns:
        raise ValueError(
            "Reporting dataset is missing columns: "
            + ", ".join(sorted(missing_reporting_columns))
        )

    if len(data) != len(reporting_data):
        raise ValueError(
            "Prepared and reporting datasets have different row counts. "
            "Regenerate both datasets using the same data preparation run."
        )

    if reporting_data["Customer_ID"].isna().any():
        raise ValueError("Customer_ID contains missing values.")

    if reporting_data["Customer_ID"].duplicated().any():
        raise ValueError(
            "Customer_ID contains duplicates. Verify the customer key "
            "before generating the report."
        )

    if "Churn" not in data.columns:
        raise ValueError("Prepared dataset is missing the Churn target.")

    # Both files are generated from the same cleaned records and order.
    # Verify that their historical target labels still align.
    if not (
        data["Churn"].reset_index(drop=True).equals(
            reporting_data["Churn"].reset_index(drop=True)
        )
    ):
        raise ValueError(
            "Churn labels do not align between the prepared and reporting "
            "datasets. Regenerate both datasets before scoring."
        )

    # Retrieve the trained model and expected input features.
    model = saved_model["model"]
    feature_names = saved_model["feature_names"]

    missing_columns = [
        column for column in feature_names
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prepared dataset is missing model features: "
            + ", ".join(missing_columns)
        )

    # Never pass Customer_ID or the actual outcome to the model.
    X = data[feature_names]

    if "Customer_ID" in X.columns or "Churn" in X.columns:
        raise ValueError(
            "Identifier or target leakage detected in model features."
        )

    print(f"Customers to score: {len(X):,}")
    print("Calculating churn probabilities...")

    churn_probabilities = model.predict_proba(X)[:, 1]

    # Guard against invalid model output.
    if (
        not pd.Series(churn_probabilities).between(0, 1).all()
    ):
        raise ValueError("The model returned invalid probabilities.")

    # Illustrative business risk bands, not calibrated future probabilities.
    risk_categories = pd.cut(
        churn_probabilities,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    report = pd.DataFrame({
        "Customer_ID": reporting_data["Customer_ID"].values,
        "Predicted_Churn_Probability": churn_probabilities.round(4),
        "Risk_Category": risk_categories.astype(str),
        "Actual_Churn": reporting_data["Churn"].values,
    })

    report["Recommended_Action"] = report["Risk_Category"].apply(
        recommend_retention_action
    )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    report.to_csv(REPORT_PATH, index=False)

    # Build a business summary.
    summary = (
        report.groupby("Risk_Category", observed=False)
        .agg(
            Customer_Count=("Customer_ID", "count"),
            Average_Churn_Probability=(
                "Predicted_Churn_Probability", "mean"
            ),
        )
        .reset_index()
    )

    summary["Percentage_of_Records"] = (
        summary["Customer_Count"] / len(report) * 100
    ).round(2)

    summary["Average_Churn_Probability"] = (
        summary["Average_Churn_Probability"].round(4)
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nRetention report generated successfully!")
    print(f"Customer-level report: {REPORT_PATH}")
    print(f"Risk summary: {SUMMARY_PATH}")

    print("\nRisk category summary:")
    print(summary.to_string(index=False))

    print("\nFirst five records:")
    print(report.head().to_string(index=False))

    return report, summary


if __name__ == "__main__":
    generate_retention_report()