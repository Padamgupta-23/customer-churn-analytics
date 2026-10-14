
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "customer_retention_report.csv"


def evaluate_retention_report():
    report = pd.read_csv(REPORT_PATH)

    if "Actual_Churn" not in report.columns:
        raise ValueError("Actual_Churn column is missing from the report.")

    # Evaluate the existing risk bands against historical labels.
    actual = report["Actual_Churn"].astype(int)
    predicted = (report["Risk_Category"] == "High").astype(int)

    print("\n=== High-Risk Group Evaluation ===")
    print("High risk = predicted churn probability above 60%")
    print("Other risk levels = predicted churn probability at or below 60%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    print("\nActual churn rate by risk category:")
    print(
        report.groupby("Risk_Category", observed=False)["Actual_Churn"]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "Record_Count",
            "mean": "Historical_Churn_Rate",
        })
        .assign(
            Historical_Churn_Rate=lambda table:
                (table["Historical_Churn_Rate"] * 100).round(2)
        )
        .to_string()
    )


if __name__ == "__main__":
    evaluate_retention_report()
