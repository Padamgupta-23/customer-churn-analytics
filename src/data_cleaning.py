
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


def prepare_customer_data(df):
    """Prepare customer data for churn analysis and modelling."""

    cleaned_df = df.copy()

    # Keep only customers whose outcome is known.
    # Joined customers do not yet have a Stayed/Churned outcome.
    cleaned_df = cleaned_df[
        cleaned_df["Customer_Status"].isin(["Stayed", "Churned"])
    ].copy()

    # Create a binary target:
    # 1 means churned, 0 means stayed.
    cleaned_df["Churn"] = (
        cleaned_df["Customer_Status"] == "Churned"
    ).astype(int)

    # Convert missing internet-service details into an explicit category
    # when the customer does not have internet service.
    internet_service_columns = [
        "Internet_Type",
        "Online_Security",
        "Online_Backup",
        "Device_Protection_Plan",
        "Premium_Support",
        "Streaming_TV",
        "Streaming_Movies",
        "Streaming_Music",
        "Unlimited_Data",
    ]

    if "Internet_Service" in cleaned_df.columns:
        no_internet = cleaned_df["Internet_Service"].eq("No")

        for column in internet_service_columns:
            if column in cleaned_df.columns:
                cleaned_df.loc[
                    no_internet & cleaned_df[column].isna(),
                    column,
                ] = "Not Applicable"

    # Multiple_Lines is not applicable when phone service is absent.
    if "Phone_Service" in cleaned_df.columns:
        no_phone = cleaned_df["Phone_Service"].eq("No")

        if "Multiple_Lines" in cleaned_df.columns:
            cleaned_df.loc[
                no_phone & cleaned_df["Multiple_Lines"].isna(),
                "Multiple_Lines",
            ] = "Not Applicable"

    # Value_Deal needs further investigation. Preserve missing values
    # here instead of assuming that missing means "No Deal".

    # Remove columns that reveal the outcome or uniquely identify a customer.
    columns_to_drop = [
        "Customer_ID",
        "Customer_Status",
        "Churn_Category",
        "Churn_Reason",
    ]

    model_df = cleaned_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # Save the prepared modelling data and a separate reporting dataset.
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # Modeling dataset: identifiers and outcome-revealing columns are excluded.
    model_df.to_csv(
        PROCESSED_PATH / "customer_churn_prepared.csv",
        index=False,
    )

    # Reporting dataset: preserve real customer IDs without exposing them
    # to the predictive model.
    reporting_columns = ["Churn"]

    if "Customer_ID" in cleaned_df.columns:
        reporting_columns.insert(0, "Customer_ID")

    reporting_df = cleaned_df[reporting_columns].copy()

    reporting_df.to_csv(
        PROCESSED_PATH / "customer_churn_reporting.csv",
        index=False,
    )

    print(
        "Saved reporting data to: "
        f"{PROCESSED_PATH / 'customer_churn_reporting.csv'}"
    )

    print("\n--- Data Preparation Summary ---")
    print(f"Original records: {len(df):,}")
    print(f"Records with known outcomes: {len(cleaned_df):,}")
    print(f"Excluded Joined records: {len(df) - len(cleaned_df):,}")

    print("\nTarget distribution:")
    print(cleaned_df["Churn"].value_counts().rename(
        index={0: "Stayed", 1: "Churned"}
    ).to_string())

    print("\nRemaining missing values in model data:")
    missing = model_df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.sort_values(ascending=False).to_string())

    print(f"\nSaved prepared data to: {PROCESSED_PATH}")

    return cleaned_df, model_df
