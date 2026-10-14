
# src/feature_audit.py

# These columns directly describe the target or identify a customer.
ALWAYS_EXCLUDE = [
    "Customer_ID",
    "Customer_Status",
    "Churn_Category",
    "Churn_Reason",
]

# Exclude these from our first model until we verify their meaning,
# calculation and availability at the intended prediction time.
REVIEW_BEFORE_USE = [
    "Total_Charges",
    "Total_Refunds",
    "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges",
    "Avg_Monthly_Data_Usage_GB",
    "Late_Payments_Last_6_Months",
    "Total_Revenue",
    "Customer_Satisfaction_Score",
    "Support_Tickets_Last_90_Days",
    "Days_Since_Last_Interaction",
    "Estimated_Annual_Value",
    "Service_Count",
    "Contract_Months",
    "Revenue_Per_Tenure_Month",
]


def audit_features(model_df):
    """Report excluded features and return a conservative feature set."""

    excluded = [
        column
        for column in ALWAYS_EXCLUDE + REVIEW_BEFORE_USE
        if column in model_df.columns and column != "Churn"
    ]

    features = model_df.drop(
        columns=["Churn"] + excluded,
        errors="ignore",
    )

    print("\n" + "=" * 55)
    print("FEATURE AUDIT")
    print("=" * 55)

    print("\nExcluded outcome/identifier columns:")
    for column in ALWAYS_EXCLUDE:
        if column in model_df.columns:
            print(f"  - {column}")

    print("\nExcluded pending review:")
    for column in REVIEW_BEFORE_USE:
        if column in model_df.columns:
            print(f"  - {column}")

    print(f"\nFeatures retained: {features.shape[1]}")
    print(f"Records available: {features.shape[0]}")

    print("\nRetained features:")
    for column in features.columns:
        print(f"  - {column}")

    return features
