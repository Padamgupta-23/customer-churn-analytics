
def investigate_missing_values(df):
    """Investigate why important columns contain missing values."""

    print("\n" + "=" * 55)
    print("INVESTIGATING MISSING VALUES")
    print("=" * 55)

    # 1. Check missing values by customer status.
    print("\n--- Missing Values by Customer Status ---")

    columns_to_check = [
        "Value_Deal",
        "Internet_Type",
        "Online_Security",
        "Multiple_Lines",
        "Churn_Category",
        "Churn_Reason",
    ]

    available_columns = [
        column for column in columns_to_check
        if column in df.columns
    ]

    missing_by_status = (
        df.groupby("Customer_Status")[available_columns]
        .apply(lambda group: group.isna().sum())
    )

    print(missing_by_status.to_string())

    # 2. Check internet-service relationships.
    print("\n--- Internet Service vs Internet Type ---")

    if {"Internet_Service", "Internet_Type"}.issubset(df.columns):
        print(
            df.groupby("Internet_Service", dropna=False)["Internet_Type"]
            .agg(
                total_customers="size",
                missing_values=lambda values: values.isna().sum(),
                unique_values=lambda values: values.nunique(),
            )
            .to_string()
        )

    # 3. Inspect churn explanations.
    print("\n--- Churn Reasons by Customer Status ---")

    if {"Customer_Status", "Churn_Reason"}.issubset(df.columns):
        print(
            df.groupby("Customer_Status")["Churn_Reason"]
            .apply(lambda values: values.value_counts(dropna=False).head(10))
            .to_string()
        )

    # 4. Compare numerical characteristics by status.
    print("\n--- Numerical Summary by Customer Status ---")

    numerical_columns = [
        "Age",
        "Tenure_in_Months",
        "Monthly_Charge",
        "Total_Revenue",
    ]

    available_numerical = [
        column for column in numerical_columns
        if column in df.columns
    ]

    print(
        df.groupby("Customer_Status")[available_numerical]
        .agg(["mean", "median"])
        .round(2)
        .to_string()
    )

    print("\nInvestigation completed.")
