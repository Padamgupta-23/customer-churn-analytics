
import pandas as pd


def inspect_customer_data(df):
    """Print a basic data-quality report for the dataset."""

    print("\n" + "=" * 55)
    print("CUSTOMER DATA QUALITY REPORT")
    print("=" * 55)

    # Basic size of the dataset.
    rows, columns = df.shape
    print(f"\nNumber of customers/records: {rows:,}")
    print(f"Number of columns: {columns}")

    # Column names and data types.
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes.to_string())

    # Missing values.
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    # Duplicate records.
    print("\n--- Duplicate Records ---")
    duplicate_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {duplicate_count:,}")

    # Unique values help us understand categorical columns.
    print("\n--- Unique Values in Categorical Columns ---")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:
        print(f"{column}: {df[column].nunique(dropna=True)} unique values")

    # Check whether the dataset contains the expected churn target.
    print("\n--- Potential Churn Target ---")

    target_column = "Customer_Status"

    if target_column in df.columns:
        print(df[target_column].value_counts(dropna=False).to_string())
    elif "Churn" in df.columns:
        print(df["Churn"].value_counts(dropna=False).to_string())
    else:
        print("No familiar churn target column found.")
        print("We will identify the correct target after inspecting the data.")

    print("\n" + "=" * 55)
