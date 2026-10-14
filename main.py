from src.model_training import train_and_evaluate_model
from src.data_cleaning import prepare_customer_data
from src.exploratory_analysis import run_exploratory_analysis
from src.data_understanding import investigate_missing_values
from src.data_loader import load_customer_data
from src.data_quality import inspect_customer_data


def main():
    print("Customer Churn Prediction & Retention Analytics")
    print("Starting data inspection...\n")

    # Step 1: Load the customer data.
    df = load_customer_data()

    # Step 2: Display a quick preview.
    print("--- First Five Records ---")
    print(df.head().to_string(index=False))

    # Step 3: Inspect the dataset's quality.
    inspect_customer_data(df)
    investigate_missing_values(df)
    run_exploratory_analysis(df)
    cleaned_df, model_df = prepare_customer_data(df)
    model, metrics = train_and_evaluate_model(model_df)

    print("\nData inspection completed successfully.")


if __name__ == "__main__":
    main()
