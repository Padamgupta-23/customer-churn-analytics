
from pathlib import Path

import pandas as pd


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep the original customer data in one predictable location.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer_Data.csv"


def load_customer_data():
    """Load the customer dataset from the raw data folder."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place Customer_Data.csv inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The customer dataset is empty.")

    return df
