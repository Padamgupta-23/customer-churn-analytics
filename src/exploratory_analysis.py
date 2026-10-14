
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")


def prepare_output_folder():
    """Create the folder for charts if it does not exist."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def plot_customer_status(df):
    """Show how customers are distributed across their statuses."""

    status_counts = df["Customer_Status"].value_counts()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=status_counts.index,
        y=status_counts.values,
    )

    plt.title("Customer Distribution by Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "customer_status_distribution.png",
        dpi=150,
    )
    plt.close()


def plot_churn_rate_by_contract(df):
    """Compare churn rates across contract types."""

    # Exclude Joined customers because their final churn outcome
    # is not represented by Stayed or Churned.
    known_outcomes = df[
        df["Customer_Status"].isin(["Stayed", "Churned"])
    ].copy()

    # Churn rate = churned customers / customers with known outcomes.
    contract_summary = (
        known_outcomes.groupby("Contract")
        .agg(
            total_customers=("Customer_Status", "size"),
            churned_customers=(
                "Customer_Status",
                lambda status: (status == "Churned").sum(),
            ),
        )
    )

    contract_summary["Churn_Rate"] = (
        contract_summary["churned_customers"]
        / contract_summary["total_customers"]
        * 100
    )

    contract_summary = contract_summary.reset_index()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=contract_summary,
        x="Contract",
        y="Churn_Rate",
    )

    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "churn_rate_by_contract.png",
        dpi=150,
    )
    plt.close()

    print("\n--- Churn Rate by Contract ---")
    print(
        contract_summary[
            ["Contract", "total_customers", "churned_customers", "Churn_Rate"]
        ].round(2).to_string(index=False)
    )


def plot_tenure_by_status(df):
    """Compare customer tenure across known outcomes."""

    known_outcomes = df[
        df["Customer_Status"].isin(["Stayed", "Churned"])
    ]

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=known_outcomes,
        x="Tenure_in_Months",
        hue="Customer_Status",
        bins=20,
        element="step",
        stat="count",
        common_norm=False,
    )

    plt.title("Customer Tenure by Status")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "tenure_distribution.png",
        dpi=150,
    )
    plt.close()

def plot_monthly_charges_by_status(df):
    """Compare monthly charges across customer statuses."""

    known_outcomes = df[
        df["Customer_Status"].isin(["Stayed", "Churned"])
    ]

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=known_outcomes,
        x="Customer_Status",
        y="Monthly_Charge",
    )

    plt.title("Monthly Charges by Customer Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Monthly Charge")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "monthly_charges_by_status.png",
        dpi=150,
    )
    plt.close()



def run_exploratory_analysis(df):
    """Generate the initial customer churn analysis charts."""

    prepare_output_folder()

    plot_customer_status(df)
    plot_churn_rate_by_contract(df)
    plot_tenure_by_status(df)
    plot_monthly_charges_by_status(df)

    # Additional business analyses
    plot_churn_reasons(df)
    plot_revenue_exposure(df)
    plot_tenure_comparison(df)

    print("\nEDA completed successfully.")
    print(f"Charts saved to: {FIGURES_PATH}")




def plot_churn_reasons(df):
    """Show the most frequently recorded reasons for churn."""

    churned_customers = df[
        df["Customer_Status"] == "Churned"
    ]

    # Ignore missing reasons rather than treating them as real reasons.
    reasons = churned_customers["Churn_Reason"].dropna()

    if reasons.empty:
        print("No churn reasons available to plot.")
        return

    reason_counts = reasons.value_counts().head(10).sort_values()

    plt.figure(figsize=(11, 7))

    reason_counts.plot(
        kind="barh",
        color="steelblue",
    )

    plt.title("Top 10 Reasons for Customer Churn")
    plt.xlabel("Number of Customers")
    plt.ylabel("Recorded Churn Reason")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "churn_reasons.png",
        dpi=150,
    )
    plt.close()

    print("\n--- Top 10 Churn Reasons ---")
    print(reason_counts.sort_values(ascending=False).to_string())


def plot_revenue_exposure(df):
    """Estimate monthly billing associated with each customer status."""

    required_columns = {"Customer_Status", "Monthly_Charge"}

    if not required_columns.issubset(df.columns):
        print("Required columns for revenue analysis are missing.")
        return

    summary = (
        df.groupby("Customer_Status")
        .agg(
            customer_count=("Monthly_Charge", "size"),
            total_monthly_charges=("Monthly_Charge", "sum"),
            average_monthly_charge=("Monthly_Charge", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values("total_monthly_charges")

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=summary,
        x="Customer_Status",
        y="total_monthly_charges",
        color="steelblue",
    )

    plt.title("Total Monthly Charges by Customer Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Total Monthly Charges")
    plt.ticklabel_format(axis="y", style="plain")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "revenue_exposure_by_status.png",
        dpi=150,
    )
    plt.close()

    print("\n--- Monthly Billing Summary by Status ---")
    print(summary.round(2).to_string(index=False))

    churned = summary[
        summary["Customer_Status"] == "Churned"
    ]

    if not churned.empty:
        exposure = churned["total_monthly_charges"].iloc[0]

        print(
            "\nMonthly charges associated with churned customers: "
            f"{exposure:,.2f}"
        )
        print(
            "This is descriptive billing exposure, not guaranteed "
            "recoverable revenue."
        )



def plot_tenure_comparison(df):
    """Compare the tenure distributions of stayed and churned customers."""

    known_outcomes = df[
        df["Customer_Status"].isin(["Stayed", "Churned"])
    ]

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=known_outcomes,
        x="Customer_Status",
        y="Tenure_in_Months",
        hue="Customer_Status",
        legend=False,
    )

    plt.title("Customer Tenure: Stayed vs Churned")
    plt.xlabel("Customer Status")
    plt.ylabel("Tenure (Months)")
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "tenure_comparison.png",
        dpi=150,
    )
    plt.close()
