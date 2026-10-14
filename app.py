from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Project paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
FEATURES_PATH = REPORTS_PATH / "feature_analysis"

REPORT_PATH = REPORTS_PATH / "customer_retention_report.csv"
SUMMARY_PATH = REPORTS_PATH / "retention_summary.csv"
METRICS_PATH = REPORTS_PATH / "model_metrics.csv"
COMPARISON_PATH = REPORTS_PATH / "model_comparison.csv"
EVALUATION_PATH = REPORTS_PATH / "selected_model_evaluation.csv"
FEATURE_PATH = FEATURES_PATH / "feature_importance.csv"
FEATURE_PLOT_PATH = FEATURES_PATH / "feature_importance.png"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


# ============================================================
# 2. Page configuration
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. Data-loading and display helpers
# ============================================================

@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    """Load a CSV file and cache its contents."""
    return pd.read_csv(path_string)


@st.cache_resource
def load_model(path_string: str):
    """Load the saved model artifact."""
    return joblib.load(path_string)


def show_image_if_available(path: Path, caption: str) -> bool:
    """Display an image if it exists; otherwise show a useful warning."""
    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
        return True

    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path

    st.warning(f"Chart image not found: {display_path}")
    return False


def as_float(value):
    """Convert a value to float, returning None for missing/invalid values."""
    try:
        result = float(value)
        return None if pd.isna(result) else result
    except (TypeError, ValueError):
        return None


def read_metric_value(metrics_df: pd.DataFrame, metric_name: str):
    """Read a metric from Metric/Score or a one-row wide-format CSV."""
    if {"Metric", "Score"}.issubset(metrics_df.columns):
        names = metrics_df["Metric"].astype(str).str.strip().str.casefold()
        matches = metrics_df.loc[
            names == metric_name.casefold(),
            "Score",
        ]
        return as_float(matches.iloc[0]) if not matches.empty else None

    normalized = {
        str(column).strip().casefold(): column
        for column in metrics_df.columns
    }
    column = normalized.get(metric_name.casefold())

    if column is not None and not metrics_df.empty:
        return as_float(metrics_df.iloc[0][column])

    return None


def format_metric(value, percent=True):
    """Format a metric for display."""
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def load_selected_evaluation():
    """Load the selected model's saved test-set evaluation row."""
    if not EVALUATION_PATH.exists():
        return None

    evaluation = load_csv(str(EVALUATION_PATH))
    if evaluation.empty:
        return None

    required = {"Model", "Threshold", "TN", "FP", "FN", "TP"}
    if not required.issubset(evaluation.columns):
        return None

    return evaluation.iloc[0]


# ============================================================
# 4. Sidebar navigation
# ============================================================

st.sidebar.title("📊 Churn Analytics")
st.sidebar.caption("Customer Retention Intelligence")
st.sidebar.divider()

pages = [
    "Overview",
    "Customer Risk Explorer",
    "Model Performance",
    "Retention Insights",
]
selected_page = st.sidebar.radio("Navigation", pages)


# ============================================================
# 5. Load and validate the retention report
# ============================================================

required_files = [REPORT_PATH, SUMMARY_PATH]
missing_files = [
    path.name for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "Required report files are missing: "
        + ", ".join(missing_files)
    )
    st.info(
        "From your project root, run `python -m src.retention_report`, "
        "then refresh this dashboard."
    )
    st.stop()

report = load_csv(str(REPORT_PATH))
summary = load_csv(str(SUMMARY_PATH))

if report.empty:
    st.warning("The customer retention report contains no records.")
    st.stop()

required_columns = {
    "Customer_ID",
    "Predicted_Churn_Probability",
    "Risk_Category",
    "Recommended_Action",
}
missing_columns = required_columns - set(report.columns)

if missing_columns:
    st.error(
        "The retention report is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

report["Predicted_Churn_Probability"] = pd.to_numeric(
    report["Predicted_Churn_Probability"],
    errors="coerce",
)
report = report.dropna(
    subset=["Predicted_Churn_Probability", "Risk_Category"]
).copy()

if not report["Predicted_Churn_Probability"].between(0, 1).all():
    st.error("Some churn probabilities fall outside the valid 0–1 range.")
    st.stop()

report["Churn_Risk_Percent"] = (
    report["Predicted_Churn_Probability"] * 100
).round(1)


# ============================================================
# 6. Shared KPIs
# ============================================================

st.title("Customer Churn Analytics")
st.caption(
    "Understand customer churn risk, explore customer segments, "
    "and identify retention priorities."
)

total_customers = len(report)
high_risk_count = int((report["Risk_Category"] == "High").sum())
medium_risk_count = int((report["Risk_Category"] == "Medium").sum())
low_risk_count = int((report["Risk_Category"] == "Low").sum())

high_risk_pct = (
    high_risk_count / total_customers * 100
    if total_customers else 0
)

if "Actual_Churn" in report.columns:
    actual_churn = pd.to_numeric(
        report["Actual_Churn"],
        errors="coerce",
    )
    historical_churn_rate = (
        float(actual_churn.mean() * 100)
        if actual_churn.notna().any()
        else None
    )
else:
    historical_churn_rate = None


# ============================================================
# 7. Overview
# ============================================================

if selected_page == "Overview":
    st.subheader("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Analyzed", f"{total_customers:,}")
    col2.metric("High-Risk Customers", f"{high_risk_count:,}")
    col3.metric("High-Risk Share", f"{high_risk_pct:.1f}%")
    col4.metric(
        "Historical Churn Rate",
        f"{historical_churn_rate:.1f}%"
        if historical_churn_rate is not None
        else "Unavailable",
    )

    st.divider()
    left, right = st.columns(2)
    risk_order = ["High", "Medium", "Low"]

    with left:
        st.subheader("Customer Risk Distribution")
        risk_counts = (
            report["Risk_Category"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
        )
        st.bar_chart(risk_counts)

    with right:
        st.subheader("Average Churn Probability by Risk Group")
        risk_probability = (
            report.groupby("Risk_Category", observed=False)[
                "Predicted_Churn_Probability"
            ]
            .mean()
            .reindex(risk_order)
            .dropna()
        )
        st.bar_chart(risk_probability)

    st.divider()
    st.subheader("Customers Requiring Attention")

    high_risk_customers = (
        report[report["Risk_Category"] == "High"]
        .sort_values("Predicted_Churn_Probability", ascending=False)
        .head(10)
    )

    display_columns = [
        "Customer_ID",
        "Churn_Risk_Percent",
        "Risk_Category",
        "Recommended_Action",
    ]
    st.dataframe(
        high_risk_customers[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These scores were generated for historical records. They are not "
        "yet validated forecasts for future customers."
    )


# ============================================================
# 8. Customer Risk Explorer
# ============================================================

elif selected_page == "Customer Risk Explorer":
    st.subheader("Customer Risk Explorer")
    st.write(
        "Search customer records, filter risk categories, sort by risk, "
        "and export the results."
    )

    filter_col1, filter_col2 = st.columns([2, 1])

    with filter_col1:
        search_id = st.text_input(
            "Search Customer ID",
            placeholder="e.g. CG-000001",
        )

    with filter_col2:
        selected_risks = st.multiselect(
            "Risk categories",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

    filtered = report[
        report["Risk_Category"].isin(selected_risks)
    ].copy()

    if search_id.strip():
        filtered = filtered[
            filtered["Customer_ID"].astype(str).str.contains(
                search_id.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    sort_order = st.selectbox(
        "Sort customers by",
        ["Highest churn risk", "Lowest churn risk", "Customer ID"],
    )

    if sort_order == "Highest churn risk":
        filtered = filtered.sort_values(
            "Predicted_Churn_Probability",
            ascending=False,
        )
    elif sort_order == "Lowest churn risk":
        filtered = filtered.sort_values(
            "Predicted_Churn_Probability",
            ascending=True,
        )
    else:
        filtered = filtered.sort_values("Customer_ID")

    st.metric("Matching Customers", f"{len(filtered):,}")

    display_columns = [
        "Customer_ID",
        "Churn_Risk_Percent",
        "Risk_Category",
        "Recommended_Action",
    ]
    if "Actual_Churn" in filtered.columns:
        display_columns.append("Actual_Churn")

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = (
        filtered.drop(
            columns=["Churn_Risk_Percent"],
            errors="ignore",
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download Filtered Customers CSV",
        data=csv_data,
        file_name="filtered_customer_retention.csv",
        mime="text/csv",
    )


# ============================================================
# 9. Model Performance and Explainability
# ============================================================

elif selected_page == "Model Performance":
    st.subheader("Model Performance & Explainability")
    st.caption(
        "Compare model performance, inspect classification errors, "
        "and understand factors associated with model predictions."
    )

    # ---- A. Headline metrics ----
    selected_evaluation = load_selected_evaluation()
    selected_model_name = "Selected Model"

    if MODEL_PATH.exists():
        saved_model = load_model(str(MODEL_PATH))
        selected_model_name = saved_model.get(
            "model_name",
            selected_model_name,
        )
    else:
        saved_model = None

    st.markdown(f"### Selected Model: {selected_model_name}")

    if METRICS_PATH.exists():
        metrics = load_csv(str(METRICS_PATH))

        accuracy = read_metric_value(metrics, "Accuracy")
        precision = read_metric_value(metrics, "Precision")
        recall = read_metric_value(metrics, "Recall")
        f1 = read_metric_value(metrics, "F1-score")
        roc_auc = read_metric_value(metrics, "ROC-AUC")

        metric_cols = st.columns(5)
        for col, label, value in zip(
            metric_cols,
            ["Accuracy", "Precision", "Churn Recall", "F1-score", "ROC-AUC"],
            [accuracy, precision, recall, f1, roc_auc],
        ):
            col.metric(label, format_metric(value))
    else:
        st.warning(f"Metrics file not found: {METRICS_PATH}")
        accuracy = precision = recall = f1 = roc_auc = None

    st.caption(
        "These headline metrics should represent the held-out test set. "
        "Recall is the fraction of actual churners detected; precision is "
        "the fraction of flagged churners who actually churned."
    )
    st.divider()

    # ---- B. Model comparison ----
    st.markdown("### Model Comparison")

    if COMPARISON_PATH.exists():
        comparison = load_csv(str(COMPARISON_PATH))

        if "Model" not in comparison.columns:
            st.warning(
                "model_comparison.csv does not contain a 'Model' column. "
                "Check its schema before plotting."
            )
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )
        else:
            metric_columns = [
                column
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score",
                    "ROC-AUC",
                ]
                if column in comparison.columns
            ]

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            if metric_columns:
                chart_data = comparison.set_index("Model")[metric_columns].copy()
                for column in metric_columns:
                    chart_data[column] = pd.to_numeric(
                        chart_data[column],
                        errors="coerce",
                    )

                st.markdown("#### Compare evaluation metrics")

                # Grouped bars make model-by-model comparisons clearer than
                # Streamlit's default stacked bar chart.
                plot_data = chart_data.T
                fig, ax = plt.subplots(figsize=(11, 5.5))

                plot_data.plot(
                    kind="bar",
                    ax=ax,
                    width=0.82,
                )

                ax.set_title("Model Performance Comparison")
                ax.set_ylabel("Score")
                ax.set_xlabel("Evaluation Metric")
                ax.set_ylim(0, 1.05)
                ax.tick_params(axis="x", rotation=15)
                ax.legend(
                    title="Model",
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.38),
                    ncol=min(3, max(1, len(plot_data.columns))),
                )
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if "ROC-AUC" in comparison.columns:
                    auc_data = comparison.copy()
                    auc_data["ROC-AUC"] = pd.to_numeric(
                        auc_data["ROC-AUC"],
                        errors="coerce",
                    )
                    auc_data = auc_data.dropna(subset=["ROC-AUC"])

                    if not auc_data.empty:
                        best_row = auc_data.loc[
                            auc_data["ROC-AUC"].idxmax()
                        ]
                        st.info(
                            f"Highest ROC-AUC in this comparison: "
                            f"{best_row['Model']} "
                            f"({best_row['ROC-AUC']:.4f})."
                        )
            else:
                st.warning(
                    "No recognized evaluation metric columns were found."
                )
    else:
        st.warning(
            "Model comparison file not found. Expected "
            "reports/model_comparison.csv."
        )

    st.divider()

    # ---- C. Confusion matrix and ROC curve ----
    st.markdown("### Classification Diagnostics")
    tab_cm, tab_roc = st.tabs(["Confusion Matrix", "ROC Curve"])

    # Use the selected model name from the saved model artifact to select
    # its matching diagnostic folder.
    if selected_model_name == "Selected Model":
        selected_folder = "logistic_regression"
    else:
        selected_folder = selected_model_name.strip().lower().replace(" ", "_")

    cm_path = FIGURES_PATH / selected_folder / "confusion_matrix.png"
    roc_path = FIGURES_PATH / selected_folder / "roc_curve.png"

    with tab_cm:
        st.markdown(f"#### {selected_model_name} — Held-out Test Set")
        show_image_if_available(
            cm_path,
            f"{selected_model_name} confusion matrix",
        )

        st.markdown("**Test-set classification results**")

        if selected_evaluation is not None:
            cm_col1, cm_col2 = st.columns(2)

            with cm_col1:
                st.metric(
                    "Correctly detected churners (TP)",
                    f"{int(selected_evaluation['TP']):,}",
                )
                st.metric(
                    "Missed churners (FN)",
                    f"{int(selected_evaluation['FN']):,}",
                )

            with cm_col2:
                st.metric(
                    "False churn alerts (FP)",
                    f"{int(selected_evaluation['FP']):,}",
                )
                st.metric(
                    "Correctly identified stayers (TN)",
                    f"{int(selected_evaluation['TN']):,}",
                )

            st.caption(
                f"Selected model: {selected_evaluation['Model']} | "
                f"Decision threshold: {float(selected_evaluation['Threshold']):.2f}"
            )
        else:
            st.warning(
                "Selected-model evaluation data was not found or had an "
                "unexpected format. Run `python main.py` to regenerate "
                "`reports/selected_model_evaluation.csv`."
            )

    with tab_roc:
        st.markdown(f"#### {selected_model_name} — Held-out Test Set")
        show_image_if_available(
            roc_path,
            f"{selected_model_name} ROC curve",
        )

        st.write(
            "ROC-AUC measures how well the model ranks churners above "
            "customers who stay across classification thresholds."
        )

        st.metric(
            f"{selected_model_name} ROC-AUC",
            format_metric(roc_auc, percent=False),
        )

        st.caption(
            "ROC-AUC is a ranking-discrimination measure. It does not mean "
            "that the same percentage of predictions is correct."
        )

    st.divider()

    # ---- D. Feature analysis ----
    st.markdown("### What Influences Churn Predictions?")

    if FEATURE_PATH.exists():
        feature_data = load_csv(str(FEATURE_PATH))
        expected_feature_columns = {
            "Feature",
            "Coefficient",
            "Absolute_Importance",
            "Direction",
        }
        missing_feature_columns = (
            expected_feature_columns - set(feature_data.columns)
        )

        if missing_feature_columns:
            st.warning(
                "Feature analysis is missing columns: "
                + ", ".join(sorted(missing_feature_columns))
            )
            st.dataframe(
                feature_data,
                use_container_width=True,
                hide_index=True,
            )
        else:
            feature_data["Coefficient"] = pd.to_numeric(
                feature_data["Coefficient"],
                errors="coerce",
            )
            feature_data["Absolute_Importance"] = pd.to_numeric(
                feature_data["Absolute_Importance"],
                errors="coerce",
            )
            feature_data = feature_data.dropna(
                subset=["Coefficient", "Absolute_Importance"]
            )

            if FEATURE_PLOT_PATH.exists():
                show_image_if_available(
                    FEATURE_PLOT_PATH,
                    "Logistic Regression feature coefficients",
                )

            higher, lower, all_features = st.tabs(
                [
                    "Associated with Higher Churn Odds",
                    "Associated with Lower Churn Odds",
                    "All Features",
                ]
            )

            with higher:
                higher_df = (
                    feature_data[feature_data["Coefficient"] > 0]
                    .sort_values("Coefficient", ascending=False)
                    .head(10)
                )
                if higher_df.empty:
                    st.info("No positive coefficients found.")
                else:
                    st.bar_chart(
                        higher_df.set_index("Feature")[["Coefficient"]]
                    )
                    st.dataframe(
                        higher_df[
                            ["Feature", "Coefficient", "Direction"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            with lower:
                lower_df = (
                    feature_data[feature_data["Coefficient"] < 0]
                    .sort_values("Coefficient", ascending=True)
                    .head(10)
                )
                if lower_df.empty:
                    st.info("No negative coefficients found.")
                else:
                    st.bar_chart(
                        lower_df.set_index("Feature")[["Coefficient"]]
                    )
                    st.dataframe(
                        lower_df[
                            ["Feature", "Coefficient", "Direction"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            with all_features:
                ranked = feature_data.sort_values(
                    "Absolute_Importance",
                    ascending=False,
                )
                st.dataframe(
                    ranked,
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                "Positive coefficients are associated with higher predicted "
                "churn odds; negative coefficients are associated with lower "
                "predicted churn odds, holding other encoded features "
                "constant. These are associations, not causal effects. "
                "Infrequent categories may have unstable estimates."
            )
    else:
        st.warning(
            "Feature-importance CSV not found: "
            f"{FEATURE_PATH.relative_to(PROJECT_ROOT)}"
        )

    st.divider()

    # ---- E. Model configuration ----
    st.markdown("### Model Configuration")

    if saved_model is not None:
        info1, info2, info3 = st.columns(3)

        info1.metric(
            "Selected Algorithm",
            saved_model.get("model_name", "Unknown"),
        )
        info2.metric(
            "Input Features",
            len(saved_model.get("feature_names", [])),
        )

        threshold = saved_model.get("threshold")
        info3.metric(
            "Selected Threshold",
            f"{threshold:.2f}"
            if isinstance(threshold, (int, float))
            else "Not saved",
        )
    else:
        st.warning(f"Saved model not found: {MODEL_PATH}")

    st.info(
        "The confusion matrix and metrics describe held-out test data. "
        "The customer retention report scores historical records and is "
        "not independent evidence of future performance. Validate on new "
        "customer data before deploying retention campaigns."
    )


# ============================================================
# 10. Retention Insights
# ============================================================

elif selected_page == "Retention Insights":
    st.subheader("Retention Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", f"{high_risk_count:,}")
    col2.metric("Medium Risk", f"{medium_risk_count:,}")
    col3.metric("Low Risk", f"{low_risk_count:,}")

    st.divider()
    st.subheader("Recommended Priorities")

    st.markdown(
        """
        **1. High risk — prioritize outreach**

        Review customer concerns and consider a personalized retention
        conversation or a suitable contract or pricing review.

        **2. Medium risk — investigate engagement**

        Review recent interactions and offer relevant support or services
        where appropriate.

        **3. Low risk — maintain the relationship**

        Continue normal customer engagement and monitor satisfaction.
        """
    )

    st.divider()
    st.subheader("Risk Segment Details")

    segment_summary = (
        report.groupby("Risk_Category", observed=False)
        .agg(
            Customer_Count=("Customer_ID", "count"),
            Average_Churn_Probability=(
                "Predicted_Churn_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    segment_summary["Customer_Percentage"] = (
        segment_summary["Customer_Count"] / len(report) * 100
    ).round(2)
    segment_summary["Average_Churn_Probability"] = (
        segment_summary["Average_Churn_Probability"].round(4)
    )

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Risk bands are illustrative prioritization rules. Before deploying "
        "retention campaigns, validate performance on unseen customers and "
        "measure whether interventions actually reduce churn."
    )


# ============================================================
# 11. Footer
# ============================================================

st.sidebar.divider()
st.sidebar.caption("Customer Churn Analytics | Portfolio Project")
