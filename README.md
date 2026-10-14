# Customer Churn Analytics

An end-to-end machine learning and analytics project that estimates customer churn risk, compares classification models, and helps prioritize customer-retention actions through an interactive Streamlit dashboard.

> **Technologies:** Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit

## Live Demo

- **Streamlit app:** https://customer-churn-analytics-model.streamlit.app/
- **GitHub repository:** https://github.com/Padamgupta-23/customer-churn-analytics

## Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who may leave can help a business prioritize outreach, investigate service issues, and evaluate retention strategies.

This project builds a binary churn-classification workflow and presents its outputs in a multi-page Streamlit dashboard. The dashboard includes portfolio-level risk summaries, customer-level risk exploration, model evaluation, and retention insights.

### Dashboard pages

- **Overview:** customer count, high-risk customer count and share, historical churn rate, and risk distribution.
- **Customer Risk Explorer:** search by customer ID, filter by risk category, sort customers by predicted risk, and download filtered records as CSV.
- **Model Performance:** compare the baseline, Logistic Regression, and Random Forest; inspect a confusion matrix, ROC curve, and Logistic Regression coefficients.
- **Retention Insights:** review risk segments and suggested retention priorities.

## Model Results

The current held-out test evaluation for the selected Logistic Regression model is:

| Metric | Score |
|---|---:|
| Accuracy | 80.95% |
| Precision | 64.82% |
| Churn recall | 74.35% |
| F1-score | 69.26% |
| ROC-AUC | 0.8689 |

The model uses 23 audited input features and a decision threshold of 0.40, selected using validation data. The project compares Logistic Regression and Random Forest against a majority-class baseline.

**Metric notes**
- Recall measures the proportion of actual churners identified by the model.
- Precision measures the proportion of customers flagged as churners who actually churned.
- ROC-AUC measures ranking discrimination across thresholds; it is not the percentage of predictions that are correct.
- Results describe the held-out test split from the current training run. They are not a guarantee of performance on future customers.

## Workflow

1. **Data loading and quality checks** — inspect dataset structure, missing values, duplicates, and target distribution.
2. **Data preparation** — retain customers with known `Stayed` or `Churned` outcomes, create a binary `Churn` target, and handle service-detail fields that are not applicable when a customer does not use that service.
3. **Feature audit** — exclude customer identifiers and columns that reveal the outcome; hold back features that need further validation for availability at prediction time.
4. **Exploratory analysis** — examine churn by contract, customer status, tenure, charges, and churn reasons.
5. **Train and validate** — split data into training, validation, and held-out test sets; fit preprocessing and classification within scikit-learn pipelines.
6. **Model and threshold selection** — choose the candidate model and classification threshold using validation F1-score, not the held-out test results.
7. **Evaluation and explainability** — report classification metrics, save confusion matrices and ROC curves, and inspect Logistic Regression coefficients.
8. **Retention reporting** — score the prepared historical records, group them into illustrative risk bands, and attach suggested actions.
9. **Interactive dashboard** — explore summaries, customer records, model diagnostics, and retention priorities in Streamlit.

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualizations:** Matplotlib, Seaborn
- **Model persistence:** Joblib
- **Dashboard:** Streamlit

## Project Structure

```text
customer-churn-analytics/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── notebooks/
├── reports/
│   ├── feature_analysis/
│   ├── figures/
│   ├── customer_retention_report.csv
│   ├── model_comparison.csv
│   ├── model_metrics.csv
│   ├── model_evaluation_summary.csv
│   ├── model_training_metadata.json
│   ├── selected_model_evaluation.csv
│   ├── retention_summary.csv
│   └── threshold_comparison.csv
└── src/
    ├── data_cleaning.py
    ├── data_loader.py
    ├── data_quality.py
    ├── data_understanding.py
    ├── exploratory_analysis.py
    ├── feature_audit.py
    ├── model_explainability.py
    ├── model_training.py
    ├── retention_evaluation.py
    ├── retention_report.py
    └── threshold_analysis.py
```

Some generated files may be recreated by the pipeline. Keep the model artifact and report files required by the dashboard available in the deployed repository, or update the app to generate/load them as part of deployment.

## Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-churn-analytics
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `requirements.txt` includes all packages imported by both the training pipeline and the Streamlit app, including `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `joblib`.

### 4. Start the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

### 5. Rebuild the analysis and model (optional)

If the required raw dataset is available locally and you want to regenerate the prepared data, model, and evaluation reports, run:

```bash
python main.py
```

Then regenerate the customer-level retention report if needed:

```bash
python -m src.retention_report
```

Run the pipeline before starting the app if the model or report artifacts are missing. The exact generated files depend on the current pipeline implementation.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Verify that `app.py`, `requirements.txt`, the saved model, and every report/data file loaded at app startup are committed and available.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Create a new app, select your repository and branch, and set the main file path to `app.py`.
5. Deploy and wait for the build to finish.
6. Open the public app URL and test all four pages, customer search, filters, charts, and CSV download.

**Important:** Streamlit Community Cloud runs the app for visitors. People who visit your GitHub repository can click the deployed app link in the README and use the web app in their browser; they do not need to install Python. If you only upload the code to GitHub and do not deploy it, visitors will see the source code but will not automatically get a running website.

Deployment checklist:
- Add the deployed URL to the **Live Demo** section above.
- Commit the correct `requirements.txt`.
- Do not commit `.venv/`, secrets, API keys, or personal files.
- Confirm the dataset license and redistribution terms before uploading the raw CSV.
- Check repository size and hosting limits before committing large model/data artifacts.
- Test the deployed app from a browser session that is not relying on your local files.

## Responsible Interpretation and Limitations

- The target represents a historical `Stayed` or `Churned` outcome; customers with a `Joined` status are excluded from binary training.
- Risk categories are illustrative thresholds for prioritization, not proof that a customer will churn.
- Retention reports score historical records. Those scores should not be presented as independently validated forecasts for future customers.
- Logistic Regression coefficients describe conditional associations in the encoded model, not causation. Rare categories may have unstable coefficients.
- Before using this workflow for real business decisions, validate on newer unseen customer data, assess probability calibration, review potential bias, and measure whether retention interventions actually reduce churn.
- Do not use the example retention recommendations as a substitute for business, legal, or customer-support review.

## Future Improvements

- Add calibration analysis and precision-recall curves.
- Evaluate a time-based split if timestamps and prediction timing are available.
- Add model monitoring and drift checks for new customer data.
- Evaluate retention interventions with controlled experiments.
- Add automated tests for data preparation, feature consistency, and report generation.
- Improve explanations with carefully validated local explanations if needed.

## Author

**Padam Gupta**  
B.Tech Computer Science & Engineering | Data Science & Machine Learning

Add your LinkedIn and other portfolio links here if you want them displayed.

---

If you use or adapt this project, verify the dataset source and license and describe the model results accurately.
