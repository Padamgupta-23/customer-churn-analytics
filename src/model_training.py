from pathlib import Path
import json

import joblib
import matplotlib

# Use a non-interactive backend so training also works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_audit import audit_features
from src.model_explainability import save_logistic_regression_importance
from src.threshold_analysis import evaluate_thresholds, plot_threshold_comparison


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
FEATURES_PATH = REPORTS_PATH / "feature_analysis"


# ============================================================
# 2. Metrics and utility functions
# ============================================================

def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate binary classification metrics for the churn class."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_probability)),
    }


def make_safe_name(model_name):
    """Convert a model name to a stable filename/directory name."""
    return model_name.strip().lower().replace(" ", "_")


def create_model_pipeline(X, classifier=None):
    """Create a preprocessing and classification pipeline."""
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numerical_columns:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if classifier is None:
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_evaluation_artifacts(
    model_name,
    y_true,
    y_pred,
    y_probability,
    threshold,
    metrics,
):
    """Save confusion matrix, ROC curve, predictions, and metrics per model."""
    safe_name = make_safe_name(model_name)
    output_dir = FIGURES_PATH / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit labels ensure a consistent Stayed / Churned matrix layout.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Stayed", "Actual_Churned"],
        columns=["Predicted_Stayed", "Predicted_Churned"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)

    # Also save a one-row file that is easy for the dashboard to read.
    pd.DataFrame(
        [
            {
                "Model": model_name,
                "Threshold": float(threshold),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                **metrics,
            }
        ]
    ).to_csv(output_dir / "evaluation_summary.csv", index=False)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"],
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        y_probability,
        name=f"{model_name} (AUC = {metrics['ROC-AUC']:.3f})",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random ranking")
    ax.set_title(f"{model_name} — ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        output_dir / "roc_curve.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    predictions_df = pd.DataFrame(
        {
            "Actual_Churn": pd.Series(y_true).reset_index(drop=True),
            "Predicted_Churn": pd.Series(y_pred).reset_index(drop=True),
            "Predicted_Churn_Probability": pd.Series(
                y_probability
            ).reset_index(drop=True),
        }
    )
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# 3. Train, validate, compare, and evaluate
# ============================================================

def train_and_evaluate_model(model_df):
    """
    Train candidate classifiers, select the threshold/model on validation
    data only, evaluate once on held-out test data, and save artifacts.

    Returns:
        best_model: fitted sklearn Pipeline
        best_test_metrics: dictionary of held-out test metrics
    """
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    if "Churn" not in model_df.columns:
        raise ValueError("model_df must contain the 'Churn' target column.")

    # Feature audit should exclude identifiers, outcome-revealing columns,
    # and features that have not been approved for prediction-time use.
    X = audit_features(model_df)
    y = model_df["Churn"].astype(int)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No model features remain after feature auditing.")

    if y.nunique() != 2:
        raise ValueError("Churn target must contain both classes: 0 and 1.")

    # Split: 60% training, 20% validation, 20% final test.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    print(f"\nTraining records:   {len(X_train):,}")
    print(f"Validation records: {len(X_validation):,}")
    print(f"Testing records:    {len(X_test):,}")
    print(f"Features before encoding: {X.shape[1]}")

    # Majority-class baseline.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "Accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "Precision": float(
            precision_score(y_test, baseline_predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, baseline_predictions, zero_division=0)
        ),
        "F1-score": float(
            f1_score(y_test, baseline_predictions, zero_division=0)
        ),
        # ROC-AUC is not useful for a constant prediction score.
        "ROC-AUC": float("nan"),
    }

    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }

    trained_models = {}
    validation_results = {}
    test_results = {}
    threshold_tables = []

    # Train candidates and select a threshold using validation data only.
    for model_name, classifier in candidate_classifiers.items():
        print(f"\nTraining {model_name}...")

        candidate_model = create_model_pipeline(
            X_train,
            classifier=classifier,
        )
        candidate_model.fit(X_train, y_train)

        validation_probabilities = candidate_model.predict_proba(
            X_validation
        )[:, 1]

        threshold_results = evaluate_thresholds(
            y_validation,
            validation_probabilities,
        )

        if threshold_results.empty or "F1-score" not in threshold_results:
            raise ValueError(
                "Threshold analysis did not return an F1-score table."
            )

        threshold_results.insert(0, "Model", model_name)
        threshold_tables.append(threshold_results)

        best_row = threshold_results.loc[
            threshold_results["F1-score"].idxmax()
        ]
        selected_threshold = float(best_row["Threshold"])
        validation_f1 = float(best_row["F1-score"])

        print(f"{model_name} validation threshold: {selected_threshold:.2f}")
        print(f"{model_name} validation F1: {validation_f1:.4f}")

        # The test set is not used to choose the model or threshold.
        test_probabilities = candidate_model.predict_proba(X_test)[:, 1]
        test_predictions = (
            test_probabilities >= selected_threshold
        ).astype(int)

        test_metrics = calculate_metrics(
            y_test,
            test_predictions,
            test_probabilities,
        )

        trained_models[model_name] = candidate_model
        validation_results[model_name] = {
            "threshold": selected_threshold,
            "F1-score": validation_f1,
        }
        test_results[model_name] = {
            "metrics": test_metrics,
            "predictions": test_predictions,
            "probabilities": test_probabilities,
            "threshold": selected_threshold,
        }

    # Ensure directories exist before writing artifacts.
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save threshold analysis and charts.
    combined_thresholds = pd.concat(threshold_tables, ignore_index=True)
    combined_thresholds.to_csv(
        REPORTS_PATH / "threshold_comparison.csv",
        index=False,
    )

    for model_name in candidate_classifiers:
        model_thresholds = combined_thresholds[
            combined_thresholds["Model"] == model_name
        ].drop(columns=["Model"])

        safe_name = make_safe_name(model_name)
        plot_threshold_comparison(
            model_thresholds,
            FIGURES_PATH / f"{safe_name}_threshold_comparison.png",
        )

    # Compare models on the same held-out test set.
    comparison_rows = [
        {"Model": "Majority Baseline", **baseline_metrics}
    ]
    for model_name, result in test_results.items():
        comparison_rows.append(
            {"Model": model_name, **result["metrics"]}
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        REPORTS_PATH / "model_comparison.csv",
        index=False,
    )

    print("\n--- Model Comparison on Held-out Test Data ---")
    print(comparison.round(4).to_string(index=False))

    # Select model based on validation F1 only, not test performance.
    best_model_name = max(
        validation_results,
        key=lambda name: validation_results[name]["F1-score"],
    )
    best_model = trained_models[best_model_name]
    best_threshold = validation_results[best_model_name]["threshold"]
    best_test_result = test_results[best_model_name]
    best_test_metrics = best_test_result["metrics"]

    print("\n" + "=" * 60)
    print("SELECTED MODEL")
    print("=" * 60)
    print(f"Model: {best_model_name}")
    print(f"Validation threshold: {best_threshold:.2f}")

    print("\n--- Final Held-out Test Metrics ---")
    for metric, score in best_test_metrics.items():
        print(f"{metric:<12}: {score:.4f}")

    print("\n--- Final Classification Report ---")
    print(
        classification_report(
            y_test,
            best_test_result["predictions"],
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    # Save metrics in the existing Metric/Score format expected by app.py.
    pd.DataFrame(
        [
            {"Metric": metric, "Score": score}
            for metric, score in best_test_metrics.items()
        ]
    ).to_csv(REPORTS_PATH / "model_metrics.csv", index=False)

    # Save diagnostics for both models, including numeric confusion matrices.
    evaluation_summaries = []
    for model_name, result in test_results.items():
        counts = save_evaluation_artifacts(
            model_name=model_name,
            y_true=y_test,
            y_pred=result["predictions"],
            y_probability=result["probabilities"],
            threshold=result["threshold"],
            metrics=result["metrics"],
        )
        evaluation_summaries.append(
            {
                "Model": model_name,
                "Threshold": result["threshold"],
                **counts,
                **result["metrics"],
            }
        )

    pd.DataFrame(evaluation_summaries).to_csv(
        REPORTS_PATH / "model_evaluation_summary.csv",
        index=False,
    )

    # Keep the selected model's confusion counts easy for the dashboard to load.
    selected_counts = next(
        row for row in evaluation_summaries
        if row["Model"] == best_model_name
    )
    pd.DataFrame([selected_counts]).to_csv(
        REPORTS_PATH / "selected_model_evaluation.csv",
        index=False,
    )

    # Save logistic-regression coefficient analysis for interpretation.
    # This is explanatory analysis, not causal evidence.
    save_logistic_regression_importance(
        trained_models["Logistic Regression"],
        FEATURES_PATH,
    )

    # Save selected model and its metadata for scoring and dashboard use.
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_names": X.columns.tolist(),
            "target_name": "Churn",
            "threshold": best_threshold,
            "validation_f1": validation_results[best_model_name]["F1-score"],
            "test_metrics": best_test_metrics,
        },
        MODEL_PATH,
    )

    # Persist a small metadata file documenting split sizes and selection rule.
    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "Validation F1-score",
        "validation_threshold": best_threshold,
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_validation)),
        "test_records": int(len(X_test)),
        "input_feature_count": int(X.shape[1]),
        "random_state": 42,
    }
    with (REPORTS_PATH / "model_training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n--- Saved Files ---")
    print(f"Selected model: {MODEL_PATH}")
    print(f"Model comparison: {REPORTS_PATH / 'model_comparison.csv'}")
    print(f"Threshold comparison: {REPORTS_PATH / 'threshold_comparison.csv'}")
    print(f"Metrics: {REPORTS_PATH / 'model_metrics.csv'}")
    print(f"Evaluation summaries: {REPORTS_PATH / 'model_evaluation_summary.csv'}")
    print(f"Diagnostic charts: {FIGURES_PATH}")

    return best_model, best_test_metrics


if __name__ == "__main__":
    # The project entry point normally calls train_and_evaluate_model(model_df).
    # Run training through `python main.py` so the normal data preparation
    # and feature-audit workflow executes first.
    print("Run `python main.py` from the project root to train the model.")

