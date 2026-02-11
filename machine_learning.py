"""
Breast Cancer Wisconsin Classification
Modular training + evaluation for 6 classifiers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# Constants ─────
MODEL_DIR = "model"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Models that require StandardScaler
SCALE_REQUIRED = {"Logistic Regression", "kNN", "XGBoost"}

# Model definitions 
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False),
    }


# Data loading 
def load_default_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name="target")
    return X, y, list(bc.feature_names)


def load_csv_dataset(filepath: str, target_col: str = "target") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(filepath)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Columns present: {list(df.columns)}"
        )
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    bc = load_breast_cancer()
    expected = list(bc.feature_names)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing feature columns: {missing}")

    X = X[expected]
    return X, y, expected


# Pre-processing ─
def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple:
    """
    Returns:
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Single model evaluation 
def evaluate_model(
    name: str,
    model,
    X_train, X_test,
    X_train_scaled, X_test_scaled,
    y_train, y_test,
) -> dict:
    """Train a single model and return its metrics + artefacts."""
    if name in SCALE_REQUIRED:
        model.fit(X_train_scaled, y_train)
        y_pred       = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Malignant", "Benign"],
                                   output_dict=True)

    return {
        "model":    model,
        "name":     name,
        "metrics": {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "AUC":       round(roc_auc_score(y_test, y_pred_proba), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
        },
        "confusion_matrix":       cm,
        "classification_report":  report,
        "y_pred":                 y_pred,
        "y_pred_proba":           y_pred_proba,
        "y_test":                 y_test,
    }


# Train all 6 models 
def train_all_models(
    X_train, X_test,
    X_train_scaled, X_test_scaled,
    y_train, y_test,
) -> dict:
    """Train and evaluate all 6 classifiers. Returns dict keyed by model name."""
    results = {}
    for name, model in get_models().items():
        results[name] = evaluate_model(
            name, model,
            X_train, X_test,
            X_train_scaled, X_test_scaled,
            y_train, y_test,
        )
    return results


# Save artefacts 
def save_artefacts(results: dict, scaler: StandardScaler) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    for name, res in results.items():
        fname = name.replace(" ", "_").lower() + "_model.pkl"
        joblib.dump(res["model"], os.path.join(MODEL_DIR, fname))
    rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
    pd.DataFrame(rows).to_csv("model_comparison_results.csv", index=False)


# Export test data to CSV 
def export_test_data(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "test_data.csv",
) -> str:
    """
    Saves the held-out test split (features + target) to a CSV file.

    The resulting file can be uploaded directly to the Streamlit app via the
    CSV uploader to run predictions against trained models.

    Columns: all 30 feature columns  +  'target'  (0 = Malignant, 1 = Benign)

    Parameters
    ----------
    X_test      : feature DataFrame (raw, unscaled)
    y_test      : target Series
    output_path : destination file path  (default: test_data.csv)

    Returns
    -------
    Absolute path of the written file.
    """
    test_df = X_test.copy().reset_index(drop=True)
    test_df["target"] = y_test.values

    test_df.to_csv(output_path, index=False)

    abs_path = os.path.abspath(output_path)
    print(f"  Test data  : {len(test_df)} rows × {test_df.shape[1]} columns")
    print(f"  Saved to   : {abs_path}")
    print(f"  Classes    : Malignant={int((test_df['target']==0).sum())}  "
          f"Benign={int((test_df['target']==1).sum())}")
    return abs_path


# Confusion matrix figure 
def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGn",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        linewidths=0.5, linecolor="#E2DDD6",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted",  fontsize=8, labelpad=8, color="#7A766E")
    ax.set_ylabel("Actual",     fontsize=8, labelpad=8, color="#7A766E")
    ax.set_title(f"{model_name}", fontsize=10, pad=10, color="#1A1915")
    ax.tick_params(colors="#7A766E", labelsize=8)
    plt.tight_layout()
    return fig


# ROC curve figure 
def plot_roc_curves(results: dict) -> plt.Figure:
    from sklearn.metrics import roc_curve

    palette = ["#2E6B4F", "#C0392B", "#2980B9", "#8E44AD", "#E67E22", "#16A085"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_pred_proba"])
        auc = res["metrics"]["AUC"]
        ax.plot(fpr, tpr, color=palette[i % len(palette)],
                lw=1.8, label=f"{name}  ({auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=8, color="#7A766E")
    ax.set_ylabel("True Positive Rate",  fontsize=8, color="#7A766E")
    ax.set_title("ROC Curves — All Models", fontsize=10, color="#1A1915")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.tick_params(colors="#7A766E", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E2DDD6")
    plt.tight_layout()
    return fig


# CLI entry-point 
if __name__ == "__main__":
    print("Loading Breast Cancer Wisconsin dataset …")
    X, y, feature_names = load_default_dataset()
    print(f"  Shape  : {X.shape}")
    print(f"  Classes: {dict(y.value_counts())}")

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"  Train  : {len(X_train)}   Test: {len(X_test)}\n")

    results = train_all_models(X_train, X_test, X_train_sc, X_test_sc, y_train, y_test)

    rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
    print(pd.DataFrame(rows).to_string(index=False))

    save_artefacts(results, scaler)
    print("\nArtefacts saved to ./model/")
    print("model_comparison_results.csv written.")

    print("\nExporting test data …")
    export_test_data(X_test, y_test, output_path="test_data.csv")
    print("\nDone. Upload test_data.csv to the Streamlit app to run predictions.")
