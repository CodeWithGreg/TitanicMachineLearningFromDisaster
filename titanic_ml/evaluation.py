"""Evaluation module for computing metrics, cross-validation, and comparison artifacts.

Provides functions for model evaluation including accuracy, precision, recall, F1,
stratified k-fold cross-validation, comparison tables, and visualization plots.
"""

from __future__ import annotations

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from titanic_ml.models import ModelRegistry


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with accuracy, precision, recall, f1.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def cross_validate_model(
    model: ModelRegistry,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """Perform stratified k-fold cross-validation and return summary statistics.

    Parameters
    ----------
    model : ModelRegistry
        A ``ModelRegistry`` instance whose underlying estimator will be used.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    cv_folds : int
        Number of stratified folds (default 5).
    random_seed : int
        Random seed for reproducible fold splits.

    Returns
    -------
    dict
        Keys: ``mean_accuracy``, ``std_accuracy``, ``fold_scores``.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_scores = cross_val_score(model.model, X, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_accuracy": float(np.std(fold_scores)),
        "fold_scores": fold_scores.tolist(),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame from per-model results.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of model name to a dict containing metric keys such as
        ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``cv_mean_accuracy``, and ``cv_std_accuracy``.

    Returns
    -------
    pd.DataFrame
        One row per model, columns for each metric.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({"model": model_name, **metrics})

    df = pd.DataFrame(rows).set_index("model")

    # Ensure expected columns are present (fill missing with NaN)
    expected_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cv_mean_accuracy",
        "cv_std_accuracy",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def plot_cv_comparison(results: dict[str, dict]) -> matplotlib.figure.Figure:
    """Bar chart of cross-validation accuracy per model.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of model name to a dict that must contain ``cv_mean_accuracy``
        and ``cv_std_accuracy`` keys.

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart figure with error bars.
    """
    model_names = list(results.keys())
    means = [results[m]["cv_mean_accuracy"] for m in model_names]
    stds = [results[m]["cv_std_accuracy"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(model_names))
    ax.bar(x_pos, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title("Model Comparison — Cross-Validation Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> matplotlib.figure.Figure:
    """Confusion matrix heatmap for a single model.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    model_name : str
        Name used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap figure of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    plt.close(fig)
    return fig
