"""Visualization module for exploratory data analysis.

Provides helper functions for generating survival rate plots, missing value
summaries, correlation heatmaps, and feature distribution plots.
"""

from __future__ import annotations

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_survival_rate(
    df: pd.DataFrame, feature: str, ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Bar chart of survival rate grouped by feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain a ``Survived`` column and the specified
        *feature* column.
    feature : str
        Column name to group by.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  When *None* a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart figure showing survival rate per group.
    """
    survival_rates = df.groupby(feature)["Survived"].mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    survival_rates.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Survival Rate")
    ax.set_title(f"Survival Rate by {feature}")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if standalone:
        fig.tight_layout()
        plt.close(fig)
    return fig


def get_percentage_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with count and percentage of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyse.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by column name with ``count`` and ``percentage``
        columns.  ``percentage`` is expressed as a value in [0, 100].
    """
    count = df.isna().sum()
    percentage = count / len(df) * 100 if len(df) > 0 else count * 0.0
    return pd.DataFrame({"count": count, "percentage": percentage})


def plot_correlation_heatmap(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Only numeric columns are included in the
        correlation matrix.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap figure of the correlation matrix.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_distribution(df: pd.DataFrame, feature: str) -> matplotlib.figure.Figure:
    """Histogram with KDE overlay for a numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the specified *feature* column.
    feature : str
        Numeric column name to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Histogram/KDE figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[feature].dropna(), kde=True, ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature}")
    fig.tight_layout()
    plt.close(fig)
    return fig
