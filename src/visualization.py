"""
visualization.py
=================

Reusable matplotlib/seaborn plotting utilities for the A/B Testing
Analysis project.

Generic chart-building functions (bar_chart, count_plot, histogram,
box_plot, violin_plot, pie_chart, heatmap, line_chart, scatter_plot)
live at the top of the module. Domain-specific comparison functions
(conversion rate, revenue, session duration, device/gender/age-group
comparisons, etc.) are thin wrappers around those generic functions so
plotting logic is defined only once.

Every plotting function:
    * accepts a DataFrame as input,
    * accepts a title, axis labels, and figure size,
    * returns the matplotlib Figure object so callers (notebooks or
      the Streamlit app) can further customize, save, or display it.

Dependencies: matplotlib, seaborn, pandas
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

DEFAULT_FIGSIZE: Tuple[int, int] = (8, 5)


def _finalize_axes(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Apply common title/axis-label formatting to an Axes object.

    Parameters
    ----------
    ax : plt.Axes
        The Axes to format.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    """
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    estimator: str = "mean",
) -> Figure:
    """
    Create a bar chart, optionally split by a hue column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column for the x-axis (categorical).
    y : str
        Column for the y-axis (numeric).
    hue : str, optional
        Column used to further split bars into sub-groups.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.
    estimator : str, default "mean"
        Aggregation function seaborn uses for each bar ("mean", "sum", etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y=y, hue=hue, estimator=estimator, errorbar=None, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def count_plot(
    df: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a count plot showing the frequency of categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Categorical column to count.
    hue : str, optional
        Column used to further split counts into sub-groups.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    order = df[x].value_counts().index
    sns.countplot(data=df, x=x, hue=hue, order=order, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel or "Count")
    fig.tight_layout()
    return fig


def histogram(
    df: pd.DataFrame,
    column: str,
    hue: Optional[str] = None,
    bins: int = 30,
    kde: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a histogram (optionally overlaid with a KDE curve).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Numeric column to plot.
    hue : str, optional
        Column used to split the histogram into overlapping groups.
    bins : int, default 30
        Number of histogram bins.
    kde : bool, default True
        Whether to overlay a kernel density estimate.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde, ax=ax)
    _finalize_axes(ax, title, xlabel or column, ylabel or "Frequency")
    fig.tight_layout()
    return fig


def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a box plot to compare the distribution of a numeric
    column across categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Categorical column for the x-axis.
    y : str
        Numeric column for the y-axis.
    hue : str, optional
        Column used to further split boxes into sub-groups.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def violin_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    split: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a violin plot to compare distributions across categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Categorical column for the x-axis.
    y : str
        Numeric column for the y-axis.
    hue : str, optional
        Column used to further split violins into sub-groups.
    split : bool, default False
        Whether to split each violin by hue (requires exactly 2 hue levels).
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=x, y=y, hue=hue, split=split, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def pie_chart(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    autopct: str = "%1.1f%%",
) -> Figure:
    """
    Create a pie chart showing the share of each category in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Categorical column to summarize.
    title : str, optional
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.
    autopct : str, default "%1.1f%%"
        Format string for the percentage labels on each wedge.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(counts.values, labels=counts.index, autopct=autopct, startangle=90)
    ax.axis("equal")
    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def heatmap(
    data: pd.DataFrame,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    cmap: str = "YlGnBu",
    annot: bool = True,
    fmt: str = ".2f",
) -> Figure:
    """
    Create a heatmap from a 2-D DataFrame (e.g. a pivot table).

    Parameters
    ----------
    data : pd.DataFrame
        2-D numeric data to visualize (e.g. from a pivot table).
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.
    cmap : str, default "YlGnBu"
        Color map used for the heatmap.
    annot : bool, default True
        Whether to annotate each cell with its value.
    fmt : str, default ".2f"
        String format for cell annotations.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=annot, cmap=cmap, fmt=fmt, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = "Correlation Heatmap",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    cmap: str = "coolwarm",
) -> Figure:
    """
    Create a correlation heatmap for the numeric columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    title : str, optional
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.
    cmap : str, default "coolwarm"
        Color map used for the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    corr = df.select_dtypes(include="number").corr()
    return heatmap(corr, title=title, figsize=figsize, cmap=cmap, annot=True, fmt=".2f")


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a line chart, typically for time-series or trend data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column for the x-axis.
    y : str
        Column for the y-axis.
    hue : str, optional
        Column used to draw multiple lines.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Create a scatter plot to explore the relationship between two
    numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column for the x-axis.
    y : str
        Column for the y-axis.
    hue : str, optional
        Column used to color points by group.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    _finalize_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Domain-specific comparison plots (thin wrappers around the generic charts
# above, pre-configured with the column names used throughout the project).
# ---------------------------------------------------------------------------


def revenue_comparison(
    df: pd.DataFrame,
    group_column: str = "group",
    revenue_column: str = "purchase_amount",
    title: str = "Average Revenue Per User by Group",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare average revenue (purchase amount) between groups using a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_column : str, default "group"
        Column identifying the experiment group.
    revenue_column : str, default "purchase_amount"
        Column containing purchase/revenue values.
    title : str, default "Average Revenue Per User by Group"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=group_column, y=revenue_column, title=title,
        xlabel="Group", ylabel="Average Purchase Amount ($)", figsize=figsize,
    )


def conversion_rate_comparison(
    df: pd.DataFrame,
    group_column: str = "group",
    conversion_column: str = "converted",
    title: str = "Conversion Rate by Group",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare conversion rates between groups using a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_column : str, default "group"
        Column identifying the experiment group.
    conversion_column : str, default "converted"
        Binary conversion indicator column.
    title : str, default "Conversion Rate by Group"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=group_column, y=conversion_column, title=title,
        xlabel="Group", ylabel="Conversion Rate", figsize=figsize,
    )


def session_duration_comparison(
    df: pd.DataFrame,
    group_column: str = "group",
    duration_column: str = "session_duration",
    title: str = "Session Duration Distribution by Group",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare session duration distributions between groups using a box plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_column : str, default "group"
        Column identifying the experiment group.
    duration_column : str, default "session_duration"
        Column containing session duration values.
    title : str, default "Session Duration Distribution by Group"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return box_plot(
        df, x=group_column, y=duration_column, title=title,
        xlabel="Group", ylabel="Session Duration (Minutes)", figsize=figsize,
    )


def purchase_amount_comparison(
    df: pd.DataFrame,
    group_column: str = "group",
    purchase_column: str = "purchase_amount",
    title: str = "Purchase Amount Distribution by Group",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare purchase amount distributions between groups using a box plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_column : str, default "group"
        Column identifying the experiment group.
    purchase_column : str, default "purchase_amount"
        Column containing purchase values.
    title : str, default "Purchase Amount Distribution by Group"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return box_plot(
        df, x=group_column, y=purchase_column, title=title,
        xlabel="Group", ylabel="Purchase Amount ($)", figsize=figsize,
    )


def device_wise_comparison(
    df: pd.DataFrame,
    device_column: str = "device_type",
    metric_column: str = "converted",
    group_column: str = "group",
    title: str = "Conversion Rate by Device Type",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare a metric (default: conversion rate) across device types,
    split by experiment group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    device_column : str, default "device_type"
        Column identifying the device type.
    metric_column : str, default "converted"
        Numeric metric to compare.
    group_column : str, default "group"
        Column identifying the experiment group (used as hue).
    title : str, default "Conversion Rate by Device Type"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=device_column, y=metric_column, hue=group_column, title=title,
        xlabel="Device Type", ylabel=metric_column.replace("_", " ").title(), figsize=figsize,
    )


def gender_wise_comparison(
    df: pd.DataFrame,
    gender_column: str = "gender",
    metric_column: str = "converted",
    group_column: str = "group",
    title: str = "Conversion Rate by Gender",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare a metric (default: conversion rate) across genders, split
    by experiment group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    gender_column : str, default "gender"
        Column identifying gender.
    metric_column : str, default "converted"
        Numeric metric to compare.
    group_column : str, default "group"
        Column identifying the experiment group (used as hue).
    title : str, default "Conversion Rate by Gender"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=gender_column, y=metric_column, hue=group_column, title=title,
        xlabel="Gender", ylabel=metric_column.replace("_", " ").title(), figsize=figsize,
    )


def age_group_comparison(
    df: pd.DataFrame,
    age_group_column: str = "age_group",
    metric_column: str = "converted",
    group_column: str = "group",
    title: str = "Conversion Rate by Age Group",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare a metric (default: conversion rate) across age groups,
    split by experiment group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (expects an age_group column, e.g. created via
        eda.create_age_groups).
    age_group_column : str, default "age_group"
        Column identifying the age group bucket.
    metric_column : str, default "converted"
        Numeric metric to compare.
    group_column : str, default "group"
        Column identifying the experiment group (used as hue).
    title : str, default "Conversion Rate by Age Group"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=age_group_column, y=metric_column, hue=group_column, title=title,
        xlabel="Age Group", ylabel=metric_column.replace("_", " ").title(), figsize=figsize,
    )


def landing_page_comparison_plot(
    df: pd.DataFrame,
    page_column: str = "landing_page",
    metric_column: str = "converted",
    title: str = "Conversion Rate by Landing Page",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Compare a metric (default: conversion rate) across landing page variants.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    page_column : str, default "landing_page"
        Column identifying the landing page variant.
    metric_column : str, default "converted"
        Numeric metric to compare.
    title : str, default "Conversion Rate by Landing Page"
        Plot title.
    figsize : tuple of int, default (8, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    return bar_chart(
        df, x=page_column, y=metric_column, title=title,
        xlabel="Landing Page", ylabel=metric_column.replace("_", " ").title(), figsize=figsize,
    )
