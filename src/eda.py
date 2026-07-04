"""
eda.py
======

Reusable exploratory-data-analysis (EDA) utilities for the
A/B Testing Analysis project.

These functions support ``notebooks/02_exploratory_data_analysis.ipynb``
and ``notebooks/04_customer_segmentation.ipynb`` and can also be
imported directly by the Streamlit dashboard.

Functions return DataFrames or scalar values rather than printing,
so calling code decides how results are displayed.

Dependencies: pandas, numpy
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a high-level overview of a dataset (dtype, non-null count,
    unique value count per column).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Overview table indexed by column name.
    """
    overview = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.notnull().sum(),
            "null_count": df.isnull().sum(),
            "unique_values": df.nunique(),
        }
    )
    return overview


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Table with columns: column, missing_count, missing_pct,
        sorted descending by missing_count.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else missing_count
    summary = pd.DataFrame(
        {"column": df.columns, "missing_count": missing_count.values, "missing_pct": missing_pct.values}
    )
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)


def numerical_summary(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str], optional
        Specific numeric columns to summarize. If None, all numeric
        columns are used.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics (count, mean, std, min, quartiles, max).
    """
    subset = df[list(columns)] if columns is not None else df.select_dtypes(include=np.number)
    return subset.describe().T


def categorical_summary(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Generate frequency counts for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str], optional
        Specific categorical columns to summarize. If None, all
        object/category dtype columns are used.

    Returns
    -------
    pd.DataFrame
        Table with columns: column, value, count, percentage.
    """
    cat_columns = list(columns) if columns is not None else df.select_dtypes(include=["object", "category"]).columns

    records = []
    for col in cat_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(normalize=True, dropna=False) * 100
        for value in counts.index:
            records.append(
                {
                    "column": col,
                    "value": value,
                    "count": counts[value],
                    "percentage": pct[value],
                }
            )
    return pd.DataFrame(records)


def calculate_conversion_rate(
    df: pd.DataFrame, group_column: Optional[str] = None, conversion_column: str = "converted"
) -> pd.Series:
    """
    Calculate conversion rate overall or grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, optional
        Column to group by (e.g. "group", "device_type"). If None,
        the overall conversion rate is returned as a one-element Series.
    conversion_column : str, default "converted"
        Binary column indicating conversion (1) or not (0).

    Returns
    -------
    pd.Series
        Conversion rate(s), indexed by group if group_column is given.
    """
    if group_column is None:
        return pd.Series({"overall": df[conversion_column].mean()})
    return df.groupby(group_column)[conversion_column].mean()


def groupwise_aggregation(
    df: pd.DataFrame,
    group_columns: Iterable[str],
    agg_map: dict,
) -> pd.DataFrame:
    """
    Perform a flexible group-wise aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_columns : Iterable[str]
        Column(s) to group by.
    agg_map : dict
        Aggregation mapping compatible with `DataFrame.agg`, e.g.
        {"converted": "mean", "purchase_amount": "mean"}.

    Returns
    -------
    pd.DataFrame
        Aggregated results with a reset index.
    """
    return df.groupby(list(group_columns)).agg(agg_map).reset_index()


def revenue_summary(
    df: pd.DataFrame, group_column: str = "group", revenue_column: str = "purchase_amount"
) -> pd.DataFrame:
    """
    Summarize revenue (purchase amount) statistics per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, default "group"
        Column identifying the experiment group.
    revenue_column : str, default "purchase_amount"
        Column containing purchase/revenue values.

    Returns
    -------
    pd.DataFrame
        Table with total_revenue, average_revenue (ARPU), and
        paying_user_count per group.
    """
    grouped = df.groupby(group_column)[revenue_column]
    summary = pd.DataFrame(
        {
            "total_revenue": grouped.sum(),
            "average_revenue": grouped.mean(),
            "paying_user_count": df[df[revenue_column] > 0].groupby(group_column)[revenue_column].count(),
        }
    ).fillna(0)
    return summary.reset_index()


def purchase_analysis(
    df: pd.DataFrame, group_column: str = "group", purchase_column: str = "purchase_amount"
) -> pd.DataFrame:
    """
    Analyze purchase behavior for paying customers only.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, default "group"
        Column identifying the experiment group.
    purchase_column : str, default "purchase_amount"
        Column containing purchase values.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics of purchase amounts among paying
        customers (purchase_amount > 0), grouped by group_column.
    """
    paying_users = df[df[purchase_column] > 0]
    return paying_users.groupby(group_column)[purchase_column].describe()


def session_duration_analysis(
    df: pd.DataFrame, group_column: str = "group", duration_column: str = "session_duration"
) -> pd.DataFrame:
    """
    Analyze session duration statistics by group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, default "group"
        Column identifying the experiment group.
    duration_column : str, default "session_duration"
        Column containing session duration values.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics of session duration, grouped by
        group_column.
    """
    return df.groupby(group_column)[duration_column].describe()


def landing_page_comparison(
    df: pd.DataFrame, page_column: str = "landing_page", conversion_column: str = "converted"
) -> pd.DataFrame:
    """
    Compare conversion performance across landing page variants.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    page_column : str, default "landing_page"
        Column identifying which landing page a user saw.
    conversion_column : str, default "converted"
        Binary conversion indicator column.

    Returns
    -------
    pd.DataFrame
        Table with users, conversions, and conversion_rate per
        landing page.
    """
    summary = df.groupby(page_column).agg(
        users=(conversion_column, "count"),
        conversions=(conversion_column, "sum"),
        conversion_rate=(conversion_column, "mean"),
    )
    return summary.reset_index()


def control_vs_treatment_comparison(
    df: pd.DataFrame,
    group_column: str = "group",
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compare control vs treatment groups across one or more metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, default "group"
        Column identifying the experiment group.
    metrics : Iterable[str], optional
        Numeric metric columns to compare (defaults to
        ["converted", "purchase_amount", "session_duration",
        "pages_visited"] where present).

    Returns
    -------
    pd.DataFrame
        Mean value of each metric, grouped by group_column.
    """
    if metrics is None:
        default_metrics = ["converted", "purchase_amount", "session_duration", "pages_visited"]
        metrics = [m for m in default_metrics if m in df.columns]

    return df.groupby(group_column)[list(metrics)].mean().reset_index()


def generate_experiment_summary(
    df: pd.DataFrame,
    group_column: str = "group",
    conversion_column: str = "converted",
    revenue_column: str = "purchase_amount",
    duration_column: str = "session_duration",
    pages_column: str = "pages_visited",
) -> pd.DataFrame:
    """
    Build the overall experiment summary table (per group plus an
    "overall" row), matching the structure exported as
    ``data/processed/experiment_summary.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment DataFrame.
    group_column : str, default "group"
        Column identifying the experiment group.
    conversion_column : str, default "converted"
        Binary conversion indicator column.
    revenue_column : str, default "purchase_amount"
        Column containing purchase/revenue values.
    duration_column : str, default "session_duration"
        Column containing session duration values.
    pages_column : str, default "pages_visited"
        Column containing number of pages visited.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: group, users, conversion_rate_pct,
        arpu, avg_session_duration, avg_pages_visited, including an
        "overall" row aggregating the full dataset.
    """
    group_cr = df.groupby(group_column)[conversion_column].mean() * 100
    group_counts = df.groupby(group_column).size()
    arpu = df.groupby(group_column)[revenue_column].mean()
    avg_session = df.groupby(group_column)[duration_column].mean()
    avg_pages = df.groupby(group_column)[pages_column].mean()

    summary = pd.DataFrame(
        {
            "group": group_cr.index,
            "users": group_counts.reindex(group_cr.index).values,
            "conversion_rate_pct": group_cr.values,
            "arpu": arpu.reindex(group_cr.index).values,
            "avg_session_duration": avg_session.reindex(group_cr.index).values,
            "avg_pages_visited": avg_pages.reindex(group_cr.index).values,
        }
    )

    overall_row = {
        "group": "overall",
        "users": len(df),
        "conversion_rate_pct": df[conversion_column].mean() * 100,
        "arpu": df[revenue_column].mean(),
        "avg_session_duration": df[duration_column].mean(),
        "avg_pages_visited": df[pages_column].mean(),
    }
    summary.loc[len(summary)] = overall_row
    return summary


def generate_segment_summary(
    df: pd.DataFrame,
    segment_columns: Iterable[str],
    group_column: str = "group",
    id_column: str = "user_id",
    conversion_column: str = "converted",
    revenue_column: str = "purchase_amount",
    duration_column: str = "session_duration",
    pages_column: str = "pages_visited",
) -> pd.DataFrame:
    """
    Build a stacked segment-level summary across multiple categorical
    columns (e.g. device_type, gender, location), matching the
    structure exported as ``data/processed/segment_summary.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment DataFrame.
    segment_columns : Iterable[str]
        Categorical columns to segment by (each processed separately
        and stacked into one long-format table).
    group_column : str, default "group"
        Column identifying the experiment group.
    id_column : str, default "user_id"
        Column used to count users.
    conversion_column : str, default "converted"
        Binary conversion indicator column.
    revenue_column : str, default "purchase_amount"
        Column containing purchase/revenue values.
    duration_column : str, default "session_duration"
        Column containing session duration values.
    pages_column : str, default "pages_visited"
        Column containing number of pages visited.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: segment_type, segment_value,
        group, users, conversion_rate_pct, arpu, avg_session_duration,
        avg_pages_visited.
    """
    segment_frames: List[pd.DataFrame] = []

    for seg_col in segment_columns:
        if seg_col not in df.columns:
            continue

        seg = df.groupby([seg_col, group_column]).agg(
            users=(id_column, "count"),
            conversion_rate_pct=(conversion_column, lambda x: x.mean() * 100),
            arpu=(revenue_column, "mean"),
            avg_session_duration=(duration_column, "mean"),
            avg_pages_visited=(pages_column, "mean"),
        ).reset_index()

        seg.insert(0, "segment_type", seg_col)
        seg = seg.rename(columns={seg_col: "segment_value"})
        segment_frames.append(seg)

    if not segment_frames:
        return pd.DataFrame()

    return pd.concat(segment_frames, ignore_index=True)


def correlation_matrix(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str], optional
        Specific numeric columns to include. If None, all numeric
        columns are used.
    method : str, default "pearson"
        Correlation method: "pearson", "kendall", or "spearman".

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    subset = df[list(columns)] if columns is not None else df.select_dtypes(include=np.number)
    return subset.corr(method=method)


def generate_pivot_table(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """
    Generate a pivot table for cross-tabular analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    index : str
        Column to use as the pivot table row index.
    columns : str
        Column to use as the pivot table columns.
    values : str
        Column whose values are aggregated.
    aggfunc : str, default "mean"
        Aggregation function to apply.

    Returns
    -------
    pd.DataFrame
        The resulting pivot table.
    """
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)


def create_age_groups(
    df: pd.DataFrame,
    age_column: str = "age",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    new_column: str = "age_group",
) -> pd.DataFrame:
    """
    Bucket a numeric age column into descriptive age groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    age_column : str, default "age"
        Column containing numeric ages.
    bins : list of int, optional
        Bin edges (defaults to [0, 24, 34, 44, 54, 100]).
    labels : list of str, optional
        Labels for each bin (defaults to
        ["18-24", "25-34", "35-44", "45-54", "55+"]).
    new_column : str, default "age_group"
        Name of the resulting categorical column.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the new age group column added.
    """
    result = df.copy()
    if age_column not in result.columns:
        return result

    default_bins = [0, 24, 34, 44, 54, 100]
    default_labels = ["18-24", "25-34", "35-44", "45-54", "55+"]

    result[new_column] = pd.cut(
        result[age_column],
        bins=bins if bins is not None else default_bins,
        labels=labels if labels is not None else default_labels,
        right=False,
    )
    return result
