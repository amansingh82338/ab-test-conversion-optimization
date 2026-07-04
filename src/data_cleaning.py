"""
data_cleaning.py
=================

Reusable data-cleaning utilities for the A/B Testing Analysis project.

These functions wrap the cleaning logic used in
``notebooks/01_data_cleaning.ipynb`` so the same operations can be
imported by other notebooks and the Streamlit dashboard without
re-writing or duplicating cleaning code.

Expected raw dataset columns (ab_testing.csv):
    user_id, timestamp, group, landing_page, converted, age,
    gender, device_type, location, session_duration,
    pages_visited, purchase_amount

Dependencies: pandas, numpy
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase, stripped, snake_case format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose columns should be standardized.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with cleaned column names.
    """
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-z_]", "", regex=True)
    )
    return cleaned


def convert_date_columns(df: pd.DataFrame, date_columns: Iterable[str]) -> pd.DataFrame:
    """
    Convert specified columns to pandas datetime dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_columns : Iterable[str]
        Names of columns to convert (e.g. ['timestamp']).

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the specified columns converted
        to datetime. Columns not present in the DataFrame are skipped.
    """
    converted = df.copy()
    for col in date_columns:
        if col in converted.columns:
            converted[col] = pd.to_datetime(converted[col], errors="coerce")
    return converted


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Optional[object] = None,
    subset: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str, default "drop"
        One of {"drop", "fill"}.
        - "drop": drop rows containing any missing values (in `subset`
          if provided, otherwise across all columns).
        - "fill": fill missing values with `fill_value`.
    fill_value : object, optional
        Value used to fill missing values when strategy="fill".
    subset : Iterable[str], optional
        Columns to consider when dropping rows with missing values.

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the DataFrame.
    """
    cleaned = df.copy()

    if strategy == "drop":
        cleaned = cleaned.dropna(subset=subset)
    elif strategy == "fill":
        cleaned = cleaned.fillna(fill_value)
    else:
        raise ValueError("strategy must be either 'drop' or 'fill'")

    return cleaned


def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Remove exact duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    subset : Iterable[str], optional
        Columns to consider when identifying duplicates.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates(subset=subset).copy()


def remove_multi_entry_users(df: pd.DataFrame, id_column: str = "user_id") -> pd.DataFrame:
    """
    Remove users who appear more than once in the dataset.

    This prevents cross-contamination between experiment groups when
    a user was accidentally recorded in both the control and
    treatment groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_column : str, default "user_id"
        Column that uniquely identifies a user.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with multi-entry users removed.
    """
    if id_column not in df.columns:
        return df.copy()

    id_counts = df[id_column].value_counts()
    multi_entry_ids = id_counts[id_counts > 1].index
    return df[~df[id_column].isin(multi_entry_ids)].copy()


def detect_incorrect_dtypes(df: pd.DataFrame, expected_dtypes: dict) -> pd.DataFrame:
    """
    Compare actual column dtypes against an expected dtype mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    expected_dtypes : dict
        Mapping of {column_name: expected_dtype_as_string}, e.g.
        {"age": "int", "converted": "int", "purchase_amount": "float"}.

    Returns
    -------
    pd.DataFrame
        A report DataFrame with columns: column, expected_dtype,
        actual_dtype, is_mismatched.
    """
    records: List[dict] = []
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        records.append(
            {
                "column": col,
                "expected_dtype": expected,
                "actual_dtype": actual,
                "is_mismatched": expected not in actual,
            }
        )
    return pd.DataFrame(records)


def convert_column_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """
    Cast specified columns to the given dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dtype_map : dict
        Mapping of {column_name: dtype}, e.g.
        {"age": int, "converted": int, "purchase_amount": float}.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with converted dtypes. Columns not
        present in the DataFrame are silently skipped.
    """
    converted = df.copy()
    for col, dtype in dtype_map.items():
        if col in converted.columns:
            converted[col] = converted[col].astype(dtype)
    return converted


def clean_categorical_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    lowercase: bool = True,
    strip_whitespace: bool = True,
) -> pd.DataFrame:
    """
    Normalize categorical text columns (e.g. group, gender, device_type).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str]
        Categorical column names to clean.
    lowercase : bool, default True
        Whether to lowercase the values.
    strip_whitespace : bool, default True
        Whether to strip leading/trailing whitespace.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with cleaned categorical columns.
    """
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        series = cleaned[col].astype(str)
        if strip_whitespace:
            series = series.str.strip()
        if lowercase:
            series = series.str.lower()
        cleaned[col] = series
    return cleaned


def clean_numeric_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> pd.DataFrame:
    """
    Clean numeric columns by coercing to numeric type and clipping
    values to a valid range (removing impossible values, e.g.
    negative session durations or negative ages).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str]
        Numeric column names to clean.
    lower_bound : float, optional
        Minimum acceptable value; values below this are clipped.
    upper_bound : float, optional
        Maximum acceptable value; values above this are clipped.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with cleaned numeric columns.
    """
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        if lower_bound is not None or upper_bound is not None:
            cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    return cleaned


def filter_valid_group_page_mapping(
    df: pd.DataFrame,
    group_column: str = "group",
    page_column: str = "landing_page",
    control_label: str = "control",
    treatment_label: str = "treatment",
    control_page: str = "old_page",
    treatment_page: str = "new_page",
) -> pd.DataFrame:
    """
    Keep only rows where the experiment group correctly matches the
    landing page shown to the user (control -> old_page,
    treatment -> new_page).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str, default "group"
        Column containing the experiment group label.
    page_column : str, default "landing_page"
        Column containing the landing page label.
    control_label, treatment_label : str
        Expected values in `group_column`.
    control_page, treatment_page : str
        Expected values in `page_column` for each group.

    Returns
    -------
    pd.DataFrame
        A filtered copy of the DataFrame containing only correctly
        mapped rows.
    """
    valid_mask = (
        (df[group_column] == control_label) & (df[page_column] == control_page)
    ) | (
        (df[group_column] == treatment_label) & (df[page_column] == treatment_page)
    )
    return df[valid_mask].copy()


def validate_dataset(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    id_column: str = "user_id",
) -> dict:
    """
    Run basic validation checks on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    required_columns : Iterable[str]
        Columns that must be present in the DataFrame.
    id_column : str, default "user_id"
        Column expected to contain unique identifiers.

    Returns
    -------
    dict
        Dictionary summarizing validation results:
        - missing_columns: list of required columns not present
        - has_duplicates: whether duplicate rows exist
        - has_null_values: whether any null values remain
        - is_id_unique: whether id_column values are unique
        - row_count: number of rows in the DataFrame
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    has_duplicates = bool(df.duplicated().any())
    has_null_values = bool(df.isnull().any().any())
    is_id_unique = (
        df[id_column].is_unique if id_column in df.columns else False
    )

    return {
        "missing_columns": missing_columns,
        "has_duplicates": has_duplicates,
        "has_null_values": has_null_values,
        "is_id_unique": is_id_unique,
        "row_count": df.shape[0],
    }


def export_cleaned_dataset(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
    """
    Export a cleaned DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    output_path : str
        Destination file path.
    index : bool, default False
        Whether to write the DataFrame index to the file.
    """
    df.to_csv(output_path, index=index)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Produce a compact summary of a dataset for quick inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Summary containing shape, dtypes, missing value counts,
        and number of duplicate rows.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
