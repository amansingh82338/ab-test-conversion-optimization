"""
utils.py
========

General-purpose helper utilities shared across the A/B Testing
Analysis notebooks and the Streamlit dashboard: number/percentage
formatting, safe arithmetic, display configuration, logging, and
figure-saving helpers.

Dependencies: standard library, pandas, matplotlib (optional, only
for save_figure), numpy (optional, only for set_random_seed)
"""

import logging
import random
from typing import Optional

import pandas as pd


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a numeric value (0-1 or already a percentage) as a
    percentage string.

    Parameters
    ----------
    value : float
        Value to format. If between -1 and 1, it is treated as a
        proportion and multiplied by 100.
    decimals : int, default 2
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted percentage string, e.g. "14.92%".
    """
    display_value = value * 100 if -1 <= value <= 1 else value
    return f"{display_value:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """
    Format a numeric value as a currency string.

    Parameters
    ----------
    value : float
        Value to format.
    symbol : str, default "$"
        Currency symbol to prepend.
    decimals : int, default 2
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted currency string, e.g. "$5.61".
    """
    return f"{symbol}{value:,.{decimals}f}"


def format_number(value: float, decimals: int = 0) -> str:
    """
    Format a numeric value with thousands separators.

    Parameters
    ----------
    value : float
        Value to format.
    decimals : int, default 0
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted number string, e.g. "294,478".
    """
    return f"{value:,.{decimals}f}"


def format_conversion_rate(conversions: int, total: int, decimals: int = 2) -> str:
    """
    Format a conversion rate as a percentage string given raw counts.

    Parameters
    ----------
    conversions : int
        Number of conversions.
    total : int
        Total number of observations.
    decimals : int, default 2
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted conversion rate string, e.g. "17.95%".
    """
    rate = safe_divide(conversions, total)
    return format_percentage(rate, decimals=decimals)


def format_lift(lift: float, decimals: int = 2) -> str:
    """
    Format a lift value (relative or absolute) as a signed percentage
    string.

    Parameters
    ----------
    lift : float
        Lift value, typically a proportion (e.g. 0.5118 for +51.18%).
    decimals : int, default 2
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted, signed lift string, e.g. "+51.18%".
    """
    sign = "+" if lift >= 0 else ""
    return f"{sign}{format_percentage(lift, decimals=decimals)}"


def format_confidence_interval(lower: float, upper: float, decimals: int = 2, as_percentage: bool = True) -> str:
    """
    Format a confidence interval as a readable string.

    Parameters
    ----------
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.
    decimals : int, default 2
        Number of decimal places to display.
    as_percentage : bool, default True
        Whether to format the bounds as percentages.

    Returns
    -------
    str
        Formatted confidence interval, e.g. "[12.30%, 18.50%]".
    """
    if as_percentage:
        lower_str = format_percentage(lower, decimals=decimals)
        upper_str = format_percentage(upper, decimals=decimals)
    else:
        lower_str = f"{lower:.{decimals}f}"
        upper_str = f"{upper:.{decimals}f}"
    return f"[{lower_str}, {upper_str}]"


def pretty_print(title: str, data: dict) -> None:
    """
    Pretty-print a dictionary of results under a titled header, for
    quick readability in notebook output.

    Parameters
    ----------
    title : str
        Section header to print above the data.
    data : dict
        Dictionary of results to display (key: value per line).
    """
    print(f"=== {title} ===")
    for key, value in data.items():
        print(f"{key}: {value}")


def preview_dataframe(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Return a quick preview of a DataFrame (head rows) for display.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_rows : int, default 5
        Number of rows to preview.

    Returns
    -------
    pd.DataFrame
        The first `n_rows` rows of the DataFrame.
    """
    return df.head(n_rows)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value instead of
    raising an error when the denominator is zero.

    Parameters
    ----------
    numerator : float
        The numerator.
    denominator : float
        The denominator.
    default : float, default 0.0
        Value to return if denominator is zero.

    Returns
    -------
    float
        The division result, or `default` if denominator is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def check_columns_exist(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Check whether all required columns exist in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    required_columns : list
        List of column names that must be present.

    Returns
    -------
    bool
        True if all required columns are present, False otherwise.
    """
    return all(col in df.columns for col in required_columns)


def get_logger(name: str = "ab_testing_analysis", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger for consistent logging
    across notebooks and the Streamlit dashboard.

    Parameters
    ----------
    name : str, default "ab_testing_analysis"
        Name of the logger.
    level : int, default logging.INFO
        Logging level.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python's `random` module and NumPy (if
    installed) to ensure reproducible results.

    Parameters
    ----------
    seed : int, default 42
        Seed value to use.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def save_figure(fig, output_path: str, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """
    Save a matplotlib figure to disk with sensible defaults.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    output_path : str
        Destination file path (e.g. "reports/conversion_rate.png").
    dpi : int, default 300
        Resolution in dots per inch.
    bbox_inches : str, default "tight"
        Bounding box setting passed to `savefig`.
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


def configure_display(max_columns: Optional[int] = None, float_format: str = "{:.4f}") -> None:
    """
    Configure pandas display options for consistent notebook output.

    Parameters
    ----------
    max_columns : int, optional
        Maximum number of columns to display. None shows all columns.
    float_format : str, default "{:.4f}"
        Format string applied to floating point display.
    """
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.float_format", float_format.format)
