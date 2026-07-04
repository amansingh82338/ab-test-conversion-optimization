"""
statistics.py
=============

Reusable statistical-testing utilities for the A/B Testing Analysis
project.

These functions support ``notebooks/03_statistical_testing.ipynb`` and
can also be imported by the Streamlit dashboard to recompute test
results on demand. Every function returns a structured output
(dict or DataFrame) rather than printing, so results can be reused
or displayed however the caller needs.

Dependencies: numpy, pandas, scipy, statsmodels
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


def calculate_conversion_rate(conversions: int, total: int) -> float:
    """
    Calculate a simple conversion rate.

    Parameters
    ----------
    conversions : int
        Number of successful conversions.
    total : int
        Total number of observations (users/sessions).

    Returns
    -------
    float
        Conversion rate as a proportion (0-1). Returns 0.0 if
        total is 0 to avoid division errors.
    """
    if total == 0:
        return 0.0
    return conversions / total


def calculate_conversion_lift(cr_treatment: float, cr_control: float) -> dict:
    """
    Calculate the relative and absolute lift of the treatment group
    over the control group.

    Parameters
    ----------
    cr_treatment : float
        Conversion rate of the treatment group.
    cr_control : float
        Conversion rate of the control group.

    Returns
    -------
    dict
        Dictionary with keys: absolute_lift, relative_lift.
        relative_lift is None if cr_control is 0.
    """
    absolute_lift = cr_treatment - cr_control
    relative_lift = (absolute_lift / cr_control) if cr_control != 0 else None

    return {
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
    }


def two_proportion_z_test(
    conv_treatment: int,
    n_treatment: int,
    conv_control: int,
    n_control: int,
    alternative: str = "larger",
) -> dict:
    """
    Perform a two-proportion Z-test comparing treatment vs control
    conversion rates.

    Parameters
    ----------
    conv_treatment : int
        Number of conversions in the treatment group.
    n_treatment : int
        Total number of observations in the treatment group.
    conv_control : int
        Number of conversions in the control group.
    n_control : int
        Total number of observations in the control group.
    alternative : str, default "larger"
        Alternative hypothesis: "two-sided", "larger", or "smaller",
        relative to the treatment group (order: [treatment, control]).

    Returns
    -------
    dict
        Dictionary with keys: z_statistic, p_value.
    """
    successes = np.array([conv_treatment, conv_control])
    nobs = np.array([n_treatment, n_control])

    z_stat, p_value = proportions_ztest(count=successes, nobs=nobs, alternative=alternative)

    return {"z_statistic": float(z_stat), "p_value": float(p_value)}


def chi_square_test(contingency_table: pd.DataFrame) -> dict:
    """
    Perform a Chi-Square test of independence on a contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        A contingency table, e.g. pd.crosstab(df['group'], df['converted']).

    Returns
    -------
    dict
        Dictionary with keys: chi2_statistic, p_value, degrees_of_freedom,
        expected_frequencies (as a DataFrame matching the input shape).
    """
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    expected_df = pd.DataFrame(
        expected, index=contingency_table.index, columns=contingency_table.columns
    )

    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "expected_frequencies": expected_df,
    }


def independent_t_test(
    sample_treatment: pd.Series, sample_control: pd.Series, equal_var: bool = False
) -> dict:
    """
    Perform an independent-samples T-test between two numeric samples
    (e.g. purchase_amount or session_duration for treatment vs control).

    Parameters
    ----------
    sample_treatment : pd.Series
        Numeric values for the treatment group.
    sample_control : pd.Series
        Numeric values for the control group.
    equal_var : bool, default False
        If False, performs Welch's T-test (unequal variances assumed).

    Returns
    -------
    dict
        Dictionary with keys: t_statistic, p_value.
    """
    t_stat, p_value = stats.ttest_ind(sample_treatment, sample_control, equal_var=equal_var)
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}


def confidence_interval_proportion(
    conversions: int, total: int, alpha: float = 0.05, method: str = "normal"
) -> Tuple[float, float]:
    """
    Calculate a confidence interval for a single proportion.

    Parameters
    ----------
    conversions : int
        Number of successful conversions.
    total : int
        Total number of observations.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for a 95% confidence interval).
    method : str, default "normal"
        Method used by statsmodels' `proportion_confint`.

    Returns
    -------
    tuple of float
        (lower_bound, upper_bound) of the confidence interval.
    """
    lower, upper = proportion_confint(conversions, total, alpha=alpha, method=method)
    return float(lower), float(upper)


def confidence_interval_difference(
    cr_treatment: float,
    n_treatment: int,
    cr_control: float,
    n_control: int,
    alpha: float = 0.05,
) -> dict:
    """
    Calculate a confidence interval for the difference between two
    proportions (treatment - control) using the normal approximation.

    Parameters
    ----------
    cr_treatment : float
        Conversion rate of the treatment group.
    n_treatment : int
        Sample size of the treatment group.
    cr_control : float
        Conversion rate of the control group.
    n_control : int
        Sample size of the control group.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for a 95% confidence interval).

    Returns
    -------
    dict
        Dictionary with keys: difference, lower_bound, upper_bound,
        margin_of_error.
    """
    diff = cr_treatment - cr_control
    se_diff = np.sqrt(
        (cr_treatment * (1 - cr_treatment) / n_treatment)
        + (cr_control * (1 - cr_control) / n_control)
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    margin_error = z_crit * se_diff

    return {
        "difference": diff,
        "lower_bound": diff - margin_error,
        "upper_bound": diff + margin_error,
        "margin_of_error": margin_error,
    }


def cohens_h(cr_treatment: float, cr_control: float) -> float:
    """
    Calculate Cohen's h effect size for the difference between two
    proportions.

    Parameters
    ----------
    cr_treatment : float
        Conversion rate of the treatment group.
    cr_control : float
        Conversion rate of the control group.

    Returns
    -------
    float
        Cohen's h effect size.
    """
    return 2 * np.arcsin(np.sqrt(cr_treatment)) - 2 * np.arcsin(np.sqrt(cr_control))


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Provide a plain-language interpretation of a p-value.

    Parameters
    ----------
    p_value : float
        The p-value to interpret.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    str
        Human-readable interpretation of the p-value.
    """
    if p_value <= alpha:
        return (
            f"The p-value ({p_value:.4g}) is less than or equal to the significance "
            f"level ({alpha}), indicating the result is statistically significant."
        )
    return (
        f"The p-value ({p_value:.4g}) is greater than the significance "
        f"level ({alpha}), indicating the result is not statistically significant."
    )


def interpret_significance(p_value: float, alpha: float = 0.05) -> bool:
    """
    Determine whether a result is statistically significant.

    Parameters
    ----------
    p_value : float
        The p-value from a statistical test.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    bool
        True if statistically significant (p_value <= alpha), else False.
    """
    return p_value <= alpha


def assess_practical_significance(
    absolute_lift: float, minimum_meaningful_effect: float
) -> dict:
    """
    Assess whether a statistically significant result is also
    practically significant (i.e. large enough to matter for the
    business).

    Parameters
    ----------
    absolute_lift : float
        Observed absolute difference between treatment and control
        (e.g. difference in conversion rate).
    minimum_meaningful_effect : float
        The smallest effect size considered meaningful for the
        business (a threshold set by stakeholders).

    Returns
    -------
    dict
        Dictionary with keys: absolute_lift, minimum_meaningful_effect,
        is_practically_significant.
    """
    return {
        "absolute_lift": absolute_lift,
        "minimum_meaningful_effect": minimum_meaningful_effect,
        "is_practically_significant": abs(absolute_lift) >= minimum_meaningful_effect,
    }


def generate_contingency_table(
    df: pd.DataFrame, row_column: str, column_column: str
) -> pd.DataFrame:
    """
    Generate a contingency table (cross-tabulation) between two
    categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    row_column : str
        Column to use for the table rows.
    column_column : str
        Column to use for the table columns.

    Returns
    -------
    pd.DataFrame
        Contingency table of counts.
    """
    return pd.crosstab(df[row_column], df[column_column])
