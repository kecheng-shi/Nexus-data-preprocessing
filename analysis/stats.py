"""Statistical helpers for Nexus analysis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["pairwise_corr"]


def _ensure_min_periods(
    frame: pd.DataFrame,
    min_periods: int | None,
) -> tuple[pd.DataFrame, int]:
    """Return a numeric-only frame with columns meeting a minimum observation threshold."""
    if min_periods is None:
        min_periods = 1
    if min_periods < 1:
        raise ValueError("min_periods must be a positive integer.")

    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return numeric, min_periods

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().sum().loc[lambda s: s >= min_periods].index.tolist()
    return numeric[valid], min_periods


def pairwise_corr(frame: pd.DataFrame, min_periods: int | None = None) -> pd.DataFrame:
    """Compute a correlation matrix with robust NaN/Inf handling.

    Pandas' ``DataFrame.corr`` delegates to ``numpy.cov`` which emits a flood of
    ``RuntimeWarning`` messages whenever entire slices contain NaNs or infs. The
    Nexus notebooks work with sparse return panels, so we sanitise the data
    first and compute correlations pairwise with explicit NaN dropping.

    Parameters
    ----------
    frame:
        Input DataFrame; non-numeric columns are ignored.
    min_periods:
        Minimum number of overlapping observations required for each pair. Pairs
        with insufficient overlap yield ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Symmetric correlation matrix aligned with the retained columns.
    """

    clean, threshold = _ensure_min_periods(frame, min_periods)
    if clean.empty:
        return pd.DataFrame(dtype=float)

    cols = clean.columns.tolist()
    result = pd.DataFrame(np.eye(len(cols), dtype=float), index=cols, columns=cols)

    for i, col_i in enumerate(cols):
        series_i = clean[col_i]
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            series_j = clean[col_j]
            joint = pd.concat([series_i, series_j], axis=1).dropna()
            if len(joint) < threshold:
                corr = np.nan
            else:
                corr = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
            result.iat[i, j] = corr
            result.iat[j, i] = corr
    return result
