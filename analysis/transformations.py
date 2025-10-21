"""Aggregation and transformation utilities for Nexus analysis."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    "to_monthly_return",
    "to_monthly_macro",
    "join_on_date",
    "standardized_linear_regression",
]


def to_monthly_return(frame: pl.DataFrame, alias: str) -> pl.DataFrame:
    """Aggregate daily/simple returns into compounded monthly returns."""
    return (
        frame.filter(pl.col("ret").is_not_null())
        .with_columns(
            [
                pl.col("date").dt.truncate("1mo").alias("month"),
                (1 + pl.col("ret")).alias("gross_return"),
            ]
        )
        .group_by("month", maintain_order=True)
        .agg((pl.col("gross_return").product() - 1).alias(alias))
        .with_columns(pl.col("month").dt.date().alias("date"))
        .select(["date", alias])
    )


def to_monthly_macro(frame: pl.DataFrame, alias: str) -> pl.DataFrame:
    """Collapse macro series to month-end levels and first differences."""
    monthly = (
        frame.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(pl.col("adj_close").last().alias("level"))
        .sort("month")
    )
    return (
        monthly.with_columns(
            [
                pl.col("level").alias(f"{alias}_level"),
                (pl.col("level") / pl.col("level").shift(1) - 1).alias(f"{alias}_change"),
            ]
        )
        .with_columns(pl.col("month").dt.date().alias("date"))
        .select(["date", f"{alias}_level", f"{alias}_change"])
    )


def join_on_date(frames: Iterable[pl.DataFrame], how: str = "inner") -> pl.DataFrame:
    """Join a list of Polars frames on their `date` column."""
    frames = [f for f in frames if f.height]
    if not frames:
        raise ValueError("No data frames supplied for join.")
    result = frames[0]
    for other in frames[1:]:
        result = result.join(other, on="date", how=how)
    return result.sort("date")


def standardized_linear_regression(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.Series, float]:
    """Return standardised betas and in-sample R^2 using a simple OLS solver."""
    X = features.copy()
    y = target.copy()

    feature_std = X.std(ddof=0)
    valid_features = feature_std[feature_std > 0].index.tolist()
    if not valid_features:
        raise ValueError("All feature columns are constant; regression not defined.")
    X_std = (X[valid_features] - X[valid_features].mean()) / feature_std[valid_features]

    y_std = y.copy()
    y_sigma = y_std.std(ddof=0)
    if y_sigma == 0:
        raise ValueError("Target series variance is zero; regression not defined.")
    y_std = (y_std - y_std.mean()) / y_sigma

    X_values = X_std.to_numpy()
    y_values = y_std.to_numpy()

    coeffs, residuals, rank, s = np.linalg.lstsq(X_values, y_values, rcond=None)
    fitted = X_values @ coeffs
    ss_res = np.sum((y_values - fitted) ** 2)
    ss_tot = np.sum((y_values - y_values.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan

    beta = pd.Series(coeffs, index=valid_features, name="beta_std")
    return beta, float(r_squared)

