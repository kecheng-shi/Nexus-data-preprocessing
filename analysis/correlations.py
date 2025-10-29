"""Correlation and co-movement utilities for Nexus notebooks."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["pair_list_from_block", "unique_pair_stats"]


def pair_list_from_block(block: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flatten a macro×asset correlation block into a sorted long-format table."""

    if block is None or block.dropna(how="all").empty:
        return pd.DataFrame(columns=["macro", "asset", "correlation", "overlap_months"])

    macros = list(block.index)
    assets = list(block.columns)
    rows = []
    for macro in macros:
        for asset in assets:
            val = block.loc[macro, asset]
            if pd.isna(val):
                continue
            overlap = (
                int(panel[[asset, macro]].dropna().shape[0])
                if panel is not None and {asset, macro} <= set(panel.columns)
                else np.nan
            )
            rows.append(
                {
                    "macro": macro,
                    "asset": asset,
                    "correlation": float(val),
                    "overlap_months": overlap,
                }
            )
    return pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)


def unique_pair_stats(corr_df: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return correlation statistics for unique asset pairs."""

    if corr_df is None or corr_df.dropna(how="all").empty:
        return pd.DataFrame(columns=["asset_a", "asset_b", "correlation", "overlap_months"])

    cols = list(corr_df.columns)
    records: list[dict[str, float | str]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a not in corr_df.index or b not in corr_df.columns:
                if b in corr_df.index and a in corr_df.columns:
                    val = corr_df.loc[b, a]
                else:
                    continue
            else:
                val = corr_df.loc[a, b]
            if pd.isna(val):
                continue
            overlap = (
                int(panel[[a, b]].dropna().shape[0])
                if panel is not None and {a, b} <= set(panel.columns)
                else np.nan
            )
            records.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "correlation": float(val),
                    "overlap_months": overlap,
                }
            )
    return pd.DataFrame(records).sort_values("correlation", ascending=False).reset_index(drop=True)

