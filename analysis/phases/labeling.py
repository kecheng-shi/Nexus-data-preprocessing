"""Business-cycle phase labelling helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["pick_columns", "assign_phase", "assign_phase_quantile"]


def pick_columns(candidates: Iterable[str], available: Sequence[str]) -> list[str]:
    """Return candidate names that are present in ``available``."""
    available_set = set(available)
    return [col for col in candidates if col in available_set]


def assign_phase(
    row,
    *,
    g_col: str = "growth_z",
    i_col: str = "inflation_z",
    p_col: str = "policy_z",
    g_low: float = -0.7,
    g_high: float = 0.7,
    i_high: float = 0.7,
    p_tight: float = 0.7,
    p_ease: float = -0.3,
) -> str:
    """Classify a macro state into business-cycle buckets."""
    g = row[g_col]
    i = row[i_col]
    p = row[p_col]
    if np.isnan(g) or np.isnan(i) or np.isnan(p):
        return "Unclassified"
    if g <= g_low:
        return "Recession"
    if g >= g_high and p <= p_ease and i <= i_high:
        return "Early Recovery"
    if g >= 0.0 and p < p_tight and i < i_high:
        return "Mid Expansion"
    if g > -0.3 and (i >= i_high or p >= p_tight):
        return "Late Cycle"
    return "Transition"


def assign_phase_quantile(
    row,
    *,
    g_col: str = "growth_z",
    i_col: str = "inflation_z",
    p_col: str = "policy_z",
    g_low: float,
    g_high_quantile: float,
    p_low_quantile: float,
    i_high_quantile: float,
    p_tight: float = 0.7,
    i_high: float = 0.7,
) -> str:
    """Quantile-based fallback classification when early-recovery months disappear."""
    g = row[g_col]
    i = row[i_col]
    p = row[p_col]
    if np.isnan(g) or np.isnan(i) or np.isnan(p):
        return "Unclassified"
    if g <= g_low:
        return "Recession"
    if g >= g_high_quantile and p <= p_low_quantile and i <= i_high_quantile:
        return "Early Recovery"
    if g >= 0.0 and p < p_tight and i < i_high:
        return "Mid Expansion"
    if g > -0.3 and (i >= i_high or p >= p_tight):
        return "Late Cycle"
    return "Transition"

