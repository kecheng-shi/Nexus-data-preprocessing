"""Plotting utilities used across Nexus notebooks."""

from __future__ import annotations

import pandas as pd

__all__ = ["segments_from_labels"]


def segments_from_labels(dates: pd.Series, labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Convert a label series into contiguous (start, end, label) spans."""
    segs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if len(dates) == 0:
        return segs
    start = dates.iloc[0]
    cur = labels.iloc[0]
    for d, lab in zip(dates.iloc[1:], labels.iloc[1:]):
        if lab != cur:
            segs.append((pd.to_datetime(start), pd.to_datetime(d), str(cur)))
            start, cur = d, lab
    segs.append((pd.to_datetime(start), pd.to_datetime(dates.iloc[-1]), str(cur)))
    return segs

