"""Ad-hoc inspection helpers for macro artefacts."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from .catalogs import MACRO_SERIES
from .data_io import RAW_DATA_DIR
from .preprocessing import load_preprocessed

__all__ = ["load_raw_macro_series", "preview_preprocessed_macro_series"]


def load_raw_macro_series(keys: Iterable[str] | None = None, n_rows: int = 5) -> List[dict[str, object]]:
    """Preview rows from the original Excel macro series for inspection."""
    if keys is None:
        keys_iter = list(MACRO_SERIES.keys())
    else:
        keys_iter = list(keys)
    previews = []
    for key in keys_iter:
        if key not in MACRO_SERIES:
            raise KeyError(f"Unknown macro key: {key}")
        cfg = MACRO_SERIES[key]
        path = RAW_DATA_DIR / f"{cfg['stem']}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Missing Excel file for {key}: {path}")
        frame = pd.read_excel(path)
        previews.append(
            {
                "key": key,
                "label": cfg["label"],
                "path": path,
                "preview": frame.head(max(n_rows, 1)),
            }
        )
    return previews


def preview_preprocessed_macro_series(keys: Iterable[str] | None = None, n_rows: int = 5) -> List[dict[str, object]]:
    """Collect head snapshots from preprocessed macro series (Parquet/CSV outputs)."""
    if keys is None:
        keys_iter = list(MACRO_SERIES.keys())
    else:
        keys_iter = list(keys)
    previews = []
    for key in keys_iter:
        if key not in MACRO_SERIES:
            raise KeyError(f"Unknown macro key: {key}")
        cfg = MACRO_SERIES[key]
        frame = load_preprocessed(cfg["stem"]).head(max(n_rows, 1))
        previews.append(
            {
                "key": key,
                "label": cfg["label"],
                "preview": frame.to_pandas() if hasattr(frame, "to_pandas") else frame,
            }
        )
    return previews

