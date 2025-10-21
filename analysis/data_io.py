"""Data loading helpers for Nexus analysis notebooks."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl
from dateutil import parser
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

# Repository-relative directories used across the analysis workflow.
RAW_DATA_DIR = Path("FULL Nexus Data")
PREPROCESSED_DIR = Path("analysis/preprocessed")
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Column aliases to reconcile heterogeneous Bloomberg exports.
SCHEMA_ALIASES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "date": (("date", "timestamp", "time", "datetime"),),
    "open": (("open", "px_open", "o"),),
    "high": (("high", "px_high", "h"),),
    "low": (("low", "px_low", "l"),),
    "close": (("close", "px_last", "last", "price"),),
    "adj_close": (("adj close", "adjusted close"),),
    "volume": (("volume", "vol", "qty", "turnover"),),
}

__all__ = [
    "RAW_DATA_DIR",
    "PREPROCESSED_DIR",
    "SCHEMA_ALIASES",
    "_first_present",
    "_coerce_datetime",
    "winsorize_series",
    "load_excel_to_polars",
]


def _first_present(columns: Iterable[str], *candidates_groups: Iterable[str]) -> Optional[str]:
    """Return the first column whose lowered form appears in the candidate groups."""
    cols = list(columns)
    lowered = [c.lower().strip() for c in cols]
    for group in candidates_groups:
        for candidate in group:
            key = candidate.lower().strip()
            if key in lowered:
                return cols[lowered.index(key)]
    return None


def _coerce_datetime(values: List[object]) -> List[Optional[datetime]]:
    """Best-effort conversion of heterogeneous Excel date representations."""
    out: List[Optional[datetime]] = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        if isinstance(value, datetime):
            out.append(value.replace(tzinfo=None))
            continue
        if isinstance(value, date):
            out.append(datetime.combine(value, datetime.min.time()))
            continue
        if isinstance(value, (int, float)):
            try:
                out.append(from_excel(value))
                continue
            except Exception:
                pass
        try:
            parsed_dt = parser.parse(str(value))
            out.append(parsed_dt.replace(tzinfo=None))
        except Exception:
            out.append(None)
    return out


def winsorize_series(series: pl.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pl.Series:
    """Clip a Polars series to the given quantile bounds."""
    clean = series.drop_nulls()
    if clean.is_empty():
        return series
    lower = clean.quantile(lower_q)
    upper = clean.quantile(upper_q)
    return series.clip(lower, upper)


def load_excel_to_polars(path: Path) -> pl.DataFrame:
    """Read a flat Bloomberg-exported workbook with openpyxl and return a DataFrame."""
    wb = load_workbook(path, data_only=True, read_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pl.DataFrame()

    header = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    data_rows = [list(row) for row in rows[1:] if any(cell is not None for cell in row)]
    if not data_rows:
        return pl.DataFrame({name: [] for name in header})

    columns = list(zip(*data_rows))
    data = {header[i]: list(columns[i]) for i in range(len(header))}
    return pl.DataFrame(data)

