"""Asset universe utilities for macro-driver studies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from .classification import infer_asset_class, is_macro_series, slugify
from .data_io import PREPROCESSED_DIR
from .preprocessing import load_preprocessed
from .transformations import to_monthly_return


@dataclass(frozen=True)
class AssetSeriesRecord:
    """Metadata for a single asset series."""

    alias: str
    stem: str
    asset_class: str
    label: str
    sample_months: int
    sample_start: Optional[pd.Timestamp]
    sample_end: Optional[pd.Timestamp]


@dataclass(frozen=True)
class AssetUniverse:
    """Container for asset-class level return panels and metadata."""

    class_returns: pd.DataFrame
    series_returns: Mapping[str, pd.Series]
    class_members: Mapping[str, Sequence[str]]
    series_metadata: pd.DataFrame
    missing_series: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        """Serialize the universe to plain Python types (helpful for notebooks)."""
        return {
            "class_returns": self.class_returns,
            "series_returns": dict(self.series_returns),
            "class_members": {cls: list(members) for cls, members in self.class_members.items()},
            "series_metadata": self.series_metadata,
            "missing_series": self.missing_series,
        }


def _iter_preprocessed_series(preprocessed_dir: Path) -> Iterable[str]:
    """Yield unique stems for preprocessed artefacts."""
    seen: set[str] = set()
    for suffix in ("_preprocessed.parquet", "_preprocessed.csv"):
        for path in sorted(preprocessed_dir.glob(f"*{suffix}")):
            stem = path.name.removesuffix(suffix)
            if stem in seen:
                continue
            seen.add(stem)
            yield stem


def _monthly_return_series(stem: str, alias: str) -> pd.Series:
    """Return a pandas Series of compounded monthly returns for `stem`."""
    frame = to_monthly_return(load_preprocessed(stem), alias)
    monthly = (
        frame.to_pandas()
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index("date")
        .sort_index()[alias]
    )
    if not monthly.empty:
        monthly.index = monthly.index.to_period("M").to_timestamp("M")
    return monthly


def build_asset_universe(
    *,
    preprocessed_dir: Path = PREPROCESSED_DIR,
    min_months: int = 1,
    drop_empty_classes: bool = True,
) -> AssetUniverse:
    """Build equal-weighted asset-class monthly returns from the full preprocessed universe.

    Parameters
    ----------
    preprocessed_dir
        Directory containing the preprocessed parquet/csv artefacts.
    min_months
        Minimum number of non-null monthly observations required to keep an asset series.
    drop_empty_classes
        If True, asset classes with no surviving members are dropped from the panel.
    """

    alias_registry: set[str] = set()
    series_returns: MutableMapping[str, pd.Series] = {}
    series_records: List[AssetSeriesRecord] = []
    missing_records: List[Dict[str, object]] = []
    class_members: MutableMapping[str, List[str]] = defaultdict(list)

    for stem in _iter_preprocessed_series(preprocessed_dir):
        if is_macro_series(stem):
            continue

        asset_class = infer_asset_class(stem)
        alias = slugify(stem, alias_registry)

        try:
            monthly = _monthly_return_series(stem, alias)
        except FileNotFoundError:
            missing_records.append(
                {
                    "alias": alias,
                    "stem": stem,
                    "asset_class": asset_class,
                    "label": stem,
                    "sample_months": 0,
                    "issue": "preprocessed file missing",
                }
            )
            continue

        monthly_clean = monthly.dropna()
        sample_months = int(monthly_clean.shape[0])
        if sample_months < max(min_months, 1):
            missing_records.append(
                {
                    "alias": alias,
                    "stem": stem,
                    "asset_class": asset_class,
                    "label": stem,
                    "sample_months": sample_months,
                    "issue": "insufficient monthly observations",
                }
            )
            continue

        series_returns[alias] = monthly
        start = monthly_clean.index.min() if not monthly_clean.empty else None
        end = monthly_clean.index.max() if not monthly_clean.empty else None

        series_records.append(
            AssetSeriesRecord(
                alias=alias,
                stem=stem,
                asset_class=asset_class,
                label=stem,
                sample_months=sample_months,
                sample_start=start,
                sample_end=end,
            )
        )
        class_members[asset_class].append(alias)

    class_series: Dict[str, pd.Series] = {}
    for cls, aliases in class_members.items():
        if not aliases:
            continue
        frame = pd.concat([series_returns[a] for a in aliases], axis=1, join="outer")
        class_series[cls] = frame.mean(axis=1, skipna=True)

    if drop_empty_classes:
        class_series = {
            cls: series
            for cls, series in class_series.items()
            if not series.dropna().empty
        }
        class_members = {
            cls: aliases
            for cls, aliases in class_members.items()
            if cls in class_series
        }

    class_panel = (
        pd.DataFrame(class_series)
        .sort_index()
        .reindex(sorted(class_series), axis=1)
        if class_series
        else pd.DataFrame()
    )

    series_metadata = pd.DataFrame(
        [
            {
                "alias": rec.alias,
                "stem": rec.stem,
                "asset_class": rec.asset_class,
                "label": rec.label,
                "sample_months": rec.sample_months,
                "sample_start": rec.sample_start,
                "sample_end": rec.sample_end,
            }
            for rec in series_records
        ]
    ).sort_values(["asset_class", "alias"])

    missing_df = pd.DataFrame(missing_records).sort_values(["issue", "asset_class", "label"])

    return AssetUniverse(
        class_returns=class_panel,
        series_returns=series_returns,
        class_members=class_members,
        series_metadata=series_metadata,
        missing_series=missing_df,
    )


__all__ = ["AssetSeriesRecord", "AssetUniverse", "build_asset_universe"]
