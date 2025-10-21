"""Preprocessing pipeline for Nexus datasets."""

from __future__ import annotations

import math
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from .data_io import (
    PREPROCESSED_DIR,
    RAW_DATA_DIR,
    SCHEMA_ALIASES,
    _coerce_datetime,
    _first_present,
    load_excel_to_polars,
    winsorize_series,
)

__all__ = [
    "preprocess_single_series",
    "preprocess_all",
    "load_preprocessed",
]


def preprocess_single_series(path: Path) -> Tuple[pl.DataFrame, Dict[str, object]]:
    """Load one Excel file and return `(tidy_frame, metadata)`."""
    raw = load_excel_to_polars(path)
    if raw.is_empty():
        raise ValueError("Workbook is empty or unreadable.")

    original_cols = raw.columns
    date_col = _first_present(original_cols, *SCHEMA_ALIASES["date"])
    open_col = _first_present(original_cols, *SCHEMA_ALIASES["open"])
    high_col = _first_present(original_cols, *SCHEMA_ALIASES["high"])
    low_col = _first_present(original_cols, *SCHEMA_ALIASES["low"])
    adj_close_col = _first_present(original_cols, *SCHEMA_ALIASES["adj_close"])
    close_col = _first_present(original_cols, *SCHEMA_ALIASES["close"])
    if adj_close_col and close_col is None:
        close_col = adj_close_col
    volume_col = _first_present(original_cols, *SCHEMA_ALIASES["volume"])

    if close_col is None and adj_close_col is None:
        candidate_cols = [c for c in original_cols if c != date_col]
        best_col = None
        best_non_null = -1
        for cand in candidate_cols:
            numeric_series = (
                raw.select(pl.col(cand).cast(pl.Float64, strict=False).alias("_num"))
                .get_column("_num")
            )
            non_null = int(numeric_series.drop_nulls().len())
            if non_null > best_non_null:
                best_non_null = non_null
                best_col = cand
        if best_col:
            close_col = best_col
        elif candidate_cols:
            close_col = candidate_cols[0]

    if date_col is None:
        fallback = original_cols[0]
        parsed = _coerce_datetime(raw[fallback].to_list())
        if any(v is not None for v in parsed):
            raw = raw.with_columns(pl.Series(fallback, parsed))
            date_col = fallback
    if date_col is None:
        raise ValueError("No date-like column detected.")

    if close_col == date_col:
        candidate_cols = [c for c in original_cols if c not in {date_col, adj_close_col}]
        best_col = None
        best_non_null = -1
        for cand in candidate_cols:
            numeric_series = (
                raw.select(pl.col(cand).cast(pl.Float64, strict=False).alias("_num"))
                .get_column("_num")
            )
            non_null = int(numeric_series.drop_nulls().len())
            if non_null > best_non_null:
                best_non_null = non_null
                best_col = cand
        close_col = best_col

    processed = raw.with_columns(pl.Series(date_col, _coerce_datetime(raw[date_col].to_list())))
    processed = processed.filter(pl.col(date_col).is_not_null())
    processed = processed.sort(date_col)
    processed = processed.unique(subset=[date_col], keep="first", maintain_order=True)

    numeric_cols = [c for c in [open_col, high_col, low_col, close_col, adj_close_col] if c]
    for col in numeric_cols:
        processed = processed.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if volume_col is not None:
        processed = processed.with_columns(pl.col(volume_col).cast(pl.Float64, strict=False).alias(volume_col))

    if numeric_cols:
        processed = processed.with_columns(
            [
                pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward").alias(col)
                for col in numeric_cols
            ]
        )
    if volume_col is not None:
        processed = processed.with_columns(
            pl.when(pl.col(volume_col) == 0).then(None).otherwise(pl.col(volume_col)).alias(volume_col)
        )
        processed = processed.with_columns(
            pl.col(volume_col).fill_null(strategy="forward").fill_null(strategy="backward").alias(volume_col)
        )

    if adj_close_col and close_col:
        processed = processed.with_columns(
            pl.when((pl.col(close_col) != 0) & pl.col(close_col).is_not_null())
            .then(pl.col(adj_close_col) / pl.col(close_col))
            .otherwise(None)
            .alias("__adj_factor")
        )
        processed = processed.with_columns(
            pl.when(pl.col("__adj_factor") > 0)
            .then(pl.col("__adj_factor"))
            .otherwise(None)
            .alias("__adj_factor")
        )
        processed = processed.with_columns(
            pl.col("__adj_factor").fill_null(strategy="forward").fill_null(strategy="backward").alias("__adj_factor")
        )
        if open_col:
            processed = processed.with_columns((pl.col(open_col) * pl.col("__adj_factor")).alias("adj_open"))
        if high_col:
            processed = processed.with_columns((pl.col(high_col) * pl.col("__adj_factor")).alias("adj_high"))
        if low_col:
            processed = processed.with_columns((pl.col(low_col) * pl.col("__adj_factor")).alias("adj_low"))
        processed = processed.with_columns((pl.col(close_col) * pl.col("__adj_factor")).alias("adj_close"))
        processed = processed.drop("__adj_factor")

    if "adj_close" not in processed.columns:
        fallback_close = adj_close_col or close_col
        if fallback_close is None:
            raise ValueError("No usable closing price column.")
        processed = processed.with_columns(pl.col(fallback_close).alias("adj_close"))

    processed = processed.rename({date_col: "date"})
    if volume_col is not None and volume_col != "volume":
        processed = processed.rename({volume_col: "volume"})
        volume_col = "volume"

    processed = processed.with_columns(pl.col("adj_close").log().diff().alias("ret"))
    processed = processed.with_columns(winsorize_series(processed["ret"]).rename("ret_w"))

    vol_20 = processed["ret"].rolling_std(window_size=20, min_periods=20)
    vol_60 = processed["ret"].rolling_std(window_size=60, min_periods=60)
    processed = processed.with_columns(
        [
            vol_20.rename("vol_20"),
            vol_60.rename("vol_60"),
        ]
    )
    processed = processed.with_columns(
        [
            (processed["vol_20"] * math.sqrt(252.0)).rename("rv_20_annual"),
            (processed["vol_60"] * math.sqrt(252.0)).rename("rv_60_annual"),
        ]
    )
    ret_vol_scaled = processed["ret"] / processed["vol_20"]
    processed = processed.with_columns(ret_vol_scaled.rename("ret_vol_scaled"))
    processed = processed.with_columns(winsorize_series(processed["ret_vol_scaled"]).rename("ret_vol_scaled_w"))

    adj_close_series = processed["adj_close"].cast(pl.Float64, strict=False)
    processed = processed.with_columns(adj_close_series.alias("adj_close"))
    delta = adj_close_series.diff().cast(pl.Float64, strict=False)
    gain_series = delta.map_elements(
        lambda x: float(x) if x is not None and x > 0 else 0.0,
        return_dtype=pl.Float64,
        skip_nulls=False,
    )
    loss_series = delta.map_elements(
        lambda x: -float(x) if x is not None and x < 0 else 0.0,
        return_dtype=pl.Float64,
        skip_nulls=False,
    )
    avg_gain = gain_series.ewm_mean(alpha=1.0 / 14.0, adjust=False, ignore_nulls=True, min_periods=14)
    avg_loss = loss_series.ewm_mean(alpha=1.0 / 14.0, adjust=False, ignore_nulls=True, min_periods=14)
    denom = avg_loss.map_elements(
        lambda x: None if x is None or x == 0 else float(x),
        return_dtype=pl.Float64,
        skip_nulls=False,
    )
    rs = avg_gain / denom
    rsi = rs.map_elements(
        lambda x: 100.0 - (100.0 / (1.0 + x)) if x is not None else None,
        return_dtype=pl.Float64,
        skip_nulls=False,
    )
    processed = processed.with_columns(rsi.rename("rsi14"))

    ema_fast = adj_close_series.ewm_mean(alpha=2.0 / (12.0 + 1.0), adjust=False, ignore_nulls=True)
    ema_slow = adj_close_series.ewm_mean(alpha=2.0 / (26.0 + 1.0), adjust=False, ignore_nulls=True)
    macd_series = ema_fast - ema_slow
    macd_signal = macd_series.ewm_mean(alpha=2.0 / (9.0 + 1.0), adjust=False, ignore_nulls=True)
    macd_hist = macd_series - macd_signal
    processed = processed.with_columns(
        [
            macd_series.rename("macd"),
            macd_signal.rename("macd_signal"),
            macd_hist.rename("macd_hist"),
        ]
    )

    bb_mid = adj_close_series.rolling_mean(window_size=20, min_periods=20)
    bb_std = adj_close_series.rolling_std(window_size=20, min_periods=20)
    processed = processed.with_columns(
        [
            bb_mid.rename("bb_mid"),
            (bb_mid + 2.0 * bb_std).rename("bb_upper"),
            (bb_mid - 2.0 * bb_std).rename("bb_lower"),
        ]
    )

    base_cols = ["date", "adj_open", "adj_high", "adj_low", "adj_close"]
    tidy_cols: List[str] = [c for c in base_cols if c in processed.columns]
    if volume_col is not None and volume_col in processed.columns:
        tidy_cols.append(volume_col)
    feature_cols = [
        "ret",
        "ret_w",
        "ret_vol_scaled",
        "ret_vol_scaled_w",
        "vol_20",
        "vol_60",
        "rv_20_annual",
        "rv_60_annual",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
    ]
    tidy_cols.extend([c for c in feature_cols if c in processed.columns])

    tidy = processed.select(tidy_cols).sort("date")
    tidy = tidy.with_columns(pl.lit(path.stem).alias("series"))
    tidy = tidy.select(["series"] + [c for c in tidy.columns if c != "series"])

    tidy = tidy.with_columns(pl.col("date").cast(pl.Datetime, strict=False))
    date_series = tidy["date"]
    non_null_dates = date_series.drop_nulls()
    if not non_null_dates.is_empty():
        start_dt = non_null_dates.min()
        end_dt = non_null_dates.max()
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        gap_series = tidy.select(pl.col("date").diff().dt.total_days()).to_series().drop_nulls()
        median_gap = float(gap_series.median()) if not gap_series.is_empty() else math.nan
    else:
        start_str = None
        end_str = None
        median_gap = math.nan
    feature_base = {"series", "date", "volume", "adj_open", "adj_high", "adj_low", "adj_close"}
    feature_count = len([c for c in tidy.columns if c not in feature_base])

    meta: Dict[str, object] = {
        "series": path.stem,
        "raw_file": path.name,
        "rows": int(tidy.height),
        "start": start_str,
        "end": end_str,
        "median_gap_days": median_gap,
        "has_volume": "volume" in tidy.columns,
        "feature_columns": feature_count,
    }

    return tidy, meta


def preprocess_all(files: Iterable[Path], overwrite: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Process a collection of Excel exports and persist tidy artefacts."""
    meta_entries: List[Dict[str, object]] = []
    error_entries: List[Dict[str, object]] = []
    for path in files:
        series_id = path.stem
        csv_path = PREPROCESSED_DIR / f"{series_id}_preprocessed.csv"
        parquet_path = PREPROCESSED_DIR / f"{series_id}_preprocessed.parquet"

        if not overwrite and csv_path.exists():
            try:
                cached = pl.read_csv(csv_path, try_parse_dates=True)
                dates = [dt for dt in cached["date"].to_list() if dt is not None]
                if dates:
                    deltas = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
                    median_gap = float(statistics.median(deltas)) if deltas else math.nan
                    start_str = dates[0].strftime("%Y-%m-%d")
                    end_str = dates[-1].strftime("%Y-%m-%d")
                else:
                    median_gap = math.nan
                    start_str = None
                    end_str = None
                meta_entries.append(
                    {
                        "series": series_id,
                        "raw_file": path.name,
                        "rows": int(cached.height),
                        "start": start_str,
                        "end": end_str,
                        "median_gap_days": median_gap,
                        "has_volume": "volume" in cached.columns,
                        "feature_columns": len(
                            [
                                c
                                for c in cached.columns
                                if c
                                not in {
                                    "series",
                                    "date",
                                    "volume",
                                    "adj_open",
                                    "adj_high",
                                    "adj_low",
                                    "adj_close",
                                }
                            ]
                        ),
                        "csv_path": str(csv_path),
                        "parquet_path": str(parquet_path) if parquet_path.exists() else "",
                        "status": "cached",
                    }
                )
                continue
            except Exception:
                pass

        try:
            tidy, meta = preprocess_single_series(path)
        except Exception as exc:
            error_entries.append({"series": series_id, "raw_file": path.name, "error": str(exc)})
            continue

        tidy.write_csv(csv_path)
        parquet_written = False
        try:
            tidy.write_parquet(parquet_path)
            parquet_written = True
        except Exception:
            parquet_written = False

        meta.update(
            {
                "csv_path": str(csv_path),
                "parquet_path": str(parquet_path) if parquet_written else "",
                "status": "processed",
            }
        )
        meta_entries.append(meta)

    meta_table = pl.DataFrame(meta_entries)
    error_table = pl.DataFrame(error_entries)
    return meta_table, error_table


@lru_cache(maxsize=None)
def load_preprocessed(stem: str) -> pl.DataFrame:
    """Load a preprocessed time series (Parquet preferred, CSV fallback)."""
    parquet_path = PREPROCESSED_DIR / f"{stem}_preprocessed.parquet"
    csv_path = PREPROCESSED_DIR / f"{stem}_preprocessed.csv"
    if parquet_path.exists():
        frame = pl.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pl.read_csv(csv_path, try_parse_dates=True)
    else:
        raise FileNotFoundError(f"No preprocessed artefact found for: {stem}")

    try:
        if "adj_close" in frame.columns and "date" in frame.columns:
            f2 = frame.with_columns(
                pl.col("date").cast(pl.Datetime, strict=False).dt.epoch("us").alias("__epoch_us")
            )
            corr_df = f2.select(pl.corr(pl.col("adj_close"), pl.col("__epoch_us")))
            corr_val = corr_df.to_series()[0] if corr_df.height else None
            ratio_df = f2.select((pl.col("adj_close") / pl.col("__epoch_us")).median())
            ratio_val = ratio_df.to_series()[0] if ratio_df.height else None
            if (
                corr_val is not None
                and abs(float(corr_val)) > 0.999
                and ratio_val is not None
                and 0.9 < float(ratio_val) < 1.1
            ):
                raw_path = RAW_DATA_DIR / f"{stem}.xlsx"
                if raw_path.exists():
                    rebuilt, _meta = preprocess_single_series(raw_path)
                    frame = rebuilt
                else:
                    ignore = {
                        "series",
                        "date",
                        "adj_open",
                        "adj_high",
                        "adj_low",
                        "adj_close",
                        "ret",
                        "ret_w",
                        "ret_vol_scaled",
                        "ret_vol_scaled_w",
                        "vol_20",
                        "vol_60",
                        "rv_20_annual",
                        "rv_60_annual",
                        "rsi14",
                        "macd",
                        "macd_signal",
                        "macd_hist",
                        "bb_mid",
                        "bb_upper",
                        "bb_lower",
                        "__epoch_us",
                    }
                    best_col = None
                    best_non_null = -1
                    for c in frame.columns:
                        if c in ignore:
                            continue
                        s = frame.select(pl.col(c).cast(pl.Float64, strict=False).alias("_num"))["_num"]
                        nn = int(s.drop_nulls().len())
                        if nn > best_non_null:
                            best_non_null = nn
                            best_col = c
                    if best_col is not None:
                        frame = frame.with_columns(pl.col(best_col).cast(pl.Float64, strict=False).alias("adj_close"))
    except Exception:
        pass

    return frame.sort("date")
