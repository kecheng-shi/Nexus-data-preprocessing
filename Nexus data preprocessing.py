#!/usr/bin/env python3
"""
Preprocess equity time series (e.g., 'CAT US.xlsx').
- Detects Date/OHLC/Adj Close/Volume columns (case-insensitive)
- Builds adjusted OHLC from Adj Close factor (if available)
- Computes log returns, winsorized returns, rolling volatility (20/60d),
  annualized realized vol, RSI(14), MACD(12,26,9), Bollinger Bands(20, ±2σ)
- Outputs CSV (always) and Parquet (if pyarrow is installed).

Usage:
  python preprocess_cat_us.py --input "CAT US.xlsx" --output_base "CAT_US_preprocessed"
"""
from __future__ import annotations

import argparse
from typing import Optional, List

import numpy as np
import pandas as pd


# ---------- helpers ----------
def _first_present(cols: List[str], *candidates_groups: List[str]) -> Optional[str]:
    cols_l = [c.lower().strip() for c in cols]
    for candidates in candidates_groups:
        for cand in candidates:
            if cand in cols_l:
                return cols[cols_l.index(cand)]
    return None


def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def winsorize(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lower=lo, upper=hi)


# ---------- indicators ----------
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, span_fast: int = 12, span_slow: int = 26, span_signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=span_fast, adjust=False).mean()
    ema_slow = prices.ewm(span=span_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal, "macd_hist": hist})


def compute_bbands(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    ma = prices.rolling(window, min_periods=window).mean()
    sd = prices.rolling(window, min_periods=window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower})


# ---------- adjustment ----------
def adjust_ohlc_with_adjclose(df: pd.DataFrame,
                              col_open: Optional[str],
                              col_high: Optional[str],
                              col_low: Optional[str],
                              col_close: Optional[str],
                              col_adjclose: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if col_adjclose is None or col_close is None:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = out[col_adjclose] / out[col_close]
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.where(factor > 0, np.nan).ffill().bfill()
    if col_open is not None:
        out["adj_open"] = out[col_open] * factor
    if col_high is not None:
        out["adj_high"] = out[col_high] * factor
    if col_low is not None:
        out["adj_low"] = out[col_low] * factor
    out["adj_close"] = out[col_close] * factor
    return out


# ---------- core ----------
def preprocess_cat_excel(input_path: str, output_base: str):
    # 1) load
    try:
        df = pd.read_excel(input_path)
    except Exception:
        df = pd.read_excel(input_path, engine="openpyxl")

    original_cols = list(df.columns)

    # 2) detect columns
    date_col = _first_present(original_cols, ["date", "timestamp", "time", "datetime"])
    open_col = _first_present(original_cols, ["open", "px_open", "o"])
    high_col = _first_present(original_cols, ["high", "px_high", "h"])
    low_col  = _first_present(original_cols, ["low", "px_low", "l"])
    close_col = _first_present(original_cols, ["adj close", "adjusted close", "close", "px_last", "price", "last"])
    # if adj close exists, prefer the raw close for factor calc
    if _first_present(original_cols, ["adj close", "adjusted close"]):
        close_col = _first_present(original_cols, ["close", "px_last", "price", "last"]) or close_col
    adj_close_col = _first_present(original_cols, ["adj close", "adjusted close"])
    volume_col = _first_present(original_cols, ["volume", "vol", "qty", "turnover"])

    # 3) dates
    if date_col is None:
        maybe_date = original_cols[0]
        parsed = pd.to_datetime(df[maybe_date], errors="coerce")
        if parsed.notna().mean() > 0.7:
            date_col = maybe_date
    if date_col is None:
        raise ValueError("No date/time column detected.")

    df[date_col] = _coerce_datetime(df[date_col])
    df = (df.dropna(subset=[date_col])
            .sort_values(date_col)
            .drop_duplicates(subset=[date_col])
            .reset_index(drop=True))

    # 4) numerics & patching
    price_cols = [c for c in [open_col, high_col, low_col, close_col, adj_close_col] if c is not None]
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if volume_col is not None:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df[price_cols] = df[price_cols].ffill().bfill()
    if volume_col is not None:
        df.loc[df[volume_col] == 0, volume_col] = np.nan
        df[volume_col] = df[volume_col].ffill().bfill()

    # 5) adjusted OHLC
    df = adjust_ohlc_with_adjclose(df, open_col, high_col, low_col, close_col, adj_close_col)
    if "adj_close" not in df.columns:
        if close_col is None:
            raise ValueError("No usable close or adjusted close column.")
        df["adj_close"] = df[close_col]

    # 6) features
    df["ret"] = np.log(df["adj_close"]).diff()
    df["ret_w"] = winsorize(df["ret"])
    df["vol_20"] = df["ret"].rolling(20, min_periods=20).std()
    df["vol_60"] = df["ret"].rolling(60, min_periods=60).std()
    df["rv_20_annual"] = df["vol_20"] * np.sqrt(252.0)
    df["rv_60_annual"] = df["vol_60"] * np.sqrt(252.0)
    df["ret_vol_scaled"] = df["ret"] / (df["vol_20"].replace(0, np.nan))
    df["ret_vol_scaled_w"] = winsorize(df["ret_vol_scaled"])

    rsi14 = compute_rsi(df["adj_close"], window=14)
    macd_df = compute_macd(df["adj_close"], span_fast=12, span_slow=26, span_signal=9)
    bb_df = compute_bbands(df["adj_close"], window=20, n_std=2.0)
    df = pd.concat([df, rsi14.rename("rsi14"), macd_df, bb_df], axis=1)

    # 7) tidy output
    adj_cols_available = [c for c in ["adj_open", "adj_high", "adj_low"] if c in df.columns]
    out_cols = [date_col] + adj_cols_available + ["adj_close"]
    if volume_col is not None:
        out_cols.append(volume_col)
    out_cols += ["ret", "ret_w", "ret_vol_scaled", "ret_vol_scaled_w",
                 "vol_20", "vol_60", "rv_20_annual", "rv_60_annual",
                 "rsi14", "macd", "macd_signal", "macd_hist",
                 "bb_mid", "bb_upper", "bb_lower"]

    tidy = df[out_cols].copy()
    rename_map = {date_col: "date"}
    if volume_col is not None:
        rename_map[volume_col] = "volume"
    tidy = tidy.rename(columns=rename_map)
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)

    # 8) write files
    csv_path = f"{output_base}.csv"
    tidy.to_csv(csv_path, index=False)

    parquet_path = None
    try:
        import pyarrow  # noqa: F401
        parquet_path = f"{output_base}.parquet"
        tidy.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    return csv_path, parquet_path


def main():
    ap = argparse.ArgumentParser(description="Preprocess equity time series (CAT US.xlsx style).")
    ap.add_argument("--input", required=True, help="Path to Excel file")
    ap.add_argument("--output_base", required=True, help="Output path prefix (no extension)")
    args = ap.parse_args()
    csv_path, pq_path = preprocess_cat_excel(args.input, args.output_base)
    print("Wrote:", csv_path)
    if pq_path:
        print("Wrote:", pq_path)
    else:
        print("Parquet not written (pyarrow missing).")


# Keep backwards-compatible name
def preprocess_cat_excel(input_path: str, output_base: str):
    return preprocess_cat_excel.__impl__(input_path, output_base)  # type: ignore[attr-defined]


# attach implementation
def _attach_impl():
    def impl(input_path, output_base):
        return __real_preprocess(input_path, output_base)
    preprocess_cat_excel.__impl__ = impl  # type: ignore[attr-defined]


def __real_preprocess(input_path: str, output_base: str):
    # Call the main processing directly
    return _process(input_path, output_base)


def _process(input_path: str, output_base: str):
    # Simple inline call to keep things tidy
    # We can just call the function defined above without indirection,
    # but we keep this structure to allow overriding in notebooks if needed.
    return _do_process(input_path, output_base)


def _do_process(input_path: str, output_base: str):
    # Final call
    # Reuse logic by calling the top-level 'preprocess_cat_excel' implementation
    # which we aliased via __impl__ to avoid recursion.
    return _simple_impl(input_path, output_base)


def _simple_impl(input_path: str, output_base: str):
    # The real worker is the top 'preprocess_cat_excel' code path via __impl__
    # Here, directly replicate the processing for clarity.
    # However, to prevent recursion, we directly execute the computation here again.
    try:
        # We will just call the earlier function's internals to keep it straightforward.
        return __compute(input_path, output_base)
    except RecursionError:
        # Fallback to recomputing without aliasing
        return preprocess_cat_excel_simple(input_path, output_base)


def __compute(input_path: str, output_base: str):
    # Re-run the main processing function once more (safe call path)
    return actually_process(input_path, output_base)


def actually_process(input_path: str, output_base: str):
    # Directly call the core function defined at the top
    return preprocess_cat_excel_core(input_path, output_base)


def preprocess_cat_excel_core(input_path: str, output_base: str):
    # To keep things robust, duplicate the minimal logic inline:
    return preprocess_cat_excel_simple(input_path, output_base)


def preprocess_cat_excel_simple(input_path: str, output_base: str):
    # Duplicate minimal logic by reading and writing through the main function
    # (avoids recursive calls in some environments)
    try:
        df = pd.read_excel(input_path)
    except Exception:
        df = pd.read_excel(input_path, engine="openpyxl")

    original_cols = list(df.columns)

    date_col = _first_present(original_cols, ["date", "timestamp", "time", "datetime"])
    open_col = _first_present(original_cols, ["open", "px_open", "o"])
    high_col = _first_present(original_cols, ["high", "px_high", "h"])
    low_col  = _first_present(original_cols, ["low", "px_low", "l"])
    close_col = _first_present(original_cols, ["adj close", "adjusted close", "close", "px_last", "price", "last"])
    if _first_present(original_cols, ["adj close", "adjusted close"]):
        close_col = _first_present(original_cols, ["close", "px_last", "price", "last"]) or close_col
    adj_close_col = _first_present(original_cols, ["adj close", "adjusted close"])
    volume_col = _first_present(original_cols, ["volume", "vol", "qty", "turnover"])

    if date_col is None:
        maybe_date = original_cols[0]
        parsed = pd.to_datetime(df[maybe_date], errors="coerce")
        if parsed.notna().mean() > 0.7:
            date_col = maybe_date
    if date_col is None:
        raise ValueError("No date/time column detected.")

    df[date_col] = _coerce_datetime(df[date_col])
    df = (df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True))

    price_cols = [c for c in [open_col, high_col, low_col, close_col, adj_close_col] if c is not None]
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if volume_col is not None:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df[price_cols] = df[price_cols].ffill().bfill()
    if volume_col is not None:
        df.loc[df[volume_col] == 0, volume_col] = np.nan
        df[volume_col] = df[volume_col].ffill().bfill()

    df = adjust_ohlc_with_adjclose(df, open_col, high_col, low_col, close_col, adj_close_col)
    if "adj_close" not in df.columns:
        if close_col is None:
            raise ValueError("No usable close or adjusted close column.")
        df["adj_close"] = df[close_col]

    df["ret"] = np.log(df["adj_close"]).diff()
    df["ret_w"] = winsorize(df["ret"])
    df["vol_20"] = df["ret"].rolling(20, min_periods=20).std()
    df["vol_60"] = df["ret"].rolling(60, min_periods=60).std()
    df["rv_20_annual"] = df["vol_20"] * np.sqrt(252.0)
    df["rv_60_annual"] = df["vol_60"] * np.sqrt(252.0)
    df["ret_vol_scaled"] = df["ret"] / (df["vol_20"].replace(0, np.nan))
    df["ret_vol_scaled_w"] = winsorize(df["ret_vol_scaled"])

    rsi14 = compute_rsi(df["adj_close"], window=14)
    macd_df = compute_macd(df["adj_close"], span_fast=12, span_slow=26, span_signal=9)
    bb_df = compute_bbands(df["adj_close"], window=20, n_std=2.0)
    df = pd.concat([df, rsi14.rename("rsi14"), macd_df, bb_df], axis=1)

    adj_cols_available = [c for c in ["adj_open", "adj_high", "adj_low"] if c in df.columns]
    out_cols = [date_col] + adj_cols_available + ["adj_close"]
    if volume_col is not None:
        out_cols.append(volume_col)
    out_cols += ["ret", "ret_w", "ret_vol_scaled", "ret_vol_scaled_w",
                 "vol_20", "vol_60", "rv_20_annual", "rv_60_annual",
                 "rsi14", "macd", "macd_signal", "macd_hist",
                 "bb_mid", "bb_upper", "bb_lower"]

    tidy = df[out_cols].copy()
    rename_map = {date_col: "date"}
    if volume_col is not None:
        rename_map[volume_col] = "volume"
    tidy = tidy.rename(columns=rename_map)
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)

    csv_path = f"{output_base}.csv"
    tidy.to_csv(csv_path, index=False)

    parquet_path = None
    try:
        import pyarrow  # noqa: F401
        parquet_path = f"{output_base}.parquet"
        tidy.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    return csv_path, parquet_path


# Wire the implementation for the public function alias
_attach_impl()

if __name__ == "__main__":
    main()
