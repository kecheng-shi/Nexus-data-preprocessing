# Equity Time Series Preprocessor

A small, batteries-included script to turn a raw Excel file like `CAT US.xlsx` into tidy, analysis-ready data with common technical indicators and volatility stats. It auto-detects Date/OHLC/Adj Close/Volume columns (case-insensitive), builds adjusted OHLC from the Adjusted Close factor when available, and writes both CSV (always) and Parquet (if `pyarrow` is installed).


---

## Features

- **Column autodetection** (case-insensitive): `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- **Adjusted OHLC** via factor = `Adj Close / Close`, forward/backward filled and applied to O/H/L/C.
- **Return & risk metrics**
  - Log returns: `ret`
  - Winsorized returns (1–99%): `ret_w`
  - Rolling vol: `vol_20`, `vol_60` (on log returns)
  - Annualized realized vol: `rv_20_annual`, `rv_60_annual` (× √252)
  - Vol-scaled returns: `ret_vol_scaled`, `ret_vol_scaled_w`
- **Technical indicators**
  - RSI(14)
  - MACD(12,26,9): `macd`, `macd_signal`, `macd_hist`
  - Bollinger Bands(20, ±2σ): `bb_mid`, `bb_upper`, `bb_lower`
- **Robust I/O**
  - Reads `.xlsx` with default engine, falls back to `openpyxl` if needed
  - Outputs `*.csv` and optionally `*.parquet` (when `pyarrow` is available)
- **Data hygiene**
  - Parses and sorts by date, drops duplicate timestamps
  - Coerces numerics, forward/backward fills price fields
  - Treats zero `Volume` as missing then fills
  - Strips timezone info for the `date` output

---

## Installation

```bash
pip install pandas numpy
# optional for Parquet:
pip install pyarrow
# optional Excel engine (fallback):
pip install openpyxl
```

No package install is required for the script itself—just add it to your repo.

---

## Usage

### Command line

```bash
python preprocess_cat_us.py --input "CAT US.xlsx" --output_base "CAT_US_preprocessed"
```

This writes:

- `CAT_US_preprocessed.csv`
- `CAT_US_preprocessed.parquet` (if `pyarrow` is installed)

### Python API

```python
from preprocess_cat_us import preprocess_cat_excel

csv_path, parquet_path = preprocess_cat_excel("CAT US.xlsx", "CAT_US_preprocessed")
print(csv_path, parquet_path)
```

---

## Input expectations

A single-sheet Excel file with at least a date/time column and a close price. Column names are matched case-insensitively against common aliases:

- **Date**: `date`, `timestamp`, `time`, `datetime`
- **Open**: `open`, `px_open`, `o`
- **High**: `high`, `px_high`, `h`
- **Low**: `low`, `px_low`, `l`
- **Close** (raw): `close`, `px_last`, `price`, `last`
- **Adj Close**: `adj close`, `adjusted close`
- **Volume**: `volume`, `vol`, `qty`, `turnover`

If `Adj Close` exists, it’s used to compute an adjustment factor; otherwise the raw `Close` is treated as `adj_close`.

> If no explicit date column is found, the script tries the **first column** and accepts it as date if ≥70% of its values parse as timestamps.

---

## Output schema

The CSV/Parquet will include some or all of the following columns (depending on inputs):

| Column               | Description |
|---                   |---|
| `date`               | Naive (no timezone) timestamp; sorted, unique |
| `adj_open`           | Adjusted open (if source open available) |
| `adj_high`           | Adjusted high (if source high available) |
| `adj_low`            | Adjusted low (if source low available) |
| `adj_close`          | Adjusted close (or raw close if no adj close) |
| `volume`             | Volume (zeros treated as NA then filled) |
| `ret`                | Log return: `ln(adj_close).diff()` |
| `ret_w`              | Winsorized `ret` (1% / 99%) |
| `ret_vol_scaled`     | `ret / vol_20` |
| `ret_vol_scaled_w`   | Winsorized `ret_vol_scaled` |
| `vol_20`             | Rolling 20-day std of `ret` |
| `vol_60`             | Rolling 60-day std of `ret` |
| `rv_20_annual`       | `vol_20 * sqrt(252)` |
| `rv_60_annual`       | `vol_60 * sqrt(252)` |
| `rsi14`              | RSI with EWMA smoothing (period 14) |
| `macd`               | EMA(12) - EMA(26) |
| `macd_signal`        | EMA(9) of `macd` |
| `macd_hist`          | `macd - macd_signal` |
| `bb_mid`             | 20-day moving average of `adj_close` |
| `bb_upper`           | `bb_mid + 2 * rolling_std(20)` |
| `bb_lower`           | `bb_mid - 2 * rolling_std(20)` |

---

## How it works (quick math)

- **Adjustment factor**: `factor = AdjClose / Close` (∞/≤0 → NA → ffill/bfill). Applied to O/H/L/C when present.
- **Returns**: `ret_t = ln(AdjClose_t) - ln(AdjClose_{t-1})`.
- **Winsorization**: clip to the 1st and 99th percentiles computed from the series.
- **Volatility**: rolling standard deviation of `ret` over 20/60 days; annualized with `sqrt(252)`.
- **RSI(14)**: EWMA of gains/losses with `alpha = 1/window`; `RSI = 100 - 100/(1 + RS)`.
- **MACD(12,26,9)**: `EMA(12) - EMA(26)`, signal = `EMA(9)` of MACD.
- **Bollinger(20, 2σ)**: 20-day mean ± 2 × 20-day std.

---

## Development notes

The module exposes a callable `preprocess_cat_excel(...)`. Internally, it uses an alias/attachment pattern to allow re-binding the implementation in notebook environments. For most users, you can ignore the indirection and simply import and call `preprocess_cat_excel`.

To add indicators, extend the **“indicators”** section and append to the `tidy` frame and `out_cols` list.

