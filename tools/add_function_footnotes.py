#!/usr/bin/env python3
"""
Insert inline footnotes as Python comments immediately before each function
definition in `Index Analysis.ipynb`. Also removes any prior top-of-cell
inline blocks (`# [codex-footnotes-inline] ...`) and previously inserted
markdown footnote cells (identified by `metadata.codex_footnotes`).

Idempotent: per-function footnotes are marked with `# [fn-footnote:<name>]`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

NB_PATH = Path("Index Analysis.ipynb")
CELL_INLINE_MARK = "# [codex-footnotes-inline]"
FN_MARK_PREFIX = "# [fn-footnote:"


def strip_cell_inline_block(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    if not lines[0].lstrip().startswith(CELL_INLINE_MARK):
        return lines
    # Drop initial inline block: marker + subsequent comment/blank lines until first non-comment.
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.strip() == "" or s.lstrip().startswith("#"):
            i += 1
            continue
        break
    return lines[i:]


def fn_block(name: str, bullets: List[str]) -> List[str]:
    header = f"# [fn-footnote:{name}]\n"
    title = f"# Footnote — {name}\n"
    body = []
    for b in bullets:
        if b.strip():
            body.append(f"# - {b}\n")
    return [header, title, "#\n", *body]


def main() -> int:
    if not NB_PATH.exists():
        print(f"Notebook not found: {NB_PATH}")
        return 2

    nb = json.loads(NB_PATH.read_text())
    cells = nb.get("cells", [])

    # Remove any markdown footnote cells inserted earlier
    new_cells = []
    removed_md = 0
    for c in cells:
        if c.get("cell_type") == "markdown" and c.get("metadata", {}).get("codex_footnotes"):
            removed_md += 1
            continue
        new_cells.append(c)
    cells = new_cells

    # Per-function footnote bullets
    notes: Dict[str, List[str]] = {
        "forward_backward_fill": [
            "Fill NaNs forward then backward for specified columns.",
            "Returns a copy to avoid mutating input frame.",
        ],
        "align_to_frequency": [
            "Resample by `freq` using ffill/bfill or interpolation.",
            "Keeps `date_col` as a column after reset_index().",
        ],
        "moving_average_filter": [
            "Simple rolling mean denoiser with min_periods=1.",
        ],
        "fourier_lowpass_filter": [
            "Zero high-frequency FFT coefficients beyond cutoff_ratio.",
            "Inverse transform and preserve original index length.",
        ],
        "zscore_normalize": [
            "Standardize to mean 0, std 1; guard for std==0.",
        ],
        "compute_log_returns": [
            "Log difference of levels; first observation is NaN.",
            "Ensure inputs are positive/non-zero upstream to avoid infs.",
        ],
        "volatility_scale": [
            "Scale returns by provided volatility; zero vol -> NaN.",
        ],
        "winsorize": [
            "Clip series at given lower/upper quantiles to reduce outliers.",
        ],
        "rolling_window_features": [
            "Return DataFrame of rolling mean and std.",
        ],
        "compute_rsi": [
            "EWM-based RSI with default 14 periods (0–100).",
        ],
        "compute_macd": [
            "EMA(12/26) MACD, 9-period signal, and histogram.",
        ],
        "compute_bbands": [
            "Rolling mean ± n_std*std; returns mid/upper/lower bands.",
        ],
        "rolling_higher_moments": [
            "Rolling skewness and kurtosis over window (default 60).",
        ],
        "detect_market_regimes": [
            "MA crossover labels: bull (fast>slow), bear (fast<slow), else stagnant.",
        ],
        "regime_specific_normalize": [
            "Z-score within contiguous regime segments; preserves index.",
        ],
        "detect_anomalies_zscore": [
            "Binary flag where |z| exceeds threshold (default 3.0).",
        ],
        "dynamic_correlation": [
            "Rolling correlation with min_periods=window.",
        ],
        "principal_component_features": [
            "Center features, eigendecompose covariance, return top PCs.",
        ],
        "mutual_information_ranking": [
            "Rank features by MI vs target (fallback: |corr|).",
        ],
        "cross_sectional_rank_normalize": [
            "Per-date ranks scaled to (0,1) via average method.",
        ],
        "block_bootstrap": [
            "Sample fixed-size blocks to preserve autocorrelation structure.",
        ],
        "canonicalize_index_frame": [
            "Normalize headers, parse dates, cast numerics, fill, deduplicate.",
        ],
        "pick_primary_column": [
            "Choose value column by common aliases; fallback to first column.",
        ],
        "_apply_lowpass": [
            "Interpolate small gaps, apply Fourier low-pass, restore NaNs.",
        ],
        "classify_name": [
            "Keyword-based mapping to a label; returns default if no match.",
        ],
    }

    fn_def_re = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>[A-Za-z_]\w*)\s*\(")

    inserted = 0
    modified_cells = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            lines = src.splitlines(True)
        else:
            lines = src

        # Remove top-of-cell inline block if present
        original_len = len(lines)
        lines = strip_cell_inline_block(lines)
        removed_inline = original_len - len(lines)

        # Iterate and insert per-function footnotes
        i = 0
        changed = removed_inline > 0
        while i < len(lines):
            m = fn_def_re.match(lines[i])
            if not m:
                i += 1
                continue
            name = m.group("name")
            indent = m.group("indent")
            # Check if previous line already has our footnote marker
            if i > 0 and lines[i-1].lstrip().startswith(f"{FN_MARK_PREFIX}{name}]"):
                i += 1
                continue
            bullets = notes.get(name)
            if bullets:
                block = fn_block(name, bullets)
                # Prepend indentation if function is nested (not expected here)
                block = [indent + ln if ln.strip() else ln for ln in block]
                lines[i:i] = block
                inserted += 1
                changed = True
                i += len(block) + 1
            else:
                i += 1

        if changed:
            modified_cells += 1
            cell["source"] = lines

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Inserted {inserted} per-function footnotes; removed {removed_md} markdown footnote cells; modified {modified_cells} code cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

