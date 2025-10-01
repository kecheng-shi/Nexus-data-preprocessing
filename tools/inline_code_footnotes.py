#!/usr/bin/env python3
"""
Add detailed footnotes directly into code cells of a Jupyter notebook using
Python comment lines (`# ...`). Also removes previously inserted markdown
footnote cells (identified by `metadata.codex_footnotes`).

Idempotent: adds an inline block only if not already present, marked by
`# [codex-footnotes-inline]` on the first line of the block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

NB_PATH = Path("Index Analysis.ipynb")
INLINE_MARK = "# [codex-footnotes-inline]"


def to_comment_block(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    commented = [INLINE_MARK + "\n"]
    for ln in lines:
        if ln.strip() == "":
            commented.append("#\n")
        else:
            commented.append("# " + ln + "\n")
    commented.append("\n")  # blank line before code
    return commented


def main() -> int:
    if not NB_PATH.exists():
        print(f"Notebook not found: {NB_PATH}")
        return 2

    nb = json.loads(NB_PATH.read_text())
    cells = nb.get("cells", [])

    # Footnote content for each code cell index
    footnotes: Dict[int, str] = {
        1: (
            "Footnotes — Data Loading\n\n"
            "- Scans `INDEX` for `*.xlsx` and loads each into `index_data` keyed by file stem.\n"
            "- Raises `FileNotFoundError` if folder is empty to fail fast.\n"
            "- Uses `pandas.read_excel` with default options; adjust engine/sheet if needed.\n"
            "- No cleaning here — Stage 1 handles canonicalization.\n"
        ),
        3: (
            "Footnotes — Utility Functions\n\n"
            "- forward_backward_fill: bidirectional fill for selected columns.\n"
            "- align_to_frequency: resample to target freq with ffill/bfill/interpolate.\n"
            "- moving_average_filter: rolling mean denoiser.\n"
            "- fourier_lowpass_filter: zero high-frequency FFT components; inverse transform.\n"
            "- zscore_normalize: center and scale (handles std=0).\n"
            "- compute_log_returns: log difference; first value is NaN.\n"
            "- volatility_scale: divide by rolling vol (protects 0 with NaN).\n"
            "- winsorize: clip to quantiles to tame outliers.\n"
            "- rolling_window_features: rolling mean/std.\n"
            "- compute_rsi: EWM-based RSI(14).\n"
            "- compute_macd: EMAs 12/26 with 9-signal; outputs macd/signal/hist.\n"
            "- compute_bbands: 20-window 2σ bands.\n"
            "- rolling_higher_moments: rolling skew/kurt (60).\n"
            "- detect_market_regimes: fast/slow MA crossover -> bull/bear/stagnant.\n"
            "- regime_specific_normalize: z-score within contiguous regime segments.\n"
            "- detect_anomalies_zscore: flag |z| > threshold.\n"
            "- dynamic_correlation: rolling correlation (60).\n"
            "- principal_component_features: PCA via covariance eigendecomposition.\n"
            "- mutual_information_ranking: sklearn MI or abs correlation fallback.\n"
            "- cross_sectional_rank_normalize: per-date uniform [0,1] ranks.\n"
            "- block_bootstrap: sample fixed-size blocks to keep autocorrelation.\n"
        ),
        5: (
            "Footnotes — Stage 1: Canonicalization and Alignment\n\n"
            "- canonicalize_index_frame: normalize headers, parse `date`, numeric cast, b/f fill, deduplicate.\n"
            "- pick_primary_column: select a value column by common aliases.\n"
            "- Align series to business days via forward fill.\n"
            "- Build `aligned_panel` (one primary column per file) and coverage summary.\n"
        ),
        7: (
            "Footnotes — Stage 2–3: Smoothing and Stationarization\n\n"
            "- Denoising: moving average and Fourier low-pass (restore original NaNs).\n"
            "- Stationary transforms: log returns, diffs, and 20-day rolling vol.\n"
            "- Volatility-scaled returns standardize variability across series.\n"
        ),
        9: (
            "Footnotes — Stage 4–5: Features and Anomaly Flags\n\n"
            "- winsorized_returns: outlier mitigation before flagging.\n"
            "- higher_moments_panel: rolling skew/kurt.\n"
            "- technical_features: roll mean/std, RSI, MACD, Bollinger Bands, return moments.\n"
            "- anomaly_flags: 1 if |z| exceeds threshold, else 0.\n"
        ),
        11: (
            "Footnotes — Stage 6–8: PCA, MI, Regimes, Cross-sectional Rank\n\n"
            "- PCA: ffill/bfill returns, ≤5 comps via covariance eigendecomposition.\n"
            "- Mutual Information: rank vs SPX target (fallback = |corr|).\n"
            "- Regimes: MA crossover labels; z-score returns within regimes.\n"
            "- Dynamic corr: rolling corr to SPX if available.\n"
            "- Cross-sectional rank: per-date [0,1] uniform ranks.\n"
        ),
        13: (
            "Footnotes — Stage 9–10: Block Bootstrap and Bundle\n\n"
            "- Block bootstrap: preserves local autocorrelation using fixed-size blocks.\n"
            "- Summaries include mean of means and path dispersion.\n"
            "- `nexus_index_bundle` consolidates all artifacts for downstream use.\n"
        ),
        15: (
            "Footnotes — Macro Impact Diagnostics\n\n"
            "- Tag series by asset/macro buckets via keyword heuristics.\n"
            "- Aggregate to monthly; build asset-class returns and standardized macro factors.\n"
            "- OLS of asset-class returns on macro factors; record betas and R².\n"
            "- Also store simple pairwise correlations.\n"
        ),
        17: (
            "Footnotes — Cross-Asset Lead–Lag\n\n"
            "- Correlate returns with future values (1–5 month leads) to infer leadership.\n"
            "- Output top leader–follower pairs; treat as exploratory (corr ≠ causation).\n"
        ),
        19: (
            "Footnotes — Regime Diagnostics\n\n"
            "- Time share by regime and conditional average returns by regime.\n"
            "- Joined with asset classes for interpretation.\n"
        ),
        21: (
            "Footnotes — Behavioural Proxies\n\n"
            "- Evaluate forward returns after RSI extremes (>70, <30) as a simple proxy.\n"
            "- Extend similarly to other features/anomalies if desired.\n"
        ),
        23: (
            "Footnotes — Quick Summary\n\n"
            "- Summarizes available macro factors, asset groups, strongest macro links, and top leads.\n"
        ),
    }

    # Remove previously inserted markdown footnote cells
    new_cells = []
    removed = 0
    for c in cells:
        if c.get("cell_type") == "markdown" and c.get("metadata", {}).get("codex_footnotes"):
            removed += 1
            continue
        new_cells.append(c)
    cells = new_cells

    # Insert inline comment blocks
    # Note: After removals, original indices may shift. We target by ordinal
    # position among code cells to be robust: map desired code cell ordinal to
    # current list.
    code_cells = [i for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    # Build a map from original absolute index to current position by rank
    # Since we know the original notebook had code cells at indices:
    # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    original_code_order = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    # Compute rank mapping: original index -> ordinal among code cells
    original_to_ordinal = {idx: k for k, idx in enumerate(original_code_order)}

    added = 0
    for original_idx, text in footnotes.items():
        ordinal = original_to_ordinal.get(original_idx)
        if ordinal is None or ordinal >= len(code_cells):
            continue
        cell_pos = code_cells[ordinal]
        cell = cells[cell_pos]
        src = cell.get("source", [])
        # Normalize to list of strings
        if isinstance(src, str):
            src = [src]
        # Skip if already has inline marker
        if src and isinstance(src[0], str) and src[0].startswith(INLINE_MARK):
            continue
        block = to_comment_block(text)
        cell["source"] = block + src
        cells[cell_pos] = cell
        added += 1

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Added inline footnotes to {added} code cell(s); removed {removed} markdown footnote cell(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

