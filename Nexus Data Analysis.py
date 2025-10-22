#!/usr/bin/env python
# coding: utf-8

# # Nexus Data Analysis[^refactor]
# _Kecheng Shi_[^author]
# 
# ---
# 
# [^refactor]: Utility helpers now live in reusable modules under `analysis/` for cleaner notebooks.
# [^author]: Maintained by Kecheng Shi; see the project README for more context.
# 

# # 1. Data Inventory[^inventory]
# 
# ---
# 
# [^inventory]: Source files reside in `FULL Nexus Data`; preprocessing outputs live in `analysis/preprocessed`.
# 

# ## 1.1 Category Snapshot
# 

# - **Equities** (52) — Vanguard Total Stock Market In - VTSAX US Equity US, Home Depot Inc_The - HD US Equity Equity, JPMorgan Chase & Co - JPM US Equity Equity, …
# - **Market Indices** (23) — Morningstar LSTA US Leveraged - SPBDLL Index Index, US Financial Conditions FCON - BFCIUS INDEX Index, Bloomberg US Govt Inflation-Li - BCIT1T Index Index, …
# - **Futures & Forwards** (16) — CSI 300 IDX FUTUR Dec25 - IFBZ5 Index Index, Generic 1st 'W ' Future - W 1 Comdty Comdty, Generic 1st 'GX' Future - GX1 Index Index, …
# - **Macro Indicators** (15) — Manufacturing SA - CPMINDX INDEX Index, YoY % NSA - CPI YOY Index Index, Month % change - DGNOCHNG Index Index, …
# - **Funds & ETFs** (13) — Vanguard Balanced Index Fund - VBIAX US Equity US, NYLI Hedge Multi-Strategy Trac - QAI US EQUITY Index, SPDR Gold Shares - GLD US Equity Equity, …
# - **FX & Rates** (12) — AUSTRALIAN DOLLAR 1 MO - AUD1M BGN Curncy BGN Curncy, Bloomberg Nominal USD 5Y Spot - BTSIUS05 Index Curncy, EURO 3 MO - EUR3M BGN Curncy BGN Curncy, …
# - **Convertible Credit** (10) — ETSY 0 1_8 10_01_26 - ETSY 0.125 10_01_26 Corp Corp (Conv), UBER 0 7_8 12_01_28 - UBER 0.875 12_01_28 Corp Corp (Conv), AMD 3.924 06_01_32 - AMD 3.924 06_01_32 Corp Corp (Conv), …
# - **Commodities** (9) — US DOLLAR_China Offshore Spot - USDCNH Comdty Comdty, US 10YR FUT OPTN  Dec25C   112 - TYZ5C 112 Comdty Comdty, 3 Month SOFR Opt  Dec25P    95 - SFRZ5P 95 Comdty Comdty, …
# - **Options & Derivatives** (9) — International Business Machine - IBM US 12_19_25 C150 Equity Equity, S&P 500 INDEX - SPX 12_19_25 P6000 Index Index, Shanghai Stock Exchange SSE 50 - SSE50 9 C3000 INDEX Index, …
# - **Corporate Credit** (8) — CAT 3 1_4 09_19_49 - CAT 3.25 09_19_49 Corp Corp, T 4 1_2 05_15_35 - T 4.50 05_15_35 Corp Corp, CCO 7 1_2 06_01_29 - CCO 7.50 06_01_29 144A Corp Corp, …
# - **Volatility Indices** (5) — Cboe Volatility Index - VIX 12_17_25 P15 Index Index, ICE BofA MOVE Index - MOVE Index Index, Cboe Volatility Index - VIX Index Index, …
# - **Government Bonds** (4) — SOAF 7.3 04_20_52 - 836205BE3 Govt Govt, BRAZIL 6 04_07_26 - BRAZIL 6 04_07_26 Govt Govt, MEX 6.05 01_11_40 - 91086QAV0 Govt Govt, …
# - **Digital Assets** (3) — Bloomberg Galaxy Crypto Index - BGCI Index Index, Bitcoin_US DOLLAR - XBTUSD BGN Curncy BGN Curncy, Ethereum_US DOLLAR - XETUSD BGN Curncy BGN Curncy
# - **Municipal Bonds** (2) — #N_A Field Not Applicable - 650028ZN6 Muni Muni, #N_A Field Not Applicable - 13063D4D Muni Muni
# - **Credit Derivatives** (1) — #N_A N_A - IBM CDS USD SR 5Y D14 Corp CDS
# 

# ## 1.2 Global Equities & Funds (65 series)
# 

# - Vanguard Total Stock Market In - VTSAX US Equity US — broad U.S. equity mutual fund.
# - SPDR S&P 500 ETF Trust - SPY US Equity Equity — flagship S&P 500 ETF benchmark.
# - Microsoft Corp - MSFT US Equity Equity — U.S. mega-cap technology stock.
# - Alibaba Group Holding Ltd - BABA US Equity Equity — China e-commerce ADR listed in the U.S.
# - iShares MSCI Emerging Markets - EEM US Equity Equity — diversified emerging markets equity ETF.
# 

# ## 1.3 Market Indices & Macro Indicators (38 series)
# 

# - US Financial Conditions FCON - BFCIUS INDEX Index — Bloomberg U.S. financial conditions composite.
# - Morningstar LSTA US Leveraged - SPBDLL Index Index — U.S. leveraged loan benchmark.
# - Bloomberg US Govt Inflation-Li - BCIT1T Index Index — 1–10 year U.S. TIPS index.
# - Manufacturing SA - CPMINDX INDEX Index — manufacturing PMI diffusion index (seasonally adjusted).
# - Citi Economic Surprise - CESIUSD INDEX Index — macro surprise tracker for the United States.
# 

# ## 1.4 Rates & FX Benchmarks (12 series)
# 

# - AUSTRALIAN DOLLAR 1 MO - AUD1M BGN Curncy BGN Curncy — short-term AUD money-market rate.
# - EURO_US DOLLAR - EURUSD Curncy Curncy — EUR/USD spot exchange rate.
# - Bloomberg Nominal USD 5Y Spot - BTSIUS05 Index Curncy — U.S. Treasury 5-year nominal yield.
# - Fed Funds Target Rate US - FDTR Index Index — Federal Funds upper bound target.
# - SHIBOR Fixing 3M - SHIF3M INDEX Index — 3-month Shanghai interbank offered rate.
# 

# ## 1.5 Commodities & Futures (25 series)
# 

# - Generic 1st 'CL' Future - CL1 Comdty Comdty — front-month WTI crude oil future.
# - Generic 1st 'NG' Future - NG1 Comdty Comdty — front-month Henry Hub natural gas future.
# - Generic 1st 'GC' Future - GC1 Comdty Comdty — front-month COMEX gold future.
# - Generic 1st 'W ' Future - W 1 Comdty Comdty — front-month CBOT wheat future.
# - CSI 300 IDX FUTUR Dec25 - IFBZ5 Index Index — December 2025 CSI 300 equity index future.
# 

# ## 1.6 Options & Volatility (14 series)
# 

# - S&P 500 INDEX - SPX 12_19_25 P6000 Index Index — SPX put option expiring December 2025.
# - Deutsche Boerse AG German Stoc - DAX 10_17_25 P23000 INDEX Index — DAX put option expiring October 2025.
# - Nikkei 225 - NKY 10 P37250 INDEX Index — Nikkei 225 put option (October tenor).
# - Cboe Volatility Index - VIX Index Index — 30-day implied volatility for the S&P 500.
# - ICE BofA MOVE Index - MOVE Index Index — U.S. Treasury rate volatility benchmark.
# 

# ## 1.7 Credit Complex (25 series)
# 

# - CAT 3 1_4 09_19_49 - CAT 3.25 09_19_49 Corp Corp — Caterpillar long-dated corporate bond.
# - ETSY 0 1_8 10_01_26 - ETSY 0.125 10_01_26 Corp Corp (Conv) — Etsy convertible note.
# - BRAZIL 6 04_07_26 - BRAZIL 6 04_07_26 Govt Govt — Brazil U.S.-dollar sovereign bond.
# - #N_A Field Not Applicable - 650028ZN6 Muni Muni — U.S. municipal bond from the Nexus pack.
# - #N_A N_A - IBM CDS USD SR 5Y D14 Corp CDS — IBM senior 5-year CDS spread.
# 

# ## 1.8 Digital Assets & Alternatives (3 series)
# 

# - Bloomberg Galaxy Crypto Index - BGCI Index Index — broad digital asset market index.
# - Bitcoin_US DOLLAR - XBTUSD BGN Curncy BGN Curncy — Bitcoin spot versus U.S. dollar.
# - Ethereum_US DOLLAR - XETUSD BGN Curncy BGN Curncy — Ethereum spot versus U.S. dollar.
# 

# # 2. Preprocessing Workflow
# 
# Reusable helpers now live under `analysis/`; the notebook focuses on orchestration, diagnostics, and quick spot checks.

# ## 2.1 Setup & Inventory
# 
# - Import shared modules, detect raw Bloomberg exports, and display a quick catalog preview.
# - Central paths: `FULL Nexus Data` for inputs and `analysis/preprocessed` for cached artefacts.
# - Bails out early if no Excel files are found so downstream blocks do not run on empty data.

# ## 2.2 Batch Pipeline
# 
# - Run `preprocess_all` to build or refresh the tidy artefacts (idempotent unless `overwrite=True`).
# - Capture per-series metadata including coverage, volume availability, and feature counts.
# - Surface errors separately for follow-up without stopping the overall run.

# In[36]:


import numpy as np
from typing import Dict, List
import pandas as pd
import polars as pl
from IPython.display import Markdown, display
from pathlib import Path

from analysis.catalogs import ASSET_SERIES, MACRO_SERIES
from analysis.classification import (
    MACRO_KEYWORDS,
    TARGET_CLASSES,
    infer_asset_class,
    is_macro_series,
    slugify,
)
from analysis.data_io import PREPROCESSED_DIR, RAW_DATA_DIR
from analysis.inspection import load_raw_macro_series, preview_preprocessed_macro_series
from analysis.preprocessing import (
    load_preprocessed,
    preprocess_all,
    preprocess_single_series,
)
from analysis.transformations import (
    join_on_date,
    standardized_linear_regression,
    to_monthly_macro,
    to_monthly_return,
)
from analysis.phases import assign_phase, assign_phase_quantile, pick_columns
from analysis.plotting import segments_from_labels
from analysis.stats import pairwise_corr

excel_files = sorted(RAW_DATA_DIR.glob('*.xlsx'))
catalog = pl.DataFrame({
    'series': [f.stem for f in excel_files],
    'raw_file': [f.name for f in excel_files],
})
print(f"Detected {catalog.height} raw Excel files.")
if catalog.height == 0:
    raise FileNotFoundError("No raw Excel files found in 'FULL Nexus Data'.")
display(catalog.head(10))


# ## 2.3 Artefact Snapshot
# 
# - Summarise the metadata table, inspect coverage ranges, and review cached status versus freshly-processed series.
# - Load a representative CSV to validate schema/feature expectations before continuing with analysis.

# In[37]:


meta_table, error_table = preprocess_all(excel_files, overwrite=False)
print(f"Series processed: {meta_table.height} | Errors: {error_table.height}")
if meta_table.height:
    preview = meta_table.sort(["status", "series"]).head(12)
    display(preview)
    coverage = meta_table.with_columns([
        pl.col("start").str.strptime(pl.Date, strict=False).alias("start_date"),
        pl.col("end").str.strptime(pl.Date, strict=False).alias("end_date"),
    ]).select([
        pl.col("start_date").min().alias("min_start"),
        pl.col("start_date").max().alias("max_start"),
        pl.col("end_date").min().alias("min_end"),
        pl.col("end_date").max().alias("max_end"),
    ])
    display(coverage)
    vol_counts = meta_table.groupby("has_volume", maintain_order=True).agg(pl.len().alias("count"))
    display(vol_counts)
    status_counts = meta_table.groupby("status", maintain_order=True).agg(pl.len().alias("count"))
    display(status_counts)
if error_table.height:
    display(error_table)


# In[38]:


if "meta_table" not in globals():
    raise NameError("Run the preprocessing cell (index 29) before previewing the outputs.")

meta_sorted = meta_table.sort("series")
display(meta_sorted.to_pandas())

csv_paths = meta_sorted["csv_path"].to_list()
if csv_paths:
    sample_frame = pl.read_csv(Path(csv_paths[0]), try_parse_dates=True)
    display(sample_frame.to_pandas())
else:
    print("No preprocessed CSV files found.")


# # 3. Macro & Asset Preparation
# 
# Bridge the preprocessed artefacts into macro and asset panels ready for factor analysis.

# In[39]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ## 3.1 Macro Series Overview
# 
# Key macro drivers tracked via `MACRO_SERIES` and available through the inspection helpers.

# - `gdp_qoq`: Annualised quarter-over-quarter change in US real GDP, a broad growth gauge.
# - `cpi_yoy`: Headline CPI year-over-year percentage change, tracking consumer price inflation.
# - `fed_funds`: Upper bound of the Federal Funds target rate, proxying US monetary policy stance.
# - `ism_pmi`: ISM Manufacturing PMI diffusion index, signalling factory-sector momentum (50 = neutral).
# - `financial_conditions`: Bloomberg US Financial Conditions Index, summarising credit, rates, and equity stress.
# - `citi_surprise`: Citi US Economic Surprise Index, measuring data releases versus economist expectations.
# - `manufacturing_sa`: US manufacturing activity index (seasonally adjusted), reflecting industrial production.
# - `ppi_goods`: Producer Price Index for goods, YoY change, capturing upstream inflation pressure.
# - `pce_change`: Personal Consumption Expenditures monthly percent change, proxying consumer spending momentum.
# - `consumer_sentiment`: University of Michigan consumer sentiment headline index, a survey-based demand indicator.
# - `payroll_change`: Nonfarm payrolls net change (thousands), measuring labour market expansion.
# - `m2_money`: M2 money supply (not seasonally adjusted), highlighting liquidity trends.
# - `cb_confidence`: Conference Board consumer confidence headline index, an alternative sentiment gauge.
# - `industrial_production`: Industrial production year-over-year percent change, tracking output across factories, mines, and utilities.
# - `retail_sales`: US retail sales monthly percent change, reflecting discretionary spending patterns.
# 

# In[40]:


preprocessed_macro_keys = None  # e.g. ["gdp_qoq", "cpi_yoy"] to focus the preview
preprocessed_macro_previews = preview_preprocessed_macro_series(keys=preprocessed_macro_keys, n_rows=5)

for entry in preprocessed_macro_previews:
    display(Markdown(f"### {entry['label']} (preprocessed)\n`{entry['key']}` head"))
    display(entry["preview"])


# # 4. Macroeconomic Drivers
# Objective: identify recurring economic periods from macro data and study how macro changes transmit into returns across asset classes. We operate at a monthly horizon to reduce noise and align disparate release cadences.
# Inputs: preprocessed macro series and all instruments from Part 1 (see `analysis/preprocessed`). Outputs: (a) per-class monthly return panel, (b) macro-change feature panel, (c) correlation/OLS sensitivity tables, (d) business-cycle phase assignments, and (e) phase-conditioned asset returns.
# Scope & caveats:
# - Monthly aggregation smooths event-time noise but can understate fast dynamics.
# - OLS betas are in-sample associations; multicollinearity among macro factors can dilute interpretability.
# - Phase thresholds are descriptive; experiment with alternative cut-offs or overlay policy regimes if desired.
# 

# ### Workflow Summary
# - Build monthly macro signals (`level` and `mom %`) from `MACRO_SERIES`.
# - Infer asset classes from the preprocessed universe and compute equal-weighted monthly returns.
# - Derive business-cycle regimes from composite macro z-scores for a unified phase view.
# - Measure macro transmission within each phase via standardised OLS betas, R², and conditional averages.
# - Visualise correlations and phase diagnostics to compare macro influence across assets and economic periods.

# ## 4.1 Monthly Macro Signals
# 
# Method
# - Collapse each macro series to month‑end (`to_monthly_macro`): record latest `level` and compute monthly percent change (`mom %`).
# - Keep only overlapping months across macro factors (inner join) to ensure consistent samples for clustering/regression.
# 
# Why monthly?
# - Aligns monthly releases (CPI, retail sales, PMI) and reduces high‑frequency market noise.
# - Quarterly series (e.g., GDP) are represented by their last available month within the quarter; the MoM transformation naturally yields sparse non‑zeros.
# 
# Data hygiene
# - Features are z‑scored downstream per column before clustering or OLS.
# 
# 

# In[41]:


# Assemble monthly macro signals
macro_monthly = [
    to_monthly_macro(load_preprocessed(cfg["stem"]), alias).rename({
        f"{alias}_level": f"{cfg['label']} (level)",
        f"{alias}_change": f"{cfg['label']} (mom %)",
    })
    for alias, cfg in MACRO_SERIES.items()
]
macro_panel = join_on_date(macro_monthly, how="inner")
macro_change_cols = [f"{cfg['label']} (mom %)" for cfg in MACRO_SERIES.values()]
macro_pd = (
    macro_panel.to_pandas()
    .assign(date=lambda df: pd.to_datetime(df["date"]))
    .set_index("date")
    .sort_index()
)
macro_changes_pd = macro_pd[macro_change_cols].dropna(how="any")


# ## 4.2 Asset-Class Monthly Returns
# 
# Universe & mapping
# - Scan `analysis/preprocessed` for all instruments; exclude macro series via stem/keyword heuristics.
# - Infer Part 1 asset classes from file naming conventions (e.g., `Equity`, `Comdty`, `Curncy`, `Govt`, options keywords).
# 
# Aggregation
# - For each class, compute equal‑weighted monthly returns from member series.
# - Rationale: consistent and transparent when market‑cap weights are unavailable; avoids single‑name dominance.
# 
# Sampling & filters
# - Require ≥ 24 months for inclusion (`MIN_OBS`) to stabilise correlations/OLS.
# - Membership is dynamic (dictated by data availability); the inventory table reports sample span and top examples.
# 
# 

# In[42]:


from collections import defaultdict

pre_paths = sorted(PREPROCESSED_DIR.glob('*_preprocessed.parquet')) or sorted(PREPROCESSED_DIR.glob('*_preprocessed.csv'))

alias_registry: set[str] = set()
asset_returns: dict[str, pd.Series] = {}
asset_meta: list[dict[str, str]] = []
asset_class_members: defaultdict[str, list[str]] = defaultdict(list)
asset_class_labels: defaultdict[str, list[str]] = defaultdict(list)

seen: set[str] = set()
for p in pre_paths:
    stem = p.name
    stem = stem.removesuffix('_preprocessed.parquet').removesuffix('_preprocessed.csv')
    if stem in seen:
        continue
    seen.add(stem)
    if is_macro_series(stem):
        continue
    asset_class = infer_asset_class(stem)
    if asset_class not in TARGET_CLASSES:
        continue
    alias = slugify(stem, alias_registry)
    try:
        monthly = to_monthly_return(load_preprocessed(stem), alias)
    except FileNotFoundError:
        continue
    mpd = (
        monthly.to_pandas()
        .assign(date=lambda df: pd.to_datetime(df['date']))
        .set_index('date')
        .sort_index()[alias]
    )
    if mpd.dropna().empty:
        continue
    asset_returns[alias] = mpd
    asset_meta.append({'alias': alias, 'stem': stem, 'asset_class': asset_class, 'label': stem})
    asset_class_members[asset_class].append(alias)
    asset_class_labels[asset_class].append(stem)

asset_class_series: dict[str, pd.Series] = {}
for cls, aliases in asset_class_members.items():
    if not aliases:
        continue
    frame = pd.concat([asset_returns[a] for a in aliases], axis=1)
    asset_class_series[cls] = frame.mean(axis=1, skipna=True)

asset_class_panel = pd.DataFrame(asset_class_series).sort_index().dropna(how='all')
asset_class_panel = asset_class_panel.reindex(sorted(asset_class_panel.columns), axis=1)

MIN_OBS = 24
asset_class_cols = [c for c in asset_class_panel.columns if asset_class_panel[c].dropna().shape[0] >= MIN_OBS]
asset_class_panel = asset_class_panel[asset_class_cols]
asset_class_members = {c: asset_class_members[c] for c in asset_class_cols}
asset_class_labels = {c: asset_class_labels[c] for c in asset_class_cols}

combined_panel = asset_class_panel.join(macro_changes_pd, how='inner').sort_index()

asset_class_inventory = []
for cls in asset_class_cols:
    s = asset_class_panel[cls].dropna()
    examples = '; '.join(asset_class_labels.get(cls, [])[:3])
    asset_class_inventory.append({
        'asset_class': cls,
        'series_count': len(asset_class_members.get(cls, [])),
        'sample_months': int(s.shape[0]),
        'sample_start': s.index.min().strftime('%Y-%m') if not s.empty else '',
        'sample_end': s.index.max().strftime('%Y-%m') if not s.empty else '',
        'representative_series': examples,
    })
asset_class_inventory_df = pd.DataFrame(asset_class_inventory).sort_values('asset_class')
print('Asset classes included in the macro-driver study:')
display(asset_class_inventory_df)


# ## 4.3 Business-Cycle Phase Classification
# 

# ### 4.3 Hidden Markov Model: Detailed Methodology
# 
# Data & Preprocessing
# - Observations are monthly macro changes at month-end: $X_{1:T} = (\mathbf{x}_1,\dots,\mathbf{x}_T)$.
# - Standardise per column: $z_{t,j} = (x_{t,j}-\mu_j)/\Sigma_j$; drop NaN/±∞ and extreme outliers; optional 3-month mean smoothing.
# - Feature sets used: growth (GDP QoQ, PMI, industrial production, retail sales, payrolls); conditions/sentiment (financial conditions, confidence, sentiment, surprise); inflation/policy (CPI/PPI/PCE, Fed Funds).
# 
# Generative Model
# - Latent state $S_t \in \{1,\dots,K\}$ is a 1st-order Markov chain with transition matrix $A=[a_{ij}]$, $a_{ij}=\Pr(S_t=j\mid S_{t-1}=i)$ and initial distribution $\pi$.
# - Emission: Gaussian HMM, $\mathbf{z}_t\mid S_t=k \sim \N(\boldsymbol{mu}_k, \igma_k)$ with diagonal or full $\igma_k$.
# - Joint: $p(X_{1:T},S_{1:T})=\pi_{S_1}\,\N(\mathbf{z}_1\mid \mu_{S_1},\igma_{S_1})\,\prod_{t=2}^T a_{S_{t-1},S_t}\,\N(\mathbf{z}_t\mid \mu_{S_t},\igma_{S_t})$.
# 
# Inference & Training (EM)
# - Forward–backward (E-step) gives $\gamma_t(k)=\Pr(S_t=k\mid X_{1:T})$ and $\xi_t(i,j)=\Pr(S_{t-1}=i,S_t=j\mid X_{1:T})$.
# - M-step:
#   - $\pi_k\leftarrow\gamma_1(k)$; $a_{ij}\leftarrow\sum_t \xi_t(i,j)/\sum_t \gamma_t(i)$.
#   - $\mu_k\leftarrow \sum_t \gamma_t(k)\,\mathbf{z}_t/\sum_t\gamma_t(k)$.
#   - $\igma_k\leftarrow \sum_t \gamma_t(k)(\mathbf{z}_t-\mu_k)(\mathbf{z}_t-\mu_k)^\top/\sum_t\gamma_t(k)$ (diag/full accordingly).
# - Viterbi path (decoding): $\delta_t(j)=\max_i\,\delta_{t-1}(i)a_{ij}\,\b_j(\mathbf{z}_t)$ with backpointers $\psi_t(j)$; $\hat{s}_{1:T}$ is the most-likely state sequence.
# 
# Model Selection (BIC/AIC)
# - Parameter count $k$ (dimension $d$, states $K$): initial $(K-1)$, transitions $K(K-1)$, means $K\,d$, covariance diag $K\,d$ or full $K\,\frac{d(d+1)}{2}$ (tied $\frac{d(d+1)}{2}$; spherical $K$).
# - $\mathrm{BIC} = k\,\log T - 2\log\hat{\cal{L}}$; we select $K\in\{2,3,4,5\}$ minimising BIC (AIC as fallback).
# 
# Regime Properties
# - Persistence (expected duration): $\E[D_k] = 1/(1-a_{kk})$.
# - Stationary distribution $\bar{\pi}$ solves $\bar{\pi}=\bar{\pi}A$, $\sum_k \bar{\pi}_k=1$.
# 
# Mapping States → Phases
# - Compute regime means $\bar{\mathbf{z}}_k$.
# - Define growth proxies (GDP/PMI/industrial/retail/payroll) and conditions proxies (financial conditions, confidence, sentiment, surprise).
# - Score: $s_k = \operatorname{mean}(\text{growth}_k) - \operatorname{mean}(\text{conditions}_k)$ (treat higher conditions = tighter; invert if already ease).
# - Assign labels by order: lowest $\rightarrow$ Contraction, highest $\rightarrow$ Expansion, middle $\rightarrow$ Slowdown/Recovery (extras $\rightarrow$ Transition).
# 
# Validation & Robustness
# - Compare Contraction vs NBER recessions (precision/recall, lead–lag).
# - Stability across seeds/rolling windows (adjusted Rand index of states).
# - Predictive check: hold-out log-likelihood across $K$ choices.
# - Inspect $\mu_k, \igma_k$ to avoid overlapping/ill-conditioned regimes; prefer `covariance_type='diag'` when $d$ is large.
# 
# Notes & Alternatives
# - Align feature signs: ensure “higher=tighter” for conditions, or invert before scoring.
# - Semi-Markov models add explicit duration distributions; Markov-switching VARs capture dynamics in observables.
# 

# 
# ### Plan: Hidden Markov Model for Business-Cycle Phases
# 
# Goal
# - Infer latent regimes (Expansion, Slowdown, Contraction, Recovery) from monthly macro changes.
# 
# Observables (from `macro_changes_pd`)
# - Growth: `GDP QoQ (annualized)`, `Manufacturing SA`, `Industrial production YoY`, `Retail sales monthly change`, `Nonfarm payroll change`.
# - Inflation & policy: `CPI YoY (headline)`, `PPI goods YoY`, `PCE monthly change`, `Fed Funds target`.
# - Sentiment & conditions: `Michigan sentiment`, `Conference Board confidence`, `US financial conditions`, `Citi economic surprise`.
# 
# Method
# - Feature prep: align to month-end, forward-fill within month if needed, then z-score per feature.
# - Model: Gaussian HMM with diagonal/full covariance; choose `n_states ∈ {2..5}` via BIC/AIC.
# - Decoding: Viterbi most-likely state and smoothed posterior probabilities per date.
# - Labelling: map anonymous states to phases by comparing regime means of growth/conditions.
# - Validation: compare contraction flags to NBER recessions; report regime persistence stats.
# - Output: timeline plot of state probabilities and final phase labels.
# 

# In[43]:


# Hidden Markov model fitting for business-cycle phases
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler

macro_df = macro_changes_pd.copy()

preferred_features = [
    "GDP QoQ (annualized) (mom %)",
    "Manufacturing SA (mom %)",
    "Industrial production YoY (mom %)",
    "Retail sales monthly change (mom %)",
    "Nonfarm payroll change (mom %)",
    "CPI YoY (headline) (mom %)",
    "PPI goods YoY (mom %)",
    "PCE monthly change (mom %)",
    "Fed Funds target (mom %)",
    "Michigan sentiment (mom %)",
    "Conference Board confidence (mom %)",
    "US financial conditions (mom %)",
    "Citi economic surprise (mom %)",
]

available_features = [col for col in preferred_features if col in macro_df.columns]
if len(available_features) < 3:
    available_features = [col for col in macro_df.columns if "(mom %)" in col]
if len(available_features) < 3:
    available_features = list(macro_df.columns)

subset = macro_df[available_features].copy()
subset = subset.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
subset = subset.rolling(3, min_periods=1).mean()
subset = subset.mask(subset.abs() > 1e6)

X = subset.dropna(how="any").sort_index()

HMM_OBJECTS = {
    "model": None,
    "states": pd.Series(dtype=int),
    "posteriors": pd.DataFrame(),
    "selection": pd.DataFrame(),
    "features": available_features,
    "scaler": None,
    "X": X,
    "Xz": pd.DataFrame(),
}

if X.empty:
    print("No macro observations available for HMM after cleaning; check upstream preprocessing.")
else:
    scaler = StandardScaler()
    Xz = pd.DataFrame(
        scaler.fit_transform(X.values),
        index=X.index,
        columns=available_features,
    )
    HMM_OBJECTS["scaler"] = scaler
    HMM_OBJECTS["Xz"] = Xz

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("hmmlearn not installed. Install with `pip install hmmlearn` and rerun this cell.")
    else:
        def make_persistent_init(K: int, stay: float = 0.9):
            start = np.full(K, 1.0 / K)
            trans = np.full((K, K), (1.0 - stay) / (K - 1))
            np.fill_diagonal(trans, stay)
            return start, trans

        def count_params(cov_type: str, n_states: int, n_features: int) -> float:
            start_params = n_states - 1
            trans_params = n_states * (n_states - 1)
            mean_params = n_states * n_features
            if cov_type == "full":
                cov_params = n_states * (n_features * (n_features + 1) / 2)
            elif cov_type == "diag":
                cov_params = n_states * n_features
            elif cov_type == "tied":
                cov_params = n_features * (n_features + 1) / 2
            elif cov_type == "spherical":
                cov_params = n_states
            else:
                raise ValueError(f"Unsupported covariance_type: {cov_type}")
            return float(start_params + trans_params + mean_params + cov_params)

        def fit_hmm_diag_persistent(data: pd.DataFrame, n_states: int, seed: int = 42):
            cov_type = "diag"
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=1000,
                random_state=seed,
                init_params="mc",
                params="stmc",
            )
            startprob, transmat = make_persistent_init(n_states, stay=0.90)
            model.startprob_ = startprob
            model.transmat_ = transmat
            model.fit(data.values)
            logL = float(model.score(data.values))
            k = count_params(cov_type, n_states, data.shape[1])
            n = len(data)
            aic = 2 * k - 2 * logL
            bic = k * np.log(n) - 2 * logL
            return model, {"logL": logL, "aic": aic, "bic": bic}

        selection_rows = []
        hmm_candidates = {}
        candidates = [4]

        for n_states in candidates:
            try:
                model, stats = fit_hmm_diag_persistent(Xz, n_states=n_states)
                hmm_candidates[n_states] = model
                selection_rows.append({"n_states": n_states, **stats})
            except Exception as e:
                print(f"Warning: HMM({n_states}) fit issue: {e}")

        if not selection_rows:
            try:
                model, stats = fit_hmm_diag_persistent(Xz, n_states=3)
                hmm_candidates[3] = model
                selection_rows.append({"n_states": 3, **stats})
            except Exception as e:
                print(f"HMM fit failed: {e}")

        if selection_rows:
            selection_table = pd.DataFrame(selection_rows).set_index("n_states")
            display(selection_table)
            best_n = int(selection_table.index[0])
            hmm_model = hmm_candidates[best_n]
            states = pd.Series(hmm_model.predict(Xz.values), index=Xz.index, name="state")
            post = hmm_model.predict_proba(Xz.values)
            posteriors = pd.DataFrame(post, index=Xz.index, columns=[f"p_state_{i}" for i in range(post.shape[1])])
        else:
            selection_table = pd.DataFrame(columns=["logL", "aic", "bic"])
            hmm_model = None
            states = pd.Series(dtype=int)
            posteriors = pd.DataFrame(index=Xz.index)

        if not states.empty:
            runs = (states != states.shift()).cumsum()
            sizes = states.groupby([states, runs]).size()
            to_fix = sizes[sizes < 2]
            if not to_fix.empty:
                s = states.copy()
                for (lab, r), _ in to_fix.items():
                    idx = (runs == r)
                    prev_label_series = s[idx].shift(1)
                    next_label_series = s[idx].shift(-1)
                    prev_label = prev_label_series.dropna().iloc[0] if not prev_label_series.dropna().empty else None
                    next_label = next_label_series.dropna().iloc[-1] if not next_label_series.dropna().empty else None
                    fill_label = prev_label if prev_label is not None else next_label
                    if fill_label is not None:
                        s[idx] = int(fill_label)
                states = s

        HMM_OBJECTS.update({
            "model": hmm_model,
            "states": states,
            "posteriors": posteriors,
            "selection": selection_table,
        })


# 
# Plan: Label Mapping & Validation
# - Compute regime means for growth/conditions features to interpret states.
# - Heuristic mapping example:
#   - High growth, easy conditions → Expansion
#   - Slowing growth, deteriorating conditions → Slowdown
#   - Negative growth proxy, tight conditions → Contraction
#   - Growth rebounds, easing conditions → Recovery
# - Validate: overlay `Contraction` vs NBER recessions, report hit/miss.
# 

# In[44]:


# Interpret HMM regimes and map to business-cycle phases
import numpy as np
import pandas as pd
from IPython.display import display

states = HMM_OBJECTS.get("states", pd.Series(dtype=int))
Xz = HMM_OBJECTS.get("Xz", pd.DataFrame())

if states.empty or Xz.empty:
    print("HMM states unavailable; run the fitting cell above.")
else:
    regime_means = (
        pd.concat([Xz, states], axis=1)
        .groupby("state")[Xz.columns]
        .mean()
        .sort_index()
    )

    def filter_columns(columns, keywords):
        kw = [k.lower() for k in keywords]
        return [c for c in columns if any(k in c.lower() for k in kw)]

    growth_cols = filter_columns(Xz.columns, ["gdp", "manufacturing", "industrial", "retail", "payroll"])
    cond_cols = filter_columns(Xz.columns, ["financial conditions", "confidence", "sentiment", "surprise"])

    # Score: higher growth and easier conditions => higher score
    def state_score(k):
        g = regime_means.loc[k, growth_cols].mean() if growth_cols else 0.0
        c = -regime_means.loc[k, cond_cols].mean() if cond_cols else 0.0
        return float(g + c)

    ordering = sorted(regime_means.index, key=state_score)
    # Fixed 4-phase mapping by order (handle K!=4 gracefully)
    phase_order = ["Contraction", "Slowdown", "Recovery", "Expansion"]
    if len(ordering) >= 4:
        label_map = {ordering[0]: phase_order[0], ordering[1]: phase_order[1], ordering[-2]: phase_order[2], ordering[-1]: phase_order[3]}
    elif len(ordering) == 3:
        label_map = {ordering[0]: "Contraction", ordering[1]: "Slowdown", ordering[2]: "Expansion"}
    elif len(ordering) == 2:
        label_map = {ordering[0]: "Contraction", ordering[1]: "Expansion"}
    else:
        label_map = {k: "Contraction" for k in ordering}

    phase_labels = states.map(label_map).rename("phase")

    # Regime durations on smoothed states
    run_id = (states != states.shift()).cumsum()
    duration = states.groupby([states, run_id]).size().rename("months")
    duration_stats = duration.groupby(level=0).describe()

    display(regime_means)
    display(duration_stats[["mean", "50%", "min", "max"]])

    phase_panel = pd.concat([
        phase_labels,
        states.rename("state"),
        HMM_OBJECTS.get("posteriors", pd.DataFrame()),
    ], axis=1)

    HMM_OBJECTS.update({
        "label_map": label_map,
        "phase_labels": phase_labels,
        "regime_means": regime_means,
        "duration_stats": duration_stats,
        "phase_panel": phase_panel,
    })


# In[45]:


# Diagnostics: NBER overlap, precision/recall/F1, and lead/lag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

phase_labels = HMM_OBJECTS.get("phase_labels", pd.Series(dtype=object))
if phase_labels.empty:
    print("Phase labels not available. Run the interpretation cell above.")
else:
    idx = pd.to_datetime(phase_labels.index)
    phase = phase_labels.reindex(idx).sort_index()

    # NBER recessions (monthly spans)
    recessions = [
        ("2001-03", "2001-11"),
        ("2007-12", "2009-06"),
        ("2020-02", "2020-04"),
    ]
    usrec = pd.Series(0, index=phase.index, dtype=int)
    for s, e in recessions:
        usrec.loc[(phase.index >= s) & (phase.index <= e)] = 1

    pred = (phase == "Contraction").astype(int)

    def prf(y_true: pd.Series, y_pred: pd.Series):
        tp = int(((y_true==1) & (y_pred==1)).sum())
        fp = int(((y_true==0) & (y_pred==1)).sum())
        fn = int(((y_true==1) & (y_pred==0)).sum())
        tn = int(((y_true==0) & (y_pred==0)).sum())
        precision = tp / (tp + fp) if (tp+fp) else float("nan")
        recall    = tp / (tp + fn) if (tp+fn) else float("nan")
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else float("nan")
        return tp, fp, fn, tn, precision, recall, f1

    tp, fp, fn, tn, precision, recall, f1 = prf(usrec, pred)
    metrics = pd.DataFrame({
        "value": [tp, fp, fn, tn, precision, recall, f1]
    }, index=["tp", "fp", "fn", "tn", "precision", "recall", "f1"]) 
    display(metrics)

    # Lead/lag window
    rows = []
    for lag in range(-6, 7):
        shifted = usrec.shift(-lag, fill_value=0).astype(int)  # negative lag => model leads
        _,_,_,_,p,r,f = prf(shifted, pred)
        rows.append({"lag_months": lag, "precision": p, "recall": r, "f1": f})
    lag_table = pd.DataFrame(rows).set_index("lag_months")
    display(lag_table)

    # Overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    ax.step(idx, pred, where="post", label="Contraction (model)", lw=2)
    for s, e in recessions:
        ax.axvspan(pd.to_datetime(s), pd.to_datetime(e) + pd.offsets.MonthEnd(0), color="gray", alpha=0.2, label="NBER recession")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Contraction vs NBER recessions")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    # De-duplicate legend labels
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", frameon=False)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()


# 
# Outputs & Next Steps
# - Plot smoothed state probabilities and final phase timeline.
# - Sanity-check vs known recessions; adjust feature set or state count if needed.
# - Use `phase` as a categorical driver in 4.4/5 (betas, performance by regime).
# 

# In[46]:


# Visualise HMM state probabilities and labelled phases
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

posteriors = HMM_OBJECTS.get("posteriors", pd.DataFrame())
phase_labels = HMM_OBJECTS.get("phase_labels", pd.Series(dtype=object))

if posteriors.empty or phase_labels.empty:
    print("HMM outputs unavailable; run the previous cells.")
else:
    posteriors = posteriors.sort_index()
    phase_series = phase_labels.sort_index().fillna("Unknown phase")

    dates = pd.to_datetime(posteriors.index)
    # Desired logical order: worst at bottom -> best at top
    desired_order = ["Contraction", "Slowdown", "Recovery", "Expansion"]
    phases_present = list(pd.Categorical(phase_series).categories)
    ordered_phases = [p for p in desired_order if p in phases_present]
    ordered_phases += [p for p in phases_present if p not in ordered_phases]

    phase_cat = phase_series.astype("category").cat.reorder_categories(ordered_phases, ordered=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: smoothed state probabilities
    cmap_states = plt.get_cmap("tab10")
    state_colors = [cmap_states(i % cmap_states.N) for i in range(posteriors.shape[1])]
    axes[0].stackplot(dates, posteriors.T, colors=state_colors, alpha=0.85)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("HMM smoothed state probabilities")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].legend(
        [f"state {i}" for i in range(posteriors.shape[1])],
        loc="upper left",
        ncol=max(1, posteriors.shape[1] // 2 + 1),
        fontsize=8,
        frameon=False,
    )

    # Bottom: mapped phases as step lines in logical vertical order
    axes[1].set_title("Business-cycle phase (mapped)")
    axes[1].set_ylabel("Phase")
    axes[1].grid(True, axis="y", alpha=0.2)

    phase_to_level = {phase: i for i, phase in enumerate(ordered_phases)}
    cmap_phase = plt.get_cmap("Set2")
    phase_colors = {phase: cmap_phase(i % cmap_phase.N) for i, phase in enumerate(ordered_phases)}

    for phase in ordered_phases:
        series = np.where(phase_cat == phase, phase_to_level[phase], np.nan)
        axes[1].step(
            dates,
            series,
            where="post",
            linewidth=2.5,
            color=phase_colors[phase],
            label=phase,
        )

    axes[1].set_yticks(list(phase_to_level.values()))
    axes[1].set_yticklabels(ordered_phases)

    transitions = phase_cat != phase_cat.shift()
    axes[1].scatter(
        dates[transitions],
        [phase_to_level[p] for p in phase_cat[transitions]],
        color="black",
        s=18,
        zorder=5,
    )

    # Clearer x-axis: adaptive ticks + minor ticks and vertical guides
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3,6,9,12]))
    axes[1].grid(True, which='major', axis='x', alpha=0.15)
    axes[1].grid(True, which='minor', axis='x', alpha=0.05)

    # Legend outside to avoid overlap
    legend = axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
        title="Phase",
    )
    if legend.get_title():
        legend.get_title().set_fontsize(9)

    # Final formatting
    fig.autofmt_xdate(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])


# ## 4.4 Correlations & Standardised Betas
# 
# Correlation matrix
# - Pearson correlations between asset‑class monthly returns and macro `mom %` changes (pairwise complete).
# - Use as orientation only; correlation does not imply causation and can be regime‑dependent.
# 
# Standardised OLS
# - Regress each asset‑class return on the macro change panel after per‑column z‑scoring (features and target).
# - `beta_std` interprets as: change in asset’s monthly return (in std‑dev units) for a 1‑std move in the macro factor, holding others fixed (in‑sample).
# - Report in‑sample R² and observation counts to contextualise fit quality.
# 
# Caveats
# - Multicollinearity among macro variables can inflate uncertainty and split attribution.
# - No intercept is fit after standardisation; relationships are approximate and descriptive, not predictive.
# 
# 

# In[47]:


# Join asset classes with macro changes
combined_panel = asset_class_panel.join(macro_changes_pd, how="inner").sort_index()

# Correlations across all included classes and macro factors
corr_cols = asset_class_cols + macro_change_cols
corr_input = combined_panel[corr_cols]
correlation_matrix = pairwise_corr(corr_input, min_periods=MIN_OBS)

# Standardised regressions per asset class
beta_records, r2_records, sample_records = [], [], []
for cls in asset_class_cols:
    reg_df = combined_panel[[cls, *macro_change_cols]].dropna()
    sample_records.append({"asset_class": cls, "observations": len(reg_df)})
    if len(reg_df) < MIN_OBS:
        continue
    beta_series, r_sq = standardized_linear_regression(reg_df[macro_change_cols], reg_df[cls])
    r2_records.append({"asset_class": cls, "r_squared": r_sq})
    for factor, val in beta_series.dropna().items():
        beta_records.append({"asset_class": cls, "factor": factor, "beta_std": val})

asset_beta_df = pd.DataFrame(beta_records)
asset_rsq_df = pd.DataFrame(r2_records).merge(pd.DataFrame(sample_records), on="asset_class", how="outer")
asset_rsq_df["observations"] = asset_rsq_df["observations"].fillna(0).astype(int)
asset_beta_matrix = (
    asset_beta_df.pivot_table(index="asset_class", columns="factor", values="beta_std", aggfunc="mean")
    .reindex(index=asset_class_cols)
    .reindex(columns=macro_change_cols, fill_value=0.0)
)

print("Regression fit quality by asset class (observations + R^2):")
display(asset_rsq_df.sort_values("r_squared", ascending=False))


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

asset_labels = asset_class_cols
macro_cols = macro_change_cols

if not asset_labels or not macro_cols:
    raise ValueError("Asset classes or macro features missing; rerun previous cells.")

macro_order = sorted(
    macro_cols,
    key=lambda col: correlation_matrix.loc[asset_labels, col].abs().mean(),
    reverse=True,
)
asset_macro_corr = correlation_matrix.loc[asset_labels, macro_order].round(3).fillna(0.0)
beta_heatmap = asset_beta_matrix.reindex(index=asset_labels, columns=macro_order).fillna(0.0)

fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

sns.heatmap(
    asset_macro_corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    cbar_kws={"label": "Correlation"},
    ax=axes[0],
)
axes[0].set_title("Monthly Correlation: Asset Classes vs Macro Drivers")
axes[0].set_xlabel("Macro driver (ordered by |corr|)")
axes[0].set_ylabel("Asset class")
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

sns.heatmap(
    beta_heatmap,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    cbar_kws={"label": "Std beta"},
    ax=axes[1],
)
axes[1].set_title("Standardised Beta Sensitivity")
axes[1].set_xlabel("Macro driver")
axes[1].set_ylabel("")
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.show()

r2_plot = asset_rsq_df.dropna(subset=["r_squared"])
if not r2_plot.empty:
    r2_plot = r2_plot.set_index("asset_class").reindex(asset_labels).dropna(subset=["r_squared"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=r2_plot, x="r_squared", y="asset_class", color="#1f77b4", ax=ax)
    ax.set_title("Regression R^2 by Asset Class")
    ax.set_xlabel("In-sample R^2")
    ax.set_ylabel("")
    for idx, row in r2_plot.iterrows():
        ax.text(row["r_squared"] + 0.005, idx, f"n={int(row.get('observations', 0))}", va="center", fontsize=9)
    plt.show()
else:
    print("No regression fits available for plotting R^2.")


# ### 4.5 Plot Comments
# 
# - Macro drivers: columns with the highest absolute correlations indicate broad cross-asset movers. Check sign consistency across related classes (e.g., stronger USD and tighter financial conditions often align with weaker equities/credit).
# - Standardised betas: a 1-sigma macro move corresponds to the shown change (in sigma) for each asset class. Compare magnitudes to rank sensitivity; note where signs flip vs the correlation view.
# - Model R²: higher values suggest the macro panel explains more variance for that class; low R² points to missing drivers or idiosyncratic behaviour.
# - Cross-check: differences between correlation and beta can reflect multicollinearity among macro factors—treat attribution directionally rather than literally.
# - Caveats: in-sample metrics; consider rolling windows and stability tests before drawing conclusions.
# 

# # 5. Cross-Asset Interaction

# ## 5.1 Correlation Structure Across Asset Classes
# 
# - Build a clean monthly return panel across the retained asset classes.
# - Summarise the strongest co-movements and the best diversifiers based on historical correlations.

# In[ ]:


CLASS_MIN_OBS = 36

interaction_panel = (
    asset_class_panel.copy()
    .dropna(axis=1, how="all")
    .dropna(axis=0, how="all")
)
interaction_panel = interaction_panel.loc[:, interaction_panel.notna().sum() >= CLASS_MIN_OBS]
interaction_panel = interaction_panel.sort_index()

if interaction_panel.empty:
    raise ValueError("Asset class panel is empty; ensure previous steps produced monthly returns.")

class_corr = pairwise_corr(interaction_panel, min_periods=CLASS_MIN_OBS)
avg_abs_corr = class_corr.abs().mean().sort_values(ascending=False)
ordered_cols = list(avg_abs_corr.index)

upper_mask = np.triu(np.ones_like(class_corr, dtype=bool), k=1)
pair_records: list[dict[str, object]] = []
for i, col_i in enumerate(class_corr.columns):
    for j, col_j in enumerate(class_corr.columns):
        if not upper_mask[i, j]:
            continue
        corr_val = class_corr.iloc[i, j]
        if pd.isna(corr_val):
            continue
        overlap = interaction_panel[[col_i, col_j]].dropna().shape[0]
        pair_records.append(
            {
                "asset_a": col_i,
                "asset_b": col_j,
                "correlation": corr_val,
                "overlap_months": overlap,
            }
        )

pair_stats = (
    pd.DataFrame(pair_records)
    .sort_values("correlation", ascending=False)
    .reset_index(drop=True)
)

top_positive_pairs = pair_stats.head(5)
top_negative_pairs = pair_stats.sort_values("correlation", ascending=True).head(5)

display(Markdown("**Top 5 positively correlated asset-class pairs**"))
display(top_positive_pairs)

display(Markdown("**Top 5 diversifying (lowest correlation) asset-class pairs**"))
display(top_negative_pairs)


# ## 5.2 Correlation Heatmap
# 
# - Visualise the cross-asset correlation matrix to highlight clusters of similar behaviour.

# In[ ]:


if class_corr.dropna(how="all").empty or not ordered_cols:
    print("Correlation matrix is empty; skip the heatmap diagnostic.")
else:
    ordered_corr = class_corr.loc[ordered_cols, ordered_cols]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        ordered_corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Asset-Class Monthly Return Correlation")
    ax.set_xlabel("Asset class")
    ax.set_ylabel("Asset class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ## 5.3 Rolling Relationship Diagnostics
# 
# - Track the stability of key correlations through a 12-month rolling window.

# In[ ]:


ROLLING_WINDOW = 12

if pair_stats.empty:
    print("Insufficient overlap to compute rolling correlations.")
else:
    pair_candidates = pd.concat(
        [
            top_positive_pairs[["asset_a", "asset_b"]],
            pair_stats.sort_values("correlation", ascending=True)[["asset_a", "asset_b"]].head(2),
        ],
        ignore_index=True,
    ).drop_duplicates().head(3)

    if pair_candidates.empty:
        print("No qualifying pair candidates for rolling correlation diagnostics.")
    else:
        fig, axes = plt.subplots(len(pair_candidates), 1, figsize=(12, 3 * len(pair_candidates)), sharex=True)
        if len(pair_candidates) == 1:
            axes = [axes]
        for ax, pair in zip(axes, pair_candidates.itertuples(index=False)):
            subset = interaction_panel[[pair.asset_a, pair.asset_b]].dropna()
            if subset.shape[0] < ROLLING_WINDOW:
                ax.text(0.5, 0.5, "Insufficient history for rolling window", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{pair.asset_a} vs {pair.asset_b}")
                ax.set_ylabel("Correlation")
                continue
            rolling_corr = subset[pair.asset_a].rolling(ROLLING_WINDOW).corr(subset[pair.asset_b])
            ax.plot(rolling_corr.index, rolling_corr, label=f"{pair.asset_a} vs {pair.asset_b}")
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"Rolling {ROLLING_WINDOW}-month correlation: {pair.asset_a} vs {pair.asset_b}")
            ax.set_ylabel("Correlation")
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.show()


# ## 5.4 Lead/Lag Structure
# 
# - Screen for statistically strong leading relationships across asset classes using simple lagged correlations.

# In[ ]:


from analysis.market_dynamics import estimate_cross_asset_influence

lead_lag_results = estimate_cross_asset_influence(interaction_panel, maxlags=3)
lead_table = lead_lag_results.get("top_leads", pd.DataFrame())

if lead_table.empty:
    print("Lead/lag table is empty (insufficient overlapping history).")
else:
    display(Markdown("**Top lead/lag relationships (monthly lag, correlation)**"))
    display(lead_table)


# ### 5.5 Interpretation Notes
# 
# - Use the heatmap and tables to spot clusters that move together versus diversifiers that reduce portfolio variance.
# - Rolling correlations flag where relationships break down, helping stress-test assumptions used in allocation models.
# - Lead/lag hits are descriptive, not predictive; combine them with fundamentals before acting on timing signals.
