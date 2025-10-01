"""Market dynamics analysis toolkit.

This module assembles utilities to:
- load macro, asset, and sentiment time series from the INDEX directory
- study macroeconomic factor influence on asset returns via OLS regressions
- estimate cross-asset lead/lag structure using VAR and causality tests
- detect market regimes with a Gaussian mixture model on returns/volatility
- explore behavioral finance proxies such as volatility and sentiment shocks

Run the module as a script to execute the full workflow using the default
series configuration shipped with the repository.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    from statsmodels.tsa.vector_ar.var_model import VAR
except ImportError:  # pragma: no cover
    sm = None
    RollingOLS = None
    VAR = None

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    GaussianMixture = None
    StandardScaler = None


@dataclass
class SeriesConfig:
    name: str
    path: str
    value_col: Optional[int] = None
    rename: Optional[str] = None
    resample: Optional[str] = None
    transform: Optional[Callable[[pd.Series], pd.Series]] = None

    def resolved_path(self, root: Path) -> Path:
        return root / self.path


def load_excel_series(config: SeriesConfig, data_root: Path) -> pd.Series:
    full_path = config.resolved_path(data_root)
    if not full_path.exists():
        raise FileNotFoundError(f"Missing file for {config.name}: {full_path}")

    raw = pd.read_excel(full_path)
    if raw.empty:
        raise ValueError(f"File {full_path} has no rows")

    columns = list(raw.columns)
    raw = raw.copy()
    date_col = columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])

    data_cols = [col for col in raw.columns[1:] if not raw[col].isna().all()]
    if not data_cols:
        raise ValueError(f"No data columns detected in {full_path}")

    if config.value_col is None:
        target_col = data_cols[-1]
    else:
        idx = config.value_col
        if idx < 0:
            idx = len(data_cols) + idx
        if idx < 0 or idx >= len(data_cols):
            raise IndexError(f"value_col {config.value_col} out of bounds for {config.name}")
        target_col = data_cols[idx]

    series = raw[[date_col, target_col]].dropna()
    series[target_col] = pd.to_numeric(series[target_col], errors="coerce")
    series = series.dropna(subset=[target_col])
    series = series.set_index(date_col)[target_col]
    series = series.sort_index()

    if config.resample:
        series = series.resample(config.resample).last()

    if config.transform:
        series = config.transform(series)

    series.name = config.rename or config.name
    return series


def load_series_frame(configs: Sequence[SeriesConfig], data_root: Path) -> pd.DataFrame:
    series_list = [load_excel_series(cfg, data_root) for cfg in configs]
    frame = pd.concat(series_list, axis=1)
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    return frame


def compute_log_returns(frame: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    levels = frame.ffill()
    levels = levels.resample(freq).last()
    returns = np.log(levels).diff()
    returns = returns.dropna(how="all")
    return returns


def compute_feature_changes(frame: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    levels = frame.ffill()
    levels = levels.resample(freq).last()
    changes = {}
    for col in levels.columns:
        series = levels[col]
        if (series > 0).all():
            change = np.log(series).diff()
        else:
            change = series.diff()
        changes[f"{col}_chg"] = change
    result = pd.DataFrame(changes)
    result = result.dropna(how="all")
    return result


def run_macro_regressions(
    returns: pd.DataFrame,
    macro_features: pd.DataFrame,
    window: Optional[int] = 126,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if sm is None:
        raise ImportError("statsmodels is required for macro regressions")

    macro_models: Dict[str, Dict[str, pd.DataFrame]] = {}
    for asset in returns.columns:
        data = pd.concat([returns[asset], macro_features], axis=1).dropna()
        if data.empty:
            continue
        y = data[asset]
        X = data.drop(columns=[asset])
        X = sm.add_constant(X)
        ols = sm.OLS(y, X).fit()

        rolling_summary = None
        if window and window < len(data) and RollingOLS is not None:
            roll_mod = RollingOLS(y, X, window=window)
            roll_res = roll_mod.fit()
            rolling_summary = roll_res.params

        macro_models[asset] = {
            "coef": ols.params.to_frame(name="coef"),
            "tvalue": ols.tvalues.to_frame(name="tvalue"),
            "pvalue": ols.pvalues.to_frame(name="pvalue"),
            "rsquared": pd.DataFrame({"rsquared": [ols.rsquared]}),
        }
        if rolling_summary is not None:
            macro_models[asset]["rolling_coef"] = rolling_summary
    return macro_models


def estimate_cross_asset_influence(
    returns: pd.DataFrame,
    maxlags: int = 5,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if VAR is None:
        raise ImportError("statsmodels is required for VAR analysis")

    clean = returns.dropna()
    model = VAR(clean)
    fit = model.fit(maxlags=maxlags, ic="aic")

    causality = {}
    for cause in clean.columns:
        for effect in clean.columns:
            if cause == effect:
                continue
            test = fit.test_causality(effect, [cause], kind="f")
            causality_key = f"{cause}->{effect}"
            crit = getattr(test, "critical_value", np.nan)
            causality[causality_key] = pd.DataFrame(
                {
                    "test_stat": [test.test_statistic],
                    "pvalue": [test.pvalue],
                    "crit": [crit],
                }
            )

    fevd = fit.fevd(10)
    fevd_frames = {
        col: fevd.decomp[:, idx, :]
        for idx, col in enumerate(clean.columns)
    }
    fevd_summary = {
        col: pd.DataFrame(
            fevd_frames[col],
            index=[f"h{h}" for h in range(1, fevd_frames[col].shape[0] + 1)],
            columns=[f"{other}_share" for other in clean.columns],
        )
        for col in clean.columns
    }

    impulse = fit.irf(10)
    horizons = pd.Index(range(impulse.irfs.shape[0]), name="horizon")
    impulse_frames = {
        f"{shock}->{target}": pd.DataFrame(
            impulse.irfs[:, target_idx, shock_idx],
            index=horizons,
            columns=["response"],
        )
        for shock_idx, shock in enumerate(clean.columns)
        for target_idx, target in enumerate(clean.columns)
    }

    return {
        "model_summary": {"aic": fit.aic, "bic": fit.bic, "lags": fit.k_ar},
        "causality": causality,
        "fevd": fevd_summary,
        "impulse_responses": impulse_frames,
    }


def detect_market_regimes(
    returns: pd.Series,
    vol_window: int = 21,
    components: int = 3,
) -> pd.DataFrame:
    if GaussianMixture is None or StandardScaler is None:
        raise ImportError("scikit-learn is required for regime detection")

    df = pd.DataFrame({"ret": returns})
    df["vol"] = returns.rolling(vol_window).std()
    df = df.dropna()

    scaler = StandardScaler()
    features = scaler.fit_transform(df)

    gmm = GaussianMixture(n_components=components, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(features)

    df["regime"] = labels

    # Map regimes by mean return ordering (high -> bull, low -> bear)
    regime_map = (
        df.groupby("regime")["ret"].mean().sort_values().reset_index().assign(label=["bear", "neutral", "bull"][:components])
    )
    mapping = dict(zip(regime_map["regime"], regime_map["label"]))
    df["regime_label"] = df["regime"].map(mapping)

    return df


def behavioral_sentiment_analysis(
    returns: pd.Series,
    sentiment: pd.DataFrame,
    high_z: float = 1.5,
    low_z: float = -1.0,
) -> Dict[str, pd.DataFrame]:
    data = pd.concat([returns, sentiment], axis=1).dropna()
    if data.empty:
        raise ValueError("No overlapping data for behavioral analysis")

    z_scores = data.apply(lambda col: (col - col.mean()) / col.std(ddof=0))

    metrics = {}
    for col in sentiment.columns:
        series_z = z_scores[col]
        future = returns.shift(-5)  # 1-week lookahead on business days
        joined = pd.concat([series_z, future], axis=1).dropna()
        joined.columns = ["z", "fwd_ret"]

        high = joined[joined["z"] >= high_z]
        low = joined[joined["z"] <= low_z]

        metrics[col] = pd.DataFrame(
            {
                "avg_fwd_ret": [joined["fwd_ret"].mean()],
                "avg_high_sentiment_ret": [high["fwd_ret"].mean() if not high.empty else np.nan],
                "avg_low_sentiment_ret": [low["fwd_ret"].mean() if not low.empty else np.nan],
                "corr_with_ret": [data[col].corr(returns)],
            }
        )

    z_scores.columns = [f"{c}_z" for c in z_scores.columns]
    return {"summary": pd.concat(metrics, axis=0), "z_scores": z_scores}


def default_config(data_root: Path) -> Dict[str, Sequence[SeriesConfig]]:
    return {
        "assets": [
            SeriesConfig("us_equity", "INDEX/Vanguard 500 Index Fund - VFIAX US Equity US.xlsx"),
            SeriesConfig("us_value", "INDEX/Vanguard Value Index Fund - VVIAX US Equity US.xlsx"),
            SeriesConfig("us_growth", "INDEX/Vanguard Growth Index Fund - VIGAX US Equity US.xlsx"),
            SeriesConfig("balanced", "INDEX/Vanguard Balanced Index Fund - VBIAX US Equity US.xlsx"),
            SeriesConfig("global_infra", "INDEX/Dow Jones Brookfield Global In - DJBGICUT INDEX Index.xlsx"),
            SeriesConfig("reit", "INDEX/FTSE NAREIT All Reits Total Re - FNARTR Index Index.xlsx"),
            SeriesConfig("investment_grade", "INDEX/iBoxx USD Liquid Investment Gr - IBOXIG Index Index.xlsx"),
            SeriesConfig("high_yield", "INDEX/iBoxx USD Liquid High Yield In - IBOXHY Index Index.xlsx"),
            SeriesConfig("crypto", "INDEX/Bloomberg Galaxy Crypto Index - BGCI Index Index.xlsx"),
        ],
        "macro": [
            SeriesConfig("gdp_qoq", "INDEX/QoQ % Change Annualized - GDP CQOQ Index Index.xlsx", resample="QS"),
            SeriesConfig("cpi_yoy", "INDEX/YoY % NSA - CPI YOY Index Index.xlsx"),
            SeriesConfig("core_cpi_yoy", "INDEX/YoY % NSA - CPI XYOY Index Index.xlsx"),
            SeriesConfig("pce_mom", "INDEX/Monthly % Change - PCE CRCH Index Index.xlsx"),
            SeriesConfig("ism_pmi", "INDEX/ISM PMI - NAPMPMI Index Index.xlsx"),
            SeriesConfig("nfp_change", "INDEX/Net Change SA - NFP TCH Index Index.xlsx"),
            SeriesConfig("unemployment", "INDEX/Total SA - USURTOT Index Index.xlsx"),
            SeriesConfig("michigan_sent", "INDEX/Univ. of Michigan Sentiment - CONSSENT INDEX Index.xlsx"),
            SeriesConfig("conference_board", "INDEX/Confidence - CONCCONF INDEX Index.xlsx"),
            SeriesConfig("m2", "INDEX/M2 (NSA) - M2NS Index Index.xlsx"),
            SeriesConfig("ten_year_yield", "INDEX/US Generic Govt 10 Yr - USGG10YR Index Index.xlsx"),
            SeriesConfig("breakeven_10y", "INDEX/BE 10 Year - USGGBE10 Index Index.xlsx"),
            SeriesConfig("fed_funds", "INDEX/Fed Funds Target Rate US - FDTR Index Index.xlsx"),
            SeriesConfig("dollar_index", "INDEX/DOLLAR INDEX SPOT - DXY Index Index.xlsx"),
            SeriesConfig("economic_surprise", "INDEX/Citi Economic Surprise - Unite - CESIUSD INDEX Index.xlsx"),
            SeriesConfig("move_index", "INDEX/ICE BofA MOVE Index - MOVE Index Index.xlsx"),
        ],
        "behavioral": [
            SeriesConfig("vix", "INDEX/Cboe Volatility Index - VIX Index Index.xlsx"),
            SeriesConfig("move", "INDEX/ICE BofA MOVE Index - MOVE Index Index.xlsx"),
            SeriesConfig("sentiment", "INDEX/Univ. of Michigan Sentiment - CONSSENT INDEX Index.xlsx"),
            SeriesConfig("confidence", "INDEX/Confidence - CONCCONF INDEX Index.xlsx"),
        ],
    }


def run_full_workflow(data_root: Path) -> Dict[str, object]:
    config = default_config(data_root)

    assets = load_series_frame(config["assets"], data_root)
    macro = load_series_frame(config["macro"], data_root)
    behavioral = load_series_frame(config["behavioral"], data_root)

    returns = compute_log_returns(assets)
    macro_changes = compute_feature_changes(macro)

    macro_models = run_macro_regressions(returns, macro_changes)

    var_assets = returns[["us_equity", "investment_grade", "high_yield", "crypto", "reit"]]
    cross_asset = estimate_cross_asset_influence(var_assets)

    regimes = detect_market_regimes(returns["us_equity"])

    behavioral_analysis = behavioral_sentiment_analysis(
        returns["us_equity"], behavioral
    )

    return {
        "asset_prices": assets,
        "asset_returns": returns,
        "macro_levels": macro,
        "macro_changes": macro_changes,
        "macro_models": macro_models,
        "cross_asset": cross_asset,
        "regimes": regimes,
        "behavioral": behavioral_analysis,
    }


def format_macro_summary(macro_models: Dict[str, Dict[str, pd.DataFrame]]) -> str:
    lines = []
    for asset, stats in macro_models.items():
        coef = stats["coef"].drop(index="const", errors="ignore").sort_values("coef", ascending=False)
        top = coef.head(5)
        lines.append(f"Asset: {asset}")
        for factor, row in top.iterrows():
            t_val = stats["tvalue"].loc[factor, "tvalue"]
            p_val = stats["pvalue"].loc[factor, "pvalue"]
            lines.append(
                f"  {factor}: coef={row['coef']:.4f}, t={t_val:.2f}, p={p_val:.3f}"
            )
        rsq = stats["rsquared"].iloc[0, 0]
        lines.append(f"  R^2: {rsq:.3f}")
    return "\n".join(lines)


def format_behavioral_summary(behavioral: Dict[str, pd.DataFrame]) -> str:
    summary = behavioral["summary"].droplevel(1)
    lines = []
    for factor, row in summary.iterrows():
        lines.append(
            f"{factor}: fwd={row['avg_fwd_ret']:.4%}, high={row['avg_high_sentiment_ret']:.4%},"
            f" low={row['avg_low_sentiment_ret']:.4%}, corr={row['corr_with_ret']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run market dynamics analysis workflow")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to repository root containing INDEX folder",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    results = run_full_workflow(args.data_root)

    logging.info("Loaded %d asset series", results["asset_prices"].shape[1])
    logging.info("Macro regression summary:\n%s", format_macro_summary(results["macro_models"]))
    logging.info("Behavioral signals:\n%s", format_behavioral_summary(results["behavioral"]))
    logging.info(
        "Latest regime label: %s", results["regimes"].iloc[-1]["regime_label"]
    )


if __name__ == "__main__":
    main()
