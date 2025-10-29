"""Hidden Markov model helpers for business-cycle phase inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = [
    "HMMArtifacts",
    "prepare_macro_features",
    "fit_persistent_hmm",
    "fit_hmm_candidates",
    "select_best_model",
    "enforce_min_state_span",
    "build_hmm_artifacts",
]


@dataclass
class HMMArtifacts:
    """Bundle of intermediate objects returned by :func:`build_hmm_artifacts`."""

    model: object | None
    states: pd.Series
    posteriors: pd.DataFrame
    selection: pd.DataFrame
    features: list[str]
    scaler: StandardScaler | None
    X: pd.DataFrame
    Xz: pd.DataFrame


def prepare_macro_features(
    macro_changes: pd.DataFrame,
    preferred_features: Sequence[str] | None = None,
    *,
    min_features: int = 3,
    rolling_window: int = 3,
    clip_threshold: float = 1e6,
) -> tuple[pd.DataFrame, list[str]]:
    """Select and clean macro features prior to HMM fitting."""

    if preferred_features:
        available = [col for col in preferred_features if col in macro_changes.columns]
    else:
        available = []

    if len(available) < min_features:
        macro_cols = [col for col in macro_changes.columns if "(mom %)" in col]
        available = macro_cols if len(macro_cols) >= min_features else list(macro_changes.columns)

    subset = macro_changes[available].copy()
    subset = subset.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if rolling_window > 1:
        subset = subset.rolling(rolling_window, min_periods=1).mean()
    if clip_threshold > 0:
        subset = subset.mask(subset.abs() > clip_threshold)
    cleaned = subset.dropna(how="any").sort_index()
    return cleaned, available


def _require_hmm() -> type:
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The `hmmlearn` package is required for HMM phase inference. "
            "Install it with `pip install hmmlearn`."
        ) from exc
    return GaussianHMM


def fit_persistent_hmm(
    data: pd.DataFrame,
    *,
    n_states: int,
    stay: float = 0.90,
    covariance_type: str = "diag",
    max_iter: int = 1000,
    seed: int = 42,
) -> tuple[object, dict[str, float]]:
    """Fit a Gaussian HMM with a persistent prior transition structure."""

    GaussianHMM = _require_hmm()
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=max_iter,
        random_state=seed,
        init_params="mc",
        params="stmc",
    )

    startprob = np.full(n_states, 1.0 / n_states)
    transmat = np.full((n_states, n_states), (1.0 - stay) / max(n_states - 1, 1))
    np.fill_diagonal(transmat, stay)

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.fit(data.values)

    logL = float(model.score(data.values))
    n_features = data.shape[1]
    start_params = n_states - 1
    trans_params = n_states * (n_states - 1)
    if covariance_type == "full":
        cov_params = n_states * (n_features * (n_features + 1) / 2)
    elif covariance_type == "diag":
        cov_params = n_states * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) / 2
    elif covariance_type == "spherical":
        cov_params = n_states
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    mean_params = n_states * n_features

    k = float(start_params + trans_params + mean_params + cov_params)
    n = len(data)
    aic = 2 * k - 2 * logL
    bic = k * np.log(n) - 2 * logL

    return model, {"logL": logL, "aic": aic, "bic": bic}


def fit_hmm_candidates(
    data: pd.DataFrame,
    candidates: Iterable[int],
    *,
    stay: float = 0.90,
    covariance_type: str = "diag",
    max_iter: int = 1000,
    seed: int = 42,
) -> tuple[dict[int, object], pd.DataFrame]:
    """Fit a set of candidate HMMs and return their selection statistics."""

    models: dict[int, object] = {}
    stats: list[dict[str, float]] = []
    for n_states in candidates:
        try:
            model, scores = fit_persistent_hmm(
                data,
                n_states=n_states,
                stay=stay,
                covariance_type=covariance_type,
                max_iter=max_iter,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - defensive
            stats.append({"n_states": n_states, "logL": np.nan, "aic": np.nan, "bic": np.nan, "error": str(exc)})
            continue
        models[n_states] = model
        stats.append({"n_states": n_states, **scores})

    selection = pd.DataFrame(stats).set_index("n_states") if stats else pd.DataFrame()
    return models, selection


def select_best_model(selection: pd.DataFrame, criterion: str = "bic") -> int | None:
    """Return the number of states that minimises the requested criterion."""

    if selection.empty or criterion not in selection.columns:
        return int(selection.index[0]) if not selection.empty else None
    ordered = selection.sort_values(criterion)
    return int(ordered.index[0]) if not ordered.empty else None


def enforce_min_state_span(states: pd.Series, min_length: int = 2) -> pd.Series:
    """Merge regimes shorter than ``min_length`` into neighbouring states."""

    if states.empty or min_length <= 1:
        return states

    cleaned = states.copy()
    run_id = (cleaned != cleaned.shift()).cumsum()
    run_lengths = cleaned.groupby([cleaned, run_id]).size()
    short_runs = run_lengths[run_lengths < min_length]
    if short_runs.empty:
        return cleaned

    for (state_label, run), _ in short_runs.items():
        idx = run_id == run
        prev_state = cleaned.loc[idx].shift(1).dropna().iloc[0] if cleaned.loc[idx].shift(1).dropna().any() else None
        next_state = cleaned.loc[idx].shift(-1).dropna().iloc[-1] if cleaned.loc[idx].shift(-1).dropna().any() else None
        replacement = prev_state if prev_state is not None else next_state
        if replacement is not None:
            cleaned.loc[idx] = int(replacement)
        else:
            cleaned.loc[idx] = int(state_label)
    return cleaned


def build_hmm_artifacts(
    macro_changes: pd.DataFrame,
    *,
    preferred_features: Sequence[str] | None = None,
    candidates: Sequence[int] = (4,),
    stay: float = 0.90,
    min_span: int = 2,
    scaler: StandardScaler | None = None,
    selection_criterion: str = "bic",
    rolling_window: int = 3,
    clip_threshold: float = 1e6,
    min_features: int = 3,
) -> HMMArtifacts:
    """End-to-end helper that mirrors the notebook workflow."""

    X, features = prepare_macro_features(
        macro_changes,
        preferred_features,
        min_features=min_features,
        rolling_window=rolling_window,
        clip_threshold=clip_threshold,
    )

    if X.empty:
        return HMMArtifacts(
            model=None,
            states=pd.Series(dtype=int),
            posteriors=pd.DataFrame(index=macro_changes.index),
            selection=pd.DataFrame(),
            features=features,
            scaler=None,
            X=X,
            Xz=pd.DataFrame(index=macro_changes.index, columns=features),
        )

    if scaler is None:
        scaler = StandardScaler()
        Xz_values = scaler.fit_transform(X.values)
    else:
        Xz_values = scaler.transform(X.values)
    Xz = pd.DataFrame(Xz_values, index=X.index, columns=features)

    models, selection = fit_hmm_candidates(
        Xz,
        candidates,
        stay=stay,
        seed=42,
    )
    best_n = select_best_model(selection, selection_criterion) if not selection.empty else None
    if best_n is None:
        states = pd.Series(dtype=int)
        posteriors = pd.DataFrame(index=X.index)
        model = None
    else:
        model = models.get(best_n)
        if model is None:
            states = pd.Series(dtype=int)
            posteriors = pd.DataFrame(index=X.index)
        else:
            raw_states = pd.Series(model.predict(Xz.values), index=X.index, name="state")
            states = enforce_min_state_span(raw_states, min_span)
            posterior_values = model.predict_proba(Xz.values)
            posteriors = pd.DataFrame(
                posterior_values,
                index=X.index,
                columns=[f"p_state_{i}" for i in range(posterior_values.shape[1])],
            )

    return HMMArtifacts(
        model=model,
        states=states,
        posteriors=posteriors,
        selection=selection,
        features=features,
        scaler=scaler,
        X=X,
        Xz=Xz,
    )
