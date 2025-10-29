"""Helpers for interpreting and evaluating HMM-derived business-cycle phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .hmm import HMMArtifacts

__all__ = [
    "filter_columns",
    "score_states",
    "interpret_states",
    "build_phase_panel",
    "precision_recall_f1",
    "lead_lag_scores",
]


def filter_columns(columns: Sequence[str], keywords: Iterable[str]) -> list[str]:
    """Return column names that contain any of the supplied keywords."""
    lowered = [k.lower() for k in keywords]
    return [col for col in columns if any(key in col.lower() for key in lowered)]


def score_states(
    regime_means: pd.DataFrame,
    *,
    growth_columns: Sequence[str],
    conditions_columns: Sequence[str],
) -> pd.Series:
    """Compute an ordinal score for each latent state."""

    scores = {}
    for state in regime_means.index:
        growth = regime_means.loc[state, growth_columns].mean() if growth_columns else 0.0
        conditions = regime_means.loc[state, conditions_columns].mean() if conditions_columns else 0.0
        scores[state] = float(growth - conditions)
    return pd.Series(scores)


@dataclass
class InterpretationResult:
    """Container returned by :func:`interpret_states`."""

    label_map: dict[int, str]
    phase_labels: pd.Series
    regime_means: pd.DataFrame
    duration_stats: pd.DataFrame


def interpret_states(
    artifacts: HMMArtifacts,
    *,
    growth_keywords: Sequence[str] = (
        "gdp",
        "manufacturing",
        "industrial",
        "retail",
        "payroll",
    ),
    conditions_keywords: Sequence[str] = (
        "financial conditions",
        "confidence",
        "sentiment",
        "surprise",
    ),
    phase_order: Sequence[str] = ("Contraction", "Slowdown", "Recovery", "Expansion"),
) -> InterpretationResult:
    """Map latent HMM states to human-readable business-cycle phases."""

    states = artifacts.states
    Xz = artifacts.Xz
    if states.empty or Xz.empty:
        return InterpretationResult(
            label_map={},
            phase_labels=pd.Series(dtype=object, name="phase"),
            regime_means=pd.DataFrame(index=[], columns=Xz.columns),
            duration_stats=pd.DataFrame(),
        )

    regime_means = (
        pd.concat([Xz, states], axis=1)
        .groupby("state")[Xz.columns]
        .mean()
        .sort_index()
    )

    growth_cols = filter_columns(Xz.columns, growth_keywords)
    condition_cols = filter_columns(Xz.columns, conditions_keywords)
    scores = score_states(regime_means, growth_columns=growth_cols, conditions_columns=condition_cols)
    ordering = scores.sort_values().index.tolist()

    if len(ordering) >= 4:
        label_map = {
            ordering[0]: phase_order[0],
            ordering[1]: phase_order[1],
            ordering[-2]: phase_order[2],
            ordering[-1]: phase_order[3],
        }
    elif len(ordering) == 3:
        label_map = {ordering[0]: phase_order[0], ordering[1]: phase_order[1], ordering[2]: phase_order[-1]}
    elif len(ordering) == 2:
        label_map = {ordering[0]: phase_order[0], ordering[1]: phase_order[-1]}
    else:
        label_map = {state: phase_order[0] for state in ordering}

    phase_labels = states.map(label_map).rename("phase")

    run_id = (states != states.shift()).cumsum()
    duration = states.groupby([states, run_id]).size().rename("months")
    duration_stats = duration.groupby(level=0).describe()

    return InterpretationResult(
        label_map=label_map,
        phase_labels=phase_labels,
        regime_means=regime_means,
        duration_stats=duration_stats,
    )


def build_phase_panel(
    artifacts: HMMArtifacts,
    interpretation: InterpretationResult,
) -> pd.DataFrame:
    """Combine states, probabilities, and labels into a single panel."""

    phase_labels = interpretation.phase_labels
    posteriors = artifacts.posteriors
    states = artifacts.states

    if phase_labels.empty and states.empty and posteriors.empty:
        return pd.DataFrame()

    pieces = [phase_labels, states.rename("state")]
    if not posteriors.empty:
        pieces.append(posteriors)
    panel = pd.concat(pieces, axis=1)
    if "state" in panel.columns:
        prob_cols = [col for col in panel.columns if col.startswith("p_state_")]

        def state_probability(row) -> float:
            state = row.get("state")
            if pd.isna(state):
                return float("nan")
            col = f"p_state_{int(state)}"
            val = row.get(col, np.nan)
            return float(val) if not pd.isna(val) else float("nan")

        panel["state_probability"] = panel.apply(state_probability, axis=1) if prob_cols else np.nan
    return panel


def precision_recall_f1(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return precision, recall, and F1 statistics for binary signals."""

    df = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    if df.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": np.nan, "recall": np.nan, "f1": np.nan}

    tp = int(((df["true"] == 1) & (df["pred"] == 1)).sum())
    fp = int(((df["true"] == 0) & (df["pred"] == 1)).sum())
    fn = int(((df["true"] == 1) & (df["pred"] == 0)).sum())
    tn = int(((df["true"] == 0) & (df["pred"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def lead_lag_scores(
    reference: pd.Series,
    indicator: pd.Series,
    *,
    max_lag: int = 6,
) -> pd.DataFrame:
    """Compute precision/recall/F1 across a symmetric lead/lag window."""

    results = []
    ref = reference.sort_index()
    ind = indicator.sort_index()
    for lag in range(-max_lag, max_lag + 1):
        shifted = ref.shift(-lag, fill_value=0)
        stats = precision_recall_f1(shifted, ind)
        results.append({"lag_months": lag, **{k: stats[k] for k in ("precision", "recall", "f1")}})
    return pd.DataFrame(results).set_index("lag_months")

