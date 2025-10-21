"""Helpers to classify Nexus series by macro relevance and asset class."""

from __future__ import annotations

import re
from typing import Iterable, Mapping, MutableSet

from .catalogs import MACRO_SERIES

# Keywords and class lists exposed so downstream notebooks can override defaults.
MACRO_KEYWORDS = [
    "gdp",
    "cpi",
    "pce",
    "pmi",
    "payroll",
    "confidence",
    "sentiment",
    "industrial",
    "production",
    "retail",
    "sales",
    "unemploy",
    "housing",
    "orders",
    "manufacturing",
    "surprise",
    "money",
    "m2",
    "inflation",
    "financial conditions",
    "fed funds",
]

TARGET_CLASSES = {
    "Equities",
    "Market Indices",
    "Futures & Forwards",
    "Funds & ETFs",
    "FX & Rates",
    "Convertible Credit",
    "Commodities",
    "Options & Derivatives",
    "Corporate Credit",
    "Volatility Indices",
    "Government Bonds",
    "Digital Assets",
    "Municipal Bonds",
    "Credit Derivatives",
}

__all__ = ["MACRO_KEYWORDS", "TARGET_CLASSES", "is_macro_series", "infer_asset_class", "slugify"]


def is_macro_series(
    name: str,
    macro_series: Mapping[str, Mapping[str, str]] | None = None,
    keywords: Iterable[str] | None = None,
) -> bool:
    """Check whether a preprocessed stem should be treated as macro data."""
    macro_cfg = macro_series or MACRO_SERIES
    macro_stems_lower = {cfg["stem"].lower() for cfg in macro_cfg.values()}
    lowered = name.lower()
    if lowered in macro_stems_lower:
        return True
    keyword_iter = keywords or MACRO_KEYWORDS
    return any(keyword in lowered for keyword in keyword_iter)


def infer_asset_class(name: str) -> str:
    """Heuristic mapping from Bloomberg security description to asset class."""
    lowered = name.lower()
    tail = name.split(" - ")[-1].lower()
    if "cds" in lowered:
        return "Credit Derivatives"
    if "muni" in lowered:
        return "Municipal Bonds"
    if "(conv" in lowered:
        return "Convertible Credit"
    if " corp" in tail and "(conv" not in lowered:
        return "Corporate Credit"
    if "govt" in tail or "treasury" in lowered or tail.endswith(" govt govt"):
        return "Government Bonds"
    if any(x in lowered for x in ["volatility", "move index", "vix"]):
        return "Volatility Indices"
    if "opt" in lowered or re.search(r"\b[cp]\d{2,}\b", tail):
        return "Options & Derivatives"
    if any(x in lowered for x in ["future", "futur"]) or re.search(r"\bfut\b", lowered):
        return "Futures & Forwards"
    if "comdty" in tail and "future" not in lowered and "opt" not in lowered:
        return "Commodities"
    if any(x in tail for x in ["curncy", "swap"]) or any(x in lowered for x in ["shibor", "libor", "estr"]):
        return "FX & Rates"
    if any(x in lowered for x in ["crypto", "bitcoin", "ethereum", "bgci", "xbt", "xet"]):
        return "Digital Assets"
    if tail.endswith(" equity us") or any(x in lowered for x in ["fund", "etf", "trust", "shares"]):
        return "Funds & ETFs"
    if "index index" in tail:
        return "Market Indices"
    if "equity" in tail:
        return "Equities"
    return "Equities"


def slugify(name: str, taken: MutableSet[str]) -> str:
    """Lowercase + dash-stabilise a name while ensuring uniqueness."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "series"
    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{base}_{counter}"
        counter += 1
    taken.add(candidate)
    return candidate

