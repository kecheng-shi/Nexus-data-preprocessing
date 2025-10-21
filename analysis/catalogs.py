"""Configuration dictionaries for key Nexus datasets."""

from __future__ import annotations

from typing import Dict

MACRO_SERIES: Dict[str, Dict[str, str]] = {
    "gdp_qoq": {
        "stem": "QoQ % Change Annualized - GDP CQOQ Index Index",
        "label": "GDP QoQ (annualized)",
    },
    "cpi_yoy": {
        "stem": "YoY % NSA - CPI YOY Index Index",
        "label": "CPI YoY (headline)",
    },
    "fed_funds": {
        "stem": "Fed Funds Target Rate US - FDTR Index Index",
        "label": "Fed Funds target",
    },
    "ism_pmi": {
        "stem": "ISM PMI - NAPMPMI Index Index",
        "label": "ISM manufacturing PMI",
    },
    "financial_conditions": {
        "stem": "US Financial Conditions FCON - BFCIUS INDEX Index",
        "label": "US financial conditions",
    },
    "citi_surprise": {
        "stem": "Citi Economic Surprise - Unite - CESIUSD INDEX Index",
        "label": "Citi economic surprise",
    },
    "manufacturing_sa": {
        "stem": "Manufacturing SA - CPMINDX INDEX Index",
        "label": "Manufacturing SA",
    },
    "ppi_goods": {
        "stem": "Goods YoY NSA - PPI YOY Index Index",
        "label": "PPI goods YoY",
    },
    "pce_change": {
        "stem": "Monthly % Change - PCE CRCH Index Index",
        "label": "PCE monthly change",
    },
    "consumer_sentiment": {
        "stem": "Univ. of Michigan Sentiment - CONSSENT INDEX Index",
        "label": "Michigan sentiment",
    },
    "payroll_change": {
        "stem": "Net Change SA - NFP TCH Index Index",
        "label": "Nonfarm payroll change",
    },
    "m2_money": {
        "stem": "M2 (NSA) - M2NS Index Index",
        "label": "M2 money supply",
    },
    "cb_confidence": {
        "stem": "Confidence - CONCCONF INDEX Index",
        "label": "Conference Board confidence",
    },
    "industrial_production": {
        "stem": "Year % change - IP YOY Index Index",
        "label": "Industrial production YoY",
    },
    "retail_sales": {
        "stem": "Monthly % Change - RSTAXMOM Index Index",
        "label": "Retail sales monthly change",
    },
}

ASSET_SERIES: Dict[str, Dict[str, str]] = {
    "spy": {
        "stem": "SPDR S&P 500 ETF Trust - SPY US Equity Equity",
        "label": "US equities (SPY)",
    },
    "us10y": {
        "stem": "US Generic Govt 10 Yr - USGG10YR Index Index",
        "label": "US 10Y Treasury",
    },
    "dxy": {
        "stem": "DOLLAR INDEX SPOT - DXY Index Index",
        "label": "USD index (DXY)",
    },
    "cl1": {
        "stem": "Generic 1st 'CL' Future - CL1 Comdty Comdty",
        "label": "WTI front-month (CL1)",
    },
}

__all__ = ["MACRO_SERIES", "ASSET_SERIES"]

