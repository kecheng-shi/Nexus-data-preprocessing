import json
from pathlib import Path

NB_PATH = Path("Nexus Data Analysis.ipynb")


def make_md_cell(key: str, title: str, lines: list[str]) -> dict:
    header = f"<!-- analysis-comment: {key} -->\n\n### {title}\n\n"
    body = "\n".join(lines).rstrip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [header + body],
    }


def cell_has_comment(cells: list[dict], idx: int, key: str) -> bool:
    if idx + 1 >= len(cells):
        return False
    nxt = cells[idx + 1]
    if nxt.get("cell_type") != "markdown":
        return False
    src = "".join(nxt.get("source", []))
    return f"<!-- analysis-comment: {key} -->" in src


def src_startswith(cell: dict, prefix: str) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = "".join(cell.get("source", []))
    src = src.lstrip()
    return src.startswith(prefix)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    insertions: list[tuple[int, dict]] = []

    for i, c in enumerate(cells):
        # 1) NBER overlap diagnostics (Section 4.3)
        if src_startswith(c, "# Diagnostics: NBER overlap"):
            key = "nber_overlap_diagnostics"
            if not cell_has_comment(cells, i, key):
                md = make_md_cell(
                    key,
                    "Commentary — NBER Overlap Diagnostics",
                    [
                        "Purpose: compare model ‘Contraction’ labels with official NBER recessions.",
                        "Reading the chart: the step series shows model contractions; grey bands mark NBER recessions. Strong alignment appears as overlapping blocks.",
                        "Tables: the precision/recall/F1 summarise hit rate; the lead/lag table indicates whether the model tends to move ahead of (positive lag) or behind (negative lag) NBER dates.",
                        "Typical pattern: good overlap around well-known recessions (2001, 2008–09, 2020). Small lead/lag is common due to monthly sampling and mapping.",
                        "Interpretation: high precision but lower recall implies conservative detection (few false positives, more misses); the reverse implies early/lenient signalling.",
                        "Next steps: if lead/lag shows consistent bias, consider shifting labels or adjusting the phase mapping threshold.",
                    ],
                )
                insertions.append((i + 1, md))

        # 2) HMM state probabilities and mapped phases
        if src_startswith(c, "# Visualise HMM state probabilities and labelled phases"):
            key = "hmm_states_and_phases"
            if not cell_has_comment(cells, i, key):
                md = make_md_cell(
                    key,
                    "Commentary — HMM States and Phases",
                    [
                        "Top panel: stacked posterior probabilities across latent HMM states; stable blocks suggest well-separated regimes.",
                        "Bottom panel: mapped business-cycle phases with transition markers. Expect long ‘Expansion’ runs, shorter ‘Contraction’ spells, and intermediate ‘Slowdown/Recovery’.",
                        "Checks: excessive rapid switching or very thin state mass hints at too many states or insufficient smoothing.",
                        "Turning points: black markers flag phase changes; clusters around major macro events are expected.",
                        "Use: these labels drive the per‑phase analysis below (correlations and betas).",
                    ],
                )
                insertions.append((i + 1, md))

        # 3) Asset vs Macro correlations
        if src_startswith(c, "# Correlation between asset returns and macro changes"):
            key = "asset_macro_correlation"
            if not cell_has_comment(cells, i, key):
                md = make_md_cell(
                    key,
                    "Commentary — Asset vs Macro Correlations",
                    [
                        "Matrix shows contemporaneous correlations between macro changes (rows) and asset‑class returns (columns).",
                        "Interpretation: sign indicates co‑movement direction; magnitude signals strength (|corr| close to 1 ⇒ strong relationship).",
                        "Common patterns: risk assets often load positively on growth proxies and negatively on real rate shocks; duration assets skew the opposite.",
                        "Caveats: correlations are not causal; they are sample‑dependent and sensitive to outliers and overlapping information.",
                        "Next steps: consider rolling correlations or partial correlations to test stability and control for confounds.",
                    ],
                )
                insertions.append((i + 1, md))

        # 4) Standardised OLS betas (overall and per‑phase heatmaps)
        if src_startswith(c, "# Standardised OLS betas"):
            key = "standardised_ols_betas"
            if not cell_has_comment(cells, i, key):
                md = make_md_cell(
                    key,
                    "Commentary — Standardised OLS Betas",
                    [
                        "Betas quantify effect sizes of macro changes on asset returns (standardised units). Centered colormap: red/blue denote positive/negative sensitivity.",
                        "Use the R²/observations table to judge model fit and reliability; thin samples or low R² warrant caution.",
                        "Per‑phase panels reveal regime dependence: sensitivities often strengthen in stress regimes and fade in expansions.",
                        "Watch for multicollinearity: similar macro drivers can dilute or flip coefficients; consider sparse or orthogonalised factors if needed.",
                        "Actionable: focus on robust, sign‑consistent betas across phases; treat isolated, low‑obs patterns as tentative.",
                    ],
                )
                insertions.append((i + 1, md))

        # 5) Cross‑asset correlation heatmaps (overall and per‑phase)
        if src_startswith(c, "# Visual diagnostics for overall and per-phase asset-class correlations."):
            key = "cross_asset_correlations"
            if not cell_has_comment(cells, i, key):
                md = make_md_cell(
                    key,
                    "Commentary — Cross‑Asset Correlations",
                    [
                        "Overall heatmap: clusters reveal risk vs defensive groupings; strong within‑cluster correlation is common.",
                        "Stock–bond relationship can be regime‑dependent: often negative in disinflationary periods, less so when inflation risk dominates.",
                        "Per‑phase matrices help identify diversification that breaks down in contractions and slowdowns.",
                        "Use ordered axes to spot persistent structures; deviations across phases point to regime‑aware allocation opportunities.",
                        "Next steps: add clustering dendrograms or network graphs to visualise community structure.",
                    ],
                )
                insertions.append((i + 1, md))

    # Apply insertions from the end to preserve indices
    for idx, md_cell in sorted(insertions, key=lambda t: t[0], reverse=True):
        cells.insert(idx, md_cell)

    if insertions:
        nb["cells"] = cells
        NB_PATH.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        print(f"Inserted {len(insertions)} commentary cell(s).")
    else:
        print("No new commentary cells added (already present).")


if __name__ == "__main__":
    main()

