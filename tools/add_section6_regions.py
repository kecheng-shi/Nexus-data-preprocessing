import json
from pathlib import Path

NB_PATH = Path("Nexus Data Analysis.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l if l.endswith("\n") else l + "\n" for l in lines],
    }


def already_has_section(cells: list[dict]) -> bool:
    for c in cells:
        if c.get("cell_type") == "markdown":
            src = "".join(c.get("source", []))
            if src.strip().startswith("# 6.") and "Regional" in src:
                return True
    return False


def build_cells() -> list[dict]:
    cells: list[dict] = []
    # Header
    cells.append(md("# 6. Regional Asset-Class Correlations"))

    # 6.1 Region mapping
    cells.append(md("## 6.1 Region Mapping and Panel Build"))
    cells.append(
        code(
            [
                "# Build region-level asset-class panels from instrument returns and metadata.",
                "import re",
                "from pathlib import Path",
                "import numpy as np",
                "import pandas as pd",
                "from IPython.display import display, Markdown",
                "",
                "REGION_LOOKUP_PATH = Path('Nexus Data Lists with VBA.xlsm')",
                "",
                "def _normalize_region_key(text: str) -> str:",
                "    return re.sub(r'[^a-z0-9]+', '', str(text).lower())",
                "",
                "def load_region_lookup(path: Path) -> dict[str, str]:",
                "    if not path.exists():",
                "        print(f'Region lookup workbook missing: {path}')",
                "        return {}",
                "    try:",
                "        region_df = pd.read_excel(path, sheet_name='Universe')",
                "    except Exception as exc:  # noqa: BLE001",
                "        print(f'Failed to load region lookup from {path}: {exc}')",
                "        return {}",
                "    region_df = region_df[['Ticker', 'TickerSuffix', 'Region']].dropna(subset=['Ticker', 'Region'])",
                "    lookup: dict[str, str] = {}",
                "    for _, row in region_df.iterrows():",
                "        ticker = str(row['Ticker']).strip()",
                "        suffix = str(row['TickerSuffix']).strip() if pd.notna(row['TickerSuffix']) else ''",
                "        region = str(row['Region']).strip()",
                "        if not ticker:",
                "            continue",
                "        candidates = {ticker, f\"{ticker} {suffix}\".strip()} if suffix else {ticker}",
                "        for candidate in filter(None, candidates):",
                "            lookup.setdefault(_normalize_region_key(candidate), region)",
                "    return lookup",
                "",
                "REGION_LOOKUP = load_region_lookup(REGION_LOOKUP_PATH)",
                "",
                "# Guard: ensure upstream artifacts exist.",
                "if 'asset_returns' not in globals() or 'asset_meta' not in globals():",
                "    display(Markdown('Prerequisites missing: run Section 4.2 asset aggregation first.'))",
                "else:",
                "    unmatched_labels: set[str] = set()",
                "",
                "    def infer_region(label: str) -> str:",
                "        if not isinstance(label, str) or not label:",
                "            return 'Global'",
                "        payload = label.split(' - ', 1)[-1] if ' - ' in label else label",
                "        region = REGION_LOOKUP.get(_normalize_region_key(payload))",
                "        if region is None:",
                "            region = REGION_LOOKUP.get(_normalize_region_key(label))",
                "        if region is None:",
                "            unmatched_labels.add(label)",
                "            return 'Global'",
                "        return region",
                "",
                "    meta_df = pd.DataFrame(asset_meta) if isinstance(asset_meta, list) else pd.DataFrame([])",
                "    if meta_df.empty:",
                "        display(Markdown('No asset metadata available to infer regions.'))",
                "    else:",
                "        meta_df['region'] = meta_df['label'].astype(str).apply(infer_region)",
                "        if unmatched_labels:",
                "            print(f'Region lookup defaulted to Global for {len(unmatched_labels)} instruments (showing up to 10).')",
                "            preview = pd.Series(sorted(unmatched_labels)).to_frame('label')",
                "            display(preview.head(10))",
                "            if preview.shape[0] > 10:",
                "                print('...')",
                "        # Build region-level panels: mean of instrument returns per asset class within region.",
                "        REGION_ASSET_CLASS_PANELS = {}",
                "        REGION_ASSET_INVENTORY = []",
                "        MIN_OBS_REGION = 24",
                "        for region, mgrp in meta_df.groupby('region'):",
                "            class_series = {}",
                "            inv_rows = []",
                "            for cls, sub in mgrp.groupby('asset_class'):",
                "                aliases = [str(a) for a in sub['alias'].dropna().tolist() if a in asset_returns]",
                "                if not aliases:",
                "                    continue",
                "                frame = pd.concat([asset_returns[a] for a in aliases], axis=1)",
                "                series = frame.mean(axis=1, skipna=True)",
                "                # Enforce observation threshold",
                "                if series.dropna().shape[0] >= MIN_OBS_REGION:",
                "                    class_series[cls] = series",
                "                    inv_rows.append({'region': region, 'asset_class': cls, 'members': len(aliases), 'sample_months': int(series.dropna().shape[0])})",
                "            panel = pd.DataFrame(class_series).sort_index().dropna(how='all') if class_series else pd.DataFrame()",
                "            if not panel.empty:",
                "                REGION_ASSET_CLASS_PANELS[region] = panel",
                "                REGION_ASSET_INVENTORY.extend(inv_rows)",
                "        if REGION_ASSET_CLASS_PANELS:",
                "            inv_df = pd.DataFrame(REGION_ASSET_INVENTORY)",
                "            display(Markdown('Included region×asset-class panels (min 24 monthly obs):'))",
                "            display(inv_df.sort_values(['region','asset_class']).reset_index(drop=True))",
                "        else:",
                "            display(Markdown('No region-level panels constructed (insufficient data).'))",
            ]
        )
    )

    # 6.2 Overall correlation by region
    cells.append(md("## 6.2 Overall Correlation by Region"))
    cells.append(
        code(
            [
                "# Compute and plot overall asset-class correlation matrices per region.",
                "import numpy as np",
                "import pandas as pd",
                "import seaborn as sns",
                "import matplotlib.pyplot as plt",
                "from analysis.stats import pairwise_corr",
                "",
                "if 'REGION_ASSET_CLASS_PANELS' not in globals() or not REGION_ASSET_CLASS_PANELS:",
                "    print('Region panels unavailable; run 6.1 first.')",
                "else:",
                "    REGION_CLASS_CORR_RESULTS = {}",
                "    for region, panel in REGION_ASSET_CLASS_PANELS.items():",
                "        if panel.shape[1] < 2:",
                "            print(f'Skipping {region}: fewer than 2 asset classes.')",
                "            continue",
                "        corr_df = pairwise_corr(panel.dropna(how='all'), min_periods=24)",
                "        # Order columns by overall correlation intensity for readability.",
                "        order = corr_df.abs().sum().sort_values(ascending=False).index.tolist()",
                "        corr_df = corr_df.loc[order, order]",
                "        REGION_CLASS_CORR_RESULTS[region] = corr_df",
                "",
                "        fig, ax = plt.subplots(figsize=(12, 9))",
                "        sns.heatmap(corr_df, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.3, cbar_kws={'label': 'Correlation'}, ax=ax)",
                "        ax.set_title(f'Asset-Class Monthly Return Correlation — {region}')",
                "        ax.set_xlabel('Asset class')",
                "        ax.set_ylabel('Asset class')",
                "        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')",
                "        plt.tight_layout()",
                "        plt.show()",
            ]
        )
    )
    cells.append(md("<!-- analysis-comment: region_overall_corrs -->\n\n### Commentary — Region Overall Correlations\n\n- Heatmaps show co-movement across asset classes within each region.\n- Expect stronger within-risk clustering in US/EU; defensives cluster in bonds/cash.\n- Compare ordering across regions to spot structural differences.\n- Use top/bottom pairs in the summary below for diversification ideas.\n"))

    # 6.3 Per-phase correlations by region
    cells.append(md("## 6.3 Per-Phase Correlations by Region"))
    cells.append(
        code(
            [
                "# Join business-cycle phases and compute region-specific correlation matrices per phase.",
                "import numpy as np",
                "import pandas as pd",
                "from collections import OrderedDict",
                "from analysis.stats import pairwise_corr",
                "from analysis.plotting import plot_phase_heatmaps",
                "",
                "# Build or reuse phase series",
                "phase_series = pd.Series(dtype=object)",
                "if 'phase_panel' in globals() and isinstance(phase_panel, pd.DataFrame) and 'phase' in phase_panel.columns:",
                "    tmp = phase_panel.copy()",
                "    tmp.index = pd.to_datetime(tmp.index)",
                "    phase_series = tmp['phase'].dropna()",
                "elif 'INTERPRETATION' in globals() and getattr(INTERPRETATION, 'phase_labels', pd.Series(dtype=object)).empty is False:",
                "    phase_series = INTERPRETATION.phase_labels.dropna()",
                "",
                "if phase_series.empty:",
                "    print('Business-cycle phases unavailable; run Section 4.3 HMM mapping first.')",
                "elif 'REGION_ASSET_CLASS_PANELS' not in globals() or not REGION_ASSET_CLASS_PANELS:",
                "    print('Region panels unavailable; run 6.1 first.')",
                "else:",
                "    REGION_PHASE_CORR_RESULTS = {}",
                "    PHASE_ORDER = ['Contraction', 'Slowdown', 'Recovery', 'Expansion']",
                "    MIN_OBS_PHASE = 3",
                "    for region, panel in REGION_ASSET_CLASS_PANELS.items():",
                "        panel = panel.copy()",
                "        panel.index = pd.to_datetime(panel.index)",
                "        joined = panel.join(phase_series.rename('phase'), how='inner')",
                "        if joined.empty or panel.shape[1] < 2:",
                "            print(f'Skipping {region}: insufficient overlap or classes.')",
                "            continue",
                "        results: OrderedDict[str, pd.DataFrame] = OrderedDict()",
                "        for phase_name in [p for p in PHASE_ORDER if p in joined['phase'].unique().tolist()]:",
                "            mask = joined['phase'] == phase_name",
                "            block = joined.loc[mask, panel.columns.tolist()]",
                "            if block.shape[0] < MIN_OBS_PHASE:",
                "                continue",
                "            corr_df = pairwise_corr(block, min_periods=max(3, MIN_OBS_PHASE))",
                "            cols = corr_df.abs().sum().sort_values(ascending=False).index.tolist()",
                "            corr_df = corr_df.loc[cols, cols]",
                "            results[phase_name] = corr_df",
                "        if results:",
                "            REGION_PHASE_CORR_RESULTS[region] = results",
                "            plot_phase_heatmaps(results, 'coolwarm', vmin=-1, vmax=1, center=0, cbar_label='Correlation', title_prefix=f'Asset-Class Correlations — {region} — ')",
            ]
        )
    )
    cells.append(md("<!-- analysis-comment: region_phase_corrs -->\n\n### Commentary — Region Per-Phase Correlations\n\n- Correlations often strengthen in contractions/slowdowns, weakening in expansions.\n- Note regime asymmetries by region (e.g., stock–bond sign flips more/less pronounced).\n- Use phase-specific breakdowns to stress-test diversification under adverse regimes.\n- Caveat: phases derived from one macro region can imperfectly align with others.\n"))

    # 6.4 Summary
    cells.append(md("## 6.4 Summary"))
    cells.append(
        code(
            [
                "# Summarise strongest and weakest co-movements per region and by phase.",
                "import pandas as pd",
                "from IPython.display import display, Markdown",
                "from analysis.correlations import unique_pair_stats",
                "",
                "def summarise_region_pairs(corr_map, panel_map):",
                "    rows_top = []",
                "    rows_low = []",
                "    for region, corr in corr_map.items():",
                "        pairs = unique_pair_stats(corr, panel_map.get(region))",
                "        if pairs.empty:",
                "            continue",
                "        rows_top.append(pairs.assign(region=region).head(5))",
                "        rows_low.append(pairs.sort_values('correlation').assign(region=region).head(5))",
                "    top = pd.concat(rows_top, ignore_index=True) if rows_top else pd.DataFrame()",
                "    low = pd.concat(rows_low, ignore_index=True) if rows_low else pd.DataFrame()",
                "    return top, low",
                "",
                "if 'REGION_CLASS_CORR_RESULTS' in globals() and REGION_CLASS_CORR_RESULTS:",
                "    top, low = summarise_region_pairs(REGION_CLASS_CORR_RESULTS, REGION_ASSET_CLASS_PANELS)",
                "    if not top.empty:",
                "        display(Markdown('**Top 5 positively correlated asset-class pairs (by region)**'))",
                "        display(top[['region','asset_a','asset_b','correlation','overlap_months']])",
                "    if not low.empty:",
                "        display(Markdown('**Top 5 diversifying (lowest correlation) asset-class pairs (by region)**'))",
                "        display(low[['region','asset_a','asset_b','correlation','overlap_months']])",
                "",
                "if 'REGION_PHASE_CORR_RESULTS' in globals() and REGION_PHASE_CORR_RESULTS:",
                "    pr_rows_top = []",
                "    pr_rows_low = []",
                "    for region, phases in REGION_PHASE_CORR_RESULTS.items():",
                "        for phase_name, corr in phases.items():",
                "            pairs = unique_pair_stats(corr, REGION_ASSET_CLASS_PANELS.get(region))",
                "            if pairs.empty:",
                "                continue",
                "            pr_rows_top.append(pairs.assign(region=region, phase=phase_name).head(5))",
                "            pr_rows_low.append(pairs.sort_values('correlation').assign(region=region, phase=phase_name).head(5))",
                "    if pr_rows_top:",
                "        top_phase = pd.concat(pr_rows_top, ignore_index=True)",
                "        display(Markdown('**Per-phase: top co-movers by region**'))",
                "        display(top_phase[['region','phase','asset_a','asset_b','correlation','overlap_months']])",
                "    if pr_rows_low:",
                "        low_phase = pd.concat(pr_rows_low, ignore_index=True)",
                "        display(Markdown('**Per-phase: diversifiers by region**'))",
                "        display(low_phase[['region','phase','asset_a','asset_b','correlation','overlap_months']])",
            ]
        )
    )

    return cells


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    if already_has_section(cells):
        print("Section 6 already present; no changes.")
        return
    new_cells = build_cells()
    nb["cells"] = cells + new_cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    print(f"Appended {len(new_cells)} cells for Section 6 (regional correlations).")


if __name__ == "__main__":
    main()
