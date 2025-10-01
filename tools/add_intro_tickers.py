#!/usr/bin/env python3
"""
Insert or update a top-of-notebook markdown introduction listing all tickers
found in the `INDEX` folder (xlsx files). Idempotent via `metadata.codex_intro_tickers`.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("Index Analysis.ipynb")
INDEX_DIR = Path("INDEX")


def build_markdown(index_dir: Path) -> str:
    files = sorted(index_dir.glob("*.xlsx"), key=lambda p: p.name.lower())
    tickers = [p.stem for p in files]
    lines = []
    lines.append("## Dataset Introduction — INDEX Tickers\n")
    lines.append("\n")
    lines.append(f"Scanned `{index_dir.resolve()}` and found {len(tickers)} Excel file(s).\n")
    if not tickers:
        lines.append("\nNo tickers detected. Ensure the `INDEX` folder contains `.xlsx` files.\n")
        return "".join(lines)
    lines.append("\nTickers (from file names):\n")
    lines.extend([f"- {t}\n" for t in tickers])
    return "".join(lines)


def main() -> int:
    if not NB_PATH.exists():
        print(f"Notebook not found: {NB_PATH}")
        return 2
    if not INDEX_DIR.exists():
        print(f"Index directory not found: {INDEX_DIR}")
        return 3

    nb = json.loads(NB_PATH.read_text())
    cells = nb.get("cells", [])

    md_text = build_markdown(INDEX_DIR)
    md_cell = {
        "cell_type": "markdown",
        "id": "codex-intro-tickers",
        "metadata": {"codex_intro_tickers": True},
        "source": md_text.splitlines(True),
    }

    # Find existing intro cell
    intro_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown" and c.get("metadata", {}).get("codex_intro_tickers"):
            intro_idx = i
            break

    if intro_idx is None:
        # Insert at top
        cells.insert(0, md_cell)
        action = "inserted"
    else:
        cells[intro_idx] = md_cell
        action = "updated"

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"{action.capitalize()} intro tickers cell with {len(md_cell['source'])} line(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

