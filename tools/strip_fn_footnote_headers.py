#!/usr/bin/env python3
"""
Remove the header lines of per-function footnotes inserted as comments in
`Index Analysis.ipynb`:

- Lines like: `# [fn-footnote:<name>]`
- Lines like: `# Footnote — <name>` (or with '-', ':')

Keeps the remaining bullet comment lines (e.g., `# - ...`) in place.
Idempotent: running again has no effect once headers are removed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NB_PATH = Path("Index Analysis.ipynb")


def main() -> int:
    if not NB_PATH.exists():
        print(f"Notebook not found: {NB_PATH}")
        return 2

    nb = json.loads(NB_PATH.read_text())
    cells = nb.get("cells", [])

    pat_marker = re.compile(r"^\s*#\s*\[fn-footnote:[^\]]+\]\s*$")
    pat_title = re.compile(r"^\s*#\s*Footnote\s*[—\-\:]\s*.*$")

    modified = 0
    removed_count = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            lines = src.splitlines(True)
        else:
            lines = list(src)
        new_lines = []
        for ln in lines:
            if pat_marker.match(ln) or pat_title.match(ln):
                removed_count += 1
                modified += 1
                continue
            new_lines.append(ln)
        if new_lines != lines:
            cell["source"] = new_lines

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Removed {removed_count} footnote header lines from {modified} code cell(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

