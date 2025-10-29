import json
from pathlib import Path

NB = Path("Nexus Data Analysis.ipynb")

def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Target the known cell index where the unterminated f-string occurs.
    idx = 29
    cells = nb.get("cells", [])
    if idx >= len(cells):
        print("Target cell index not found; no change.")
        return
    cell = cells[idx]
    if cell.get("cell_type") != "code":
        print("Target is not a code cell; no change.")
        return
    src = cell.get("source", [])
    # Defensive: search for the two-line pattern and replace with triple-quoted variant
    replaced = False
    for i in range(len(src) - 1):
        if src[i].lstrip().startswith("display(Markdown(f\"### ") and "(preprocessed)" in src[i]:
            # Merge with the next line and rewrite
            src[i] = (
                "    display(Markdown(f\"\"\"### {entry['label']} (preprocessed)\n"
                "`{entry['key']}` head\"\"\"))\n"
            )
            src.pop(i + 1)
            replaced = True
            break
    if replaced:
        cell["source"] = src
        NB.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        print("Patched cell to use triple-quoted f-string for Markdown display.")
    else:
        print("Pattern not found; no change.")

if __name__ == "__main__":
    main()

