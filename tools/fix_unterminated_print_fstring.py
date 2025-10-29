import json
from pathlib import Path

NB_PATH = Path("Nexus Data Analysis.ipynb")


def fix_print_newline_pattern(lines):
    out = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        lstr = line.lstrip()
        indent = line[: len(line) - len(lstr)]
        if lstr.startswith('print(f"') and not lstr.rstrip().endswith(')\n') and i + 1 < len(lines):
            nxt = lines[i + 1]
            nstr = nxt.lstrip()
            # Merge when next line carries the content and closes ")
            if nstr.strip().endswith('")'):
                inner = nstr.strip()[:-2]  # drop ")
                out.append(f"{indent}print(f\"{inner}\")\n")
                i += 2
                changed = True
                continue
        out.append(line)
        i += 1
    return out, changed


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    touched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        new_src, did = fix_print_newline_pattern(src)
        if did:
            cell["source"] = new_src
            touched = True
    if touched:
        NB_PATH.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        print("Fixed unterminated print f-strings split across lines.")
    else:
        print("No split print f-strings found.")


if __name__ == "__main__":
    main()

