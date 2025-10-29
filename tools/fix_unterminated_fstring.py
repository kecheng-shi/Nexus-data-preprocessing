import json
from pathlib import Path

NB_PATH = Path("Nexus Data Analysis.ipynb")


def fix_cell_sources(sources):
    out = list(sources)
    # Look for the two-line unterminated f-string pattern and collapse to triple-quoted form.
    for i in range(len(out) - 1):
        line = out[i]
        nxt = out[i + 1]
        lstr = line.lstrip()
        nstr = nxt.strip()
        if (
            lstr.startswith('display(Markdown(f"### ')
            and '(preprocessed)' in lstr
            and '"""' not in lstr
            and nstr.startswith('`{entry[')
            and nstr.endswith('"))\\n')
        ):
            out[i] = (
                "    display(Markdown(f\"\"\"### {entry['label']} (preprocessed)\n"
                "`{entry['key']}` head\"\"\"))\n"
            )
            out.pop(i + 1)
            return out, True
    return out, False


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        new_src, did = fix_cell_sources(src)
        if did:
            cell["source"] = new_src
            changed = True

    if changed:
        NB_PATH.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
        print("Fixed unterminated f-string in display(Markdown(...))")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
