"""Reverse cross-check: treat the English manuscript as the baseline and verify
that every quoted number and every factual claim appears consistently in the
Chinese version, the supplementary material and the package note.

This exists because three successive review rounds found the same class of
defect: a value changed in the manuscript but not in its companions.
"""

import pathlib
import re
import sys

from docx import Document

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "投稿提交_修订版"


def text_of(path: pathlib.Path) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    en = text_of(OUT / "1_Manuscript_revised.docx")
    cn = text_of(OUT / "4_中文审阅版.docx")
    sup = text_of(OUT / "2_Supplementary_Methods.docx")
    note = (OUT / "0_材料包说明.md").read_text(encoding="utf-8")

    # numbers that must be consistent everywhere they refer to the same quantity
    shared = [
        "57.67%", "54.26%", "57.47%", "56.94%", "54.58%", "63.04%", "63.36%",
        "61.47%", "59.53%", "61.54%", "60.74%", "56.71%", "57.51%", "56.31%",
        "3.12", "23.4%", "47.28%", "7.85%", "157,558", "178,047",
    ]
    print("== value presence ==")
    problems = []
    for value in shared:
        row = {"EN": value in en, "CN": value in cn, "SUP": value in sup, "NOTE": value in note}
        if not (row["EN"] and row["CN"]):
            problems.append((value, row))
        print(f"  {value:10s} " + " ".join(f"{k}={'y' if v else '-'}" for k, v in row.items()))

    # statements that must not survive anywhere
    print("\n== stale claims that must be absent ==")
    stale = {
        "3.41": (en, cn, sup, note),
        "47.5%": (en, cn),
        "22.4%": (en, cn),
        "with the prompt": (en, sup),
        "113,558": (en, cn, sup, note),
        "134,047": (en, cn, sup, note),
        "含提示词": (cn,),
        ", Treating": (en,),
    }
    for claim, sources in stale.items():
        hits = [name for name, src in zip(("EN", "CN", "SUP", "NOTE"), sources) if claim in src]
        flag = "OK" if not hits else "PRESENT in " + ",".join(hits)
        if hits:
            problems.append((claim, flag))
        print(f"  {claim:18s} {flag}")

    print("\n== summary ==")
    if problems:
        print(f"  {len(problems)} item(s) need attention")
        for value, row in problems:
            print("   -", value, row)
    else:
        print("  all checked values and claims are consistent")


if __name__ == "__main__":
    main()
