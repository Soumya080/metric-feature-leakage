"""Check that every \\cite key in the paper resolves to a bib entry, and vice versa.

There is no LaTeX toolchain on the authoring machine, so an unresolved key would
otherwise surface as a "?" in a compiled PDF on Overleaf rather than as an error here.
Run this after touching either file.

Usage:
    python paper/check_refs.py
"""

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main_v2.tex")
BIB = os.path.join(HERE, "references.bib")

CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY_RE = re.compile(r"^@\w+\{([^,]+),", re.M)
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:page)?ref\{([^}]*)\}")
COUNT_RE = re.compile(r"\\begin\{(table|figure)\}")


def main() -> int:
    tex = io.open(TEX, encoding="utf-8").read()
    bib = io.open(BIB, encoding="utf-8").read()

    # Comment lines would otherwise contribute phantom citations.
    tex_body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))

    cited = {k.strip() for m in CITE_RE.findall(tex_body) for k in m.split(",") if k.strip()}
    defined = {m.strip() for m in ENTRY_RE.findall(bib)}

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)

    print(f"cited: {len(cited)}   defined: {len(defined)}")
    if missing:
        print(f"\nBROKEN — cited but not in references.bib ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
    else:
        print("\nAll cite keys resolve.")

    if unused:
        print(f"\nIn references.bib but never cited ({len(unused)}):")
        for k in unused:
            print(f"  {k}")

    dupes = [k for k in defined if bib.count("{" + k + ",") > 1]
    if dupes:
        print(f"\nDuplicate bib keys: {dupes}")
        return 1

    # Cross-references. Deleting a float and forgetting its \ref is the natural way to
    # break this file while trimming for a page limit, and LaTeX reports it only as a "??"
    # in the PDF -- easy to miss in a ten-page skim.
    labels = set(LABEL_RE.findall(tex_body))
    refs = set(REF_RE.findall(tex_body))
    dangling = sorted(refs - labels)
    orphans = sorted(labels - refs)

    print(f"\nlabels: {len(labels)}   refs: {len(refs)}")
    if dangling:
        print(f"BROKEN — \\ref with no \\label ({len(dangling)}): {', '.join(dangling)}")
    else:
        print("All cross-references resolve.")
    if orphans:
        # Not a defect. A \label nobody points at costs nothing and is worth keeping on a
        # section in case a later edit wants to cross-reference it; it is listed only so
        # that a label orphaned by DELETING its reference is visible.
        print(f"(informational) labels not currently referenced: {', '.join(orphans)}")

    floats = COUNT_RE.findall(tex_body)
    words = len(re.findall(r"[A-Za-z][A-Za-z\-]+",
                           tex_body.split(r"\begin{document}")[-1]))
    print(f"body words: {words}   tables: {floats.count('table')}   "
          f"figures: {floats.count('figure')}")

    return 1 if (missing or dangling) else 0


if __name__ == "__main__":
    raise SystemExit(main())
