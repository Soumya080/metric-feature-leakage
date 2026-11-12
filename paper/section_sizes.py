"""Word count per section and per float, to decide what to cut against a page limit.

Cutting to a page limit by eye tends to shave the parts that are easy to shorten rather
than the parts that cost the most page for the least argument. This prints both, so the
decision is made against sizes rather than impressions.

Floats are costed in approximate page fractions: at NeurIPS single-column widths a figure
spanning 0.8\textwidth runs about a third of a page including caption, and a small booktabs
table about a fifth. Rough, but the ranking is what matters.

Usage:
    python paper/section_sizes.py
"""

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main_v2.tex")

WORDS_PER_PAGE = 578          # calibrated on the neurips_2026.sty build: 3,182w + 2 figs -> body ends p6,
                              # references start p7. Earlier 430/500 figures came from builds under the
                              # FALLBACK geometry, which is materially wider-margined than the real template.
FIGURE_PAGES = 0.50
TABLE_PAGES = 0.30


def strip_comments(text):
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


def words(text):
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z\-']+", text))


def main():
    tex = strip_comments(io.open(TEX, encoding="utf-8").read())
    body = tex.split(r"\begin{document}")[-1]
    # The page limit counts the body. References and anything after \appendix do not, so
    # counting them here reports an overage that does not exist and provokes cuts to material
    # that costs nothing. Measure the body; report the appendix separately.
    body, _, appendix = body.partition(r"\appendix")

    parts = re.split(r"\\section\{([^}]*)\}", body)
    chunks = [("Title/abstract", parts[0])]
    chunks += [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]

    rows = []
    for name, block in chunks:
        n_fig = len(re.findall(r"\\begin\{figure\}", block))
        n_tab = len(re.findall(r"\\begin\{table\}", block))
        w = words(block)
        pages = w / WORDS_PER_PAGE + n_fig * FIGURE_PAGES + n_tab * TABLE_PAGES
        rows.append((name, w, n_fig, n_tab, pages))

    total_w = sum(r[1] for r in rows)
    total_p = sum(r[4] for r in rows)

    print(f"{'section':<34}{'words':>7}{'fig':>5}{'tab':>5}{'~pages':>9}")
    print("-" * 60)
    for name, w, f, t, p in rows:
        print(f"{name[:33]:<34}{w:>7}{f:>5}{t:>5}{p:>9.2f}")
    print("-" * 60)
    print(f"{'TOTAL':<34}{total_w:>7}"
          f"{sum(r[2] for r in rows):>5}{sum(r[3] for r in rows):>5}{total_p:>9.2f}")
    print()
    print(f"floats cost ~{sum(r[2] for r in rows) * FIGURE_PAGES + sum(r[3] for r in rows) * TABLE_PAGES:.2f} "
          f"pages of the {total_p:.2f}")
    if appendix.strip():
        aw = words(appendix)
        print(f"appendix: {aw} words (~{aw / WORDS_PER_PAGE:.2f} pages) -- NOT counted above, "
              f"it falls outside the limit")
    print(f"to reach 8.0 pages, cut ~{max(0, total_p - 8.0) * WORDS_PER_PAGE:.0f} words "
          f"(or one figure + ~{max(0, (total_p - 8.0 - FIGURE_PAGES)) * WORDS_PER_PAGE:.0f} words)")


if __name__ == "__main__":
    raise SystemExit(main())
