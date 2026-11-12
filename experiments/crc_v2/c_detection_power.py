#!/usr/bin/env python3
"""C: detection power -- how much leak gets through each kind of check, by noise level.

No new fits. Joins three committed artifacts on the MRR@20 leak (the paper's leak), pooled
over NQ/TriviaQA/HotpotQA, Table 2 protocol:

  R1  results/crc_v2/crc_v2/r1_classifier_sweep/summary.json   value-keyed crc32 offset (recoding)
  R5  results/crc_v2/crc_v2/r5_random_noise/{summary.json, leakcheck.jsonl}   independent noise
  R3  results/crc_v2/crc_v2/r3_leakcheck_sweep/leakcheck_sweep.json   leakcheck on the R1 arms

For each construction and half-width s it reports
  gain_pp           GB accuracy minus honest GB accuracy (what the leak still buys)
  equality_check    does exact agreement with the reward exceed leakcheck's 25% threshold?
  leakcheck         number of suspicions (rank-based) and verdicts (exact) on the pooled matrix

and draws results/figures/fig11_detection_power.{pdf,png}. Writes --out-dir/table.json
(refuses non-empty).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as C
from leakcheck.detectors import EXACT_MATCH_THRESHOLD

CRC = os.path.join(C.CRC_RESULTS, "crc_v2")
FIGDIR = os.path.join(C.RESULTS_DIR, "figures")
SCALES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
COL = {"value_keyed": "#3b6ea5", "independent": "#c0573e"}


def tag(s):
    return f"s{int(round(s * 100)):03d}"


def build_table():
    r1 = json.load(open(os.path.join(CRC, "r1_classifier_sweep", "summary.json"), encoding="utf-8"))["arms"]
    r3 = json.load(open(os.path.join(CRC, "r3_leakcheck_sweep", "leakcheck_sweep.json"), encoding="utf-8"))["arms"]
    r5 = json.load(open(os.path.join(CRC, "r5_random_noise", "summary.json"), encoding="utf-8"))["arms"]
    r5_lc = [json.loads(l) for l in open(os.path.join(CRC, "r5_random_noise", "leakcheck.jsonl"), encoding="utf-8")]

    rows = []
    h1 = r1["honest_minimal"]["gbt"]["mean"]
    for s in SCALES:
        arm = r1[f"mrr20_{tag(s)}"]["gbt"]
        pooled = r3[f"mrr20_{tag(s)}"]["pooled"]
        ex = max(st["exact_match_rate"] for st in pooled["slot_stats"] if st["slot"] in (22, 23))
        rows.append({"construction": "value_keyed", "scale": s, "gain_pp": (arm["mean"] - h1) * 100,
                     "exact_match_max": ex, "equality_check_fires": ex > EXACT_MATCH_THRESHOLD,
                     "leakcheck_suspicions": pooled["n_suspicions"], "leakcheck_verdicts": pooled["n_verdicts"]})
    h5 = r5["honest_minimal"]["gbt"]["mean"]
    for s in SCALES:
        name = "mrr20_exact" if s == 0 else f"mrr20_rand_{tag(s)}"
        g = r5[name]["gbt"]
        lc = [r for r in r5_lc if r["arm"] == name]
        ex = max(max(r["exact_match_22_bm25"], r["exact_match_23_dense"]) for r in lc)
        rows.append({"construction": "independent", "scale": s, "gain_pp": (g["mean"] - h5) * 100,
                     "gain_pp_draw_min": (min(g["draw_means"]) - h5) * 100,
                     "gain_pp_draw_max": (max(g["draw_means"]) - h5) * 100,
                     "exact_match_max": ex, "equality_check_fires": ex > EXACT_MATCH_THRESHOLD,
                     "leakcheck_suspicions": float(np.mean([r["n_suspicions"] for r in lc])),
                     "leakcheck_verdicts": float(np.mean([r["n_verdicts"] for r in lc]))})
    return rows


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK,
                         "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    x = np.arange(len(SCALES))
    labels = ["0"] + [f"{s:g}" for s in SCALES[1:]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for c, lab in (("value_keyed", "value-keyed recoding"), ("independent", "independent noise")):
        rs = [r for r in rows if r["construction"] == c]
        y = [r["gain_pp"] for r in rs]
        a1.plot(x, y, "-o", color=COL[c], ms=3.5, lw=1.4, label=lab)
        if c == "independent":
            a1.fill_between(x, [r["gain_pp_draw_min"] for r in rs], [r["gain_pp_draw_max"] for r in rs],
                            color=COL[c], alpha=0.18, lw=0)
        a2.plot(x, [r["leakcheck_suspicions"] for r in rs], "-o", color=COL[c], ms=3.5, lw=1.4, label=lab)
    for a in (a1, a2):
        a.set_xticks(x, labels)
        a.set_xlabel("noise half-width $s$")
        a.grid(axis="y", color=GRID, lw=0.7)
        a.axvspan(-0.4, 0.4, color=GRID, alpha=0.6, lw=0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    a1.set_ylabel("accuracy gain over honest (pp)")
    a1.set_title("(a) what the leak still buys", fontsize=9, loc="left")
    a1.set_ylim(bottom=0)
    a1.legend(loc="lower left", bbox_to_anchor=(0.12, 0.0), fontsize=7.5)
    a2.set_ylabel("leakcheck suspicions")
    a2.set_title("(b) what leakcheck flags", fontsize=9, loc="left")
    a2.set_ylim(bottom=0)
    a1.text(0, a1.get_ylim()[1] * 0.04, "equality\ncheck fires", ha="center", va="bottom", fontsize=7, color=INK_2)
    a2.text(0, a2.get_ylim()[1] * 0.04, "verdict", ha="center", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig11_detection_power.{ext}"), dpi=200 if ext == "png" else None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "c_detection_power"))
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    rows = build_table()
    C.write_json(os.path.join(out_dir, "table.json"), {"header": C.run_header(args), "leak": "mrr_at_20",
                                                        "equality_threshold": EXACT_MATCH_THRESHOLD, "rows": rows})
    draw(rows)
    print(f"{'construction':12s} {'s':>5} {'gain':>6} {'eq.fires':>8} {'susp':>5} {'verd':>4}")
    for r in rows:
        print(f"{r['construction']:12s} {r['scale']:5.2f} {r['gain_pp']:+6.2f} {str(r['equality_check_fires']):>8} "
              f"{r['leakcheck_suspicions']:5.1f} {r['leakcheck_verdicts']:4.1f}")
    print(f"Done: {out_dir}; figure results/figures/fig11_detection_power.pdf")


if __name__ == "__main__":
    main()
