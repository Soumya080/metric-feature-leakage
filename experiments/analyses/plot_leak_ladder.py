#!/usr/bin/env python3
"""Figure 1: the leak ladder in accuracy (Table 2) and in apparent capture of the oracle gap.

(a) accuracy at predicting the strictly-best arm, default GB, CV seed 0 (Table 2's column),
    against the 60.9% majority baseline;
(b) apparent capture of the oracle gap by the tuned router of Table 3, mean over 3 datasets x 5
    split seeds, whiskers = min/max over those 15 cells.
Reads results/analyses/leak_capture_summary/summary.json; Table 2 accuracies are the published
values in results/leak_gradient.json (seed 0). The strip under panel (a) marks the rungs on which
an equality check (a leakcheck verdict on slots 22-25) fires, read from
results/analyses/ladder_checks/summary.json ("ladder_verdicts"). Panel (a)'s y-axis starts at 0
Writes results/figures/fig12_leak_ladder.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cap = json.load(open(os.path.join(ROOT, "results", "analyses", "leak_capture_summary", "summary.json")))["rungs"]
grad = json.load(open(os.path.join(ROOT, "results", "leak_gradient.json")))
arms = grad.get("arms", grad)
verd = json.load(open(os.path.join(ROOT, "results", "analyses", "ladder_checks", "summary.json")))["ladder_verdicts"]

ORDER = ["honest_minimal", "leak_one_arm", "leak_hit", "leak_noisy", "leaky"]
LABEL = ["honest\n(no leak)", "one arm\n(BM25 only)", "Hit@20\n(hit/miss)",
         "recoded\n(keyed)", "exact\n(the bug)"]
BLUE, ORANGE, RED, GREY = "#2b78d4", "#e8693a", "#d64545", "#555555"
acc = [arms[k]["gbt"] * 100 for k in ORDER]
majority = arms["honest_minimal"]["majority"] * 100
cm = [cap[k]["mean"] * 100 for k in ORDER]
lo = [cm[i] - cap[k]["min"] * 100 for i, k in enumerate(ORDER)]
hi = [cap[k]["max"] * 100 - cm[i] for i, k in enumerate(ORDER)]
cols = [BLUE] + [ORANGE] * 4

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7))
x = range(len(ORDER))
a.bar(x, acc, 0.62, color=cols, zorder=3)
a.axhline(majority, color=RED, ls="--", lw=1.2, zorder=4)
a.text(-0.45, 97, f"dashed: majority baseline {majority:.1f}%", color=RED, ha="left", va="top", fontsize=8)
for i, v in enumerate(acc):
    a.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
# Equality-check strip: does an exact-agreement check (leakcheck verdict) catch this rung?
for i, k in enumerate(ORDER):
    if k == "honest_minimal":
        lab, c = "n/a", "white"
    else:
        lab, c = ("caught", "white") if verd[k]["verdict"] else ("missed", "#fff2b3")
    a.text(i, 8, lab, ha="center", va="center", fontsize=7, fontweight="bold", color=c)
a.text(-0.45, 90, "in bars: does an equality check catch it?", color=GREY, ha="left", va="top", fontsize=7.5)
a.set_ylim(0, 98)
a.set_ylabel("accuracy, best arm (%)")
a.set_title("(a) what a classifier sees", fontsize=9, loc="left")
b.bar(x, cm, 0.62, color=cols, zorder=3, yerr=[lo, hi], capsize=3, ecolor=GREY, error_kw={"lw": 1})
b.axhline(0, color=GREY, lw=0.8)
for i, v in enumerate(cm):
    b.text(i, v + hi[i] + 2, f"{v:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
b.set_ylim(-15, 105)
b.set_ylabel("apparent capture of oracle gap (%)")
b.set_title("(b) what a routing paper would report", fontsize=9, loc="left")
for ax in (a, b):
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABEL, fontsize=7.2)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
fig.tight_layout()
out = os.path.join(ROOT, "results", "figures", "fig12_leak_ladder")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print("wrote", out, "acc", [round(v, 1) for v in acc], "capture", [round(v, 1) for v in cm])
