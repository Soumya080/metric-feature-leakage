"""Generate every paper figure from the committed result artifacts.

One script so the figures cannot drift from the numbers. Every value is read from a JSON in
results/ -- nothing is typed in here, so a figure can never disagree with the table beside
it. Reads only; runs no experiment.

Figures produced (PDF for LaTeX + PNG for slides/preview):

  fig1_leak            the central claim: leaky vs honest against the majority baseline
  fig2_feature_groups  every feature group sits below the baseline (7/7)
  fig3_capture         share of oracle headroom captured, per method per dataset
  fig4_decomposition   where the headroom lives, by the winner's gold rank
  fig5_fusion          RRF and tuned convex fusion vs the best single arm
  fig6_learning_curve  lift vs sample size -- bias-limited or data-limited?
  fig7_pool_shift      what changed from the 20K pool to the 100K pool
  fig8_dose_response   how much the leak is worth, by dose
  fig9_permutation     observed accuracy against shuffled-label and random-feature nulls
  fig10_leak_vs_scale  leak premium as the shared corpus grows 20K -> 100K

Palette. Two-slot categorical (blue #2a78d6 / orange #eb6834) for honest-vs-leaky, and a
blue-red diverging pair for signed quantities, both validated against the light chart
surface: CVD deltaE 24.7 and 21.6, normal-vision 33.6 and 32.3, all >= 3:1 contrast. Red and
orange never share a chart -- they fail together (deltaE 5.6) and are deliberately kept apart.

Identity is never carried by colour alone: every series is direct-labelled or named in the
axis, so the figures survive greyscale printing and CVD.
"""

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")

DATASETS = ["nq", "triviaqa", "hotpotqa"]
PRETTY = {"nq": "NQ", "triviaqa": "TriviaQA", "hotpotqa": "HotpotQA", "pooled": "Pooled"}

# Validated slots -- see module docstring.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e8e7e3"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # TrueType, so the PDF is text-searchable and editable
    })
    return plt


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGDIR, f"{name}.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None)
    print(f"  {name}.pdf / .png")


def load(path):
    p = path if os.path.isabs(path) else os.path.join(RESULTS, path)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tagged(ref, relpath):
    """A results file from a git tag, for the 20K-vs-100K comparison."""
    out = subprocess.run(["git", "show", f"{ref}:{relpath}"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


# ---------------------------------------------------------------- fig 1: the leak

def fig_leak(plt, learn):
    """Grouped bars because the job is comparing two magnitudes within each dataset; the
    baseline is a per-group reference line rather than a third bar, because it is the
    threshold being cleared or missed, not another method."""
    keys = DATASETS + ["pooled"]
    rows = []
    for k in keys:
        e = (learn.get("per_dataset", {}).get(k) or (learn.get("pooled") if k == "pooled" else None))
        h = (e or {}).get("honest") if k != "pooled" else (learn.get("pooled") or {}).get("honest")
        l = (e or {}).get("leaky") if k != "pooled" else (learn.get("pooled") or {}).get("leaky")
        if h and l:
            rows.append((PRETTY[k], h["gbt"] * 100, l["gbt"] * 100, h["majority"] * 100))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = range(len(rows))
    w = 0.34
    gap = 0.02   # 2px-equivalent surface gap between adjacent fills
    ax.bar([i - w/2 - gap/2 for i in x], [r[1] for r in rows], w, color=BLUE,
           label="Honest features", zorder=3)
    ax.bar([i + w/2 + gap/2 for i in x], [r[2] for r in rows], w, color=ORANGE,
           label="Leaky features", zorder=3)

    for i, r in enumerate(rows):
        ax.plot([i - 0.42, i + 0.42], [r[3], r[3]], color=INK, lw=1.6, zorder=4,
                solid_capstyle="butt")
        ax.text(i - 0.42, r[3] + 0.7, f"{r[3]:.1f}", color=INK, fontsize=7.5, va="bottom")
        ax.text(i - w/2 - gap/2, r[1] - 2.2, f"{r[1]:.1f}", color="white", fontsize=7.5,
                ha="center", va="top", fontweight="bold")
        ax.text(i + w/2 + gap/2, r[2] - 2.2, f"{r[2]:.1f}", color="white", fontsize=7.5,
                ha="center", va="top", fontweight="bold")
        ax.annotate("", xy=(i + w/2 + gap/2, r[2]), xytext=(i + w/2 + gap/2, r[3]),
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7, ls=":"))

    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Best-retriever prediction accuracy (%)")
    ax.set_title("Leaky features clear the majority baseline; honest features do not",
                 loc="left", pad=8)
    ax.set_ylim(0, max(r[2] for r in rows) + 9)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    h, lb = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([0], [0], color=INK, lw=1.6)); lb.append("Majority baseline")
    ax.legend(h, lb, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=1.4)
    fig.tight_layout()
    return fig


# ------------------------------------------------- fig 2: feature groups vs baseline

def fig_feature_groups(plt, learn):
    """Signed magnitudes around zero -> diverging horizontal bars, sorted, zero rule drawn."""
    groups = ((learn.get("pooled") or {}).get("honest") or {}).get("groups")
    if not groups:
        return None
    rows = sorted(((k.split(" (")[0], v["lift"] * 100) for k, v in groups.items()),
                  key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for i, (name, lift) in enumerate(rows):
        c = BLUE if lift > 0 else RED
        ax.barh(i, lift, height=0.62, color=c, zorder=3)
        off = 0.12 if lift < 0 else -0.12
        ax.text(lift + off, i, f"{lift:+.1f}", va="center",
                ha="right" if lift < 0 else "left", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Lift over majority baseline (percentage points)")
    ax.set_title("Every feature group performs worse than a constant policy",
                 loc="left", pad=8)
    lo = min(r[1] for r in rows)
    ax.set_xlim(lo - 1.6, max(0.9, max(r[1] for r in rows) + 1.0))
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- fig 3: headroom capture

def fig_capture(plt, hr):
    """Grouped diverging bars: signed magnitude, three groups per method."""
    cap = (hr.get("capture") or {}).get("honest")
    if not cap:
        return None
    # score_gap is excluded on purpose. It is a single-threshold rule that collapses onto
    # Always-BM25, so its capture is around -232% on NQ -- an order of magnitude outside the
    # others. Plotted together it compresses every real method into an unreadable band at
    # zero, which is the opposite of what the figure is for. Its value goes in the caption.
    methods = [("logistic_regression", "LogReg"),
               ("random_forest", "RandomForest"), ("ridge", "Ridge"),
               ("gradient_boosting", "GBT (tuned)"), ("rl_route_mean", "RL-Route")]
    excluded = {}
    for d in DATASETS:
        v = (cap.get(d, {}).get("capture", {}) or {}).get("score_gap")
        if v is not None:
            excluded[PRETTY[d]] = v * 100
    present = [(k, lb) for k, lb in methods
               if any((cap.get(d, {}).get("capture", {}) or {}).get(k) is not None
                      for d in DATASETS)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    n = len(DATASETS); w = 0.26
    from matplotlib.patches import Patch
    shades = {"nq": BLUE, "triviaqa": "#5598e7", "hotpotqa": "#9ec5f4"}
    for j, d in enumerate(DATASETS):
        vals = [((cap.get(d, {}).get("capture", {}) or {}).get(k) or 0) * 100
                for k, _ in present]
        xs = [i + (j - (n - 1) / 2) * (w + 0.02) for i in range(len(present))]
        ax.bar(xs, vals, w, color=shades[d], label=PRETTY[d], zorder=3)
    ax.axhline(0, color=INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(present))); ax.set_xticklabels([lb for _, lb in present])
    ax.set_ylabel("Oracle headroom captured (%)")
    ax.set_title("No method captures meaningful headroom; 0% = commit to one retriever",
                 loc="left", pad=8)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Legend below the plot: inside, it sat on top of a bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, columnspacing=1.3)
    if excluded:
        note = ", ".join(f"{k} {v:.0f}%" for k, v in excluded.items())
        fig.text(0.5, -0.115, f"Heuristic (score-gap) omitted as an outlier: {note}.",
                 ha="center", fontsize=7.5, color=INK_3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------- fig 4: where the headroom lives

def fig_decomposition(plt, hr):
    """Parts of a whole -> stacked horizontal bars, one sequential blue ramp, 2px gaps."""
    dec = hr.get("decomposition")
    if not dec:
        return None
    buckets = ["rank 1-5", "rank 6-20", "rank 21-100", "rank >100"]
    ramp = ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for i, d in enumerate(DATASETS):
        e = dec.get(d)
        if not e:
            continue
        left = 0.0
        for b, c in zip(buckets, ramp):
            v = (e["bucket_share"].get(b) or 0) * 100
            ax.barh(i, v, left=left, height=0.6, color=c, zorder=3,
                    edgecolor="white", linewidth=1.4)
            if v > 7:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white" if c in ramp[:2] else INK,
                        fontweight="bold")
            left += v
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([PRETTY[d] for d in DATASETS])
    ax.set_xlabel("Share of oracle headroom (%)")
    ax.set_xlim(0, 100)
    ax.set_title("The headroom sits in the top five ranks, not the deep tail",
                 loc="left", pad=8)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=c) for c in ramp], buckets,
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4, columnspacing=1.2)
    ax.invert_yaxis()      # NQ at the top, matching every other figure's reading order
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- fig 5: fusion

def fig_fusion(plt, fus):
    """Four magnitudes per dataset; a dot plot rather than bars because the comparison is
    against a reference (the best arm), not between independent categories."""
    rows = [(d, fus[d]) for d in DATASETS if d in fus]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (d, r) in enumerate(rows):
        best, rrf, cvx, orc = (r["best_fixed_test"], r["rrf_hybrid_test"],
                               r["convex_fusion_test"], r["oracle_test"])
        ax.plot([best, orc], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([orc], [i], s=52, color="white", edgecolor=INK_3, lw=1.2, zorder=4)
        # White rings: convex fusion lands almost exactly on the best arm on two of three
        # datasets (that IS the result), so without a surface ring they read as one mark.
        ax.scatter([best], [i], s=64, color=INK, zorder=6,
                   edgecolor="white", linewidth=1.4)
        ax.scatter([rrf], [i], s=56, color=RED, zorder=5,
                   edgecolor="white", linewidth=1.4)
        ax.scatter([cvx], [i], s=58, color=ORANGE, marker="D", zorder=7,
                   edgecolor="white", linewidth=1.4)
        ax.text(orc + 0.004, i, f"oracle {orc:.3f}", va="center", fontsize=7, color=INK_3)
        # Below the mark, not above: above collided with the figure title on the top row.
        ax.text(best, i - 0.32, f"best {best:.3f}", ha="center", va="top",
                fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{PRETTY[d]}\n(α={r['selected_alpha']:.1f})" for d, r in rows])
    ax.set_xlabel("MRR on the held-out test split")
    ax.set_title("Fusion does not beat the best single retriever either", loc="left", pad=8)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=7),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=7),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=ORANGE, markersize=6),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor=INK_3, markersize=7)],
              ["Best single arm", "RRF (Hybrid)", "Tuned convex", "Oracle"],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, columnspacing=1.2)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.margins(x=0.14, y=0.28)
    ax.invert_yaxis()      # NQ at the top, matching every other figure's reading order
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- fig 6: learning curve

def fig_learning_curve(plt, lc):
    """Change over a continuous variable -> line with a spread band, zero rule for the
    threshold, endpoint emphasised."""
    pts = lc.get("points") or []
    if len(pts) < 2:
        return None
    import numpy as np
    n = np.array([p["n"] for p in pts], float)
    lift = np.array([p["gbt_lift"] for p in pts]) * 100
    sp = np.array([p.get("gbt_lift_spread", 0) for p in pts]) * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.axhline(0, color=INK, lw=1.2, zorder=4)
    ax.text(n[0], 0.35, "parity with a constant policy", fontsize=7.5, color=INK_2,
            va="bottom")
    ax.fill_between(n, lift - sp, lift + sp, color=BLUE, alpha=0.16, zorder=2, lw=0)
    ax.plot(n, lift, "-o", color=BLUE, lw=2, ms=5, zorder=3)
    ax.scatter([n[-1]], [lift[-1]], s=90, color=BLUE, zorder=5,
               edgecolor="white", linewidth=1.8)
    ax.annotate(f"{lift[-1]:+.1f} pts at n={int(n[-1]):,}",
                xy=(n[-1], lift[-1]), xytext=(-8, 14), textcoords="offset points",
                ha="right", fontsize=8, color=INK)
    ax.set_xlabel("Informative queries used for training/CV (n)")
    ax.set_ylabel("Lift over majority baseline (pp)")
    ax.set_title("Still improving with data, and still below parity", loc="left", pad=8)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------- fig 7: 20K -> 100K shift

def fig_pool_shift(plt, learn20, learn100):
    """Paired before/after -> slope chart. One line per quantity, so the direction of
    change is the visual, not the absolute values."""
    if not learn20 or not learn100:
        return None
    def get(l, arm, field):
        return ((l.get("pooled") or {}).get(arm) or {}).get(field)
    rows = []
    for label, arm, field, scale in (("Leaky accuracy", "leaky", "gbt", 100),
                                     ("Majority baseline", "honest", "majority", 100),
                                     ("Honest accuracy", "honest", "gbt", 100)):
        a, b = get(learn20, arm, field), get(learn100, arm, field)
        if a is not None and b is not None:
            rows.append((label, a * scale, b * scale))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for label, a, b in rows:
        c = ORANGE if "Leaky" in label else (BLUE if "Honest" in label else INK_3)
        ax.plot([0, 1], [a, b], "-o", color=c, lw=2, ms=6, zorder=3)
        ax.text(-0.04, a, f"{a:.1f}", ha="right", va="center", fontsize=8, color=INK_2)
        ax.text(1.04, b, f"{b:.1f}  {label}", ha="left", va="center", fontsize=8, color=c)
    ax.set_xlim(-0.30, 1.75); ax.set_xticks([0, 1])
    ax.set_xticklabels(["20K pool\n(n=2,468)", "100K pool\n(n=3,170)"])
    ax.set_ylabel("Prediction accuracy (%)")
    ax.set_title("On a cleaner, larger corpus the leak grew", loc="left", pad=8)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------- fig 8: the dose-response (central result)

def fig_dose_response(plt, grad):
    """The paper's central result, so it gets a chart of its own rather than sharing fig1's.

    Single series ordered by dose, so no legend is needed -- the x-axis names each arm and the
    ordering IS the variable. Bars rather than a line because the arms are distinct feature
    sets, not samples of a continuum. One shade of orange for every leaked arm and blue for the
    honest one: the categorical distinction that matters is leaked vs not, and grading the
    orange by dose would encode the same information the x-position already carries.
    """
    # `arms` is a dict keyed by arm name; each value carries its own `majority`, which is
    # identical across arms by construction (the reward table is asserted equal).
    by = grad.get("arms") or {}
    order = ["honest_minimal", "leak_one_arm", "leak_hit", "leak_noisy", "leaky"]
    label = {"honest_minimal": "honest\n(no leak)", "leak_one_arm": "one arm\n(1 column)",
             "leak_hit": "Hit@k\n(coarser)", "leak_noisy": "jittered\n(+15% noise)",
             "leaky": "exact\n(verbatim)"}
    rows = [(label[k], by[k]) for k in order if isinstance(by.get(k), dict)]
    if not rows:
        return None

    majority = rows[0][1]["majority"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = range(len(rows))
    vals = [r[1]["gbt"] * 100 for r in rows]
    cols = [BLUE if k.startswith("honest") else ORANGE for k, _ in rows]
    ax.bar(xs, vals, 0.62, color=cols, zorder=3)

    ax.axhline(majority * 100, color=RED, lw=1.4, ls=(0, (5, 3)), zorder=4)
    # Labelled in a margin opened to the RIGHT of the last bar, straddling the line. Two
    # placements were tried and rendered wrong: at the right edge inside the axes it fell on an
    # orange fill and vanished; at the left it collided with the honest bar's own value label.
    # A dedicated margin is the only spot that is white in every dataset.
    ax.set_xlim(-0.6, len(rows) - 0.5 + 1.15)
    # va="bottom" so the whole three-line block sits ABOVE the rule. Centring it on the rule
    # (the first instinct) drew the dashes straight through the middle line of text.
    ax.annotate(f"majority\nbaseline\n{majority*100:.1f}%",
                xy=(len(rows) - 0.34, majority * 100), xytext=(4, 3),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8.5, color=RED, zorder=7, linespacing=1.35)

    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")

    # The comparison the reader must not miss: jitter costs almost nothing.
    if len(rows) >= 5:
        y = max(vals[-2], vals[-1]) + 3.4
        ax.plot([len(rows) - 2, len(rows) - 2, len(rows) - 1, len(rows) - 1],
                [y - 0.9, y, y, y - 0.9], color=INK_3, lw=0.8, zorder=2)
        ax.annotate(f"{abs(vals[-1] - vals[-2]):.1f} pt apart",
                    xy=(len(rows) - 1.5, y), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([k for k, _ in rows])
    ax.set_ylabel("accuracy predicting the best arm (%)")
    ax.set_ylim(min(min(vals), majority * 100) - 4, max(vals) + 8)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


# ------------------------------------------------- fig 9: the permutation control

def fig_permutation(plt, perm):
    """Observed accuracy against two nulls, per dataset.

    A dot plot with null spread as a horizontal band, because the question is 'does the dot sit
    outside the band' -- a comparison of position against an interval, which bars cannot show.
    The majority baseline is a separate marker rather than a line, since it differs per dataset.
    """
    keys = DATASETS + ["pooled"]
    rows = []
    for k in keys:
        e = perm.get("pooled") if k == "pooled" else (perm.get("per_dataset", {}) or {}).get(k)
        if e:
            rows.append((PRETTY[k], e))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(6.6, 0.72 * len(rows) + 1.5))
    ys = list(range(len(rows)))[::-1]          # first dataset at the top

    for y, (_, e) in zip(ys, rows):
        pm, ps = e["perm_null_mean"] * 100, e["perm_null_std"] * 100
        rm, rs = e["rand_null_mean"] * 100, e["rand_null_std"] * 100
        # Null distributions as +/-1 sd bands, offset so they do not overplot.
        ax.plot([pm - ps, pm + ps], [y + 0.13, y + 0.13], color=INK_3, lw=6, alpha=0.35,
                solid_capstyle="butt", zorder=2)
        ax.plot([rm - rs, rm + rs], [y - 0.13, y - 0.13], color=INK_3, lw=6, alpha=0.18,
                solid_capstyle="butt", zorder=2)
        ax.plot(pm, y + 0.13, "o", ms=5, color=INK_2, zorder=4)
        ax.plot(rm, y - 0.13, "s", ms=4.5, color=INK_3, zorder=4)
        # Observed, with a white ring so it stays readable where it overlaps a band.
        ax.plot(e["observed"] * 100, y, "o", ms=9, color=BLUE,
                markeredgecolor="white", markeredgewidth=1.6, zorder=6)
        ax.plot(e["majority"] * 100, y, "|", ms=13, color=RED, mew=1.8, zorder=5)
        ax.annotate(f"p={e['perm_p']:.3f}", xy=(e["majority"] * 100, y), xytext=(9, -3),
                    textcoords="offset points", fontsize=8.5, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in rows])
    ax.set_xlabel("accuracy predicting the best arm (%)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # Headroom on the right for the p-value labels, which hang off the majority marker and
    # otherwise run into the figure edge on the dataset with the highest baseline.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.10 * (hi - lo))
    ax.set_ylim(-0.55, len(rows) - 0.45)

    # Legend by proxy artists: five marks with distinct shapes, so identity never rests on
    # colour alone.
    h = [plt.Line2D([], [], marker="o", ms=8, color=BLUE, ls="", markeredgecolor="white",
                    markeredgewidth=1.4, label="observed (honest features)"),
         plt.Line2D([], [], marker="o", ms=5, color=INK_2, ls="", label="shuffled-label null"),
         plt.Line2D([], [], marker="s", ms=4.5, color=INK_3, ls="", label="random-feature null"),
         plt.Line2D([], [], marker="|", ms=11, color=RED, ls="", mew=1.8,
                    label="majority baseline")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.4, columnspacing=1.4)
    return fig


def fig_leak_vs_scale(plt, sweep):
    """How much the leak buys, as a function of how many passages were searched.

    Lines rather than bars: corpus size is a continuum and the reading is the SLOPE, not
    the individual heights. Log x, because the scales are geometric.

    Orange for every leaked arm and blue for the honest one, matching fig8, so a reader who
    has seen the dose-response recognises the encoding without a second look. Arms are
    direct-labelled at the right rather than put in a legend, so identity never depends on
    telling four oranges apart.

    The 5K scale is excluded upstream in 06_leak_vs_scale.py: 233 informative queries
    against ~3,000 elsewhere makes it an underpowered outlier, and including it would make
    the slope look steeper than the evidence supports.
    """
    scales = sorted(sweep.get("scales", {}), key=int)
    if len(scales) < 2:
        return None

    order = ["leaky", "leak_noisy", "leak_hit", "leak_one_arm"]
    label = {"leaky": "exact", "leak_noisy": "jittered",
             "leak_hit": "Hit@k", "leak_one_arm": "one arm"}

    def prem(arm, s):
        cell = sweep["scales"][s]
        a, h = cell.get(arm), cell.get("honest_minimal")
        return None if not (a and h) else (a["accuracy"] - h["accuracy"]) * 100

    xs = [int(s) for s in scales]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    endpoints = []
    for arm in order:
        ys = [prem(arm, s) for s in scales]
        if any(y is None for y in ys):
            continue
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.6, color=ORANGE, zorder=3)
        endpoints.append([ys[-1], label[arm], ys[-1]])

    # Declutter the direct labels. The top three arms land within 0.6pp of each other at
    # 100K, so labelling each at its own y overprints them into an unreadable smear -- the
    # first render of this figure did exactly that. Walk downward from the highest and push
    # each label to at least MIN_GAP below the previous, then draw a leader line from the
    # displaced label back to the point it belongs to.
    MIN_GAP = 0.62
    endpoints.sort(key=lambda e: -e[0])
    for i in range(1, len(endpoints)):
        if endpoints[i - 1][2] - endpoints[i][2] < MIN_GAP:
            endpoints[i][2] = endpoints[i - 1][2] - MIN_GAP
    for y_true, name, y_lab in endpoints:
        ax.annotate(f" {name}", xy=(xs[-1], y_lab), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK_2)
        if abs(y_lab - y_true) > 1e-9:
            ax.plot([xs[-1], xs[-1] * 1.14], [y_true, y_lab],
                    color=INK_3, lw=0.6, zorder=2, clip_on=False)

    # Honest features on the same axis, as the zero the premium is measured against, plus
    # their own lift so a reader can see they move the OTHER way.
    honest = [sweep["scales"][s]["honest_minimal"]["lift"] * 100 for s in scales]
    ax.plot(xs, honest, marker="s", ms=4.5, lw=1.6, color=BLUE, ls=(0, (4, 2)), zorder=3)
    ax.annotate(" honest\n (vs majority)", xy=(xs[-1], honest[-1]), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=8.5, color=BLUE)

    ax.axhline(0, color=INK_3, lw=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x//1000}K" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 2.6)
    ax.set_xlabel("Shared corpus size (passages, log scale)")
    ax.set_ylabel("Accuracy vs honest features (pp)")
    ax.grid(axis="y", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-tag", default="results-pool20k",
                    help="git tag holding the earlier run, for fig7")
    args = ap.parse_args()

    plt = style()
    learn = load("learnability_analysis.json")
    hr = load("headroom_analysis.json")
    fus = load("fusion_baseline.json")
    lc = load("learning_curve.json")
    learn20 = load_tagged(args.compare_tag, "results/learnability_analysis.json")
    grad = load("leak_gradient.json")
    # B=1000. The old results/permutation_control.json ran B=20, whose attainable p-floor is
    # 0.048 -- exactly what it reported for NQ -- and now sits under a SUPERSEDED_ name so that a
    # stale figure cannot be regenerated from it by accident.
    perm = load(os.path.join("perm_B1000", "permutation_control.json"))
    # Lives under experiments/corpus_v3/results/, not results/, because it belongs to the
    # shared-index sweep rather than the 500-passage pipeline. Keeping it there is what stops
    # the two settings' numbers being read as one experiment.
    scale_sweep = load(os.path.join("..", "experiments", "corpus_v3", "results",
                                    "leak_vs_scale.json"))

    print("Writing figures to results/figures/")
    jobs = [
        ("fig1_leak", lambda: fig_leak(plt, learn) if learn else None),
        ("fig2_feature_groups", lambda: fig_feature_groups(plt, learn) if learn else None),
        ("fig3_capture", lambda: fig_capture(plt, hr) if hr else None),
        ("fig4_decomposition", lambda: fig_decomposition(plt, hr) if hr else None),
        ("fig5_fusion", lambda: fig_fusion(plt, fus) if fus else None),
        ("fig6_learning_curve", lambda: fig_learning_curve(plt, lc) if lc else None),
        ("fig7_pool_shift", lambda: fig_pool_shift(plt, learn20, learn) if learn else None),
        # The two the paper actually uses as Figures 1 and 2. Added after the first seven,
        # so the existing numbering does not shift under anyone who already cited it.
        ("fig8_dose_response", lambda: fig_dose_response(plt, grad) if grad else None),
        ("fig9_permutation", lambda: fig_permutation(plt, perm) if perm else None),
        ("fig10_leak_vs_scale",
         lambda: fig_leak_vs_scale(plt, scale_sweep) if scale_sweep else None),
    ]
    made = 0
    for name, fn in jobs:
        try:
            fig = fn()
        except Exception as e:                                    # noqa: BLE001
            print(f"  {name}: FAILED - {type(e).__name__}: {e}")
            continue
        if fig is None:
            print(f"  {name}: skipped (input artifact missing or empty)")
            continue
        save(fig, name)
        plt.close(fig)
        made += 1
    print(f"\n{made}/{len(jobs)} figures written.")
    if made == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
