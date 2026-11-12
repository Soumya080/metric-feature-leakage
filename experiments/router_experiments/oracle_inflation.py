"""
oracle_inflation.py — How much of the reported routing headroom is real, and how much is
noise being maximised over?

The oracle in `learnability_analysis.py` is computed as `R.max(axis=1).mean()`: for every
query, take the best of the three observed rewards. That is an upward-biased estimate of
what a perfect router could actually achieve, and the bias has a name — it is the
max-over-noisy-arms problem, an instance of Jensen's inequality. Each observed reward is the
true quality of that retriever on that query plus measurement noise (which passage the
answer string happened to match, where the distractor draw put the gold passage). Taking a
max selects, disproportionately, arms whose noise realisation was favourable. So

    E[max over arms of (truth + noise)]  >=  max over arms of E[truth + noise]

and the gap is pure inflation: headroom that no router can capture, because there is nothing
there to predict. The tighter the arms are clustered, the larger the share of the max that is
noise — and this repo's arms are very tightly clustered (Hybrid is RRF of the other two).

This matters because oracle headroom is the entire motivation for routing. If most of the
26–51% headroom reported here is inflation, then "no method captures the headroom" is not a
failure of the methods; the headroom was never there.

MEASURING IT WITHOUT A SECOND CORPUS
------------------------------------
A same-split oracle re-uses the same reward draw both to pick the arm and to score it, which
is what creates the bias. Splitting the *metric* is not possible here (there is one draw per
query/arm), so this script uses the standard alternatives:

  ARM-SPLIT (the honest one). Pick the arm on one half of the queries and evaluate that
      *decision rule* on the other half. Concretely: the best-fixed arm chosen on half A,
      scored on half B. This does not de-bias the per-query oracle -- nothing can, from a
      single draw -- but it does quantify how much of "best fixed" is itself a selection
      artifact, which is the same mechanism at a scale we CAN cross-validate.

  NOISE INJECTION (the diagnostic). Add zero-mean noise of a known scale to the reward table
      and recompute the oracle. A headroom made of real per-query differences is insensitive
      to small noise; a headroom made of near-ties inflates steeply, because added noise
      gives the max more to select from. The slope of oracle-vs-noise is therefore a direct
      readout of how much of the headroom rests on ties, and it needs no extra data.

  TIE-MARGIN DECOMPOSITION (the explanation). What fraction of the headroom comes from
      queries where the winning arm beat the runner-up by a negligible margin? Those are
      precisely the queries where the winner is decided by noise rather than by quality.

Reported together, these three say whether the paper's motivating quantity survives
scrutiny. Interpretation is stated in the output rather than left to the reader.

Usage:
    python experiments/router_experiments/oracle_inflation.py
    python experiments/router_experiments/oracle_inflation.py --n-splits 200 --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]

# Noise scales as a fraction of the reward table's own std, so the sweep is scale-free and
# comparable across datasets whose MRR ranges differ by 3x.
NOISE_FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)

# "Negligible" margin for the tie decomposition, as a fraction of the best-fixed mean. A
# winner that beats the runner-up by less than this is not distinguishable from a tie at the
# resolution this metric actually has.
MARGIN_FRACTIONS = (0.01, 0.05, 0.10, 0.25)


def load_reward_matrix(path, metric="mrr"):
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    queries = env.get("queries", env)
    return np.array([[float(queries[q]["retrievers"][a][metric]) for a in ACTIONS]
                     for q in queries], dtype=float)


def arm_split_headroom(R, n_splits, rng):
    """Best-fixed chosen on half A, scored on half B, vs. chosen and scored on the same half.

    The difference is the selection bias in 'best fixed' -- the same mechanism that inflates
    the per-query oracle, at the one scale where a held-out estimate is available.
    """
    n = len(R)
    same, cross = [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        arm_a = int(R[a].mean(axis=0).argmax())
        same.append(float(R[a].mean(axis=0)[arm_a]))   # chosen and scored on A
        cross.append(float(R[b].mean(axis=0)[arm_a]))  # chosen on A, scored on B
    return float(np.mean(same)), float(np.mean(cross))


def noise_sweep(R, rng, n_reps=50):
    """Oracle as a function of injected zero-mean noise. Steep slope => tie-driven headroom."""
    sd = float(R.std())
    best_fixed = float(R.mean(axis=0).max())
    out = []
    for frac in NOISE_FRACTIONS:
        if frac == 0.0:
            oracle = float(R.max(axis=1).mean())
        else:
            vals = [float((R + rng.normal(0.0, frac * sd, R.shape)).max(axis=1).mean())
                    for _ in range(n_reps)]
            oracle = float(np.mean(vals))
        out.append({"noise_frac": frac, "oracle": oracle,
                    "headroom": oracle - best_fixed})
    return out, best_fixed, sd


def margin_decomposition(R):
    """Share of total headroom contributed by queries won by a negligible margin.

    Headroom is sum(max - best_fixed_arm_value) over queries; a query's contribution is how
    much the oracle gains on it over committing to the best fixed arm. We bucket those
    contributions by the winner's margin over the runner-up.
    """
    best_arm = int(R.mean(axis=0).argmax())
    gain = R.max(axis=1) - R[:, best_arm]          # per-query oracle gain, >= 0
    total = float(gain.sum())
    srt = np.sort(R, axis=1)
    margin = srt[:, -1] - srt[:, -2]               # winner minus runner-up
    scale = float(R.mean(axis=0).max())
    out = []
    for f in MARGIN_FRACTIONS:
        thresh = f * scale
        share = float(gain[margin <= thresh].sum()) / total if total > 0 else float("nan")
        out.append({"margin_frac_of_best_fixed": f, "margin_abs": thresh,
                    "share_of_headroom": share,
                    "n_queries": int((margin <= thresh).sum())})
    return out, total


def analyse(path, n_splits, seed):
    R = load_reward_matrix(path)
    rng = np.random.default_rng(seed)
    best_fixed = float(R.mean(axis=0).max())
    oracle = float(R.max(axis=1).mean())

    same, cross = arm_split_headroom(R, n_splits, rng)
    sweep, _, sd = noise_sweep(R, rng)
    margins, total_gain = margin_decomposition(R)

    return {
        "n": int(len(R)),
        "best_fixed_arm": ACTIONS[int(R.mean(axis=0).argmax())],
        "best_fixed": best_fixed,
        "oracle": oracle,
        "headroom": oracle - best_fixed,
        "headroom_rel": (oracle - best_fixed) / best_fixed if best_fixed > 0 else float("nan"),
        "reward_std": sd,
        "arm_split": {"same_split": same, "cross_split": cross,
                      "selection_bias": same - cross, "n_splits": n_splits},
        "noise_sweep": sweep,
        "margin_decomposition": margins,
        "total_oracle_gain": total_gain,
    }


def render_markdown(per_ds, seed, n_splits):
    L = ["# Oracle Inflation Analysis\n",
         f"_Seed {seed} · {n_splits} random arm-splits · noise sweep averaged over 50 draws_\n",
         "\nOracle headroom is the motivation for routing, and it is computed as a max over "
         "three noisy arms — which is upward-biased by Jensen's inequality. These three "
         "measurements say how much of the headroom is real.\n",
         "\n## Table 1 — Headroom, and the selection bias in choosing an arm\n",
         "| dataset | n | best fixed | oracle | headroom | same-split | cross-split | selection bias |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, e in per_ds.items():
        a = e["arm_split"]
        L.append(f"| {ds} | {e['n']} | {e['best_fixed_arm']} {e['best_fixed']:.4f} | "
                 f"{e['oracle']:.4f} | **+{e['headroom']:.4f}** "
                 f"({e['headroom_rel']*100:.0f}%) | {a['same_split']:.4f} | "
                 f"{a['cross_split']:.4f} | {a['selection_bias']:+.4f} |")
    L += ["\n`same-split` picks the best fixed arm on one half of the queries and scores it on "
          "that same half; `cross-split` picks on one half and scores on the other. Their "
          "difference is how much of `best fixed` is a selection artifact. A small bias here "
          "means arm selection is stable; it does not by itself de-bias the per-query oracle, "
          "which cannot be cross-validated from a single reward draw.\n",
          "\n## Table 2 — Oracle under injected zero-mean noise\n",
          "| dataset | " + " | ".join(f"+{int(f*100)}% sd" for f in NOISE_FRACTIONS) + " | inflation slope |",
          "|---|" + "---:|" * (len(NOISE_FRACTIONS) + 1)]
    for ds, e in per_ds.items():
        row = [f"{s['headroom']:.4f}" for s in e["noise_sweep"]]
        base = e["noise_sweep"][0]["headroom"]
        top = e["noise_sweep"][-1]["headroom"]
        slope = (top - base) / base * 100 if base > 0 else float("nan")
        L.append(f"| {ds} | " + " | ".join(row) + f" | **{slope:+.0f}%** |")
    L += ["\nHeadroom after adding noise of the stated fraction of the reward table's own "
          "standard deviation. Real per-query differences are insensitive to small noise; "
          "headroom built on near-ties inflates steeply, because the max has more noise to "
          "select from. The slope is the increase in headroom from 0% to 100% sd.\n",
          "\n**Read this as a sensitivity, not a proof of inflation.** A max over three arms "
          "inflates under *any* added noise, so a positive slope is expected and is not by "
          "itself evidence that the reported headroom is an artifact. What the column "
          "licenses is a conditional: if the reward table carries measurement noise of a "
          "given size — and it does, since which passage the answer string matches depends "
          "on the distractor draw — then that column says how much of the headroom that noise "
          "accounts for. Quote the +10% column, not the +100% one.\n",
          "\n## Table 3 — Which queries the headroom comes from\n",
          "| dataset | " + " | ".join(f"margin <= {int(f*100)}%" for f in MARGIN_FRACTIONS) + " |",
          "|---|" + "---:|" * len(MARGIN_FRACTIONS)]
    for ds, e in per_ds.items():
        L.append(f"| {ds} | " + " | ".join(
            f"{m['share_of_headroom']*100:.1f}% ({m['n_queries']}q)"
            for m in e["margin_decomposition"]) + " |")
    L += ["\nShare of total oracle headroom contributed by queries where the winning "
          "retriever beat the runner-up by no more than the stated fraction of the best-fixed "
          "mean. Headroom concentrated in tiny margins is headroom decided by measurement "
          "noise rather than by retriever quality — a router cannot learn to predict it "
          "because there is no stable signal to predict.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--n-splits", type=int, default=200,
                   help="Random half-splits for the arm-selection bias estimate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    args = p.parse_args()

    env_dir = args.env_dir or os.path.join(PROJECT_ROOT, "src", "router")
    per_ds = {}
    for ds in ("nq", "triviaqa", "hotpotqa"):
        path = os.path.join(env_dir, f"{ds}_env_data{args.env_suffix}.json")
        if not os.path.exists(path):
            print(f"  {ds}: env data missing at {path} — skipped.")
            continue
        per_ds[ds] = analyse(path, args.n_splits, args.seed)
        e = per_ds[ds]
        print(f"  {ds}: headroom +{e['headroom']:.4f} "
              f"({e['headroom_rel']*100:.0f}%), arm-selection bias "
              f"{e['arm_split']['selection_bias']:+.4f}")

    if not per_ds:
        print("No env data found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    js = os.path.join(args.outdir, "oracle_inflation.json")
    md = os.path.join(args.outdir, "oracle_inflation.md")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "n_splits": args.n_splits,
                   "noise_fractions": list(NOISE_FRACTIONS),
                   "margin_fractions": list(MARGIN_FRACTIONS),
                   "per_dataset": per_ds}, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_markdown(per_ds, args.seed, args.n_splits))

    print()
    print(render_markdown(per_ds, args.seed, args.n_splits))
    print(f"Saved: {js}\nSaved: {md}")


if __name__ == "__main__":
    main()
