"""06 — Leak severity as a function of corpus size.

The question this whole directory exists to answer: does metric-as-feature contamination
inflate a router more, or less, as the corpus grows?

It is not obvious in either direction, and the nearest literature points the other way.
Privacy/membership-inference leakage *decreases* with scale, because a model with more data
memorises less. The mechanism here is different: a bigger corpus makes retrieval harder,
which makes rewards sparser and label-free features weaker, while the leaked feature stays
exactly as informative as it ever was -- it is the answer. If that reasoning holds, the leak
should buy *more* at scale, and the regime where routing matters most is the regime where a
leak is most inflationary.

Runs the leak-gradient protocol of experiments/router_experiments/leak_gradient.py at each
corpus size, holding everything else fixed. Accuracy is for predicting the strictly-best
arm, 5-fold stratified CV x 5 repeats, pooled over datasets, against the majority-class
baseline at that scale.

Requires 05_build_arms.py to have run for the scales requested.

Usage:
    python experiments/corpus_v3/06_leak_vs_scale.py
    python experiments/corpus_v3/06_leak_vs_scale.py --scale 20000 50000 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common  # noqa: E402

sys.path.insert(0, os.path.join(common.PROJECT_ROOT, "experiments", "router_experiments"))
from learnability_analysis import (  # noqa: E402
    ACTIONS, N_REPEATS, N_SPLITS, load_queries, reward_matrix,
)

SEED = 0
# Same six arms and the same display order as leakage_ablation, so the scaled table can be
# read directly against the 500-passage one in the paper.
ARM_SUFFIX = {
    "honest_minimal": "",
    "honest_calibrated": "_cal",
    "leak_one_arm": "_leak1",
    "leak_hit": "_leakhit",
    "leak_noisy": "_leaknoisy",
    "leaky": "_leaky",
}
ARM_ORDER = ["honest_minimal", "honest_calibrated", "leak_one_arm",
             "leak_hit", "leak_noisy", "leaky"]


def env_path(scale: int, dataset: str, suffix: str) -> str:
    return os.path.join(common.ARTIFACT_DIR, f"scale_{scale}", "env",
                        f"{dataset}_env_data{suffix}.json")


def pooled_accuracy(scale: int, suffix: str):
    """GBT accuracy for one arm at one scale, pooled across datasets."""
    X_parts, R_parts = [], []
    for ds in common.DATASETS:
        p = env_path(scale, ds, suffix)
        if not os.path.exists(p):
            continue
        _, X, R = reward_matrix(load_queries(p), "mrr")
        X_parts.append(X)
        R_parts.append(R)
    if not X_parts:
        return None

    X = np.vstack(X_parts)
    R = np.vstack(R_parts)

    # Only queries with a single strictly-best arm carry a routing label at all.
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    counts = np.bincount(yi, minlength=len(ACTIONS))
    if len(yi) < N_SPLITS * 2 or (counts[counts > 0] < N_SPLITS).any():
        return None

    majority = float(counts.max() / len(yi))
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    scores = cross_val_score(GradientBoostingClassifier(random_state=SEED),
                             Xi, yi, cv=cv, scoring="accuracy")
    return {
        "n_informative": int(len(yi)),
        "n_total": int(len(X)),
        "majority": majority,
        "accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "lift": float(scores.mean() - majority),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=int, nargs="+", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    scales = args.scale or [s for s in (5_000, 20_000, 50_000, 100_000)
                            if os.path.isdir(os.path.join(common.ARTIFACT_DIR,
                                                          f"scale_{s}", "env"))]
    if not scales:
        print("No built arms found. Run 05_build_arms.py first.")
        return 1

    out = {"seed": SEED, "cv": f"{N_SPLITS}x{N_REPEATS}", "scales": {}}

    for scale in scales:
        print(f"\nscale {common.fmt_int(scale)}")
        per_arm = {}
        for arm in ARM_ORDER:
            res = pooled_accuracy(scale, ARM_SUFFIX[arm])
            if res is None:
                print(f"  {arm:20s} missing")
                continue
            per_arm[arm] = res
            print(f"  {arm:20s} acc={res['accuracy']*100:5.2f}%  "
                  f"majority={res['majority']*100:5.2f}%  "
                  f"lift={res['lift']*100:+5.2f}  n={res['n_informative']}")
        out["scales"][str(scale)] = per_arm

    # --- The curve: what the leak buys over honest features, at each scale ---------
    print(f"\n{'=' * 74}")
    print("LEAK PREMIUM  (arm accuracy minus honest_minimal accuracy, percentage points)")
    print(f"{'=' * 74}")
    header = f"{'arm':<20}" + "".join(f"{common.fmt_int(s):>13}" for s in scales)
    print(header)
    for arm in ARM_ORDER:
        if arm == "honest_minimal":
            continue
        row = f"{arm:<20}"
        for s in scales:
            a = out["scales"][str(s)]
            if arm in a and "honest_minimal" in a:
                row += f"{(a[arm]['accuracy'] - a['honest_minimal']['accuracy']) * 100:>+13.2f}"
            else:
                row += f"{'--':>13}"
        print(row)

    print(f"\n{'honest lift vs majority':<20}" +
          "".join(f"{out['scales'][str(s)]['honest_minimal']['lift'] * 100:>+13.2f}"
                  if 'honest_minimal' in out["scales"][str(s)] else f"{'--':>13}"
                  for s in scales))
    print(f"{'informative n':<20}" +
          "".join(f"{out['scales'][str(s)]['honest_minimal']['n_informative']:>13}"
                  if 'honest_minimal' in out["scales"][str(s)] else f"{'--':>13}"
                  for s in scales))

    path = args.output or os.path.join(common.RESULTS_DIR, "leak_vs_scale.json")
    common.write_manifest(path, out)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
