#!/usr/bin/env python3
"""Power and intervals on the population that Table 3's capture is normalised by.

Computes:
  * The paper's minimum detectable capture (two_arm_ablation.mde) used the whole-corpus SD and
    headroom with n_test = round(0.15 N). Table 3 normalises capture by TEST-split headroom.
    This script recomputes the MDE on the actual split_seed=42 test queries, three arms and two
    arms (BM25 + Dense), one- and two-sided, alpha 0.05, power 0.80.
  * A paired bootstrap (resampling test queries) gives a 95% percentile interval on every
    method's capture, so the "+0.4%" headline carries an interval.

Same optimistic MDE construction as two_arm_ablation.mde: the difference series is the per-query
oracle gain over best-fixed, the largest difference any router could produce.

Reads results/canonical_results.json only. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import common as C

Z_TWO = 1.959964 + 0.841621   # two-sided alpha .05, power .80 -> 2.8016 (Z^2 = 7.849)
Z_ONE = 1.644854 + 0.841621   # one-sided alpha .05, power .80 -> 2.4865
ARMS3 = ("bm25", "dense", "hybrid")
ARMS2 = ("bm25", "dense")
METHODS = ("score_gap", "logistic_regression", "random_forest", "ridge", "gradient_boosting", "rl_route")


def mde(R: np.ndarray) -> dict:
    best = int(R.mean(0).argmax())
    diff = R.max(1) - R[:, best]
    n, sd, head = len(diff), float(diff.std(ddof=1)), float(diff.mean())
    out = {"n_test": n, "best_fixed_col": best, "diff_sd": sd, "headroom_abs": head,
           "headroom_rel": head / float(R[:, best].mean())}
    for name, z in (("two_sided", Z_TWO), ("one_sided", Z_ONE)):
        out[f"mde_abs_{name}"] = z * sd / math.sqrt(n)
        out[f"min_capture_{name}"] = out[f"mde_abs_{name}"] / head
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "split_power"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    path = os.path.join(C.RESULTS_DIR, "canonical_results.json")
    import json
    canon = json.load(open(path, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    res = {"header": {**C.run_header(args), "input": path, "input_sha256": C.sha256_file(path)}, "datasets": {}}
    for ds in C.DATASETS:
        sc = canon[ds]["per_query"]["scores"]
        R3 = np.column_stack([np.asarray(sc[a], float) for a in ARMS3])
        R2 = np.column_stack([np.asarray(sc[a], float) for a in ARMS2])
        best = int(R3.mean(0).argmax())
        fixed, orc = R3[:, best], R3.max(1)
        n = len(fixed)
        idx = rng.integers(0, n, size=(args.boot, n))
        f_b, o_b = fixed[idx].mean(1), orc[idx].mean(1)
        caps = {}
        for m in METHODS:
            v = np.asarray(sc[m], float)
            point = (v.mean() - fixed.mean()) / (orc.mean() - fixed.mean())
            boot = (v[idx].mean(1) - f_b) / (o_b - f_b)
            caps[m] = {"mean_reward": float(v.mean()), "capture": float(point),
                       "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        res["datasets"][ds] = {"best_fixed": ARMS3[best], "best_fixed_reward": float(fixed.mean()),
                               "oracle_reward": float(orc.mean()),
                               "mde_three_arms": mde(R3), "mde_two_arms_bm25_dense": mde(R2), "capture": caps}
    C.write_json(os.path.join(out_dir, "summary.json"), res)
    for ds, d in res["datasets"].items():
        m3, m2 = d["mde_three_arms"], d["mde_two_arms_bm25_dense"]
        print(f"{ds:9s} n={m3['n_test']} headroom {100*m3['headroom_rel']:.1f}%  MDE/headroom 3-arm "
              f"two-sided {100*m3['min_capture_two_sided']:.0f}% one-sided {100*m3['min_capture_one_sided']:.0f}% | "
              f"2-arm two-sided {100*m2['min_capture_two_sided']:.0f}% one-sided {100*m2['min_capture_one_sided']:.0f}%")
        for m, c in d["capture"].items():
            print(f"    {m:20s} {c['mean_reward']:.4f} capture {100*c['capture']:+6.1f}% "
                  f"CI [{100*c['ci95'][0]:+6.1f}, {100*c['ci95'][1]:+6.1f}]")
    print("Done:", out_dir)


if __name__ == "__main__":
    main()
