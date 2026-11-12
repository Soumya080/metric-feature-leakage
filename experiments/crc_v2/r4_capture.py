#!/usr/bin/env python3
"""R4 (+O1): apparent oracle-gap capture of leaky routers, in Table 3's terms.

Uses canonical_evaluation's own functions, unchanged, so "tuned gradient boosting" means
what it means in Table 3: create_train_val_test_split(split_seed) -> compute_static_baselines
-> compute_supervised_baselines (LR, RF, ridge, RandomizedSearchCV-tuned GBT; fit on train,
scored once on test). The validation split is used only to pick the deployable fixed arm.

capture = (method - fixed) / (oracle - fixed), unclipped; None if oracle - fixed < 1e-12.
Two comparators, kept in separate fields:
  test_best_fixed       the paper's Table 3 comparator (hindsight; diagnostic)
  val_selected_fixed    arm with the best validation mean (deployable)

Split seed 42 is Table 3's split; honest_minimal at seed 42 must reproduce Table 3's GBT
row (0.1642 / 0.2000 / 0.0753) -- checked and reported. Extra --split-seeds give O1
(robustness of Table 3 across partitions). Contaminated arms are invalid-at-decision-time
controls: their capture is "what a leaked router would appear to achieve", not a router.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import common as C
from canonical_evaluation import compute_static_baselines, compute_supervised_baselines  # noqa: E402
from r1_classifier_sweep import arm_paths
from src.router.routing_env import create_train_val_test_split  # noqa: E402

DEFAULT_ARMS = ["honest_minimal", "leak_one_arm", "leak_hit", "leak_noisy", "leaky",
                "mrr20_s000", "mrr20_s015", "rankreward_s000", "rankreward_s015"]
TABLE3_GBT = {"nq": 0.1642, "triviaqa": 0.2000, "hotpotqa": 0.0753}
METHODS = ("logistic_regression", "random_forest", "ridge", "gradient_boosting")


def capture(method: float, fixed: float, oracle: float):
    gap = oracle - fixed
    return None if gap <= 1e-12 else (method - fixed) / gap


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-dir", default=C.CRC_ENV_DIR)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "r4_capture"))
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--datasets", nargs="+", choices=C.DATASETS, default=list(C.DATASETS))
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[42],
                    help="42 = Table 3's split. Add e.g. 7 13 99 2024 for O1.")
    args = ap.parse_args(argv)

    arms = arm_paths(args.env_dir)
    bad = [a for a in args.arms if a not in arms or not all(os.path.exists(arms[a][d]) for d in args.datasets)]
    if bad:
        raise SystemExit(f"unknown arm or env file missing: {bad}")
    out_dir = C.prepare_out_dir(args.out_dir)

    rows, checks = [], {}
    for arm in args.arms:
        for ds in args.datasets:
            for split_seed in args.split_seeds:
                t = time.time()
                tr, va, te = create_train_val_test_split(arms[arm][ds], split_seed=split_seed)
                test_means, _ = compute_static_baselines(te)
                val_means, _ = compute_static_baselines(va)
                sup, _ = compute_supervised_baselines(tr, te)
                oracle = test_means["oracle"]
                tb = max(C.ACTIONS, key=lambda a: test_means[a])
                vs = max(C.ACTIONS, key=lambda a: val_means[a])
                for m in METHODS:
                    rows.append({
                        "arm": arm, "dataset": ds, "split_seed": split_seed, "method": m,
                        "n_train": len(tr.query_pool), "n_val": len(va.query_pool), "n_test": len(te.query_pool),
                        "method_reward": sup[m], "oracle": oracle,
                        "test_best_fixed_arm": tb, "test_best_fixed_reward": test_means[tb],
                        "capture_vs_test_best": capture(sup[m], test_means[tb], oracle),
                        "val_selected_fixed_arm": vs, "val_selected_fixed_reward": test_means[vs],
                        "capture_vs_val_selected": capture(sup[m], test_means[vs], oracle),
                    })
                if arm == "honest_minimal" and split_seed == 42:
                    got = sup["gradient_boosting"]
                    checks[ds] = {"got": got, "table3": TABLE3_GBT[ds], "ok_4dp": round(got, 4) == TABLE3_GBT[ds]}
                gb = [r for r in rows if r["method"] == "gradient_boosting"][-1]
                print(f"  {arm:18s} {ds:9s} split={split_seed} gbt={gb['method_reward']:.4f} "
                      f"capture(test-best)={gb['capture_vs_test_best']} ({time.time() - t:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "capture_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    C.write_json(os.path.join(out_dir, "capture.json"),
                 {"header": C.run_header(args), "table3_regression_check": checks, "rows": rows})
    for ds, c in checks.items():
        print(f"  Table 3 check {ds}: got {c['got']:.4f} vs {c['table3']:.4f} -> {'OK' if c['ok_4dp'] else 'MISMATCH'}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
