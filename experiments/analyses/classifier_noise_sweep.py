#!/usr/bin/env python3
"""Evaluate step: best-arm accuracy across the noise-scale variants + Table 2 anchors.

Protocol is Table 2's, reused rather than reimplemented: leak_gradient.evaluate() --
strictly-unique best arm, ties excluded, RepeatedStratifiedKFold 5x5, gradient boosting
(random_state=seed) and logistic regression, pooled over NQ/TriviaQA/HotpotQA. Run for CV
seeds 0-4, as Table 2's seed-range column is.

Arms evaluated:
  anchors   the five existing Table 2 rungs, read in place from src/router/ (regression
            check: honest 59.9, leak_noisy 72.7, leaky 73.1 must reproduce)
  variants  results/analyses/env_data/*_crc_{mrr20,rankreward}_sNNN[_clip].json from
            build_noise_variants.py. rankreward_s000 is the true-reward copy row.

Retention = (acc(variant) - acc(honest)) / (acc(family at s=0) - acc(honest)), per model.
It is descriptive; nothing here is a significance test.

Writes rows incrementally to rows.jsonl so an interrupted run loses at most one fit;
--resume continues into an existing out-dir, skipping (arm, seed) pairs already present.
--calibrate times one arm at one seed and prints an extrapolated total, then stops.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import common as C
from leak_gradient import evaluate  # noqa: E402  (Table 2's exact protocol)

PAPER_ANCHORS = {"honest_minimal": 0.599, "leak_noisy": 0.727, "leaky": 0.731}


def arm_paths(env_dir: str):
    arms = {name: {ds: C.anchor_path(ds, s) for ds in C.DATASETS} for name, s in C.ANCHOR_SUFFIXES.items()}
    for tag in C.all_variant_tags():
        arms[tag] = {ds: C.variant_path(ds, tag, env_dir) for ds in C.DATASETS}
    return arms


def summarize(rows, arms_order):
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    summ = {}
    for arm in arms_order:
        rs = by.get(arm)
        if not rs:
            continue
        s = {"n_seeds": len(rs), "n_informative": rs[0]["n_informative"], "majority": rs[0]["majority"]}
        for m in ("gbt", "logreg"):
            v = np.array([r[m] for r in rs])
            s[m] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                    "seed_range_pp": float((v.max() - v.min()) * 100),
                    "lift_vs_majority_pp": float((v.mean() - rs[0]["majority"]) * 100)}
        summ[arm] = s
    honest = summ.get("honest_minimal")
    if honest:
        for arm, s in summ.items():
            fam = arm.split("_s")[0] if arm.startswith(("mrr20_", "rankreward_")) else None
            ref = summ.get(f"{fam}_s000") if fam else summ.get("leaky")
            for m in ("gbt", "logreg"):
                s[m]["delta_vs_honest_pp"] = (s[m]["mean"] - honest[m]["mean"]) * 100
                den = (ref[m]["mean"] - honest[m]["mean"]) if ref else None
                s[m]["retention"] = (s[m]["mean"] - honest[m]["mean"]) / den if den and abs(den) > 1e-9 else None
                s[m]["retention_reference"] = f"{fam}_s000" if fam else "leaky"
    # Table 2 reports CV seed 0 (leak_gradient.evaluate's default), not the seed mean.
    checks = {}
    for arm, want in PAPER_ANCHORS.items():
        seed0 = [r for r in by.get(arm, []) if r["seed"] == 0]
        if seed0:
            got = seed0[0]["gbt"]
            checks[arm] = {"got_seed0": got, "paper": want, "ok_to_0.1pp": abs(round(got, 3) - want) < 1.5e-3}
    return summ, checks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-dir", default=C.ANALYSIS_ENV_DIR)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "classifier_noise_sweep"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=None, help="subset of arm names (default: all)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args(argv)

    arms = arm_paths(args.env_dir)
    order = list(arms) if not args.arms else args.arms
    unknown = [a for a in order if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    missing = [a for a in order for p in arms[a].values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"env files missing for {sorted(set(missing))}; run build_noise_variants.py first")

    if args.calibrate:
        X, R = C.pooled(arms["leaky"])
        t = time.time()
        evaluate(X, R, seed=args.seeds[0], n_jobs=args.n_jobs)
        dt = time.time() - t
        n = len(order) * len(args.seeds)
        print(f"one arm x one seed (GB + LR, 5x5 CV): {dt:.1f}s; full sweep = {n} fits "
              f"~ {dt * n / 3600:.2f} h on this machine at --n-jobs {args.n_jobs}")
        return

    if args.resume and os.path.isdir(args.out_dir):
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = C.prepare_out_dir(args.out_dir)
        C.write_json(os.path.join(out_dir, "run_header.json"), C.run_header(args))
    rows_path = os.path.join(out_dir, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            done = {(r["arm"], r["seed"]) for r in map(json.loads, f)}

    ref_R = None
    for arm in order:
        X, R = C.pooled(arms[arm])
        if ref_R is None:
            ref_R = R
        elif not np.array_equal(R, ref_R):
            raise SystemExit(f"INTEGRITY FAILURE: reward table of {arm} differs from {order[0]}")
        for seed in args.seeds:
            if (arm, seed) in done:
                continue
            t = time.time()
            r = evaluate(X, R, seed=seed, n_jobs=args.n_jobs)
            if r is None:
                raise SystemExit(f"{arm}: too few informative queries")
            row = {"arm": arm, "seed": seed, "n_features": int(X.shape[1]), "seconds": round(time.time() - t, 1), **r}
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {arm:24s} seed={seed} gbt={r['gbt']*100:5.2f} logreg={r['logreg']*100:5.2f} ({row['seconds']}s)", flush=True)

    with open(rows_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    summ, checks = summarize(rows, order)
    C.write_json(os.path.join(out_dir, "summary.json"), {"anchor_checks": checks, "arms": summ})
    for arm, c in checks.items():
        print(f"  anchor {arm}: seed 0 got {c['got_seed0']*100:.2f} vs paper {c['paper']*100:.1f} -> {'OK' if c['ok_to_0.1pp'] else 'MISMATCH'}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
