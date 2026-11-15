#!/usr/bin/env python3
"""Table 2's logistic regression refit on standardised features.

leak_gradient.evaluate fits LogisticRegression(max_iter=3000) on unscaled features, so the
paper's LR retention numbers partly measure preprocessing. This re-runs the LR half of Table 2's
protocol (strictly-best arm, ties excluded, RepeatedStratifiedKFold 5x5, CV seeds 0-4) with a
StandardScaler fitted inside each training fold, on:
  * the five Table 2 rungs (src/router/*_env_data{,_leak1,_leakhit,_leaknoisy,_leaky}.json);
  * random_noise_sweep.py's additive noise at s = 0.15 (draws 0-2) and the exact mrr20 matrix;
  * multiplicative_noise.py's noise at s = 0.15 (draws 0-2).
LR only, so it is cheap. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from random_noise_sweep import leak_matrix, load_base, noise
from multiplicative_noise import leak_matrix_mult


def lr_acc(X, R, seed, scaled, n_jobs):
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if scaled \
        else LogisticRegression(max_iter=3000)
    return float(cross_val_score(est, Xi, yi, cv=cv, scoring="accuracy", n_jobs=n_jobs).mean()), \
        float(np.bincount(yi).max() / len(yi))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "scaled_logreg"))
    ap.add_argument("--cv-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)

    mats = {}
    for name, suf in C.ANCHOR_SUFFIXES.items():
        X, R = C.pooled({ds: C.anchor_path(ds, suf) for ds in C.DATASETS}, metric="mrr")
        mats[f"table2_{name}"] = [(X, R)]
    Xb, Rb, vals = load_base()
    mats["mrr20_exact_honest25"] = [(leak_matrix(Xb, vals["mrr20"], 0.0, noise(len(Xb), 0)), Rb)]
    mats["mrr20_add_s015"] = [(leak_matrix(Xb, vals["mrr20"], 0.15, noise(len(Xb), d)), Rb) for d in (0, 1, 2)]
    mats["mrr20_mult_s015"] = [(leak_matrix_mult(Xb, vals["mrr20"], 0.15, noise(len(Xb), d)), Rb) for d in (0, 1, 2)]

    res = {"header": C.run_header(args), "arms": {}}
    for arm, lst in mats.items():
        out = {}
        for scaled in (False, True):
            accs, maj = [], None
            for X, R in lst:
                for s in args.cv_seeds:
                    a, maj = lr_acc(X, R, s, scaled, args.n_jobs)
                    accs.append(a)
            out["scaled" if scaled else "unscaled"] = {"mean": float(np.mean(accs)), "min": float(min(accs)),
                                                       "max": float(max(accs)), "n": len(accs)}
        out["majority"] = maj
        res["arms"][arm] = out
        print(f"{arm:28s} unscaled {100*out['unscaled']['mean']:6.2f}  scaled {100*out['scaled']['mean']:6.2f}  "
              f"(majority {100*maj:.2f})", flush=True)
    C.write_json(os.path.join(out_dir, "summary.json"), res)
    print("Done:", out_dir)


if __name__ == "__main__":
    main()
