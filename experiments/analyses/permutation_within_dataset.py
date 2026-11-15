#!/usr/bin/env python3
"""Is the pooled permutation signal routing signal, or dataset identity?

Figure 2's pooled label-permutation null (results/perm_B1000) shuffles best-arm labels ACROSS
the three datasets. That also destroys the link between dataset identity and class prior, which
the real classifier can exploit because the features identify the dataset (capitalisation,
entities, embedding PCA). So "pooled p = 0.003, the honest features do carry signal" could in
principle be dataset identity.

This null shuffles labels WITHIN each dataset, keeping that link intact. Everything else is
permutation_control._cv_score's protocol: GradientBoostingClassifier defaults, 5x5
RepeatedStratifiedKFold, CV seed 0, informative (strictly-best) rows of the honest 61-D env
files. Replicate b uses numpy default_rng(5000 + b).

If the within-dataset null centres where the across-dataset one does (59.15%, B = 1000), the
pooled signal is not dataset identity.

Reads src/router/{ds}_env_data.json only. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import common as C
from learnability_analysis import N_REPEATS, N_SPLITS, load_queries, reward_matrix  # noqa: E402


def cv_score(X, y, n_jobs=-1):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
    return float(cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=cv,
                                 scoring="accuracy", n_jobs=n_jobs).mean())


def pooled_informative():
    Xs, ys, ds_ids = [], [], []
    for i, ds in enumerate(C.DATASETS):
        _, X, R = reward_matrix(load_queries(C.anchor_path(ds, "")))
        m = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
        Xs.append(X[m]); ys.append(R[m].argmax(axis=1)); ds_ids.append(np.full(int(m.sum()), i))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ds_ids)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "permutation_within_dataset"))
    ap.add_argument("--n-perm", type=int, default=60)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)

    X, y, d = pooled_informative()
    t = time.time()
    observed = cv_score(X, y, args.n_jobs)
    print(f"n={len(y)} observed={observed*100:.3f} ({time.time()-t:.0f}s/fit)", flush=True)
    null = []
    for b in range(args.n_perm):
        rng = np.random.default_rng(5000 + b)
        yp = y.copy()
        for i in range(len(C.DATASETS)):
            idx = np.where(d == i)[0]
            yp[idx] = y[idx][rng.permutation(len(idx))]
        null.append(cv_score(X, yp, args.n_jobs))
        print(f"  b={b} {null[-1]*100:.3f}", flush=True)
    null = np.array(null)
    k = int((null >= observed).sum())
    C.write_json(os.path.join(out_dir, "summary.json"), {
        "header": C.run_header(args), "n": int(len(y)), "observed": observed, "B": args.n_perm,
        "within_ds_null_mean": float(null.mean()), "within_ds_null_std": float(null.std()),
        "k": k, "p": (1 + k) / (1 + args.n_perm), "null": null.tolist(),
        "reference_across_ds_null_mean_B1000": 0.5914988346456693})
    print(f"within-dataset null {null.mean()*100:.2f} +/- {null.std()*100:.2f}; "
          f"observed {observed*100:.2f}; p = {(1 + k) / (1 + args.n_perm):.3f}")


if __name__ == "__main__":
    main()
