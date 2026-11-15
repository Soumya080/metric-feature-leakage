#!/usr/bin/env python3
"""Positive control: a routing problem with GENUINE signal, with and without a leak.

Answers the review's point 3 ("leakage is sufficient, but you do not show it is necessary;
gains beyond a baseline are not evidence of leakage"). The paper's own data has no exploitable
honest signal, so it cannot show what a genuine gain looks like next to a leak. This builds one:

  utility u[i, a] = alpha * (X[i, :3] @ W[:, a]) + N(0, 1)        X ~ N(0, 1), 20 columns
  reward  R[i, a] = 1 / (1 + floor(exp(1.5 - u[i, a])))           reciprocal-rank-like, in (0, 1]

alpha = 0 is a null (no signal); alpha > 0 makes the winning arm genuinely predictable from
the first three features. For each alpha it reports, under Table 2's protocol
(GradientBoostingClassifier defaults, 5x5 RepeatedStratifiedKFold, strictly-best rows):

  honest            the 20 honest features only
  + exact copy      plus the reward of every arm, verbatim
  + random noise    plus every arm's reward + Uniform(-0.15, 0.15), independent per cell
  + value-keyed     plus the paper's crc32 value-keyed +/-0.15 offset (a recoding)

and leakcheck.audit on each matrix (verdicts, suspicions, and which columns they name).
What it is meant to show: a genuine gain exists without any verdict; a leak still inflates
accuracy beyond the genuine gain; and only the exact copy is a verdict -- leakcheck separates
"the router learned something" from "the features contain the answer" only as far as the
exact/suspicion split allows, which is stated rather than hidden.

Synthetic, read-only, CPU. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common as C
import leakcheck
from leakcheck.taxonomy import _deterministic_jitter

N_HONEST = 20
SIGNAL_COLS = (0, 1, 2)


def make_problem(alpha: float, n: int, seed: int):
    rng = np.random.default_rng([seed, int(round(alpha * 1000))])
    X = rng.standard_normal((n, N_HONEST))
    W = np.random.default_rng(seed).standard_normal((len(SIGNAL_COLS), 3))
    W[:, 1] += 0.6  # arm 1 better on average, so a majority class exists as in real data
    u = alpha * (X[:, SIGNAL_COLS] @ W) + rng.standard_normal((n, 3))
    R = 1.0 / (1.0 + np.floor(np.exp(1.5 - u)))
    return X, R


def variants(X, R, seed: int):
    noise = np.random.default_rng([seed, 99]).uniform(-0.15, 0.15, R.shape)
    keyed = np.column_stack([_deterministic_jitter(R[:, a], 0.15, salt=f"arm{a}") for a in range(R.shape[1])])
    return {"honest": X,
            "exact_copy": np.hstack([X, R]),
            "random_noise": np.hstack([X, R + noise]),
            "value_keyed": np.hstack([X, keyed])}


def evaluate(X, R, seed: int, n_jobs: int):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    majority = float(np.bincount(yi).max() / len(yi))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    acc = cross_val_score(GradientBoostingClassifier(random_state=seed), Xi, yi, cv=cv,
                          scoring="accuracy", n_jobs=n_jobs)
    return {"n_informative": int(len(yi)), "majority": majority, "acc": float(acc.mean()),
            "acc_std": float(acc.std()), "lift": float(acc.mean()) - majority}


def audit(X, R):
    rep = leakcheck.audit(X, R, arm_names=["arm0", "arm1", "arm2"])
    return {"leaked_verdict": bool(rep.leaked), "n_verdicts": len(rep.verdicts),
            "n_suspicions": len(rep.suspicions),
            "verdict_columns": sorted({int(f.feature) for f in rep.verdicts}),
            "suspicion_columns": sorted({int(f.feature) for f in rep.suspicions}),
            "suspicions_on_signal_cols": sorted({int(f.feature) for f in rep.suspicions} & set(SIGNAL_COLS)),
            "suspicions_on_leak_cols": sorted(c for c in {int(f.feature) for f in rep.suspicions} if c >= N_HONEST)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "positive_control"))
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)

    rows = []
    for alpha in args.alphas:
        X, R = make_problem(alpha, args.n, args.seed)
        for name, Xv in variants(X, R, args.seed).items():
            t = time.time()
            row = {"alpha": alpha, "variant": name, **evaluate(Xv, R, args.seed, args.n_jobs), **audit(Xv, R),
                   "seconds": round(time.time() - t, 1)}
            rows.append(row)
            print(f"alpha={alpha:.1f} {name:13s} acc={row['acc']*100:5.1f} majority={row['majority']*100:5.1f} "
                  f"lift={row['lift']*100:+5.1f}  verdicts={row['n_verdicts']} suspicions={row['n_suspicions']} "
                  f"(signal cols {row['suspicions_on_signal_cols']}, leak cols {row['suspicions_on_leak_cols']})",
                  flush=True)
    C.write_json(os.path.join(out_dir, "summary.json"), {"header": C.run_header(args), "rows": rows})
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
