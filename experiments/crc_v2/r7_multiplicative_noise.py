#!/usr/bin/env python3
"""R7: R5's independent-noise sweep with MULTIPLICATIVE instead of additive noise.

Answers review_2 item A2. R5 perturbs the leaked MRR@20 additively,
    slot = v + s * u,          u ~ Uniform(-1, 1) per query and arm,
so "s = 0.15" is +/-0.15 in absolute reward units, which is large next to typical non-zero
MRR@20 values (1/k for k <= 20). A reader of "15% noise" will usually assume relative noise:
    slot = v * (1 + s * u).
Under relative noise, zeros stay zero and 1/k vs 1/(k+1) stays mostly separable, so the leak may
keep much more of its gain. This script measures that with exactly R5's construction otherwise:
same honest base, same slots (22 = BM25, 23 = Dense, 24 = 23 - 22, 25 honest), same noise draws
(numpy default_rng([draw, arm])), same Table 2 protocol (leak_gradient.evaluate: GB + LR,
5x5 CV, ties excluded), same CV seeds 0-4. Retention = (acc - honest) / (exact - honest).

Reads src/router/{ds}_env_data.json only. Writes --out-dir (refuses non-empty; --resume).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import common as C
from leak_gradient import evaluate  # noqa: E402  (Table 2's exact protocol)
from r5_random_noise_sweep import audit, load_base, noise, summarize


def leak_matrix_mult(X: np.ndarray, vals: np.ndarray, scale: float, u: np.ndarray) -> np.ndarray:
    Xl = X.copy()
    Xl[:, 22] = vals[:, 0] * (1.0 + scale * u[:, 0])
    Xl[:, 23] = vals[:, 1] * (1.0 + scale * u[:, 1])
    Xl[:, 24] = Xl[:, 23] - Xl[:, 22]
    return Xl


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "r7_multiplicative_noise"))
    ap.add_argument("--scales", nargs="+", type=float, default=[0.15, 0.50, 1.00])
    ap.add_argument("--draws", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cv-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--anchors-from", default=None,
                    help="R5 summary.json: reuse its honest_minimal and mrr20_exact means (identical "
                         "matrices and protocol) instead of re-fitting them")
    args = ap.parse_args(argv)

    X, R, vals = load_base()
    v = vals["mrr20"]
    plan = [] if args.anchors_from else [("honest_minimal", None, 0.0, 0), ("mrr20_exact", "mrr20", 0.0, 0)]
    plan +=[(f"mrr20_mult_s{int(round(s * 100)):03d}", "mrr20", s, d) for s in args.scales for d in args.draws]

    if args.resume and os.path.isdir(args.out_dir):
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = C.prepare_out_dir(args.out_dir)
        C.write_json(os.path.join(out_dir, "run_header.json"),
                     {**C.run_header(args), "n_rows_pooled": int(len(X)), "noise": "multiplicative v*(1+s*u)"})
    rows_path, audit_path = os.path.join(out_dir, "rows.jsonl"), os.path.join(out_dir, "leakcheck.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {(r["arm"], r["draw"], r["seed"]) for r in map(json.loads, open(rows_path, encoding="utf-8"))}
    audited = set()
    if os.path.exists(audit_path):
        audited = {(r["arm"], r["draw"]) for r in map(json.loads, open(audit_path, encoding="utf-8"))}

    for arm, fam, s, d in plan:
        Xl = X if fam is None else leak_matrix_mult(X, v, s, noise(len(X), d))
        if (arm, d) not in audited:
            # exact-agreement rate of the perturbed leak with the full-rank reward
            ex = {k: float(np.mean(np.isclose(Xl[:, slot], R[:, a]) & (R[:, a] > 0)))
                  for k, slot, a in (("bm25", 22, 0), ("dense", 23, 1))}
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"arm": arm, "family": fam, "scale": s, "draw": d,
                                    "nonzero_exact_rate": ex, **audit(Xl, R)}) + "\n")
        for seed in args.cv_seeds:
            if (arm, d, seed) in done:
                continue
            t = time.time()
            r = evaluate(Xl, R, seed=seed, n_jobs=args.n_jobs)
            row = {"arm": arm, "family": fam, "scale": s, "draw": d, "seed": seed,
                   "seconds": round(time.time() - t, 1), **r}
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {arm:20s} draw={d} seed={seed} gbt={r['gbt']*100:5.2f} logreg={r['logreg']*100:5.2f} "
                  f"({row['seconds']}s)", flush=True)

    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    summ = summarize(rows)
    anchors = None
    if args.anchors_from:
        r5 = json.load(open(args.anchors_from, encoding="utf-8"))["arms"]
        anchors = {"source": os.path.abspath(args.anchors_from),
                   "honest": {m: r5["honest_minimal"][m]["mean"] for m in ("gbt", "logreg")},
                   "exact": {m: r5["mrr20_exact"][m]["mean"] for m in ("gbt", "logreg")}}
        for arm, s in summ.items():
            for m in ("gbt", "logreg"):
                h, e = anchors["honest"][m], anchors["exact"][m]
                s[m]["delta_vs_honest_pp"] = (s[m]["mean"] - h) * 100
                s[m]["retention"] = (s[m]["mean"] - h) / (e - h)
    C.write_json(os.path.join(out_dir, "summary.json"), {"anchors": anchors, "arms": summ})
    print(f"\n{'arm':20s} {'GB':>6} {'kept':>6} {'LR':>6} {'kept':>6}")
    for arm, s in summ.items():
        k = lambda x: "   -  " if x is None else f"{x:6.2f}"  # noqa: E731
        print(f"{arm:20s} {s['gbt']['mean']*100:6.2f} {k(s['gbt']['retention'])} "
              f"{s['logreg']['mean']*100:6.2f} {k(s['logreg']['retention'])}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
