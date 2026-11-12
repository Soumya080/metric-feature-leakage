#!/usr/bin/env python3
"""R5: the R1 noise sweep with genuinely random, per-query noise.

Why this exists: R1 (and the paper's leak_noisy rung) perturbs with crc32 of the VALUE, so
every query with the same reward gets the same offset. MRR@20 has 21 distinct values, so that
"noise" is a lossless relabelling a tree can decode at any scale (validated: 0 collisions at
s = 1.0; GB keeps 92% of the gain). This run answers the question the paper's wording implies:
how much of the leak survives INDEPENDENT noise?

Construction (in memory; no env files written). Pooled over NQ/TriviaQA/HotpotQA in the same
row order as Table 2 (leak_gradient.load_arm):
    u_arm ~ Uniform(-1, 1), one independent draw per query and arm, from
            numpy default_rng([draw, arm_index])  -- arm 0 = bm25, 1 = dense
    slot 22 = v_bm25  + s * u_bm25
    slot 23 = v_dense + s * u_dense
    slot 24 = slot 23 - slot 22
    slot 25 = honest value (as in R1)
  v = mrr_at_20 (family "mrr20") or rank_reward (family "rankreward").
  The same u is reused across scales within a draw (common random numbers), so the curve over
  s is not confounded by a different noise realisation at each point. Several draws give the
  noise-realisation spread.

Evaluated with Table 2's protocol (leak_gradient.evaluate: GB + LR, 5x5 CV, ties excluded)
at --cv-seeds, plus leakcheck.audit on the pooled matrix against rank_reward.
Retention = (acc - acc_honest) / (acc_exact_family - acc_honest), per model; exact = s 0.

Reads src/router/{ds}_env_data.json only. Writes --out-dir (refuses non-empty; --resume
continues an interrupted run, skipping (arm, draw, cv_seed) rows already present).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import common as C
import leakcheck
from leakcheck import detectors as D
from leak_gradient import evaluate  # noqa: E402  (Table 2's exact protocol)

SCALES = tuple(s for s in C.NOISE_SCALES if s > 0)


def load_base():
    """Pooled honest features plus the per-arm values each family leaks."""
    X, R, v = [], [], {"mrr20": [], "rankreward": []}
    for ds in C.DATASETS:
        q = C.load_env(C.anchor_path(ds, ""))
        for qid, e in q.items():
            X.append(e["features"])
            R.append([float(e["retrievers"][a]["rank_reward"]) for a in C.ACTIONS])
            for fam, key in (("mrr20", "mrr_at_20"), ("rankreward", "rank_reward")):
                v[fam].append([float(e["retrievers"][a][key]) for a in ("bm25", "dense")])
    X, R = np.asarray(X, dtype=float), np.asarray(R, dtype=float)
    if not np.array_equal(R, C.pooled({ds: C.anchor_path(ds, "") for ds in C.DATASETS}, metric="mrr")[1]):
        raise SystemExit("INTEGRITY FAILURE: rank_reward != mrr in the honest env files")
    return X, R, {k: np.asarray(val) for k, val in v.items()}


def noise(n: int, draw: int) -> np.ndarray:
    """(n, 2) independent Uniform(-1, 1) per query, columns bm25 and dense."""
    return np.column_stack([np.random.default_rng([draw, a]).uniform(-1.0, 1.0, n) for a in (0, 1)])


def leak_matrix(X: np.ndarray, vals: np.ndarray, scale: float, u: np.ndarray) -> np.ndarray:
    Xl = X.copy()
    Xl[:, 22] = vals[:, 0] + scale * u[:, 0]
    Xl[:, 23] = vals[:, 1] + scale * u[:, 1]
    Xl[:, 24] = Xl[:, 23] - Xl[:, 22]
    return Xl


def audit(X: np.ndarray, R: np.ndarray) -> dict:
    rep = leakcheck.audit(X, R, arm_names=list(C.ACTIONS))
    ex = {s: D.exact_match(X[:, s], R[:, a])[0] for s, a in ((22, 0), (23, 1))}
    rho = {s: D.rank_agreement(X[:, s], R[:, a])[0] for s, a in ((22, 0), (23, 1))}
    return {"leaked_verdict": bool(rep.leaked), "n_verdicts": len(rep.verdicts), "n_suspicions": len(rep.suspicions),
            "n_outside_22_25": sum(f.feature not in C.LEAK_SLOTS for f in rep.findings),
            "exact_match_22_bm25": ex[22], "exact_match_23_dense": ex[23],
            "abs_spearman_22_bm25": rho[22], "abs_spearman_23_dense": rho[23]}


def summarize(rows):
    def mean_of(arm, m):
        v = [r[m] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else None

    out = {}
    honest = {m: mean_of("honest_minimal", m) for m in ("gbt", "logreg")}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        fam = arm.split("_")[0] if arm != "honest_minimal" else None
        s = {"family": fam, "scale": rs[0]["scale"], "n_rows": len(rs), "draws": sorted({r["draw"] for r in rs})}
        for m in ("gbt", "logreg"):
            vals = np.array([r[m] for r in rs])
            per_draw = [float(np.mean([r[m] for r in rs if r["draw"] == d])) for d in s["draws"]]
            exact = mean_of(f"{fam}_exact", m) if fam else None
            den = (exact - honest[m]) if exact is not None and honest[m] is not None else None
            s[m] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
                    "draw_means": per_draw, "draw_range_pp": float((max(per_draw) - min(per_draw)) * 100),
                    "delta_vs_honest_pp": (float(vals.mean()) - honest[m]) * 100 if honest[m] is not None else None,
                    "retention": (float(vals.mean()) - honest[m]) / den if den and abs(den) > 1e-9 else None}
        out[arm] = s
    return out


R1_SUMMARY_CANDIDATES = [os.path.join(C.CRC_RESULTS, "r1_classifier_sweep", "summary.json"),
                         os.path.join(C.CRC_RESULTS, "crc_v2", "r1_classifier_sweep", "summary.json")]
R1_EQUIVALENT = {"honest_minimal": "honest_minimal", "mrr20_exact": "mrr20_s000", "rankreward_exact": "rankreward_s000"}


def r1_regression_check(summ, cv_seeds):
    """The s = 0 arms are the same matrices as R1's (no noise is added), so with the same CV
    seeds they must reproduce R1's means exactly. Skipped if R1's summary is not found or the
    CV seeds differ from R1's 0-4."""
    path = next((p for p in R1_SUMMARY_CANDIDATES if os.path.exists(p)), None)
    if path is None or sorted(cv_seeds) != [0, 1, 2, 3, 4]:
        return {}
    r1 = json.load(open(path, encoding="utf-8"))["arms"]
    out = {}
    for arm, r1_arm in R1_EQUIVALENT.items():
        if arm in summ and r1_arm in r1:
            got, want = summ[arm]["gbt"]["mean"], r1[r1_arm]["gbt"]["mean"]
            out[arm] = {"got_gbt": got, "r1_gbt": want, "ok": abs(got - want) < 1e-9, "r1_summary": path}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "r5_random_noise"))
    ap.add_argument("--families", nargs="+", choices=C.FAMILIES, default=list(C.FAMILIES))
    ap.add_argument("--scales", nargs="+", type=float, default=list(SCALES))
    ap.add_argument("--draws", nargs="+", type=int, default=[0, 1, 2], help="independent noise realisations")
    ap.add_argument("--cv-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="time one fit and print the estimated total")
    args = ap.parse_args(argv)

    X, R, vals = load_base()
    plan = [("honest_minimal", None, 0.0, 0)]
    for fam in args.families:
        plan.append((f"{fam}_exact", fam, 0.0, 0))
        plan += [(f"{fam}_rand_s{int(round(s * 100)):03d}", fam, s, d) for s in args.scales for d in args.draws]
    n_fits = len(plan) * len(args.cv_seeds)

    if args.calibrate:
        t = time.time()
        evaluate(leak_matrix(X, vals["mrr20"], 0.15, noise(len(X), 0)), R, seed=0, n_jobs=args.n_jobs)
        dt = time.time() - t
        print(f"one fit (GB + LR, 5x5 CV): {dt:.1f}s; {n_fits} fits ~ {dt * n_fits / 3600:.2f} h")
        return

    if args.resume and os.path.isdir(args.out_dir):
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = C.prepare_out_dir(args.out_dir)
        C.write_json(os.path.join(out_dir, "run_header.json"),
                     {**C.run_header(args), "n_rows_pooled": int(len(X)), "plan_size": len(plan), "n_fits": n_fits})
    rows_path, audit_path = os.path.join(out_dir, "rows.jsonl"), os.path.join(out_dir, "leakcheck.jsonl")
    done, audited = set(), set()
    if os.path.exists(rows_path):
        done = {(r["arm"], r["draw"], r["seed"]) for r in map(json.loads, open(rows_path, encoding="utf-8"))}
    if os.path.exists(audit_path):
        audited = {(r["arm"], r["draw"]) for r in map(json.loads, open(audit_path, encoding="utf-8"))}

    for arm, fam, s, d in plan:
        Xl = X if fam is None else leak_matrix(X, vals[fam], s, noise(len(X), d))
        if (arm, d) not in audited:
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"arm": arm, "family": fam, "scale": s, "draw": d, **audit(Xl, R)}) + "\n")
        for seed in args.cv_seeds:
            if (arm, d, seed) in done:
                continue
            t = time.time()
            r = evaluate(Xl, R, seed=seed, n_jobs=args.n_jobs)
            if r is None:
                raise SystemExit(f"{arm}: too few informative queries")
            row = {"arm": arm, "family": fam, "scale": s, "draw": d, "seed": seed,
                   "seconds": round(time.time() - t, 1), **r}
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {arm:22s} draw={d} seed={seed} gbt={r['gbt']*100:5.2f} logreg={r['logreg']*100:5.2f} "
                  f"({row['seconds']}s)", flush=True)

    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    summ = summarize(rows)
    checks = r1_regression_check(summ, args.cv_seeds)
    C.write_json(os.path.join(out_dir, "summary.json"), {"r1_regression_check": checks, "arms": summ})
    for arm, c in checks.items():
        print(f"  R1 check {arm}: gbt {c['got_gbt']:.6f} vs R1 {c['r1_gbt']:.6f} -> {'OK' if c['ok'] else 'MISMATCH'}")
    print(f"\n{'arm':22s} {'GB':>6} {'kept':>6} {'LR':>6} {'kept':>6}")
    for arm, s in summ.items():
        k = lambda x: "   -  " if x is None else f"{x:6.2f}"  # noqa: E731
        print(f"{arm:22s} {s['gbt']['mean']*100:6.2f} {k(s['gbt']['retention'])} "
              f"{s['logreg']['mean']*100:6.2f} {k(s['logreg']['retention'])}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
