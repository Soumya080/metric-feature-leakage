#!/usr/bin/env python3
"""H: which detector catches which leak? A benchmark with planted ground truth.

Part 1 (synthetic). Each problem has 20 honest features, three of which carry GENUINE signal
about which arm wins (as in b_positive_control), and reciprocal-rank-like rewards. One extra
column is then appended according to the scenario, built from arm 0's reward r:

  none            pure noise column (nothing leaks)
  exact           r                                   truncated  r if r >= 1/5 else 0 (MRR@5-like)
  monotone        rank of r                           value_keyed r + crc32 value-keyed +/-0.15 offset
  scrambled       distinct values of r mapped to a random order (order-free recoding)
  noise_015       r + Uniform(-0.15, 0.15)            noise_050  r + Uniform(-0.50, 0.50)
  confound        a legitimate group id; the group shifts which arm wins (RouterBench's case)
  split           two columns, r + z and z (z ~ N(0, 0.5)): the leak lives in their difference
  pca_mixed       r rotated together with 5 honest columns (the leak is smeared over 6 columns)

Detectors, each flagging a column if it fires against any arm:
  equality        exact agreement on > 25% of non-zero-reward rows (leakcheck's threshold)
  pearson         |Pearson r| > 0.35
  auc_scan        bioLeak-style univariate scan: |AUC - 0.5| x 2 >= 0.90 for "arm a is best"
  importance      held-out permutation importance of a GB router > 0.01 accuracy
  leakcheck_v1    any v1 finding (verdict or suspicion); leakcheck_v1_verdict: verdicts only
  leakcheck_v2    any audit_v2 finding (adds DECODABLE; groups passed in the confound scenario)

Reported per scenario: flag rate on the planted column(s), on the 3 genuine-signal columns, and
on the 17 pure-noise columns. For "none" and "confound" the planted column is NOT a leak, so its
flag rate is a false-flag rate.

Part 2 (the paper's data). leakcheck v1 vs v2 on the pooled 61-D matrices: honest, the
value-keyed MRR@20 recoding at s = 0.15/0.50/1.00 (R1 env files), and independent noise at
s = 0.15/0.50 (R5 construction), with dataset id as groups.

Read-only. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common as C
import leakcheck
from leakcheck import decode, detectors
from leakcheck.taxonomy import _deterministic_jitter

N_HONEST, SIGNAL = 20, (0, 1, 2)
SCENARIOS = ("none", "exact", "truncated", "monotone", "value_keyed", "scrambled", "noise_015",
             "noise_050", "confound", "split", "pca_mixed")
DETECTORS = ("equality", "pearson", "auc_scan", "importance", "leakcheck_v1_verdict",
             "leakcheck_v1", "leakcheck_v2")


def problem(scenario, n, seed):
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario)])
    X = rng.standard_normal((n, N_HONEST))
    W = np.random.default_rng(seed).standard_normal((3, 3))
    W[:, 1] += 0.6
    u = 0.5 * (X[:, SIGNAL] @ W) + rng.standard_normal((n, 3))
    groups = None
    if scenario == "confound":
        groups = rng.integers(0, 4, size=n)
        u += np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2], [0, 0, 0]])[groups]
    R = 1.0 / (1.0 + np.floor(np.exp(1.5 - u)))
    r = R[:, 0]
    if scenario == "none":
        extra = rng.standard_normal((n, 1))
    elif scenario == "exact":
        extra = r[:, None]
    elif scenario == "truncated":
        extra = np.where(r >= 0.2, r, 0.0)[:, None]
    elif scenario == "monotone":
        extra = np.argsort(np.argsort(r, kind="mergesort"), kind="mergesort")[:, None].astype(float)
    elif scenario == "value_keyed":
        extra = _deterministic_jitter(r, 0.15, salt="h")[:, None]
    elif scenario == "scrambled":
        uniq, inv = np.unique(r, return_inverse=True)
        extra = (rng.permutation(len(uniq)) * 0.37 + 5.0)[inv][:, None]
    elif scenario == "noise_015":
        extra = (r + rng.uniform(-0.15, 0.15, n))[:, None]
    elif scenario == "noise_050":
        extra = (r + rng.uniform(-0.5, 0.5, n))[:, None]
    elif scenario == "confound":
        extra = (groups + rng.normal(0, 0.01, n))[:, None]
    elif scenario == "split":
        z = rng.normal(0, 0.5, n)
        extra = np.column_stack([r + z, z])
    elif scenario == "pca_mixed":
        block = np.column_stack([X[:, 15:20], (r - r.mean()) / r.std()])
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        X = np.hstack([X[:, :15], block @ Q])
        return X, R, list(range(15, 21)), groups
    X = np.hstack([X, extra])
    return X, R, list(range(N_HONEST, X.shape[1])), groups


def _auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return 0.5 if n1 == 0 or n0 == 0 else (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def flags(X, R, groups, seed, n_perm):
    d, k = X.shape[1], R.shape[1]
    out = {name: np.zeros(d, dtype=bool) for name in DETECTORS}
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    best = R.argmax(axis=1)
    for j in range(d):
        for a in range(k):
            if detectors.exact_match(X[:, j], R[:, a])[0] > detectors.EXACT_MATCH_THRESHOLD:
                out["equality"][j] = True
            c = np.corrcoef(X[:, j], R[:, a])[0, 1] if X[:, j].std() > 0 else 0.0
            if abs(c) > 0.35:
                out["pearson"][j] = True
            y = best[strict] == a
            if abs(_auc(X[strict, j], y) - 0.5) * 2 >= 0.90:
                out["auc_scan"][j] = True
    v1 = leakcheck.audit(X, R)
    for f in v1.findings:
        if f.feature >= 0:
            out["leakcheck_v1"][f.feature] = True
            if f.is_verdict:
                out["leakcheck_v1_verdict"][f.feature] = True
    v2 = leakcheck.audit_v2(X, R, groups=groups, n_perm=n_perm, seed=seed)
    confound_cols = set()
    for f in v2.findings:
        if f.feature >= 0:
            out["leakcheck_v2"][f.feature] = True
            if "GROUP CONFOUND" in f.detail:
                confound_cols.add(f.feature)

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    idx = np.where(strict)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    gb = GradientBoostingClassifier(random_state=seed).fit(X[tr], best[tr])
    imp = permutation_importance(gb, X[te], best[te], n_repeats=5, random_state=seed).importances_mean
    out["importance"] = imp > 0.01
    return out, confound_cols


def synthetic_part(seeds, n, n_perm):
    rows = []
    for sc in SCENARIOS:
        agg = {det: {"planted": [], "signal": [], "noise": []} for det in DETECTORS}
        confound_labelled = []
        t = time.time()
        for s in seeds:
            X, R, planted, groups = problem(sc, n, s)
            fl, conf = flags(X, R, groups, s, n_perm)
            noise_cols = [j for j in range(X.shape[1]) if j not in planted and j not in SIGNAL]
            for det, m in fl.items():
                agg[det]["planted"].append(float(m[planted].any()))
                agg[det]["signal"].append(float(m[list(SIGNAL)].mean()))
                agg[det]["noise"].append(float(m[noise_cols].mean()))
            if sc == "confound":
                confound_labelled.append(float(any(j in conf for j in planted)))
        row = {"scenario": sc, "is_leak": sc not in ("none", "confound"), "n_seeds": len(seeds)}
        for det in DETECTORS:
            row[det] = {key: float(np.mean(v)) for key, v in agg[det].items()}
        if confound_labelled:
            row["v2_labels_planted_as_group_confound"] = float(np.mean(confound_labelled))
        rows.append(row)
        print(f"{sc:11s} " + " ".join(f"{det[:12]:>12s}={row[det]['planted']:.2f}" for det in DETECTORS)
              + f"  ({time.time()-t:.0f}s)", flush=True)
    return rows


def real_part(n_perm):
    import r5_random_noise_sweep as R5
    X, R, vals = R5.load_base()
    groups = np.concatenate([np.full(len(C.load_env(C.anchor_path(ds, ""))), i) for i, ds in enumerate(C.DATASETS)])
    arms = {"honest": X}
    for s in (0.15, 0.50, 1.00):
        p = {ds: C.variant_path(ds, C.variant_tag("mrr20", s), env_dir=os.path.join(C.CRC_RESULTS, "crc_v2", "env_data"))
             for ds in C.DATASETS}
        if all(os.path.exists(v) for v in p.values()):
            Xs, Rs = C.pooled(p, metric="mrr")
            if not np.array_equal(Rs, R):
                raise SystemExit("reward mismatch between R1 env files and the honest pool")
            arms[f"value_keyed_s{int(s*100):03d}"] = Xs
    for s in (0.15, 0.50):
        arms[f"independent_s{int(s*100):03d}"] = R5.leak_matrix(X, vals["mrr20"], s, R5.noise(len(X), 0))
    out = {}
    for name, Xa in arms.items():
        res = {}
        for tag, rep in (("v1", leakcheck.audit(Xa, R, arm_names=list(C.ACTIONS))),
                         ("v2", leakcheck.audit_v2(Xa, R, arm_names=list(C.ACTIONS), groups=groups, n_perm=n_perm))):
            leak = sorted({f.feature for f in rep.findings if f.feature in (22, 23)})
            other = sorted({f.feature for f in rep.findings if f.feature >= 0 and f.feature not in C.LEAK_SLOTS})
            conf = sorted({f.feature for f in rep.findings if "GROUP CONFOUND" in f.detail})
            res[tag] = {"verdicts": len(rep.verdicts), "suspicions": len(rep.suspicions),
                        "flags_on_leak_slots_22_23": leak, "flags_outside_22_25": other,
                        "labelled_group_confound": conf}
        out[name] = res
        print(f"real {name:18s} v1 leak-slot flags {res['v1']['flags_on_leak_slots_22_23']} (other {len(res['v1']['flags_outside_22_25'])}) | "
              f"v2 leak-slot flags {res['v2']['flags_on_leak_slots_22_23']} (other {len(res['v2']['flags_outside_22_25'])}, "
              f"group-confound {res['v2']['labelled_group_confound']})", flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "h_detector_benchmark"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    summary = {"header": C.run_header(args), "decode_min_r2": decode.MIN_R2,
               "synthetic": synthetic_part(args.seeds, args.n, args.n_perm)}
    C.write_json(os.path.join(out_dir, "summary.json"), summary)
    if not args.skip_real:
        summary["real"] = real_part(args.n_perm)
        C.write_json(os.path.join(out_dir, "summary.json"), summary)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
