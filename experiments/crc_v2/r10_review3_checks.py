#!/usr/bin/env python3
"""R10: the numbers behind the review_3 camera-ready edits.

Four read-only checks:
  (a) shared_index_arms  - arm means on the shared-index rebuild (experiments/corpus_v3, 100K
      passages), over queries whose answer occurs in the corpus (gold_rank not null), the same
      population Table 2 uses. Reward = reciprocal rank of the first answer-bearing passage;
      MRR@20 and MRR@5 also reported.
  (b) mde_bracket        - Table 3's minimum detectable capture under two variance models: the
      oracle's per-query gain (what the paper quotes) and a "selective" router that follows the
      oracle on a random share c of queries and keeps best-fixed elsewhere.
  (c) method_pvalues     - Table 3's one-sided paired bootstrap (canonical_evaluation's
      procedure, null-shifted, seed 12345, B = 10,000) with ties counted (>= observed - 1e-12),
      plus how many test queries each method's choice changes the reward on.
  (d) leaked_router_test - the same test applied to the tuned GB router trained on the leaky
      rungs at split seed 42: would RQ2's test have detected the leak RQ1 manufactures?
  (e) ladder_verdicts    - leakcheck.audit verdicts on each pooled Table 2 rung.

Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import common as C
from canonical_evaluation import compute_static_baselines, compute_supervised_baselines  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

Z_ONE = 1.644854 + 0.841621
SHARED = os.path.join(C.PROJECT_ROOT, "experiments", "corpus_v3", "results", "scale_100000")


def shared_index_arms():
    out = {}
    for ds in C.DATASETS:
        with open(os.path.join(SHARED, f"{ds}_per_query_results.json"), encoding="utf-8") as f:
            rows = json.load(f)
        keep = [r for r in rows if any(r["retrievers"][a].get("gold_rank") for a in C.ACTIONS)]
        res = {"n_all": len(rows), "n_answerable": len(keep)}
        for name, fn in {
            "rr_full": lambda x: 1.0 / x["gold_rank"] if x.get("gold_rank") else 0.0,
            "mrr_at_20": lambda x: x["metrics_at_k"]["20"]["mrr"],
            "mrr_at_5": lambda x: x["metrics_at_k"]["5"]["mrr"],
        }.items():
            means = {a: float(np.mean([fn(r["retrievers"][a]) for r in keep])) for a in C.ACTIONS}
            orc = float(np.mean([max(fn(r["retrievers"][a]) for a in C.ACTIONS) for r in keep]))
            res[name] = {"means": means, "oracle": orc, "best_fixed": max(means, key=means.get)}
        out[ds] = res
    return out


def mde_bracket(canon):
    out = {}
    for ds in C.DATASETS:
        s = canon[ds]["per_query"]["scores"]
        R = np.array([s[a] for a in C.ACTIONS]).T
        b = int(R.mean(0).argmax())
        g = R.max(1) - R[:, b]
        n, h = len(g), float(g.mean())
        oracle_model = Z_ONE * g.std(ddof=1) / math.sqrt(n) / h
        # Selective router: diff_i = B_i * g_i, B_i ~ Bernoulli(c). mean = c h,
        # var = c E[g^2] - c^2 h^2. Smallest c with Z * sqrt(var / n) <= c h.
        eg2 = float((g ** 2).mean())
        cs = np.linspace(0.001, 1.0, 100000)
        ok = Z_ONE * np.sqrt((cs * eg2 - cs ** 2 * h ** 2) / n) <= cs * h
        sel = float(cs[ok.argmax()]) if ok.any() else None
        out[ds] = {"n_test": n, "headroom_abs": h, "oracle_variance_model": oracle_model,
                   "selective_router_model": sel}
    return out


def boot_p(a, b, B=10000, seed=12345):
    rng = np.random.RandomState(seed)
    d = np.asarray(a) - np.asarray(b)
    obs = d.mean()
    sh = d - obs
    cnt = sum(rng.choice(sh, size=len(d), replace=True).mean() >= obs - 1e-12 for _ in range(B))
    return (cnt + 1) / (B + 1)


def method_pvalues(canon):
    out = {}
    for ds in C.DATASETS:
        s = canon[ds]["per_query"]["scores"]
        best = max(C.ACTIONS, key=lambda a: np.mean(s[a]))
        out[ds] = {"best_fixed": best}
        for m in ("score_gap", "logistic_regression", "random_forest", "ridge", "gradient_boosting"):
            diff = np.array(s[m]) - np.array(s[best])
            out[ds][m] = {"p_one_sided": float(boot_p(s[m], s[best])),
                          "n_queries_changed": int((np.abs(diff) > 1e-12).sum()),
                          "mean_diff": float(diff.mean())}
    return out


def leaked_router_test(arms_to_test):
    from r1_classifier_sweep import arm_paths
    paths = arm_paths(C.CRC_ENV_DIR)
    out = {}
    for arm in arms_to_test:
        out[arm] = {}
        for ds in C.DATASETS:
            tr, va, te = create_train_val_test_split(paths[arm][ds], split_seed=42)
            test_means, test_pq = compute_static_baselines(te)
            sup, raw = compute_supervised_baselines(tr, te)
            best = max(C.ACTIONS, key=lambda a: test_means[a])
            gb = raw["gradient_boosting"]
            fixed = test_pq[best]
            orc = test_means["oracle"]
            out[arm][ds] = {
                "gb_mean": float(np.mean(gb)), "best_fixed": best, "best_fixed_mean": float(np.mean(fixed)),
                "capture": float((np.mean(gb) - np.mean(fixed)) / (orc - np.mean(fixed))),
                "p_one_sided": float(boot_p(gb, fixed)), "n_test": len(gb)}
            print(f"  {arm:12s} {ds:9s} GB {np.mean(gb):.4f} vs {best} {np.mean(fixed):.4f} "
                  f"p={out[arm][ds]['p_one_sided']:.4f}", flush=True)
    return out


def ladder_verdicts():
    import leakcheck
    out = {}
    for name, suffix in C.ANCHOR_SUFFIXES.items():
        Xs, Rs = [], []
        for ds in C.DATASETS:
            q = C.load_env(C.anchor_path(ds, suffix))
            ids = list(q)
            Xs.append(np.array([q[i]["features"] for i in ids], float))
            Rs.append(np.array([[float(q[i]["retrievers"][a]["mrr"]) for a in C.ACTIONS] for i in ids]))
        X, R = np.vstack(Xs), np.vstack(Rs)
        rep = leakcheck.audit(X, R)
        leak_slots = [f for f in rep.findings if f.feature in C.LEAK_SLOTS]
        out[name] = {"verdict": bool(any(f.is_verdict for f in leak_slots)),
                     "suspicion": bool(any(not f.is_verdict for f in leak_slots)),
                     "findings_on_leak_slots": [str(f) for f in leak_slots]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "r10_review3_checks"))
    ap.add_argument("--leaked-arms", nargs="+", default=["leaky", "leak_noisy"])
    ap.add_argument("--skip-leaked", action="store_true")
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    canon_path = os.path.join(C.RESULTS_DIR, "canonical_results.json")
    with open(canon_path, encoding="utf-8") as f:
        canon = json.load(f)
    res = {"header": {**C.run_header(args), "canonical_sha256": C.sha256_file(canon_path)}}
    res["shared_index_arms"] = shared_index_arms()
    res["mde_bracket"] = mde_bracket(canon)
    res["method_pvalues"] = method_pvalues(canon)
    res["ladder_verdicts"] = ladder_verdicts()
    if not args.skip_leaked:
        res["leaked_router_test"] = leaked_router_test(args.leaked_arms)
    C.write_json(os.path.join(out_dir, "summary.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "header"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
