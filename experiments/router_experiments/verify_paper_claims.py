"""Recompute two claims in the paper that no committed artifact backs.

Both were flagged during a number-verification pass over main.tex: the text asserts
them, but nothing in results/ contains them, so they could not be checked.

  1. "features 22--23 alone reach 73.9%, higher than all 61 features with the leak present"
     -- learnability_analysis.py scores the group [22,23,24,25], never the pair [22,23].
     Reproduced here with that script's exact protocol so the number is comparable to the
     rest of the table: same CV object, same estimator, same seed, same pooled population.

  2. "eleven methods, maximum capture +0.4%"
     -- headroom_analysis.json puts only six methods on the capture scale; convex fusion
     and the cascade live in separate files. This enumerates all of them in one place and
     recomputes capture on the SAME test splits, so the count in the paper is checkable
     from a single artifact.

Writes results/paper_claim_checks.json. Read-only with respect to every other result.

Usage:
    python experiments/router_experiments/verify_paper_claims.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (PROJECT_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from learnability_analysis import (  # noqa: E402
    ACTIONS, N_REPEATS, N_SPLITS, load_queries, reward_matrix,
)
from canonical_evaluation import get_dataset_paths  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS = ("nq", "triviaqa", "hotpotqa")
SEED = 0  # learnability_analysis's default; changing it changes every number below


# =====================================================================
# Claim 1 — accuracy of narrow leaked column subsets
# =====================================================================

def _score(X, y, cols, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    s = cross_val_score(GradientBoostingClassifier(random_state=seed),
                        X[:, cols], y, cv=cv, scoring="accuracy")
    return float(s.mean()), float(s.std())


def claim_one() -> dict:
    """Pooled accuracy for column subsets of the LEAKY arm.

    The interesting comparison is [22,23] -- the two columns that literally hold BM25's
    and Dense's reward -- against all 61 features of the same arm. If the pair matches or
    beats the full set, the other 59 columns are contributing nothing once the answer is
    present, which is the paper's claim.
    """
    leaky = get_dataset_paths("_leaky")
    pooled_X, pooled_R = [], []
    per_dataset = {}

    for ds in DATASETS:
        path = leaky.get(ds)
        if not path or not os.path.exists(path):
            print(f"  {ds}: leaky env data missing, skipped ({path})")
            continue
        _, X, R = reward_matrix(load_queries(path), "mrr")
        pooled_X.append(X)
        pooled_R.append(R)
        per_dataset[ds] = {"n": int(len(X)), "dim": int(X.shape[1])}

    if not pooled_X:
        return {"error": "no leaky env data found — run prepare_env_data --feature-mode leaky"}

    X = np.vstack(pooled_X)
    R = np.vstack(pooled_R)

    # Same informative-query filter as learnability_analysis: exactly one strictly-best arm.
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    counts = np.bincount(yi, minlength=len(ACTIONS))
    majority = float(counts.max() / len(yi))
    dim = Xi.shape[1]

    subsets = {
        "leaked_pair (22,23)": [22, 23],
        "leaked_block (22-25)": [22, 23, 24, 25],
        "f22_only (bm25 mrr)": [22],
        "f23_only (dense mrr)": [23],
        "all_features": list(range(dim)),
        "everything_but_leak": [c for c in range(dim) if c not in (22, 23, 24, 25)],
    }

    out = {
        "arm": "leaky",
        "n_informative": int(len(yi)),
        "majority": majority,
        "seed": SEED,
        "cv": f"{N_SPLITS}-fold stratified x {N_REPEATS} repeats",
        "per_dataset": per_dataset,
        "subsets": {},
    }
    for label, cols in subsets.items():
        cols = [c for c in cols if c < dim]
        if not cols:
            continue
        acc, std = _score(Xi, yi, cols)
        out["subsets"][label] = {
            "n_features": len(cols),
            "accuracy": acc,
            "std": std,
            "lift_vs_majority": acc - majority,
        }
        print(f"    {label:26s} n={len(cols):3d}  acc={acc*100:5.2f}% "
              f"(lift {(acc-majority)*100:+5.2f})")
    return out


# =====================================================================
# Claim 2 — the eleven methods, on one capture scale
# =====================================================================

def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def claim_two() -> dict:
    """Every method the paper counts, with its capture of oracle headroom.

    capture = (method - best_fixed) / (oracle - best_fixed), on the test split, so 0%
    means "no better than committing to one retriever" and 100% means perfect routing.
    """
    canonical = _load("canonical_results.json")
    fusion = _load("fusion_baseline.json")
    cascade = _load("cascade_router.json")
    if canonical is None:
        return {"error": "results/canonical_results.json missing"}

    per_dataset = {}
    for ds in DATASETS:
        c = canonical.get(ds)
        if not c:
            continue
        arms = {a: float(c[a]) for a in ACTIONS if a in c}
        best_name = max(arms, key=arms.get)
        best_fixed = arms[best_name]
        oracle = float(c["oracle"])
        headroom = oracle - best_fixed

        def capture(value):
            if headroom <= 0 or value is None:
                return None
            return (float(value) - best_fixed) / headroom

        methods = {}
        # Fixed arms: included because the paper's count says "eleven methods", and a
        # fixed arm is a method a practitioner might ship.
        for a in ACTIONS:
            if a in c:
                methods[f"always_{a}"] = capture(c[a])
        for key in ("score_gap", "logistic_regression", "random_forest", "ridge",
                    "gradient_boosting", "rl_route_mean"):
            if key in c:
                methods[key] = capture(c[key])

        if fusion and ds in fusion and "convex_fusion_test" in fusion[ds]:
            methods["convex_fusion"] = capture(fusion[ds]["convex_fusion_test"])

        # The cascade optimises quality-per-cost, not quality, so its capture is reported
        # for completeness and is not the number it should be judged on.
        if cascade and ds in cascade:
            entry = cascade[ds]
            probes = entry if isinstance(entry, dict) else {}
            best_q = None
            for probe_name, probe in probes.items():
                if isinstance(probe, dict) and "quality" in probe:
                    q = float(probe["quality"])
                    best_q = q if best_q is None else max(best_q, q)
            if best_q is not None:
                methods["cascade_best_probe"] = capture(best_q)

        per_dataset[ds] = {
            "best_fixed_name": best_name,
            "best_fixed": best_fixed,
            "oracle": oracle,
            "headroom": headroom,
            "headroom_relative": headroom / best_fixed if best_fixed else None,
            "n_methods": len(methods),
            "capture": methods,
        }

    counted = sorted({m for v in per_dataset.values() for m in v["capture"]})
    all_vals = [v for d in per_dataset.values() for v in d["capture"].values()
                if v is not None]
    return {
        "method_names": counted,
        "n_methods": len(counted),
        "max_capture": max(all_vals) if all_vals else None,
        "per_dataset": per_dataset,
    }


def main() -> int:
    print("=" * 72)
    print("Claim 1 — do features 22-23 alone match the full leaked feature set?")
    print("=" * 72)
    one = claim_one()

    print()
    print("=" * 72)
    print("Claim 2 — the eleven methods, on one capture scale")
    print("=" * 72)
    two = claim_two()
    if "error" not in two:
        print(f"  {two['n_methods']} methods: {', '.join(two['method_names'])}")
        for ds, v in two["per_dataset"].items():
            best = max((x for x in v["capture"].values() if x is not None), default=None)
            print(f"    {ds:9s} best_fixed={v['best_fixed']:.4f} oracle={v['oracle']:.4f} "
                  f"headroom_rel={v['headroom_relative']*100:.1f}%  "
                  f"max capture={best*100:+.2f}%")
        if two["max_capture"] is not None:
            print(f"  MAX capture across all methods and datasets: "
                  f"{two['max_capture']*100:+.2f}%")
    else:
        print(" ", two["error"])

    out_path = os.path.join(RESULTS_DIR, "paper_claim_checks.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"claim_1_leaked_subsets": one, "claim_2_method_capture": two},
                  f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
