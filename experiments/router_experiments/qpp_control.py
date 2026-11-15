"""Do the query-performance predictors the IR field actually uses rescue routing?

WHY THIS EXISTS
    The sharpest objection to this paper's negative result is that our label-free features are
    two summary statistics -- the mean and standard deviation of the top-5 retrieval scores --
    where information retrieval has spent twenty years building predictors for exactly the
    quantity a router needs: how well is this query about to do? NQC is, in effect, the
    principled version of one of our two statistics, so it is the natural control.

    So "no signal in these features" has to become "no signal in the features the field would
    actually use", or the limitation stands. This computes the standard post-retrieval
    predictors per arm and refits.

WHY IT IS SAFE TO RUN THIS LATE
    It is a standalone control, exactly like embedding_control.py. The dose-response arms are
    untouched, so nothing here can move the leak premium. The only possible outcomes are that
    the objection closes or that the paper's claim must narrow -- and we would rather learn the
    second here than in review.

WHICH SETTING, AND WHY IT IS NOT THE 500-PASSAGE ONE
    QPP needs raw retrieval scores. The 500-passage env data stores only metrics (mrr, gold_rank,
    hit_at_5) and a precomputed feature vector -- the score distribution was never persisted, so
    these predictors CANNOT be computed there without re-running retrieval. The shared-index
    sweep does persist scores_top10, so this runs at 100K passages. That is a different setting
    from Section 5 and the accuracies are not comparable to 59.9%/60.9%; the majority baseline is
    recomputed here and every comparison is within this setting.

WHAT IS A REAL PREDICTOR AND WHAT IS A PROXY
    NQC, WIG and SMV are the published estimators, computed from the top-k score distribution.
    Clarity is NOT computed: it needs a language model over the retrieved documents' text, which
    means loading the corpus, and its usual role -- how focused is the result set -- is partly
    covered by the score-distribution entropy included below. That substitution is named rather
    than hidden, because calling an entropy "Clarity" would misattribute a specific published
    method.

Usage:
    python experiments/router_experiments/qpp_control.py
    python experiments/router_experiments/qpp_control.py --scale 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ("nq", "triviaqa", "hotpotqa")
ARMS = ("bm25", "dense", "hybrid")
SEED = 0
N_SPLITS, N_REPEATS = 5, 5
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SWEEP = os.path.join(PROJECT_ROOT, "experiments", "corpus_v3", "results")


def qpp_features(scores, mu_c, sigma_c, n_query_terms):
    """Standard post-retrieval predictors from one arm's top-k score list.

    mu_c / sigma_c are collection-level score statistics, standing in for the "collection
    score" these estimators are defined against. The published forms assume a score for the
    collection treated as a single document, which a sparse/dense hybrid index does not give
    us directly; the corpus-wide mean of retrieved scores is the standard practical
    substitute and is applied identically to every arm and query, so it cannot advantage one.
    """
    s = np.asarray([x for x in scores if x is not None], dtype=float)
    if s.size == 0:
        return [0.0] * 8
    k = s.size
    mu = float(s.mean())
    sd = float(s.std())

    # NQC: standard deviation of the top-k scores, normalised by the collection score. High
    # spread means one document clearly dominates, which predicts an easy query.
    nqc = sd / mu_c if mu_c else 0.0

    # WIG: how far the top-k scores rise above the collection score, length-normalised.
    wig = (mu - mu_c) / (np.sqrt(max(n_query_terms, 1)) * (sigma_c or 1.0))

    # SMV: magnitude and variance together.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mu > 0, s / mu, 1.0)
        logs = np.log(np.where(ratio > 0, ratio, 1.0))
    smv = float(np.abs(s * logs).sum() / (k * mu_c)) if mu_c else 0.0

    # Score-distribution entropy: a focus measure standing in for Clarity, which would need
    # the retrieved documents' text. Named as a proxy, not as Clarity.
    z = s - s.max()
    p = np.exp(z) / np.exp(z).sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(k)) if k > 1 else 0.0

    # Shape of the decay from rank 1 -- the gap the top document holds over its successors.
    gap1 = float(s[0] - s[1]) if k > 1 else 0.0
    gap_rel = gap1 / mu if mu else 0.0

    return [nqc, wig, smv, ent, gap1, gap_rel, mu, sd]


FEATURE_NAMES = ["nqc", "wig", "smv", "entropy", "gap1", "gap_rel", "mean_topk", "std_topk"]


def build(dataset, scale):
    path = os.path.join(SWEEP, f"scale_{scale}", f"{dataset}_per_query_results.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path, encoding="utf-8"))

    # Collection-level score statistics per arm, from every query's retrieved scores.
    pool = {a: [] for a in ARMS}
    for r in rows:
        for a in ARMS:
            rec = r.get("retrievers", {}).get(a, {})
            pool[a].extend([x for x in (rec.get("scores_top10") or []) if x is not None])
    mu_c = {a: float(np.mean(v)) if v else 0.0 for a, v in pool.items()}
    sd_c = {a: float(np.std(v)) if v else 1.0 for a, v in pool.items()}

    X, R = [], []
    for r in rows:
        nterms = len(str(r.get("question", "")).split())
        feats, rewards = [], []
        ok = True
        for a in ARMS:
            rec = r.get("retrievers", {}).get(a, {})
            gr = rec.get("gold_rank")
            if gr is None:
                ok = False
                break
            rewards.append(1.0 / gr if gr and gr > 0 else 0.0)
            feats.extend(qpp_features(rec.get("scores_top10") or [], mu_c[a], sd_c[a], nterms))
        if not ok:
            continue
        feats.append(float(nterms))
        X.append(feats)
        R.append(rewards)
    return np.array(X, dtype=float), np.array(R, dtype=float)


def evaluate(X, y, seed=SEED):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    counts = np.bincount(y)
    if len(y) < N_SPLITS * 2 or (counts[counts > 0] < N_SPLITS).any():
        return None
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    s = cross_val_score(GradientBoostingClassifier(random_state=seed), X, y,
                        cv=cv, scoring="accuracy", n_jobs=-1)
    return {"acc": float(s.mean()), "std": float(s.std()),
            "majority": float(counts.max() / len(y)), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=100000)
    ap.add_argument("--outdir", default=RESULTS_DIR)
    args = ap.parse_args()

    out = {"scale": args.scale, "seed": SEED, "cv": f"{N_SPLITS}x{N_REPEATS}",
           "feature_names": FEATURE_NAMES, "n_features": len(ARMS) * len(FEATURE_NAMES) + 1,
           "per_dataset": {}}
    Xs, ys = [], []

    for ds in DATASETS:
        built = build(ds, args.scale)
        if built is None:
            print(f"  {ds}: no retrieval file at scale {args.scale}, skipping")
            continue
        X, R = built
        strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
        Xd, yd = X[strict], R[strict].argmax(axis=1)
        res = evaluate(Xd, yd)
        if res is None:
            print(f"  {ds}: too few informative queries")
            continue
        out["per_dataset"][ds] = res
        Xs.append(Xd)
        ys.append(yd)
        print(f"  {ds:<10} n={res['n']:<5} QPP {res['acc']*100:6.2f}%  "
              f"majority {res['majority']*100:6.2f}%  lift {(res['acc']-res['majority'])*100:+.2f}")

    if Xs:
        Xp, yp = np.vstack(Xs), np.concatenate(ys)
        res = evaluate(Xp, yp)
        out["pooled"] = res
        print(f"  {'pooled':<10} n={res['n']:<5} QPP {res['acc']*100:6.2f}%  "
              f"majority {res['majority']*100:6.2f}%  lift {(res['acc']-res['majority'])*100:+.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"qpp_control_scale{args.scale}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    lifts = [v["acc"] - v["majority"] for v in out["per_dataset"].values()]
    if lifts and max(lifts) <= 0:
        print("Every QPP lift is negative: the standard IR difficulty predictors do not beat a "
              "constant policy either, so the negative is about the target, not our features.")
    else:
        print("At least one QPP lift is POSITIVE. The paper's claim must narrow -- report which "
              "dataset and by how much, and say so in Limitations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
