"""
per_retriever_router.py — Route by predicting each retriever's quality, not by classifying.

The supervised baseline in canonical_evaluation.py trains one 3-way classifier on argmax
labels. That framing is a poor fit for this problem:

  * It discards magnitude. A query where BM25 beats Dense 0.9 to 0.1 and one where it wins
    0.11 to 0.10 produce the same label, even though only the first is worth routing.
  * It is dominated by ties. Under MRR@5, 70-83% of queries have every retriever at 0, so
    argmax silently emits whichever action sits at index 0 and the classifier spends most
    of its capacity fitting a tie-break artifact.
  * It cannot use retriever-specific evidence. Query Performance Prediction's core finding
    is that a retriever's OWN score distribution predicts its OWN success; a single shared
    classifier has no way to express "use BM25's margin to judge BM25".

This module instead fits one regressor per retriever, predicting that retriever's reward
from shared query features plus that retriever's own score features, and routes to the
argmax of the three predictions. Every query contributes to every model (no ties to
discard), the target is smooth, and each model sees the evidence QPP says is relevant.
This is the pointwise form of LTRR's learning-to-rank-retrievers setup.

Reported against the baselines that matter: Best Fixed (the strongest single retriever),
Oracle (the ceiling), and the 3-way classifier this replaces — with a paired bootstrap so
the comparison is a significance statement, not a decimal-point comparison.

Usage:
    python experiments/router_experiments/per_retriever_router.py
    python experiments/router_experiments/per_retriever_router.py --env-suffix _smoke --datasets nq
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths, paired_bootstrap_test  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]
SPLIT_SEED = 42

# Feature layout (see QueryFeatureExtractor.get_feature_order):
#   0-21  query features (lexical / entity / structural) — shared by all retrievers
#   22-25 retrieval-score features (avg_bm25, avg_dense, score_gap, bm25_variance)
#   26-28 cross-retriever agreement — relational, shared
#   29+   PCA query-embedding components — shared
SHARED_IDX = list(range(0, 22)) + list(range(26, 29))
# Which score feature belongs to which retriever. score_gap (24) and bm25_variance (25)
# are relational/BM25-specific; hybrid is a fusion of both, so it sees both sides.
OWN_SCORE_IDX = {
    "bm25": [22, 24, 25],
    "dense": [23, 24],
    "hybrid": [22, 23, 24, 25],
}


def feature_indices(dim, retriever, use_embeddings=True):
    idx = list(SHARED_IDX) + OWN_SCORE_IDX[retriever]
    if use_embeddings and dim > 29:
        idx += list(range(29, dim))
    return sorted(i for i in idx if i < dim)


def load_split(path):
    env_train, env_val, env_test = create_train_val_test_split(path, split_seed=SPLIT_SEED)
    out = []
    for env in (env_train, env_val, env_test):
        qids = list(env.query_pool)
        X = np.array([env.master_table[q]["features"] for q in qids], dtype=float)
        R = np.array([[float(env.master_table[q]["retrievers"][a]["mrr"]) for a in ACTIONS]
                      for q in qids])
        # Report on MRR@k when it is stored separately, so the headline metric stays the
        # standard retrieval metric even when training optimises the denser rank reward.
        first = env.master_table[qids[0]]["retrievers"]["bm25"]
        mrr_key = next((kk for kk in first if kk.startswith("mrr_at_")), None)
        M = (np.array([[float(env.master_table[q]["retrievers"][a][mrr_key]) for a in ACTIONS]
                       for q in qids]) if mrr_key else R)
        out.append({"qids": qids, "X": X, "R": R, "M": M})
    return out


def fit_router(train, val, model_name="gbt", use_embeddings=True, seed=0):
    """One regressor per retriever on the training reward; route by argmax of predictions."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV

    dim = train["X"].shape[1]
    models = {}
    for j, a in enumerate(ACTIONS):
        cols = feature_indices(dim, a, use_embeddings)
        if model_name == "gbt":
            m = GradientBoostingRegressor(random_state=seed)
        else:
            m = RidgeCV(alphas=np.logspace(-3, 3, 13))
        m.fit(train["X"][:, cols], train["R"][:, j])
        models[a] = (m, cols)

    def route(split):
        preds = np.column_stack([models[a][0].predict(split["X"][:, models[a][1]])
                                 for a in ACTIONS])
        return preds.argmax(axis=1)

    return models, route


def evaluate(actions, split):
    """Realised MRR@k of the chosen actions (the metric the paper reports)."""
    return split["M"][np.arange(len(actions)), actions]


def analyze_dataset(name, path, seed=0):
    train, val, test = load_split(path)
    dim = train["X"].shape[1]

    means = test["M"].mean(axis=0)
    best_idx = int(train["M"].mean(axis=0).argmax())   # chosen on TRAIN, evaluated on test
    best_fixed = test["M"][:, best_idx]
    oracle = test["M"].max(axis=1)

    res = {
        "n_train": len(train["qids"]), "n_val": len(val["qids"]), "n_test": len(test["qids"]),
        "feature_dim": dim,
        "per_retriever_test_mrr": {a: float(m) for a, m in zip(ACTIONS, means)},
        "best_fixed_name": ACTIONS[best_idx],
        "best_fixed": float(best_fixed.mean()),
        "oracle": float(oracle.mean()),
        "methods": {},
    }

    for model_name in ("gbt", "ridge"):
        for use_emb in ((True, False) if dim > 29 else (False,)):
            tag = f"per_retriever_{model_name}" + ("" if use_emb else "_noembed")
            _, route = fit_router(train, val, model_name, use_emb, seed)
            scores = evaluate(route(test), test)
            res["methods"][tag] = {
                "mrr": float(scores.mean()),
                "vs_best_fixed": float(scores.mean() - best_fixed.mean()),
                "p_vs_best_fixed": float(paired_bootstrap_test(scores, best_fixed)[1]),
                "per_query": [float(x) for x in scores],
            }

    # The 3-way classifier this replaces, trained the way canonical_evaluation does it.
    from sklearn.ensemble import GradientBoostingClassifier
    y = train["R"].argmax(axis=1)
    if len(np.unique(y)) > 1:
        clf = GradientBoostingClassifier(random_state=seed).fit(train["X"], y)
        scores = evaluate(clf.predict(test["X"]), test)
        res["methods"]["argmax_classifier"] = {
            "mrr": float(scores.mean()),
            "vs_best_fixed": float(scores.mean() - best_fixed.mean()),
            "p_vs_best_fixed": float(paired_bootstrap_test(scores, best_fixed)[1]),
            "per_query": [float(x) for x in scores],
        }
    return res


def render(all_res):
    L = ["# Per-Retriever Routing\n",
         "One regressor per retriever predicting its own reward, routed by argmax, versus",
         "the single 3-way argmax classifier it replaces. Reported metric is MRR@k on the",
         "held-out test split; `p` is a paired bootstrap against Best Fixed.\n",
         "| dataset | n_test | best fixed | oracle | method | MRR | Δ vs best-fixed | p |",
         "|---|---:|---:|---:|---|---:|---:|---:|"]
    for ds, r in all_res.items():
        for tag, m in r["methods"].items():
            L.append(f"| {ds} | {r['n_test']} | {r['best_fixed']:.4f} | {r['oracle']:.4f} | "
                     f"{tag} | {m['mrr']:.4f} | {m['vs_best_fixed']:+.4f} | {m['p_vs_best_fixed']:.3f} |")
    L += ["\nA positive Δ with p < 0.05 means routing genuinely beats committing to the "
          "single best retriever. Δ ≤ 0 means it does not — which, stated plainly, is the "
          "result rather than something to tune away.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str,
                   default=os.path.join(RESULTS_DIR, "per_retriever_router.json"))
    p.add_argument("--output-md", type=str,
                   default=os.path.join(RESULTS_DIR, "per_retriever_router.md"))
    args = p.parse_args()

    paths = get_dataset_paths(args.env_suffix)
    if args.env_dir:
        paths = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in paths.items()}
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        paths = {k: v for k, v in paths.items() if k in keep}

    all_res = {}
    for ds, path in paths.items():
        if not os.path.exists(path):
            print(f"  {ds}: env data missing — skipped.")
            continue
        print(f"\n--- {ds} ---")
        r = analyze_dataset(ds, path, args.seed)
        all_res[ds] = r
        print(f"  best-fixed={r['best_fixed_name']} {r['best_fixed']:.4f}  "
              f"oracle={r['oracle']:.4f}  n_test={r['n_test']}  dim={r['feature_dim']}")
        for tag, m in r["methods"].items():
            flag = "  <-- beats best-fixed" if m["vs_best_fixed"] > 0 and m["p_vs_best_fixed"] < 0.05 else ""
            print(f"    {tag:28s} MRR={m['mrr']:.4f}  d={m['vs_best_fixed']:+.4f}  "
                  f"p={m['p_vs_best_fixed']:.3f}{flag}")

    if not all_res:
        print("No env data found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render(all_res))
    print(f"\nSaved: {args.output}\nSaved: {args.output_md}")


if __name__ == "__main__":
    main()
