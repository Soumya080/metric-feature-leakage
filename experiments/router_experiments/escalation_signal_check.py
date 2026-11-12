"""
escalation_signal_check.py — Can a retriever's own failure be detected AFTER running it?

This is the go/no-go for the cascade formulation, and it should be run before any cascade
policy is built.

The learnability analysis establishes that *which* retriever wins cannot be predicted from
the query: no feature set beats the majority baseline. The proposed escape is to stop
predicting and start verifying — run the cheap retriever first, look at what came back, and
escalate only if it looks bad. That only works if failure is detectable from the cheap
retriever's own output. This script measures exactly that, and nothing else.

Framed as a binary task per retriever: given evidence available after running ONLY that
retriever, did it fail (no answer-bearing passage within k)? Reported as ROC-AUC against
0.5, across three nested feature sets that isolate where any signal comes from:

    pre_retrieval    query features only — what a pre-retrieval router sees. This is the
                     control: it should sit near chance, and if it does not, the failure
                     label is partly predictable from the query and the cascade's advantage
                     is smaller than it looks.
    + self_score     the retriever's own score distribution. This is classical QPP.
    + cross_score    the retriever's top-5 judged by the OTHER retriever's scoring
                     function — the post-retrieval evidence the cascade would actually use.

A large gap between pre_retrieval and cross_score is the result that justifies a cascade.
If all three sit near chance, escalation would be random and the cascade should be dropped
before it is built, not after.

CAUSAL CONSTRAINT — the reason this file is careful about feature indices:
Deciding whether to run Dense must not use Dense's results. Any feature derived from the
retriever we have not run yet is unavailable at decision time, and including it would be a
fresh leak of exactly the kind the leakage audit is about. So for the BM25 probe only these
are admissible:

    query features, BM25's own scores, and bm25_under_dense

`bm25_under_dense` is admissible and is worth stating plainly, because it looks like it
should not be: scoring BM25's top-5 under the dense function needs embeddings of five
passages plus the query, NOT a dense retrieval over the corpus. In deployment those passage
embeddings are already in the index. Five encodings is a rounding error against an ANN
search over the full corpus, so the cascade stays cheap.

Excluded for the BM25 probe: avg_dense, score_gap, dense_under_bm25, every hybrid_* term,
and both _gap features (they subtract a quantity that requires having run the other
retriever).

Usage:
    python experiments/router_experiments/escalation_signal_check.py --env-suffix _xscore
    python experiments/router_experiments/escalation_signal_check.py --env-suffix _xscore --k 20
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]
SPLIT_SEED = 42

# Handcrafted block (see QueryFeatureExtractor.get_feature_order):
#   0-21  query features — available before any retrieval
#   22-25 avg_bm25, avg_dense, score_gap, bm25_variance
#   26-28 cross-retriever agreement (needs BOTH ranked lists -> never available to a probe)
QUERY_IDX = list(range(0, 22))
SELF_SCORE_IDX = {"bm25": [22, 25], "dense": [23]}
# Offsets within the cross_score block, in the order written by prepare_env_data:
#   0 bm25_under_dense, 1 dense_under_bm25, 2 hybrid_under_bm25, 3 hybrid_under_dense,
#   4 dense_gap, 5 bm25_gap
CROSS_OFFSET = {"bm25": [0], "dense": [1]}


def load_layout(path):
    """Feature-group boundaries from the sidecar prepare_env_data writes next to the env."""
    side = os.path.splitext(path)[0] + "_layout.json"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_pool(path):
    """All queries with features, per-action MRR@k, and per-action full-corpus gold rank."""
    envs = create_train_val_test_split(path, split_seed=SPLIT_SEED)
    X, mrr, rank = [], [], []
    mrr_key = None
    for env in envs:
        for q in env.query_pool:
            rec = env.master_table[q]
            rets = rec["retrievers"]
            if mrr_key is None:
                mrr_key = next((kk for kk in rets["bm25"] if kk.startswith("mrr_at_")), "mrr")
            X.append(rec["features"])
            mrr.append([float(rets[a][mrr_key]) for a in ACTIONS])
            rank.append([float(rets[a].get("gold_rank") or 10**9) for a in ACTIONS])
    return np.array(X, dtype=float), np.array(mrr), np.array(rank), mrr_key


def feature_sets(retriever, dim, layout):
    """Nested admissible feature sets for this retriever's probe. See CAUSAL CONSTRAINT."""
    sets = {"pre_retrieval": list(QUERY_IDX)}
    self_idx = [i for i in SELF_SCORE_IDX.get(retriever, []) if i < dim]
    sets["+self_score"] = sets["pre_retrieval"] + self_idx

    cross = []
    if layout:
        start, end = layout.get("cross_score", [0, 0])
        cross = [start + o for o in CROSS_OFFSET.get(retriever, []) if start + o < end]
    sets["+cross_score"] = sets["+self_score"] + cross
    return sets, bool(cross)


def auc_cv(X, y, seed=0, n_splits=5, n_repeats=5):
    """Repeated stratified CV AUC. Returns (mean, std) or None if a class is missing."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    if len(np.unique(y)) < 2:
        return None
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    if not scores:
        return None
    return float(np.mean(scores)), float(np.std(scores))


def analyze(name, path, k_fail, seed):
    X, mrr, rank, mrr_key = load_pool(path)
    layout = load_layout(path)
    dim = X.shape[1]
    out = {"n": int(X.shape[0]), "feature_dim": dim, "metric": mrr_key,
           "k_fail": k_fail, "has_cross_scores": False, "probes": {}}

    for j, ret in enumerate(["bm25", "dense"]):
        # Failure = this retriever put no answer-bearing passage within k. Defined from the
        # full-corpus gold rank rather than from MRR@k so it stays meaningful when the
        # training reward is the dense 1/rank signal.
        y = (rank[:, j] > k_fail).astype(int)
        sets, has_cross = feature_sets(ret, dim, layout)
        out["has_cross_scores"] = out["has_cross_scores"] or has_cross

        probe = {"failure_rate": float(y.mean()),
                 "majority_baseline": float(max(y.mean(), 1 - y.mean())),
                 "sets": {}}
        for tag, idx in sets.items():
            if not idx:
                continue
            res = auc_cv(X[:, idx], y, seed=seed)
            probe["sets"][tag] = ({"auc": res[0], "auc_std": res[1], "n_features": len(idx)}
                                  if res else {"auc": None, "note": "single-class target"})
        pre = probe["sets"].get("pre_retrieval", {}).get("auc")
        post = probe["sets"].get("+cross_score", {}).get("auc")
        probe["post_minus_pre"] = (post - pre) if (pre is not None and post is not None) else None
        out["probes"][ret] = probe
    return out


def render(all_res, threshold):
    L = ["# Escalation Signal: is a retriever's failure detectable from its own output?\n",
         "Binary probe per retriever — did it fail to rank an answer-bearing passage within",
         "k? — using only evidence available after running THAT retriever. ROC-AUC under",
         "repeated stratified CV; 0.5 is chance.\n",
         "`pre_retrieval` is the control (query features only, what a pre-retrieval router",
         "sees). `+cross_score` adds the retriever's top-5 judged by the other retriever's",
         "scoring function. The gap between them is the value of looking after retrieving,",
         "and it is the number that decides whether a cascade is worth building.\n",
         "| dataset | retriever | fail rate | pre-retrieval AUC | +self AUC | +cross AUC | post-pre |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for ds, r in all_res.items():
        for ret, p in r["probes"].items():
            def g(t):
                v = p["sets"].get(t, {}).get("auc")
                s = p["sets"].get(t, {}).get("auc_std")
                return f"{v:.3f}±{s:.3f}" if v is not None else "—"
            d = p["post_minus_pre"]
            delta = f"{d:+.3f}" if d is not None else "—"
            L.append(f"| {ds} | {ret} | {p['failure_rate']:.1%} | {g('pre_retrieval')} | "
                     f"{g('+self_score')} | {g('+cross_score')} | {delta} |")
    L += ["\n## Verdict\n",
          f"A cascade is worth building if `+cross_score` AUC clears ~{0.5 + threshold:.2f} "
          "and improves materially on `pre_retrieval`. That means the cheap retriever's own "
          "output tells you when it failed, so escalation can be targeted rather than "
          "random.\n",
          "If every column sits near 0.5, failure is not detectable, escalation would be "
          "arbitrary, and the cascade should be dropped BEFORE the policy is built — which "
          "is the point of running this first.\n",
          "Note the asymmetry this whole line of work rests on: predicting which retriever "
          "will win requires knowing the query-passage interaction in advance, while "
          "detecting that a retriever just failed only requires looking at what it "
          "returned. The second is strictly easier, and only the second is needed.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="_xscore",
                   help="Env suffix to read. Defaults to _xscore, the arm built with "
                        "--cross-scores; without cross-score features this check can only "
                        "report the pre_retrieval and self_score columns.")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--k", type=int, default=20,
                   help="Failure cutoff: no answer-bearing passage within this rank (20).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.05,
                   help="AUC margin over chance required to call the signal usable.")
    p.add_argument("--output", type=str,
                   default=os.path.join(RESULTS_DIR, "escalation_signal.json"))
    p.add_argument("--output-md", type=str,
                   default=os.path.join(RESULTS_DIR, "escalation_signal.md"))
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
            print(f"  {ds}: env data missing ({path}) — skipped.")
            continue
        print(f"\n--- {ds} ---")
        r = analyze(ds, path, args.k, args.seed)
        all_res[ds] = r
        if not r["has_cross_scores"]:
            print("  NOTE: no cross-score features in this env — rebuild with "
                  "`prepare_env_data.py --cross-scores --suffix _xscore` for the full check.")
        for ret, p_ in r["probes"].items():
            pre = p_["sets"].get("pre_retrieval", {}).get("auc")
            post = p_["sets"].get("+cross_score", {}).get("auc")
            print(f"  {ret:6s} fail={p_['failure_rate']:.1%}  "
                  f"pre={pre if pre is None else round(pre, 3)}  "
                  f"cross={post if post is None else round(post, 3)}  "
                  f"delta={p_['post_minus_pre'] if p_['post_minus_pre'] is None else round(p_['post_minus_pre'], 3)}")

    if not all_res:
        print("\nNo env data found. Build it first (stage 3 / prepare_env_data.py).")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render(all_res, args.threshold))
    print(f"\nSaved: {args.output}\nSaved: {args.output_md}")


if __name__ == "__main__":
    main()
