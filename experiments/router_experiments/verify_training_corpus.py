"""
verify_training_corpus.py — Sanity-check the RL training corpus before trusting any result.

The env data is the last artifact before training, and every upstream bug this project has
hit (MRR-derived features, template gold passages, random negatives sold as hard ones,
silently-zero cross-retriever features) was invisible in the aggregate metrics and obvious
in the env data. This script looks at the env data directly and fails loudly on the
specific failure modes we already know about, rather than eyeballing a final MRR.

Checks, per dataset:
  1. Shape        — every record has the same feature dim; no NaN/inf.
  2. Dead features— columns that are constant across all queries carry no signal. Reported,
                    not fatal (some binary flags legitimately fire rarely).
  3. Leakage      — no feature column is (near-)identical to any action's reward. This is
                    the exact bug that inflated the original results.
  4. Reward table — distribution per action, oracle gap, how often each action is uniquely
                    best. If one action is always best there is nothing to route.
  5. Cross-retriever features — present, non-degenerate, in [0,1].
  6. Splits       — train/val/test are disjoint, cover the pool, and are large enough for
                    the downstream GBT (RandomizedSearchCV with cv=3 needs >=3 per class).

Usage:
    python experiments/router_experiments/verify_training_corpus.py
    python experiments/router_experiments/verify_training_corpus.py --env-suffix _smoke
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402
from src.router.feature_extractor import QueryFeatureExtractor  # noqa: E402

ACTIONS = ["bm25", "dense", "hybrid"]
SPLIT_SEED = 42
XR_IDX = [26, 27, 28]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    return env.get("queries", env)


def check_dataset(name, path, metric="mrr"):
    queries = load(path)
    qids = list(queries)
    problems, warnings = [], []

    feats = np.array([queries[q]["features"] for q in qids], dtype=float)
    n, dim = feats.shape
    names = QueryFeatureExtractor.get_feature_order()

    print(f"\n{'='*66}\n{name.upper()}  —  n={n}, feature_dim={dim}\n{'='*66}")

    # --- 1. shape / finiteness -------------------------------------------------
    if dim != len(names):
        problems.append(f"feature dim {dim} != canonical order {len(names)}")
    bad = np.argwhere(~np.isfinite(feats))
    if bad.size:
        problems.append(f"{len(bad)} non-finite feature values (first at row {bad[0][0]}, col {bad[0][1]})")

    # --- 2. dead columns -------------------------------------------------------
    spread = np.ptp(feats, axis=0)  # ndarray.ptp() was removed in NumPy 2.0
    dead = [names[i] if i < len(names) else f"f{i}" for i in np.where(spread == 0)[0]]
    if dead:
        warnings.append(f"{len(dead)} constant (dead) feature(s): {dead}")

    # --- 3. leakage: is any feature a copy of a reward? ------------------------
    rewards = {a: np.array([float(queries[q]["retrievers"][a][metric]) for q in qids])
               for a in ACTIONS}
    for a, rvec in rewards.items():
        for j in range(dim):
            col = feats[:, j]
            if np.allclose(col, rvec, atol=1e-12):
                fname = names[j] if j < len(names) else f"f{j}"
                problems.append(
                    f"LEAK: feature[{j}] ({fname}) is identical to reward of action {a!r}"
                )
    # near-identical too: a 0.999 correlation is a leak wearing a hat
    for a, rvec in rewards.items():
        if np.ptp(rvec) == 0:
            continue
        for j in range(dim):
            col = feats[:, j]
            if np.ptp(col) == 0:
                continue
            r = abs(float(np.corrcoef(col, rvec)[0, 1]))
            if r > 0.99:
                fname = names[j] if j < len(names) else f"f{j}"
                warnings.append(f"feature[{j}] ({fname}) corr={r:.4f} with reward of {a!r}")

    # --- 4. reward table -------------------------------------------------------
    stacked = np.vstack([rewards[a] for a in ACTIONS])
    means = {a: float(rewards[a].mean()) for a in ACTIONS}
    oracle = stacked.max(axis=0)
    best_fixed_name = max(means, key=means.get)
    headroom = float(oracle.mean() - means[best_fixed_name])

    best_counts = Counter()
    unique_best = 0
    for i in range(n):
        col = stacked[:, i]
        winners = [ACTIONS[k] for k in np.where(col == col.max())[0]]
        if len(winners) == 1:
            unique_best += 1
            best_counts[winners[0]] += 1

    print("  reward means      : " + "  ".join(f"{a}={means[a]:.4f}" for a in ACTIONS))
    print(f"  best fixed        : {best_fixed_name} ({means[best_fixed_name]:.4f})")
    print(f"  oracle            : {oracle.mean():.4f}")
    print(f"  HEADROOM          : {headroom:+.4f}"
          + (f"  ({headroom/means[best_fixed_name]*100:.1f}% relative)"
             if means[best_fixed_name] > 0 else ""))
    print(f"  queries w/ unique best action : {unique_best}/{n} ({unique_best/n*100:.1f}%)")
    print(f"  unique-best distribution      : {dict(best_counts)}")

    if headroom <= 0:
        problems.append("headroom <= 0: no routing gain exists in this corpus at all")
    if unique_best == 0:
        problems.append("no query has a unique best action — routing cannot be learned")
    elif unique_best / n < 0.05:
        warnings.append(f"only {unique_best/n*100:.1f}% of queries have a unique best action; "
                        f"the routing signal is very sparse")
    if len(best_counts) == 1:
        warnings.append(f"only {list(best_counts)[0]!r} is ever uniquely best — "
                        f"a constant policy would be optimal")
    if all(r.mean() == 0 for r in rewards.values()):
        problems.append("all rewards are 0 — retrieval found nothing; check the corpus")

    # --- 5. cross-retriever features ------------------------------------------
    if dim > max(XR_IDX):
        xr = feats[:, XR_IDX]
        print(f"  cross-retriever   : overlap@5 mean={xr[:,0].mean():.3f} "
              f"(min={xr[:,0].min():.2f} max={xr[:,0].max():.2f}), "
              f"top1_agree={xr[:,2].mean():.1%}")
        if np.all(xr == 0):
            problems.append("all cross-retriever features are 0 — 02_run_baselines.py "
                            "likely ran without persisting 'retrieved_ids_top10'")
        if xr.min() < 0 or xr.max() > 1:
            problems.append(f"cross-retriever features outside [0,1]: "
                            f"min={xr.min()}, max={xr.max()}")
    else:
        problems.append(f"feature dim {dim} has no cross-retriever features (expected 29)")

    # --- 6. splits -------------------------------------------------------------
    tr, va, te = create_train_val_test_split(path, split_seed=SPLIT_SEED)
    s_tr, s_va, s_te = set(tr.query_pool), set(va.query_pool), set(te.query_pool)
    print(f"  splits            : train={len(s_tr)} val={len(s_va)} test={len(s_te)}")
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        problems.append("train/val/test splits OVERLAP — this is a leak")
    if len(s_tr | s_va | s_te) != n:
        warnings.append(f"splits cover {len(s_tr|s_va|s_te)} of {n} queries")

    # GBT tuning uses RandomizedSearchCV(cv=3) over oracle-action labels.
    train_labels = Counter()
    for q in s_tr:
        col = np.array([float(queries[q]["retrievers"][a][metric]) for a in ACTIONS])
        train_labels[ACTIONS[int(col.argmax())]] += 1
    thin = {k: v for k, v in train_labels.items() if v < 3}
    if thin:
        problems.append(f"train split has <3 examples for oracle class(es) {thin} — "
                        f"canonical_evaluation.py's GBT (cv=3) will crash")
    print(f"  train oracle-label counts     : {dict(train_labels)}")

    return problems, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-suffix", type=str, default="")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--metric", type=str, default="mrr")
    args = parser.parse_args()

    paths = get_dataset_paths(args.env_suffix)
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        paths = {k: v for k, v in paths.items() if k in keep}

    all_problems, all_warnings = {}, {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"\n{name}: env data not found at {path} — skipping.")
            continue
        p, w = check_dataset(name, path, metric=args.metric)
        if p:
            all_problems[name] = p
        if w:
            all_warnings[name] = w

    print(f"\n{'='*66}\nVERDICT\n{'='*66}")
    if all_warnings:
        for ds, ws in all_warnings.items():
            for w in ws:
                print(f"  WARN  [{ds}] {w}")
    if all_problems:
        for ds, ps in all_problems.items():
            for p in ps:
                print(f"  FAIL  [{ds}] {p}")
        print("\nTraining corpus is NOT sound. Fix the above before trusting any result.")
        sys.exit(1)

    print("  All checks passed — corpus is trainable and leak-free.")
    if all_warnings:
        print("  (Warnings above are informational, not blocking.)")


if __name__ == "__main__":
    main()
