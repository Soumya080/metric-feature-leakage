"""Fusion baselines: is routing better than just combining the lists you already have?

The question this answers is the most damaging one the paper faces, because it attacks the
problem rather than the method: at decision time all three ranked lists already exist, so
why select one instead of merging them?

Part of the answer is already in Table 1 and was not being stated. Hybrid IS Reciprocal
Rank Fusion over BM25 and Dense across the full corpus, so a fusion arm has been sitting in
the results the whole time -- and on HotpotQA it is the best arm. This script adds the
stronger fusion that was missing: a convex combination of min-max normalised scores,

    score = alpha * norm(BM25) + (1 - alpha) * norm(Dense)

which Bruch, Gai & Ingber (TOIS 2023) report beats RRF in- and out-of-domain.

alpha is selected on the TRAINING split and reported on test, using the same partition
RouterEnvironment uses. Selecting it on all queries would be test-set tuning -- the thing
this codebase exists to stop doing -- and would flatter fusion exactly as reward leakage
flattered the router.

Interpretation. If fusion beats every routing method, the paper's framing changes: the
useful recommendation becomes "do not route, fuse", which is a more actionable finding than
a routing result. If fusion does not, the objection is answered with a measurement instead
of an argument. Either outcome is worth having; neither requires the number to come out a
particular way.

Usage:
    python experiments/router_experiments/fusion_baseline.py
    python experiments/router_experiments/fusion_baseline.py --env-dir /path --per-query-dir /path
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.routing_env import split_query_ids            # noqa: E402
from src.router.provenance import build as build_provenance   # noqa: E402

DATASETS = ["nq", "triviaqa", "hotpotqa"]
ACTIONS = ["bm25", "dense", "hybrid"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def rank_to_reward(rank, n_corpus):
    """1/rank, matching prepare_env_data.py's dense reward.

    None means the retriever never ranked an answer-bearing passage anywhere. Treated as
    rank n_corpus + 1 -- strictly worse than any achieved rank -- rather than dropped,
    which would silently score fusion on an easier subset than the arms it is compared to.
    """
    if rank is None:
        rank = (n_corpus or 500) + 1
    return 1.0 / float(rank)


def load_dataset(env_path, per_query_path):
    """Join env data (which defines the query set and split) to per-query fusion ranks.

    The env data is already gold-type filtered, so it -- not the raw per-query file --
    decides which queries count. Any query without a fusion record is reported rather than
    silently skipped: a partial join would quietly change the comparison set.
    """
    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    env = env.get("queries", env)

    with open(per_query_path, "r", encoding="utf-8") as f:
        pq = json.load(f)
    # 02_run_baselines.py writes a bare list; other producers wrap it under "results".
    rows = pq if isinstance(pq, list) else pq.get("results", pq)
    by_id = {str(r["query_id"]): r for r in rows}

    joined, missing = {}, 0
    for qid, entry in env.items():
        rec = by_id.get(str(qid))
        if rec is None or not rec.get("convex_fusion_gold_rank"):
            missing += 1
            continue
        joined[qid] = {
            "rewards": {a: float(entry["retrievers"][a]["mrr"]) for a in ACTIONS},
            "fusion": rec["convex_fusion_gold_rank"],
            "n_corpus": rec.get("n_corpus", rec.get("corpus_size", 500)),
        }
    return joined, missing, len(env)


def evaluate(joined, split_seed=42):
    """Select alpha on train, report on test, alongside the arms and the oracle."""
    train_ids, _, test_ids = split_query_ids(list(joined), split_seed=split_seed)
    train_ids = [q for q in train_ids if q in joined]
    test_ids = [q for q in test_ids if q in joined]

    alphas = sorted(next(iter(joined.values()))["fusion"], key=float)

    def fusion_mean(ids, alpha):
        return float(np.mean([rank_to_reward(joined[q]["fusion"][alpha],
                                             joined[q]["n_corpus"]) for q in ids]))

    train_curve = {a: fusion_mean(train_ids, a) for a in alphas}
    best_alpha = max(train_curve, key=train_curve.get)

    arms = {a: float(np.mean([joined[q]["rewards"][a] for q in test_ids])) for a in ACTIONS}
    oracle = float(np.mean([max(joined[q]["rewards"].values()) for q in test_ids]))
    best_fixed_name = max(arms, key=arms.get)

    return {
        "n_train": len(train_ids), "n_test": len(test_ids),
        "selected_alpha": float(best_alpha),
        "alpha_selected_on": "train split",
        "train_curve": {a: train_curve[a] for a in alphas},
        "test_curve": {a: fusion_mean(test_ids, a) for a in alphas},
        "convex_fusion_test": fusion_mean(test_ids, best_alpha),
        "arms_test": arms,
        "best_fixed_name": best_fixed_name,
        "best_fixed_test": arms[best_fixed_name],
        "rrf_hybrid_test": arms["hybrid"],
        "oracle_test": oracle,
    }


def render(all_res):
    L = ["# Fusion Baselines: route, or just combine the lists?", "",
         "At decision time all three ranked lists already exist. This asks whether "
         "selecting one beats merging them.", "",
         "`RRF (Hybrid)` is not a new baseline — Hybrid **is** reciprocal rank fusion over "
         "BM25 and Dense across the full corpus, so it has been in Table 1 all along. "
         "`Convex` is the addition: `alpha * norm(BM25) + (1-alpha) * norm(Dense)`, with "
         "alpha chosen on the **training** split and reported on test.", "",
         "| dataset | n test | best arm | RRF (Hybrid) | Convex (alpha) | Oracle | Convex - best arm |",
         "|---|---:|---|---:|---:|---:|---:|"]
    for ds, r in all_res.items():
        delta = r["convex_fusion_test"] - r["best_fixed_test"]
        L.append(f"| {ds} | {r['n_test']} | {r['best_fixed_name']} "
                 f"{r['best_fixed_test']:.4f} | {r['rrf_hybrid_test']:.4f} | "
                 f"{r['convex_fusion_test']:.4f} (a={r['selected_alpha']:.1f}) | "
                 f"{r['oracle_test']:.4f} | **{delta:+.4f}** |")
    L += ["",
          "A positive final column means fusion beats committing to the single best "
          "retriever. If it is positive while every routing method is negative, the honest "
          "recommendation from this work is to fuse rather than route — and that belongs in "
          "the paper regardless of how it reflects on the original method.", "",
          "## Selected alpha, and what the sweep looks like on test", ""]
    for ds, r in all_res.items():
        L += [f"**{ds}** — alpha selected on train: **{r['selected_alpha']:.1f}**", "",
              "| alpha | " + " | ".join(f"{a}" for a in r["test_curve"]) + " |",
              "|---" * (len(r["test_curve"]) + 1) + "|",
              "| test MRR | " + " | ".join(f"{v:.4f}" for v in r["test_curve"].values()) + " |",
              ""]
    L += ["_alpha = 0 is pure Dense, alpha = 1 is pure BM25; both reproduce the "
          "corresponding arm exactly, which is the implementation's self-check._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-dir", default=os.path.join(PROJECT_ROOT, "src", "router"))
    ap.add_argument("--per-query-dir",
                    default=os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "results"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_res, prov = {}, {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        env_path = os.path.join(args.env_dir, f"{ds}_env_data.json")
        pq_path = os.path.join(args.per_query_dir, f"{ds}_per_query_results.json")
        if not (os.path.exists(env_path) and os.path.exists(pq_path)):
            print(f"  {ds}: missing env data or per-query results — skipped.")
            continue

        joined, missing, n_env = load_dataset(env_path, pq_path)
        if missing:
            print(f"  {ds}: WARNING {missing}/{n_env} queries have no convex_fusion_gold_rank "
                  f"- re-run 02_run_baselines.py (post-2026-08-05) to populate them.")
        if not joined:
            print(f"  {ds}: no joinable queries — skipped.")
            continue

        r = evaluate(joined, split_seed=args.split_seed)
        all_res[ds] = r
        prov[ds] = build_provenance(env_path=env_path, metric="mrr",
                                    split_seed=args.split_seed)
        print(f"  {ds}: n_test={r['n_test']}  best arm {r['best_fixed_name']} "
              f"{r['best_fixed_test']:.4f} | RRF {r['rrf_hybrid_test']:.4f} | "
              f"convex {r['convex_fusion_test']:.4f} (a={r['selected_alpha']:.1f}) "
              f"| delta {r['convex_fusion_test'] - r['best_fixed_test']:+.4f}")

    if not all_res:
        raise SystemExit("No datasets evaluated.")

    payload = dict(all_res)
    payload["_provenance"] = {"metric": "mrr", "split_seed": args.split_seed,
                              "alpha_selected_on": "train split", "datasets": prov}
    with open(os.path.join(args.out_dir, "fusion_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(args.out_dir, "fusion_baseline.md"), "w", encoding="utf-8") as f:
        f.write(render(all_res))
    print(f"\nSaved: {os.path.join(args.out_dir, 'fusion_baseline.json')}")
    print(f"Saved: {os.path.join(args.out_dir, 'fusion_baseline.md')}")


if __name__ == "__main__":
    main()
