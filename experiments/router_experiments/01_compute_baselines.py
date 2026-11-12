"""
01_compute_baselines.py — Compute all fixed-strategy baseline metrics.

Produces the foundation numbers every other experiment compares against.
Outputs: results/baselines.json

Usage:
    cd <repo>
    python experiments/router_experiments/01_compute_baselines.py
"""

import json
import os
import sys
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
METRICS = ["hit_at_5", "mrr", "f1", "recall", "em"]
RETRIEVERS = ["bm25", "dense", "hybrid"]
DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}


def compute_baselines(data, test_keys, dataset_name):
    """Compute all fixed-strategy baselines on the test split."""
    results = {}
    n = len(test_keys)
    
    # 1. Fixed-strategy baselines (always pick one retriever)
    for ret_name in RETRIEVERS:
        ret_results = {}
        for metric in METRICS:
            total = sum(data[k]["retrievers"][ret_name].get(metric, 0.0) for k in test_keys)
            ret_results[metric] = round(total / n, 4)
        results[f"always_{ret_name}"] = ret_results
    
    # 2. Random baseline (averaged over 1000 trials for stability)
    random_trials = {m: [] for m in METRICS}
    for trial in range(1000):
        random.seed(SEED + trial)
        trial_scores = {m: 0.0 for m in METRICS}
        for k in test_keys:
            chosen = random.choice(RETRIEVERS)
            for m in METRICS:
                trial_scores[m] += data[k]["retrievers"][chosen].get(m, 0.0)
        for m in METRICS:
            random_trials[m].append(trial_scores[m] / n)
    
    results["random"] = {m: round(np.mean(random_trials[m]), 4) for m in METRICS}
    results["random_std"] = {m: round(np.std(random_trials[m]), 4) for m in METRICS}
    
    # 3. Oracle baseline (always pick the best retriever per query)
    oracle_results = {m: 0.0 for m in METRICS}
    for k in test_keys:
        for m in METRICS:
            best = max(data[k]["retrievers"][r].get(m, 0.0) for r in RETRIEVERS)
            oracle_results[m] += best
    results["oracle"] = {m: round(oracle_results[m] / n, 4) for m in METRICS}
    
    # 4. Summary stats
    best_fixed_name = max(RETRIEVERS, key=lambda r: results[f"always_{r}"]["mrr"])
    results["_meta"] = {
        "dataset": dataset_name,
        "test_size": n,
        "best_fixed_strategy": best_fixed_name,
        "best_fixed_mrr": results[f"always_{best_fixed_name}"]["mrr"],
        "oracle_mrr": results["oracle"]["mrr"],
        "headroom_mrr": round(results["oracle"]["mrr"] - results[f"always_{best_fixed_name}"]["mrr"], 4),
    }
    
    return results


def main():
    all_results = {}
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        with open(ds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reproduce the same train/test split as train_reinforce.py
        keys = list(data.keys())
        random.seed(SEED)
        random.shuffle(keys)
        split = int(len(keys) * 0.8)
        test_keys = keys[split:]
        
        print(f"Total queries: {len(keys)}")
        print(f"Test queries:  {len(test_keys)}")
        
        results = compute_baselines(data, test_keys, ds_name)
        all_results[ds_name] = results
        
        # Print results table
        print(f"\n{'Method':<20s} {'Hit@5':>8s} {'MRR':>8s} {'F1':>8s} {'EM':>8s}")
        print("-" * 50)
        for method in ["random", "always_bm25", "always_dense", "always_hybrid", "oracle"]:
            r = results[method]
            print(f"{method:<20s} {r['hit_at_5']:>8.4f} {r['mrr']:>8.4f} {r.get('f1',0):>8.4f} {r.get('em',0):>8.4f}")
        
        meta = results["_meta"]
        print(f"\nBest fixed: {meta['best_fixed_strategy']} (MRR={meta['best_fixed_mrr']:.4f})")
        print(f"Oracle: MRR={meta['oracle_mrr']:.4f}")
        print(f"Headroom: {meta['headroom_mrr']:.4f}")
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved baselines to: {output_path}")


if __name__ == "__main__":
    main()
