"""
19_scale_corpus.py — SUPERSEDED. Do not cite this in the paper.

WHAT THIS ACTUALLY DOES: trains the router on 10/25/50/75/100% of the *query set* and
reports how performance varies with the number of TRAINING QUERIES. That is a
sample-efficiency curve.

WHAT ITS NAME AND ORIGINAL DOCSTRING CLAIMED: a corpus-scaling experiment answering
"does this work at scale?". It does not. The retrieval corpus is a fixed 500 passages
per query at every point on the curve — nothing about corpus size changes. Citing this
as evidence for corpus scaling would be a false claim, and corpus scale is explicitly
deferred to future work (see the paper's Limitations).

SECOND PROBLEM: it uses create_train_test_split (train/test only) and passes no env_val
to ReinforceTrainer, so best-checkpoint selection silently falls back to TEST-set
performance — the exact train/test leak fixed elsewhere in this repo (canonical_evaluation.py,
21_fixed_split_multi_seed.py, 23_multi_seed_ablation.py all use create_train_val_test_split
with env_val=). Any number this script produces is optimistically biased.

If a sample-efficiency curve is wanted, port it to the create_train_val_test_split +
env_val= pattern from canonical_evaluation.py first, and rename it. Until then this
script is kept only for reference and is not wired into run_full_pipeline.py.

Usage:
    cd <repo>
    python experiments/router_experiments/19_scale_corpus.py --i-understand-this-is-leaky
"""

import os
import sys
import json
import random
import numpy as np
import torch
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
TQA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json")

# ---- Configuration ----
SCALE_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
SEEDS = [42, 123, 456]  # 3 seeds for efficiency
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
REWARD_TYPE = "mrr"


def run_scaling_experiment(data_path: str, dataset_name: str):
    """Run the router at different corpus sizes and measure performance.
    
    For each fraction:
        1. Subsample the queries
        2. Split into train/test (80/20)
        3. Train REINFORCE
        4. Record: MRR, oracle accuracy, convergence speed, training time
    
    Returns:
        Dict with per-fraction results.
    """
    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT — {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load full dataset to get total size
    with open(data_path, "r") as f:
        full_data = json.load(f)
    total_queries = len(full_data)
    all_qids = list(full_data.keys())
    
    print(f"Total available queries: {total_queries}")
    
    scaling_results = []
    
    for fraction in SCALE_FRACTIONS:
        n_queries = max(10, int(total_queries * fraction))
        
        print(f"\n{'='*50}")
        print(f"Fraction: {fraction:.0%} ({n_queries} queries)")
        print(f"{'='*50}")
        
        seed_results = []
        
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Subsample queries
            sampled_qids = random.sample(all_qids, n_queries)
            
            # Create a temporary env data file with subset
            subset_data = {qid: full_data[qid] for qid in sampled_qids}
            temp_path = os.path.join(
                PROJECT_ROOT, "src", "router", 
                f"_temp_scale_{dataset_name}_{fraction:.2f}_{seed}.json"
            )
            
            with open(temp_path, "w") as f:
                json.dump(subset_data, f)
            
            try:
                # Create train/test split
                env_train, env_test = create_train_test_split(
                    temp_path, train_ratio=0.8, seed=seed
                )
                
                agent = PolicyAgent(
                    input_dim=env_train.state_dim,
                    hidden_dim=HIDDEN_DIM,
                    output_dim=RouterEnvironment.NUM_ACTIONS,
                    learningrate=LEARNING_RATE
                )
                
                trainer = ReinforceTrainer(
                    env_train=env_train, env_test=env_test, agent=agent,
                    reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF, seed=seed
                )
                
                t0 = time.time()
                history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=300)
                wall_time = time.time() - t0
                
                test_mrrs = history["test_mean_reward"]
                
                # Baselines for this subset
                baselines = {}
                for action in range(3):
                    ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
                    total = sum(env_test.step(qid, action, target_metric=REWARD_TYPE) 
                               for qid in env_test.query_pool)
                    baselines[ret_name] = total / len(env_test.query_pool) if len(env_test.query_pool) > 0 else 0
                
                best_fixed = max(baselines.values()) if baselines else 0
                
                # Oracle
                oracle_total = sum(
                    env_test.step(qid, env_test.get_oracle_action(qid, REWARD_TYPE), 
                                target_metric=REWARD_TYPE)
                    for qid in env_test.query_pool
                )
                oracle_mrr = oracle_total / len(env_test.query_pool) if len(env_test.query_pool) > 0 else 0
                
                seed_results.append({
                    "seed": seed,
                    "final_test_mrr": test_mrrs[-1],
                    "best_test_mrr": max(test_mrrs),
                    "best_fixed_mrr": best_fixed,
                    "oracle_mrr": oracle_mrr,
                    "improvement": test_mrrs[-1] - best_fixed,
                    "oracle_accuracy": history["oracle_accuracy"][-1],
                    "convergence_epoch": next(
                        (i for i, r in enumerate(test_mrrs) if r >= 0.95 * max(test_mrrs)),
                        NUM_EPOCHS
                    ),
                    "wall_time": round(wall_time, 1),
                    "train_queries": len(env_train.query_pool),
                    "test_queries": len(env_test.query_pool),
                })
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Aggregate per fraction
        fraction_result = {
            "fraction": fraction,
            "n_queries": n_queries,
            "train_queries": seed_results[0]["train_queries"],
            "test_queries": seed_results[0]["test_queries"],
            "router_mrr_mean": round(np.mean([r["final_test_mrr"] for r in seed_results]), 4),
            "router_mrr_std": round(np.std([r["final_test_mrr"] for r in seed_results]), 4),
            "best_fixed_mrr_mean": round(np.mean([r["best_fixed_mrr"] for r in seed_results]), 4),
            "oracle_mrr_mean": round(np.mean([r["oracle_mrr"] for r in seed_results]), 4),
            "improvement_mean": round(np.mean([r["improvement"] for r in seed_results]), 4),
            "oracle_accuracy_mean": round(np.mean([r["oracle_accuracy"] for r in seed_results]), 4),
            "convergence_mean": round(np.mean([r["convergence_epoch"] for r in seed_results]), 1),
            "wall_time_mean": round(np.mean([r["wall_time"] for r in seed_results]), 1),
            "per_seed": seed_results,
        }
        
        scaling_results.append(fraction_result)
        
        fr = fraction_result
        print(f"\n  Results ({fraction:.0%}):")
        print(f"  Router MRR:   {fr['router_mrr_mean']:.4f} ± {fr['router_mrr_std']:.4f}")
        print(f"  Best Fixed:   {fr['best_fixed_mrr_mean']:.4f}")
        print(f"  Oracle:       {fr['oracle_mrr_mean']:.4f}")
        print(f"  Improvement:  {fr['improvement_mean']:+.4f}")
        print(f"  Convergence:  {fr['convergence_mean']:.0f} epochs")
        print(f"  Wall Time:    {fr['wall_time_mean']:.0f}s")
    
    return scaling_results


def analyze_scaling_trends(results: list):
    """Analyze whether performance scales with corpus size."""
    print(f"\n{'='*60}")
    print("SCALING TREND ANALYSIS")
    print(f"{'='*60}")
    
    fractions = [r["fraction"] for r in results]
    router_mrrs = [r["router_mrr_mean"] for r in results]
    improvements = [r["improvement_mean"] for r in results]
    oracle_gaps = [r["oracle_mrr_mean"] - r["best_fixed_mrr_mean"] for r in results]
    headroom_captured = [
        r["improvement_mean"] / (r["oracle_mrr_mean"] - r["best_fixed_mrr_mean"]) * 100
        if r["oracle_mrr_mean"] > r["best_fixed_mrr_mean"] else 0
        for r in results
    ]
    
    print(f"\n{'Fraction':>10s} {'Router MRR':>12s} {'Improvement':>12s} "
          f"{'Oracle Gap':>12s} {'% Captured':>12s}")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{r['fraction']:>10.0%} {router_mrrs[i]:>12.4f} "
              f"{improvements[i]:>+12.4f} {oracle_gaps[i]:>12.4f} "
              f"{headroom_captured[i]:>11.1f}%")
    
    # Linear regression on log(n_queries) vs MRR
    n_queries = [r["n_queries"] for r in results]
    log_n = np.log(n_queries)
    
    # Fit linear trend
    if len(log_n) > 1:
        slope, intercept = np.polyfit(log_n, router_mrrs, 1)
        print(f"\nLog-linear trend: MRR = {slope:.4f} * log(N) + {intercept:.4f}")
        
        # Predict for larger corpora
        for future_n in [1000, 5000, 10000]:
            predicted_mrr = slope * np.log(future_n) + intercept
            print(f"  Predicted MRR at N={future_n:,}: {predicted_mrr:.4f}")
    
    return {
        "fractions": fractions,
        "router_mrrs": router_mrrs,
        "improvements": improvements,
        "oracle_gaps": oracle_gaps,
        "headroom_captured": headroom_captured,
    }


def main():
    if not os.path.exists(NQ_PATH):
        print(f"ERROR: {NQ_PATH} not found")
        sys.exit(1)
    
    # Run scaling on NQ
    nq_results = run_scaling_experiment(NQ_PATH, "nq")
    
    # Analyze trends
    trends = analyze_scaling_trends(nq_results)
    
    # Optionally run on TriviaQA too
    tqa_results = None
    if os.path.exists(TQA_PATH):
        tqa_results = run_scaling_experiment(TQA_PATH, "triviaqa")
    
    # Save
    output = {
        "nq": {
            "scaling_results": nq_results,
            "trends": trends,
        },
    }
    
    if tqa_results:
        output["triviaqa"] = {
            "scaling_results": tqa_results,
        }
    
    output_path = os.path.join(RESULTS_DIR, "scaling_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    # Refuse to run silently: the numbers this produces are test-set-selected (see the
    # module docstring), so an accidental invocation must not quietly generate results
    # that could end up in the paper.
    if "--i-understand-this-is-leaky" not in sys.argv:
        print(__doc__)
        print("REFUSING TO RUN. This script has a known test-set model-selection leak and "
              "its results must not be reported. Pass --i-understand-this-is-leaky to "
              "override for debugging only.")
        sys.exit(1)
    main()
