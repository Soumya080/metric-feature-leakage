"""
15_grpo_comparison.py — Compare REINFORCE vs PPO vs GRPO across 5 seeds.

This is the full algorithmic comparison experiment for the paper.
Produces a 3-way comparison table with:
    - MRR (mean ± std) for all three algorithms
    - Oracle accuracy, convergence speed, training time
    - Per-seed breakdown and aggregate statistics

Usage:
    cd <repo>
    python experiments/router_experiments/15_grpo_comparison.py
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_val_test_split
from src.router.train_reinforce import ReinforceTrainer
from src.router.train_ppo import PPOTrainer
from src.router.train_grpo import GRPOTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configuration ----
SEEDS = [42, 123, 456, 789, 1024]
# Split seed is FIXED and independent of the training seed, so that all three algorithms
# and all seeds are compared on the identical test set (matching canonical_evaluation.py).
# Previously the split was re-derived from the training seed, so each seed saw a different
# train/test partition — that conflates split variance with algorithm variance and makes
# the reported std meaningless as a measure of algorithmic stability.
SPLIT_SEED = 42
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
REWARD_TYPE = "mrr"

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


def train_single(algo: str, seed: int, data_path: str):
    """Train a single seed with the given algorithm."""
    print(f"\n{'='*40} {algo} | SEED={seed} {'='*40}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env_train, env_val, env_test = create_train_val_test_split(data_path, split_seed=SPLIT_SEED)

    agent = PolicyAgent(
        input_dim=env_train.state_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=RouterEnvironment.NUM_ACTIONS,
        learningrate=LEARNING_RATE
    )

    t0 = time.time()

    if algo == "REINFORCE":
        trainer = ReinforceTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF, seed=seed,
            env_val=env_val
        )
    elif algo == "PPO":
        trainer = PPOTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF,
            clip_epsilon=0.2, ppo_epochs=4, seed=seed,
            env_val=env_val
        )
    elif algo == "GRPO":
        trainer = GRPOTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF,
            group_size=8, kl_coeff=0.01, seed=seed,
            env_val=env_val
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=150)
    wall_time = time.time() - t0
    
    test_mrrs = history["test_mean_reward"]
    
    # Baselines
    baselines = {}
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        total = sum(env_test.step(qid, action, target_metric=REWARD_TYPE) 
                    for qid in env_test.query_pool)
        baselines[ret_name] = total / len(env_test.query_pool)
    
    result = {
        "algorithm": algo,
        "seed": seed,
        # final_test_mrr is the number to report in the paper: the policy at the end of
        # training, evaluated once on the held-out test set.
        "final_test_mrr": test_mrrs[-1],
        # DIAGNOSTIC ONLY — do NOT quote this in the paper. It is the maximum over every
        # epoch's test-set evaluation, i.e. it peeks at the test set once per epoch and
        # keeps the luckiest value. Reporting it would be test-set cherry-picking.
        "best_test_mrr_DIAGNOSTIC_ONLY": max(test_mrrs),
        "final_oracle_accuracy": history["oracle_accuracy"][-1],
        "final_entropy": history["entropy"][-1],
        "convergence_epoch": next(
            (i for i, r in enumerate(test_mrrs) if r >= 0.95 * max(test_mrrs)),
            NUM_EPOCHS
        ),
        "wall_time_seconds": round(wall_time, 1),
        "best_fixed_mrr": max(baselines.values()),
    }
    result["improvement"] = result["final_test_mrr"] - result["best_fixed_mrr"]
    
    return result


def main():
    algorithms = ["REINFORCE", "PPO", "GRPO"]
    all_results = {algo: [] for algo in algorithms}
    
    for algo in algorithms:
        for seed in SEEDS:
            result = train_single(algo, seed, NQ_PATH)
            all_results[algo].append(result)
            print(f"  -> {algo} Seed {seed}: MRR={result['final_test_mrr']:.4f}")
    
    # ---- Aggregate ----
    print(f"\n{'='*80}")
    print(f"REINFORCE vs PPO vs GRPO — FULL COMPARISON")
    print(f"{'='*80}")
    
    comparison = {}
    for algo in algorithms:
        results = all_results[algo]
        mrrs = [r["final_test_mrr"] for r in results]
        best_mrrs = [r["best_test_mrr_DIAGNOSTIC_ONLY"] for r in results]
        oracle_accs = [r["final_oracle_accuracy"] for r in results]
        conv_epochs = [r["convergence_epoch"] for r in results]
        wall_times = [r["wall_time_seconds"] for r in results]
        improvements = [r["improvement"] for r in results]
        
        comparison[algo] = {
            "test_mrr_mean": round(np.mean(mrrs), 4),
            "test_mrr_std": round(np.std(mrrs), 4),
            "best_mrr_mean": round(np.mean(best_mrrs), 4),
            "best_mrr_std": round(np.std(best_mrrs), 4),
            "oracle_acc_mean": round(np.mean(oracle_accs), 4),
            "convergence_mean": round(np.mean(conv_epochs), 1),
            "wall_time_mean": round(np.mean(wall_times), 1),
            "improvement_mean": round(np.mean(improvements), 4),
        }
        
        c = comparison[algo]
        print(f"\n{algo:12s}: MRR={c['test_mrr_mean']:.4f}+/-{c['test_mrr_std']:.4f} | "
              f"Best={c['best_mrr_mean']:.4f} | "
              f"Conv={c['convergence_mean']:.0f}ep | "
              f"Time={c['wall_time_mean']:.0f}s | "
              f"Delta={c['improvement_mean']:+.4f}")
    
    # Determine winner
    best_algo = max(comparison.keys(), key=lambda a: comparison[a]["test_mrr_mean"])
    print(f"\n-> Best algorithm: {best_algo} "
          f"(MRR = {comparison[best_algo]['test_mrr_mean']:.4f})")
    
    # Save
    output = {
        "dataset": "nq",
        "num_seeds": len(SEEDS),
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "comparison": comparison,
        "per_seed": all_results,
    }
    
    output_path = os.path.join(RESULTS_DIR, "algorithm_comparison.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
