"""
03_train_multi_seed.py — Run REINFORCE training across 5 seeds for error bars.

Produces mean ± std for all metrics. This is MANDATORY for any publication.

Usage:
    cd <repo>
    python experiments/router_experiments/03_train_multi_seed.py
"""

import os
import sys
import json
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configuration ----
SEEDS = [42, 123, 456, 789, 1024]
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
REWARD_TYPE = "mrr"
DATASET = "nq"  # Change to "triviaqa" for second dataset

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
TQA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json")
DATA_PATH = NQ_PATH if DATASET == "nq" else TQA_PATH


def train_single_seed(seed, data_path):
    """Train REINFORCE with a single seed. Returns final test metrics."""
    print(f"\n{'='*40} SEED={seed} {'='*40}")
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env_train, env_test = create_train_test_split(data_path, train_ratio=0.8, seed=seed)
    
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
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=50)
    
    # Extract final metrics
    final = {
        "seed": seed,
        "final_test_mrr": history["test_mean_reward"][-1],
        "best_test_mrr": max(history["test_mean_reward"]),
        "final_oracle_mrr": history["test_oracle_reward"][-1],
        "final_oracle_accuracy": history["oracle_accuracy"][-1],
        "final_entropy": history["entropy"][-1],
        "action_distribution": history["action_distribution"][-1],
        "convergence_epoch": next(
            (i for i, r in enumerate(history["test_mean_reward"]) 
             if r >= 0.95 * max(history["test_mean_reward"])),
            NUM_EPOCHS
        ),
    }
    
    # Also compute baselines for this specific split
    test_keys = env_test.query_pool
    baselines = {}
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        total = sum(env_test.step(qid, action, target_metric=REWARD_TYPE) for qid in test_keys)
        baselines[ret_name] = total / len(test_keys)
    final["baselines"] = baselines
    final["best_fixed_mrr"] = max(baselines.values())
    final["improvement_over_best_fixed"] = final["final_test_mrr"] - final["best_fixed_mrr"]
    
    return final, history


def main():
    all_seed_results = []
    all_histories = {}
    
    for seed in SEEDS:
        result, history = train_single_seed(seed, DATA_PATH)
        all_seed_results.append(result)
        all_histories[str(seed)] = {
            "epoch": history["epoch"],
            "test_mean_reward": history["test_mean_reward"],
            "train_mean_reward": history["train_mean_reward"],
            "oracle_accuracy": history["oracle_accuracy"],
            "entropy": history["entropy"],
        }
        
        print(f"\nSeed {seed}: Test MRR={result['final_test_mrr']:.4f}, "
              f"Best Fixed={result['best_fixed_mrr']:.4f}, "
              f"Delta={result['improvement_over_best_fixed']:+.4f}")
    
    # Aggregate statistics
    test_mrrs = [r["final_test_mrr"] for r in all_seed_results]
    best_mrrs = [r["best_test_mrr"] for r in all_seed_results]
    improvements = [r["improvement_over_best_fixed"] for r in all_seed_results]
    oracle_accs = [r["final_oracle_accuracy"] for r in all_seed_results]
    
    summary = {
        "dataset": DATASET,
        "num_seeds": len(SEEDS),
        "seeds": SEEDS,
        "test_mrr_mean": round(np.mean(test_mrrs), 4),
        "test_mrr_std": round(np.std(test_mrrs), 4),
        "best_mrr_mean": round(np.mean(best_mrrs), 4),
        "best_mrr_std": round(np.std(best_mrrs), 4),
        "improvement_mean": round(np.mean(improvements), 4),
        "improvement_std": round(np.std(improvements), 4),
        "oracle_acc_mean": round(np.mean(oracle_accs), 4),
        "oracle_acc_std": round(np.std(oracle_accs), 4),
        "per_seed": all_seed_results,
    }
    
    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY ({DATASET})")
    print(f"{'='*60}")
    print(f"Test MRR:    {summary['test_mrr_mean']:.4f} ± {summary['test_mrr_std']:.4f}")
    print(f"Best MRR:    {summary['best_mrr_mean']:.4f} ± {summary['best_mrr_std']:.4f}")
    print(f"Improvement: {summary['improvement_mean']:+.4f} ± {summary['improvement_std']:.4f}")
    print(f"Oracle Acc:  {summary['oracle_acc_mean']:.4f} ± {summary['oracle_acc_std']:.4f}")
    
    # Save
    output_path = os.path.join(RESULTS_DIR, f"multi_seed_{DATASET}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    history_path = os.path.join(RESULTS_DIR, f"multi_seed_histories_{DATASET}.json")
    with open(history_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print(f"Saved: {history_path}")


if __name__ == "__main__":
    main()
