"""
14_ppo_vs_reinforce.py — Compare PPO and REINFORCE across 5 seeds.

Produces a head-to-head comparison table for the paper:
    - Mean MRR ± std for both algorithms
    - Oracle accuracy for both
    - Convergence speed (epochs to 95% of best)
    - Training wall-clock time

Usage:
    cd <repo>
    python experiments/router_experiments/14_ppo_vs_reinforce.py
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configuration ----
SEEDS = [42, 123, 456, 789, 1024]
# Fixed split seed, independent of the training seed — see the note in
# 15_grpo_comparison.py. Previously each training seed also changed the train/test
# partition, conflating split variance with algorithm variance.
SPLIT_SEED = 42
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
REWARD_TYPE = "mrr"
DATASET = "nq"

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
TQA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json")
DATA_PATH = NQ_PATH if DATASET == "nq" else TQA_PATH


def train_with_algorithm(algo: str, seed: int, data_path: str):
    """Train one seed with specified algorithm. Returns metrics dict."""
    print(f"\n{'='*40} {algo} SEED={seed} {'='*40}")
    
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

    start_time = time.time()

    if algo == "REINFORCE":
        trainer = ReinforceTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF, seed=seed,
            env_val=env_val
        )
        history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=100)
    elif algo == "PPO":
        trainer = PPOTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF,
            clip_epsilon=0.2, ppo_epochs=4, seed=seed,
            env_val=env_val
        )
        history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=100)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    wall_time = time.time() - start_time
    
    # Extract metrics
    test_mrrs = history["test_mean_reward"]
    result = {
        "algorithm": algo,
        "seed": seed,
        "final_test_mrr": test_mrrs[-1],
        # DIAGNOSTIC ONLY — max over per-epoch test evaluations, i.e. test-set
        # cherry-picking. Do not quote in the paper; use final_test_mrr.
        "best_test_mrr_DIAGNOSTIC_ONLY": max(test_mrrs),
        "final_oracle_accuracy": history["oracle_accuracy"][-1],
        "final_entropy": history["entropy"][-1],
        "convergence_epoch": next(
            (i for i, r in enumerate(test_mrrs) if r >= 0.95 * max(test_mrrs)),
            NUM_EPOCHS
        ),
        "wall_time_seconds": round(wall_time, 1),
    }
    
    # Compute baselines for this split
    baselines = {}
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        total = sum(env_test.step(qid, action, target_metric=REWARD_TYPE) for qid in env_test.query_pool)
        baselines[ret_name] = total / len(env_test.query_pool)
    result["best_fixed_mrr"] = max(baselines.values())
    result["improvement"] = result["final_test_mrr"] - result["best_fixed_mrr"]
    
    return result, history


def main():
    all_results = {"REINFORCE": [], "PPO": []}
    all_histories = {"REINFORCE": {}, "PPO": {}}
    
    for algo in ["REINFORCE", "PPO"]:
        for seed in SEEDS:
            result, history = train_with_algorithm(algo, seed, DATA_PATH)
            all_results[algo].append(result)
            all_histories[algo][str(seed)] = {
                "test_mean_reward": history["test_mean_reward"],
                "oracle_accuracy": history["oracle_accuracy"],
                "entropy": history["entropy"],
            }
            
            print(f"\n{algo} Seed {seed}: "
                  f"Test MRR={result['final_test_mrr']:.4f}, "
                  f"Best={result['best_test_mrr']:.4f}, "
                  f"Time={result['wall_time_seconds']}s")
    
    # ---- Aggregate comparison ----
    print(f"\n{'='*70}")
    print(f"PPO vs REINFORCE COMPARISON ({DATASET})")
    print(f"{'='*70}")
    
    comparison = {}
    for algo in ["REINFORCE", "PPO"]:
        results = all_results[algo]
        test_mrrs = [r["final_test_mrr"] for r in results]
        best_mrrs = [r["best_test_mrr_DIAGNOSTIC_ONLY"] for r in results]
        oracle_accs = [r["final_oracle_accuracy"] for r in results]
        conv_epochs = [r["convergence_epoch"] for r in results]
        wall_times = [r["wall_time_seconds"] for r in results]
        improvements = [r["improvement"] for r in results]
        
        comparison[algo] = {
            "test_mrr_mean": round(np.mean(test_mrrs), 4),
            "test_mrr_std": round(np.std(test_mrrs), 4),
            "best_mrr_mean": round(np.mean(best_mrrs), 4),
            "best_mrr_std": round(np.std(best_mrrs), 4),
            "oracle_acc_mean": round(np.mean(oracle_accs), 4),
            "oracle_acc_std": round(np.std(oracle_accs), 4),
            "convergence_epoch_mean": round(np.mean(conv_epochs), 1),
            "convergence_epoch_std": round(np.std(conv_epochs), 1),
            "wall_time_mean": round(np.mean(wall_times), 1),
            "wall_time_std": round(np.std(wall_times), 1),
            "improvement_mean": round(np.mean(improvements), 4),
            "improvement_std": round(np.std(improvements), 4),
        }
        
        c = comparison[algo]
        print(f"\n{algo}:")
        print(f"  Test MRR:      {c['test_mrr_mean']:.4f} ± {c['test_mrr_std']:.4f}")
        print(f"  Best MRR:      {c['best_mrr_mean']:.4f} ± {c['best_mrr_std']:.4f}")
        print(f"  Oracle Acc:    {c['oracle_acc_mean']:.4f} ± {c['oracle_acc_std']:.4f}")
        print(f"  Convergence:   {c['convergence_epoch_mean']:.0f} ± {c['convergence_epoch_std']:.0f} epochs")
        print(f"  Wall Time:     {c['wall_time_mean']:.1f} ± {c['wall_time_std']:.1f}s")
        print(f"  Improvement:   {c['improvement_mean']:+.4f} ± {c['improvement_std']:.4f}")
    
    # Statistical test: is the difference significant?
    reinforce_mrrs = [r["final_test_mrr"] for r in all_results["REINFORCE"]]
    ppo_mrrs = [r["final_test_mrr"] for r in all_results["PPO"]]
    
    try:
        from scipy import stats
        # PAIRED t-test: both algorithms are run on the same seeds and the same fixed
        # train/val/test split, so the samples are matched. ttest_ind (unpaired) was
        # previously used, which discards that pairing and is less powerful / not the
        # right null model here.
        t_stat, p_value = stats.ttest_rel(reinforce_mrrs, ppo_mrrs)
        print(f"\nPaired t-test (n={len(SEEDS)} seeds): t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Significant difference (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
        print(f"NOTE: n={len(SEEDS)} is small; treat this as indicative, not conclusive.")
    except ImportError:
        print("\nscipy not available for t-test. Install with: pip install scipy")
    
    # Save everything
    output = {
        "dataset": DATASET,
        "num_seeds": len(SEEDS),
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "comparison": comparison,
        "per_seed": all_results,
    }
    
    output_path = os.path.join(RESULTS_DIR, "ppo_vs_reinforce.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    histories_path = os.path.join(RESULTS_DIR, "ppo_vs_reinforce_histories.json")
    with open(histories_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print(f"Saved: {histories_path}")


if __name__ == "__main__":
    main()
