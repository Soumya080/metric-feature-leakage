"""
07_cross_dataset.py — Train on Dataset A, test on Dataset B.

This is the generalization experiment that makes your paper strong.
A router that only works on its training distribution is not useful.

Usage:
    cd <repo>
    python experiments/router_experiments/07_cross_dataset.py
"""

import json
import os
import sys
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import (
    RouterEnvironment,
    create_cross_dataset_envs,
    create_train_val_test_split,
)
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
TQA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json")

SEED = 42
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15


def run_cross_dataset(train_path, test_path, train_name, test_name):
    """Train on one dataset, test on another."""
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET: Train on {train_name}, Test on {test_name}")
    print(f"{'='*60}")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    env_train, env_test = create_cross_dataset_envs(train_path, test_path)

    # Model selection must NOT use env_test here: env_test is the *target* dataset, and
    # the whole claim of this experiment is zero-shot transfer to it. Picking the best
    # checkpoint by target-dataset reward would make the transfer numbers no longer
    # zero-shot. Instead hold out a validation split from the SOURCE dataset.
    source_train, source_val, _ = create_train_val_test_split(train_path, split_seed=SEED)

    agent = PolicyAgent(
        input_dim=source_train.state_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=3,
        learningrate=LEARNING_RATE
    )

    trainer = ReinforceTrainer(
        env_train=source_train, env_test=env_test, agent=agent,
        reward_type="mrr", entropy_coeff=ENTROPY_COEFF, seed=SEED,
        env_val=source_val
    )
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=50)
    
    # Compute baselines on the test dataset
    test_keys = env_test.query_pool
    baselines = {}
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        total = sum(env_test.step(qid, action, target_metric="mrr") for qid in test_keys)
        baselines[ret_name] = round(total / len(test_keys), 4)
    
    return {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "train_size": len(source_train),
        "val_size": len(source_val),
        "test_size": len(env_test),
        "final_test_mrr": round(history["test_mean_reward"][-1], 4),
        # DIAGNOSTIC ONLY — max over per-epoch target-dataset evaluations. Quoting this
        # would not be a zero-shot transfer number. Use final_test_mrr.
        "best_test_mrr_DIAGNOSTIC_ONLY": round(max(history["test_mean_reward"]), 4),
        "oracle_mrr": round(history["test_oracle_reward"][-1], 4),
        "baselines": baselines,
        "best_fixed_mrr": round(max(baselines.values()), 4),
        "improvement": round(history["test_mean_reward"][-1] - max(baselines.values()), 4),
        "history_test_mrr": history["test_mean_reward"],
        "history_epochs": history["epoch"],
    }


def main():
    results = {}
    
    # Experiment 1: Train NQ → Test TriviaQA
    results["nq_to_triviaqa"] = run_cross_dataset(NQ_PATH, TQA_PATH, "NQ", "TriviaQA")
    
    # Experiment 2: Train TriviaQA → Test NQ
    results["triviaqa_to_nq"] = run_cross_dataset(TQA_PATH, NQ_PATH, "TriviaQA", "NQ")
    
    # Summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Router MRR:    {r['final_test_mrr']:.4f}")
        print(f"  Best Fixed:    {r['best_fixed_mrr']:.4f}")
        print(f"  Oracle:        {r['oracle_mrr']:.4f}")
        print(f"  Improvement:   {r['improvement']:+.4f}")
    
    # Save (remove large history arrays for compact JSON)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {key: val for key, val in v.items() 
                           if key not in ["history_test_mrr", "history_epochs"]}
    
    output_path = os.path.join(RESULTS_DIR, "cross_dataset.json")
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
