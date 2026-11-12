#!/usr/bin/env python3
"""
failure_mode_analysis.py - Qualitative examples where RL-Route routes correctly and
tuned Gradient Boosting does not (and vice versa), for the paper's results discussion.

Table 1 argues RL-Route's advantage through aggregate MRR alone. This script surfaces
concrete per-query examples — the question text, each retriever's raw score, and each
method's routing decision — so a reviewer can see *why* the router wins on specific
queries, not just that it wins on average.

Reuses the exact same train/val/test split and GBT tuning as canonical_evaluation.py so
the comparison is apples-to-apples with the main results table.

Usage:
    python experiments/router_experiments/failure_mode_analysis.py
    python experiments/router_experiments/failure_mode_analysis.py --dataset triviaqa --top_n 10
    python experiments/router_experiments/failure_mode_analysis.py --seed 42 --epochs 20   # quick test
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from src.router.policy_network import PolicyAgent
from src.router.routing_env import create_train_val_test_split, deterministic_tiebreak
from src.router.train_reinforce import ReinforceTrainer
from canonical_evaluation import (
    DATASETS, SPLIT_SEED, HIDDEN_DIM, LEARNING_RATE, ENTROPY_COEFF, REWARD_TYPE,
    compute_supervised_baselines,
)

RETRIEVER_NAMES = ["bm25", "dense", "hybrid"]


def train_gbt_with_predictions(env_train, env_test):
    """Refit the tuned GBT (same hyperparameter search as canonical_evaluation.py) and
    return per-query predicted actions on the test set, aligned with env_test.query_pool."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV

    def prepare_xy(env):
        X, y_cls = [], []
        for qid in env.query_pool:
            X.append(env.get_state(qid))
            scores = [env.get_all_rewards(qid, REWARD_TYPE)[a] for a in range(3)]
            max_score = max(scores)
            tied = [a for a in range(3) if scores[a] == max_score]
            y_cls.append(tied[deterministic_tiebreak(qid, len(tied))])
        return np.array(X), np.array(y_cls)

    X_train, y_train = prepare_xy(env_train)
    X_test, _ = prepare_xy(env_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    gbt_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_leaf': [1, 5, 10],
    }
    gbt = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gbt_param_dist,
                              n_iter=20, cv=3, random_state=42, n_jobs=-1)
    gbt.fit(X_train, y_train)
    return gbt.predict(X_test)


def train_rl_with_predictions(env_train, env_val, env_test, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = PolicyAgent(input_dim=env_train.state_dim, hidden_dim=HIDDEN_DIM, output_dim=3, learningrate=LEARNING_RATE)
    trainer = ReinforceTrainer(
        env_train=env_train, env_test=env_test, agent=agent,
        reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF, seed=seed, env_val=env_val,
    )
    trainer.train(num_epochs=epochs, log_interval=epochs)

    actions = []
    for qid in env_test.query_pool:
        state = trainer._normalize_state(env_test.get_state(qid))
        action, _ = agent.select_action(state, evaluate=True)
        actions.append(action)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Qualitative RL-Route vs. GBT failure-mode examples.")
    parser.add_argument("--dataset", type=str, default="nq", choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=42, help="Single RL training seed (this is a qualitative script, not a multi-seed statistic).")
    parser.add_argument("--epochs", type=int, default=150, help="Use a small value (e.g. 20) for a quick timing-calibration run.")
    parser.add_argument("--top_n", type=int, default=5, help="How many examples to print per direction (RL-wins, GBT-wins).")
    args = parser.parse_args()

    ds_path = DATASETS[args.dataset]
    env_train, env_val, env_test = create_train_val_test_split(ds_path, split_seed=SPLIT_SEED)

    print(f"Dataset: {args.dataset} | test queries: {len(env_test)}")
    print("Training tuned GBT...")
    gbt_actions = train_gbt_with_predictions(env_train, env_test)

    print(f"Training RL-Route (seed={args.seed}, epochs={args.epochs})...")
    rl_actions = train_rl_with_predictions(env_train, env_val, env_test, args.seed, args.epochs)

    rows = []
    for i, qid in enumerate(env_test.query_pool):
        question = env_test.master_table[qid].get("question", "<no question text>")
        all_rewards = env_test.get_all_rewards(qid, target_metric=REWARD_TYPE)
        oracle_action = env_test.get_oracle_action(qid, target_metric=REWARD_TYPE)

        gbt_a, rl_a = int(gbt_actions[i]), int(rl_actions[i])
        rows.append({
            "qid": qid,
            "question": question,
            "scores": {RETRIEVER_NAMES[a]: round(all_rewards[a], 4) for a in range(3)},
            "oracle": RETRIEVER_NAMES[oracle_action],
            "gbt_action": RETRIEVER_NAMES[gbt_a],
            "gbt_reward": round(all_rewards[gbt_a], 4),
            "rl_action": RETRIEVER_NAMES[rl_a],
            "rl_reward": round(all_rewards[rl_a], 4),
            "margin": round(all_rewards[rl_a] - all_rewards[gbt_a], 4),
        })

    rl_wins = sorted([r for r in rows if r["margin"] > 0], key=lambda r: -r["margin"])
    gbt_wins = sorted([r for r in rows if r["margin"] < 0], key=lambda r: r["margin"])
    ties = [r for r in rows if r["margin"] == 0]

    print(f"\n{'='*70}\nSUMMARY: {args.dataset.upper()}\n{'='*70}")
    print(f"RL-Route wins:  {len(rl_wins)} / {len(rows)} queries")
    print(f"GBT wins:       {len(gbt_wins)} / {len(rows)} queries")
    print(f"Ties:           {len(ties)} / {len(rows)} queries")

    def print_examples(title, examples):
        print(f"\n--- {title} (top {min(args.top_n, len(examples))} by margin) ---")
        for r in examples[:args.top_n]:
            print(f"\nQ: {r['question']}")
            print(f"   Scores:  BM25={r['scores']['bm25']:.4f}  Dense={r['scores']['dense']:.4f}  Hybrid={r['scores']['hybrid']:.4f}  (Oracle: {r['oracle']})")
            print(f"   GBT chose {r['gbt_action']:>6s} -> reward {r['gbt_reward']:.4f}")
            print(f"   RL  chose {r['rl_action']:>6s} -> reward {r['rl_reward']:.4f}  (margin {r['margin']:+.4f})")

    print_examples("RL-Route WINS", rl_wins)
    print_examples("GBT WINS", gbt_wins)

    out_path = os.path.join(os.path.dirname(__file__), "results", f"failure_mode_analysis_{args.dataset}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset, "seed": args.seed, "epochs": args.epochs,
            "summary": {"rl_wins": len(rl_wins), "gbt_wins": len(gbt_wins), "ties": len(ties), "n_test": len(rows)},
            "rl_wins_examples": rl_wins[:20], "gbt_wins_examples": gbt_wins[:20],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
