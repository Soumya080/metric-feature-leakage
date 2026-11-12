#!/usr/bin/env python3
"""
online_adaptation.py - The "Money Shot" Experiment for RL-Route.
Proves that the RL router can adapt to distribution shifts online (non-stationary streams),
whereas a frozen Gradient Boosting classifier fails when the dataset changes.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.routing_env import create_train_val_test_split, deterministic_tiebreak
from src.router.policy_network import PolicyAgent

DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}

SPLIT_SEED = 42
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
ENTROPY_COEFF = 0.1

def align_state(raw_state, target_dim):
    """Pad with zeros or truncate so every state has the same width.

    Different datasets can produce different feature counts (e.g. 46 vs 57)
    depending on corpus size.  The scaler, RL agents, and normalization stats
    are all fitted on the first dataset's dimension, so we must align.
    """
    s = np.array(raw_state, dtype=float)
    if len(s) == target_dim:
        return s
    if len(s) < target_dim:
        return np.concatenate([s, np.zeros(target_dim - len(s))])
    return s[:target_dim]


def prepare_xy(env, metric='mrr'):
    X, y_cls = [], []
    for qid in env.query_pool:
        X.append(env.get_state(qid))
        scores = [env.get_all_rewards(qid, metric)[a] for a in range(3)]
        max_score = max(scores)
        tied = [a for a in range(3) if scores[a] == max_score]
        y_cls.append(tied[deterministic_tiebreak(qid, len(tied))])
    return np.array(X), np.array(y_cls)

def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    
    print("Loading datasets and creating splits...")
    envs = {}
    for name, path in DATASETS.items():
        train_env, val_env, test_env = create_train_val_test_split(path, split_seed=SPLIT_SEED)
        envs[name] = {"train": train_env, "val": val_env, "test": test_env}
        
    nq_train = envs["nq"]["train"]
    state_dim = nq_train.state_dim
    
    # 1. Pre-train GBT on NQ
    print("Pre-training Gradient Boosting on NQ...")
    X_train, y_train = prepare_xy(nq_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    gbt = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gbt.fit(X_train_scaled, y_train)
    
    # 2. Setup RL Agents (Online and Frozen)
    print("Setting up RL Agents...")
    agent_online = PolicyAgent(input_dim=state_dim, hidden_dim=HIDDEN_DIM, output_dim=3, learningrate=LEARNING_RATE)
    # We could pre-train agent_online on NQ, but demonstrating it learning from scratch 
    # or adapting online is the goal. For simplicity, we'll let it learn entirely online.
    
    agent_frozen = PolicyAgent(input_dim=state_dim, hidden_dim=HIDDEN_DIM, output_dim=3, learningrate=LEARNING_RATE)
    # Copy weights (both untrained, or both pre-trained. We'll leave them untrained to show pure online learning)
    agent_frozen.policynetwork.load_state_dict(agent_online.policynetwork.state_dict())
    
    # Normalization stats for RL (using NQ train as the starting baseline)
    raw_states = [nq_train.get_state(q) for q in nq_train.query_pool]
    norm_mean = np.mean(raw_states, axis=0)
    norm_std = np.std(raw_states, axis=0) + 1e-8
    
    # 3. Create Continuous Stream (Test sets: NQ -> TriviaQA -> HotpotQA)
    print("Creating continuous data stream (NQ -> TriviaQA -> HotpotQA)...")
    stream = []
    boundaries = {}
    current_idx = 0
    for ds_name in ["nq", "triviaqa", "hotpotqa"]:
        test_env = envs[ds_name]["test"]
        boundaries[ds_name] = {"start": current_idx, "end": current_idx + len(test_env.query_pool)}
        for qid in test_env.query_pool:
            stream.append((qid, ds_name, test_env))
            current_idx += 1
            
    print(f"Total stream length: {len(stream)} queries")
    
    # Metrics tracking
    results = {
        "gbt": [],
        "rl_frozen": [],
        "rl_online": []
    }
    
    # RL Online running baseline
    running_baseline = 0.0
    baseline_decay = 0.99
    
    print("Processing stream...")
    for i, (qid, ds_name, env) in enumerate(stream):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(stream)} queries...")
            
        raw_state = align_state(env.get_state(qid), state_dim)
        
        # --- GBT (Frozen) ---
        state_scaled = scaler.transform([raw_state])
        gbt_action = gbt.predict(state_scaled)[0]
        gbt_reward = env.step(qid, gbt_action, 'mrr')
        results["gbt"].append(float(gbt_reward))
        
        # Normalize for RL
        state_norm = (raw_state - norm_mean) / norm_std
        
        # --- RL (Frozen) ---
        frozen_action, _ = agent_frozen.select_action(state_norm, evaluate=True)
        frozen_reward = env.step(qid, frozen_action, 'mrr')
        results["rl_frozen"].append(float(frozen_reward))
        
        # --- RL (Online Adaptation) ---
        # IMPORTANT: the REPORTED metric must come from the greedy (evaluate=True) action —
        # the policy's actual current best guess — not from the stochastic exploration
        # sample. Conflating the two means exploration noise gets counted against the
        # router's reported performance, artificially deflating it (this previously made
        # "online RL" look worse than a frozen, randomly-initialized policy in some
        # segments, which cannot be a real adaptation effect). The stochastic sample is
        # still used for the actual policy-gradient update below, as it must be.
        greedy_action, _ = agent_online.select_action(state_norm, evaluate=True)
        greedy_reward = env.step(qid, greedy_action, 'mrr')
        results["rl_online"].append(float(greedy_reward))

        online_action, log_prob = agent_online.select_action(state_norm, evaluate=False)  # stochastic, for exploration + the update below
        online_reward = env.step(qid, online_action, 'mrr')

        # Perform 1-step online policy gradient update
        running_baseline = baseline_decay * running_baseline + (1 - baseline_decay) * online_reward
        advantage = online_reward - running_baseline

        agent_online.optimizer.zero_grad()
        state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
        action_probs = agent_online.policynetwork(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy()

        loss = -log_prob.squeeze() * advantage - ENTROPY_COEFF * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_online.policynetwork.parameters(), max_norm=1.0)
        agent_online.optimizer.step()

    # Calculate rolling averages
    window = 100
    def rolling_avg(data):
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
    final_output = {
        "boundaries": boundaries,
        "rolling_averages": {
            "gbt": rolling_avg(results["gbt"]),
            "rl_frozen": rolling_avg(results["rl_frozen"]),
            "rl_online": rolling_avg(results["rl_online"])
        },
        "segment_averages": {}
    }
    
    for ds_name, bounds in boundaries.items():
        start, end = bounds["start"], bounds["end"]
        final_output["segment_averages"][ds_name] = {
            "gbt": float(np.mean(results["gbt"][start:end])),
            "rl_frozen": float(np.mean(results["rl_frozen"][start:end])),
            "rl_online": float(np.mean(results["rl_online"][start:end]))
        }
        
    out_path = os.path.join(PROJECT_ROOT, "results", "online_adaptation_results.json")
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\nSegment Averages:")
    print(json.dumps(final_output["segment_averages"], indent=2))
    print(f"\nSaved online adaptation results to {out_path}")

if __name__ == "__main__":
    main()
