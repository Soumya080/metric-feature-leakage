"""
06_feature_ablation.py — Train REINFORCE with feature subsets to find what matters.

Feature groups:
  - Lexical (6):  query_length, log_query_length, avg_word_length, stopword_ratio, unique_word_ratio, has_quotes
  - Entity (7):   num_entities, entity_present_* (6 types)
  - Structural (9): q_type_* (7 types), contains_numbers, contains_proper_nouns
  - Retrieval (4): avg_bm25, avg_dense, score_gap, bm25_score_variance

Usage:
    cd <repo>
    python experiments/router_experiments/06_feature_ablation.py
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
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature group index ranges (0-indexed into the 26-dim feature vector)
FEATURE_GROUPS = {
    "lexical":    [0, 1, 2, 3, 4, 5],             # 6 features
    "entity":     [6, 7, 8, 9, 10, 11, 12],        # 7 features
    "structural": [13, 14, 15, 16, 17, 18, 19, 20, 21],  # 9 features
    "retrieval":  [22, 23, 24, 25],                 # 4 features
}

ABLATION_CONFIGS = {
    "all_features":           list(range(26)),
    "lexical_only":           FEATURE_GROUPS["lexical"],
    "entity_only":            FEATURE_GROUPS["entity"],
    "structural_only":        FEATURE_GROUPS["structural"],
    "retrieval_only":         FEATURE_GROUPS["retrieval"],
    "no_lexical":             FEATURE_GROUPS["entity"] + FEATURE_GROUPS["structural"] + FEATURE_GROUPS["retrieval"],
    "no_entity":              FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["structural"] + FEATURE_GROUPS["retrieval"],
    "no_structural":          FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["entity"] + FEATURE_GROUPS["retrieval"],
    "no_retrieval":           FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["entity"] + FEATURE_GROUPS["structural"],
    "lexical_and_retrieval":  FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["retrieval"],
}

SEED = 42
NUM_EPOCHS = 200  # Shorter for ablation (speed)
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


class MaskedRouterEnvironment(RouterEnvironment):
    """RouterEnvironment that returns only a subset of features."""
    
    def __init__(self, data_path, query_ids_subset=None, feature_indices=None):
        super().__init__(data_path, query_ids_subset)
        self.feature_indices = feature_indices or list(range(26))
    
    def get_state(self, query_id):
        full_features = super().get_state(query_id)
        return [full_features[i] for i in self.feature_indices]
    
    @property
    def state_dim(self):
        return len(self.feature_indices)


def run_ablation(config_name, feature_indices, data_path, seed):
    """Train REINFORCE with a specific feature subset."""
    print(f"\n{'='*40} {config_name} (dim={len(feature_indices)}) {'='*40}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and split data manually to inject MaskedRouterEnvironment
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    keys = list(raw_data.keys())
    random.seed(seed)
    random.shuffle(keys)
    split = int(len(keys) * 0.8)
    train_keys = keys[:split]
    test_keys = keys[split:]
    
    env_train = MaskedRouterEnvironment(data_path, train_keys, feature_indices)
    env_test = MaskedRouterEnvironment(data_path, test_keys, feature_indices)
    
    agent = PolicyAgent(
        input_dim=len(feature_indices),
        hidden_dim=HIDDEN_DIM,
        output_dim=3,
        learningrate=LEARNING_RATE
    )
    
    trainer = ReinforceTrainer(
        env_train=env_train, env_test=env_test, agent=agent,
        reward_type="mrr", entropy_coeff=ENTROPY_COEFF, seed=seed
    )
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=50)
    
    return {
        "config": config_name,
        "num_features": len(feature_indices),
        "feature_indices": feature_indices,
        "final_test_mrr": round(history["test_mean_reward"][-1], 4),
        "best_test_mrr": round(max(history["test_mean_reward"]), 4),
        "final_oracle_acc": round(history["oracle_accuracy"][-1], 4),
    }


def main():
    results = {}
    
    for config_name, feature_indices in ABLATION_CONFIGS.items():
        result = run_ablation(config_name, feature_indices, DATA_PATH, SEED)
        results[config_name] = result
        print(f"  -> {config_name}: MRR={result['final_test_mrr']:.4f} (best={result['best_test_mrr']:.4f})")
    
    # Summary
    print(f"\n{'='*60}")
    print("FEATURE ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25s} {'#Feat':>5s} {'Final MRR':>10s} {'Best MRR':>10s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["best_test_mrr"]):
        print(f"{name:<25s} {r['num_features']:>5d} {r['final_test_mrr']:>10.4f} {r['best_test_mrr']:>10.4f}")
    
    output_path = os.path.join(RESULTS_DIR, "feature_ablation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
