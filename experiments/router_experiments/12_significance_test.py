"""
12_significance_test.py — Paired bootstrap test: REINFORCE vs Always-Dense.

A reviewer will ask "is the improvement statistically significant?"
This answers with a p-value.

Usage:
    cd <repo>
    python experiments/router_experiments/12_significance_test.py
"""
import json, os, sys, random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 42
NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Test if system A is significantly better than system B."""
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    delta_observed = np.mean(scores_a) - np.mean(scores_b)
    
    count_better = 0
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        delta_boot = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if delta_boot <= 0:
            count_better += 1
    
    p_value = count_better / n_bootstrap
    return delta_observed, p_value


def main():
    with open(NQ_PATH) as f:
        data = json.load(f)
    
    keys = list(data.keys())
    random.seed(SEED)
    random.shuffle(keys)
    test_keys = keys[int(len(keys)*0.8):]
    
    # Load trained model
    model_path = os.path.join(RESULTS_DIR, "best_reinforce_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "best_reinforce_model.pt")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dim = len(data[test_keys[0]]["features"])
    agent = PolicyAgent(input_dim=state_dim, hidden_dim=128, output_dim=3)
    agent.policynetwork.load_state_dict(checkpoint["model_state_dict"])
    
    # Normalization
    train_features = np.array([data[k]["features"] for k in keys[:int(len(keys)*0.8)]])
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    
    # Get per-query MRR for REINFORCE and Always-Dense
    rl_scores = []
    dense_scores = []
    bm25_scores = []
    
    for qid in test_keys:
        entry = data[qid]
        
        # RL choice
        features = ((np.array(entry["features"]) - feat_mean) / feat_std).tolist()
        action, _ = agent.select_action(features, evaluate=True)
        rl_ret = ["bm25", "dense", "hybrid"][action]
        rl_scores.append(entry["retrievers"][rl_ret]["mrr"])
        
        dense_scores.append(entry["retrievers"]["dense"]["mrr"])
        bm25_scores.append(entry["retrievers"]["bm25"]["mrr"])
    
    rl_scores = np.array(rl_scores)
    dense_scores = np.array(dense_scores)
    bm25_scores = np.array(bm25_scores)
    
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*60}")
    
    for name, baseline in [("Always Dense", dense_scores), ("Always BM25", bm25_scores)]:
        delta, p_val = paired_bootstrap_test(rl_scores, baseline)
        sig = "YES (p<0.05)" if p_val < 0.05 else "NO (p≥0.05)"
        print(f"\n  REINFORCE vs {name}:")
        print(f"    Mean diff:    {delta:+.4f}")
        print(f"    p-value:      {p_val:.4f}")
        print(f"    Significant:  {sig}")
    
    results = {
        "rl_mean_mrr": round(float(np.mean(rl_scores)), 4),
        "dense_mean_mrr": round(float(np.mean(dense_scores)), 4),
        "bm25_mean_mrr": round(float(np.mean(bm25_scores)), 4),
        "rl_vs_dense_delta": round(float(np.mean(rl_scores) - np.mean(dense_scores)), 4),
        "rl_vs_dense_pvalue": round(float(paired_bootstrap_test(rl_scores, dense_scores)[1]), 4),
        "rl_vs_bm25_delta": round(float(np.mean(rl_scores) - np.mean(bm25_scores)), 4),
        "rl_vs_bm25_pvalue": round(float(paired_bootstrap_test(rl_scores, bm25_scores)[1]), 4),
    }
    
    out = os.path.join(RESULTS_DIR, "significance_tests.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
