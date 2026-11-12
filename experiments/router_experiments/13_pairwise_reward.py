"""
13_pairwise_reward.py — Ablation: Pairwise vs Absolute reward shaping.

Trains REINFORCE with both reward types and compares:
- Oracle accuracy (expected: pairwise >> absolute)
- MRR (expected: pairwise >= absolute)
- Convergence speed

Usage:
    cd <repo>
    python experiments/router_experiments/13_pairwise_reward.py
"""
import os, sys, json, random, numpy as np, torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


def train_with_reward_type(data_path, reward_shaping, seed):
    """Train REINFORCE with a specific reward shaping method."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env_train, env_test = create_train_test_split(data_path, train_ratio=0.8, seed=seed)
    
    state_dim = env_train.state_dim
    agent = PolicyAgent(input_dim=state_dim, hidden_dim=128, output_dim=3, learningrate=3e-3)
    
    trainer = ReinforceTrainer(
        env_train=env_train, env_test=env_test, agent=agent,
        reward_type="mrr", entropy_coeff=0.15,
        reward_shaping=reward_shaping,  # "absolute" or "pairwise"
        seed=seed,
    )
    
    history = trainer.train(num_epochs=300, log_interval=50)
    
    # Get best-fixed baseline for comparison
    best_fixed = max(
        sum(env_test.step(q, a, "mrr") for q in env_test.query_pool) / len(env_test)
        for a in range(3)
    )
    
    return {
        "seed": seed,
        "reward_shaping": reward_shaping,
        "final_mrr": round(history["test_mean_reward"][-1], 4),
        "best_mrr": round(max(history["test_mean_reward"]), 4),
        "final_oracle_acc": round(history["oracle_accuracy"][-1], 4),
        "best_oracle_acc": round(max(history["oracle_accuracy"]), 4),
        "best_fixed_mrr": round(best_fixed, 4),
        "improvement": round(max(history["test_mean_reward"]) - best_fixed, 4),
    }


def main():
    results = {"absolute": [], "pairwise": []}
    
    for shaping in ["absolute", "pairwise"]:
        print(f"\n{'='*60}")
        print(f"REWARD SHAPING: {shaping.upper()}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            r = train_with_reward_type(NQ_PATH, shaping, seed)
            results[shaping].append(r)
            print(f"  Seed {seed}: MRR={r['best_mrr']:.4f} OracleAcc={r['best_oracle_acc']:.1%} Delta={r['improvement']:+.4f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for shaping in ["absolute", "pairwise"]:
        mrrs = [r["best_mrr"] for r in results[shaping]]
        accs = [r["best_oracle_acc"] for r in results[shaping]]
        imps = [r["improvement"] for r in results[shaping]]
        print(f"\n  {shaping.upper()}:")
        print(f"    MRR:        {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
        print(f"    Oracle Acc: {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"    Delta vs Fixed: {np.mean(imps):+.4f} ± {np.std(imps):.4f}")
    
    out = os.path.join(RESULTS_DIR, "pairwise_reward_ablation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
