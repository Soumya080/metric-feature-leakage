"""
10_lean_router.py — Train the lean 4-feature router (retrieval scores only).

This is your BEST model based on ablation evidence.
4 features: avg_bm25, avg_dense, score_gap, bm25_score_variance

Usage:
    cd <repo>
    python experiments/router_experiments/10_lean_router.py
"""
import os, sys, json, random, numpy as np, torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Retrieval feature indices (positions 22-25 in the 26-dim vector)
RETRIEVAL_INDICES = [22, 23, 24, 25]
SEEDS = [42, 123, 456, 789, 1024]
NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
TQA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json")


class LeanRouterEnvironment(RouterEnvironment):
    """Environment that only exposes 4 retrieval score features."""
    def __init__(self, data_path, query_ids_subset=None):
        super().__init__(data_path, query_ids_subset)
    
    def get_state(self, query_id):
        full = super().get_state(query_id)
        return [full[i] for i in RETRIEVAL_INDICES]
    
    @property
    def state_dim(self):
        return len(RETRIEVAL_INDICES)


def train_lean_router(data_path, dataset_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    with open(data_path) as f:
        data = json.load(f)
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)
    split = int(len(keys) * 0.8)
    
    env_train = LeanRouterEnvironment(data_path, keys[:split])
    env_test = LeanRouterEnvironment(data_path, keys[split:])
    
    agent = PolicyAgent(input_dim=4, hidden_dim=64, output_dim=3, learningrate=3e-3)
    
    trainer = ReinforceTrainer(
        env_train=env_train, env_test=env_test, agent=agent,
        reward_type="mrr", entropy_coeff=0.15, seed=seed
    )
    
    history = trainer.train(num_epochs=300, log_interval=50)
    
    # Compute baselines
    baselines = {}
    for a in range(3):
        r = RouterEnvironment.ACTION_TO_RETRIEVER[a]
        baselines[r] = sum(env_test.step(q, a, "mrr") for q in env_test.query_pool) / len(env_test)
    
    return {
        "seed": seed, "dataset": dataset_name,
        "final_mrr": round(history["test_mean_reward"][-1], 4),
        "best_mrr": round(max(history["test_mean_reward"]), 4),
        "best_fixed": round(max(baselines.values()), 4),
        "improvement": round(max(history["test_mean_reward"]) - max(baselines.values()), 4),
        "oracle_acc": round(history["oracle_accuracy"][-1], 4),
    }


def main():
    all_results = {"nq": [], "triviaqa": []}
    
    for ds_name, path in [("nq", NQ_PATH), ("triviaqa", TQA_PATH)]:
        print(f"\n{'='*60}\n{ds_name.upper()} — Lean 4-Feature Router\n{'='*60}")
        for seed in SEEDS:
            r = train_lean_router(path, ds_name, seed)
            all_results[ds_name].append(r)
            print(f"  Seed {seed}: MRR={r['best_mrr']:.4f} (Delta={r['improvement']:+.4f})")
        
        mrrs = [r["best_mrr"] for r in all_results[ds_name]]
        imps = [r["improvement"] for r in all_results[ds_name]]
        print(f"\n  Mean MRR: {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
        print(f"  Mean Delta:   {np.mean(imps):+.4f} ± {np.std(imps):.4f}")
    
    out = os.path.join(RESULTS_DIR, "lean_router_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
