import argparse
import json
import os
import sys
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature group index ranges (0-indexed into the 26-dim feature vector)
FEATURE_GROUPS = {
    "lexical":         [0, 1, 2, 3, 4, 5],             # 6 features
    "entity":          [6, 7, 8, 9, 10, 11, 12],        # 7 features
    "structural":      [13, 14, 15, 16, 17, 18, 19, 20, 21],  # 9 features
    "retrieval":       [22, 23, 24, 25],                 # 4 features
    "cross_retriever": [26, 27, 28],                     # 3 features
    # Filled in at runtime from the actual feature dimension: PCA components of the query
    # embedding occupy every index from 29 onward, and how many there are depends on
    # prepare_env_data.py's --embed-dims.
    "embedding":       [],
}


def _resolve_embedding_group(data_path):
    """Point FEATURE_GROUPS['embedding'] at whatever embedding dims this env data has."""
    import json as _json
    with open(data_path, "r", encoding="utf-8") as f:
        env = _json.load(f)
    queries = env.get("queries", env)
    dim = len(next(iter(queries.values()))["features"])
    FEATURE_GROUPS["embedding"] = list(range(29, dim))
    ABLATION_CONFIGS["all_features"] = list(range(dim))
    if FEATURE_GROUPS["embedding"]:
        ABLATION_CONFIGS["embedding_only"] = list(FEATURE_GROUPS["embedding"])
        ABLATION_CONFIGS["retrieval_cross_embedding"] = (
            FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["cross_retriever"]
            + FEATURE_GROUPS["embedding"])
    return dim


ABLATION_CONFIGS = {
    "all_features":           list(range(29)),
    "retrieval_only":         FEATURE_GROUPS["retrieval"],
    "no_retrieval":           FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["entity"] + FEATURE_GROUPS["structural"],
    "lexical_and_retrieval":  FEATURE_GROUPS["lexical"] + FEATURE_GROUPS["retrieval"],
    # The two configs that test the paper's "compact label-free state" claim: whether
    # per-retriever confidence (retrieval) plus inter-retriever disagreement
    # (cross_retriever) is sufficient on its own, and what disagreement contributes alone.
    "retrieval_plus_cross":   FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["cross_retriever"],
    "cross_only":             FEATURE_GROUPS["cross_retriever"],
}

SPLIT_SEED = 42
TRAINING_SEEDS = [42, 123, 456, 789, 1024, 2025, 31337, 8675309, 90210, 1729,
                  271828, 161803, 512, 65537, 20260829]  # 15 seeds, matches canonical_evaluation.py
NUM_EPOCHS = 200
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")

class MaskedRouterEnvironment(RouterEnvironment):
    """RouterEnvironment that returns only a subset of features."""
    def __init__(self, data_path, query_ids_subset=None, feature_indices=None):
        super().__init__(data_path, query_ids_subset)
        self.feature_indices = feature_indices or list(range(29))
    
    def get_state(self, query_id):
        full_features = super().get_state(query_id)
        return [full_features[i] for i in self.feature_indices]
    
    @property
    def state_dim(self):
        return len(self.feature_indices)

def get_fixed_split_keys(data_path, split_seed, train_ratio=0.70, val_ratio=0.15):
    """Fixed train/val/test split (same methodology as create_train_val_test_split in
    routing_env.py) so that best-checkpoint selection below can use a held-out val set
    instead of silently falling back to test-set performance."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        keys = sorted(raw_data.keys())  # sorted() for determinism, matches routing_env.py
    elif isinstance(raw_data, list):
        keys = sorted(str(entry.get("query_id", i)) for i, entry in enumerate(raw_data))

    random.seed(split_seed)
    random.shuffle(keys)
    n = len(keys)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return keys[:train_end], keys[train_end:val_end], keys[val_end:]

def paired_bootstrap_two_sided(scores_a, scores_b, n_resamples=10000, seed=42):
    """Two-sided paired bootstrap on the mean difference (A - B).

    Two-sided (not one-sided) on purpose: the ablation question is "do these two feature
    sets differ?", not "is A better than B" — using a one-sided test here would bias toward
    the conclusion the paper wants. Seeded so the p-value is reproducible.
    """
    rng = np.random.RandomState(seed)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    diffs = a - b
    observed = diffs.mean()
    shifted = diffs - observed  # impose the null (no difference)
    idx = rng.randint(0, len(diffs), size=(n_resamples, len(diffs)))
    null_means = shifted[idx].mean(axis=1)
    p_value = float((np.abs(null_means) >= abs(observed)).mean())
    return float(observed), p_value


def run_ablation(config_name, feature_indices, data_path, train_keys, val_keys, test_keys, training_seeds):
    print(f"\n{'='*40}\nConfig: {config_name} (dim={len(feature_indices)})\n{'='*40}")

    env_train = MaskedRouterEnvironment(data_path, train_keys, feature_indices)
    env_val = MaskedRouterEnvironment(data_path, val_keys, feature_indices)
    env_test = MaskedRouterEnvironment(data_path, test_keys, feature_indices)
    
    seed_results = []
    per_query_by_seed = []
    for seed in training_seeds:
        print(f"Training seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        agent = PolicyAgent(
            input_dim=len(feature_indices),
            hidden_dim=HIDDEN_DIM,
            output_dim=3,
            learningrate=LEARNING_RATE
        )
        
        trainer = ReinforceTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type="mrr", entropy_coeff=ENTROPY_COEFF, seed=seed,
            env_val=env_val
        )

        # Suppress output during training
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=NUM_EPOCHS)
        sys.stdout.close()
        sys.stdout = original_stdout
        
        final_test_mrr = round(history["test_mean_reward"][-1], 4)
        seed_results.append({
            "seed": seed,
            "test_mrr": final_test_mrr
        })

        # Per-query scores for this seed's final policy, so configs can be compared with a
        # PAIRED test rather than by eyeballing overlapping mean±std. The paper's "Lean
        # Router" claim rests on a +0.7% gap between two configs whose seed-level std is of
        # the same order — that needs a significance test to be a claim at all.
        seed_scores = []
        for qid in env_test.query_pool:
            state = trainer._normalize_state(env_test.get_state(qid))
            action, _ = agent.select_action(state, evaluate=True)
            seed_scores.append(float(env_test.step(qid, action, target_metric="mrr")))
        per_query_by_seed.append(seed_scores)

        print(f"  Seed {seed} MRR: {final_test_mrr:.4f}")

    test_mrrs = [r["test_mrr"] for r in seed_results]
    mean_mrr = float(np.mean(test_mrrs))
    std_mrr = float(np.std(test_mrrs))

    # Average each query's score across seeds -> one score vector per config, which is what
    # the paired bootstrap between configs consumes.
    mean_per_query = np.mean(np.array(per_query_by_seed), axis=0).tolist() if per_query_by_seed else []

    print(f"Aggregated {config_name} MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")

    return {
        "mean": mean_mrr,
        "std": std_mrr,
        "per_seed": seed_results,
        "mean_per_query": mean_per_query,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Feature ablation (Lean Router), multi-seed.")
    parser.add_argument("--seeds", type=str, default=None,
                        help=f"Comma-separated seed list, or a single int N to use the first N of the default {len(TRAINING_SEEDS)}-seed list.")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Training epochs per seed (default {NUM_EPOCHS}). Use a small value for a quick timing-calibration run.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/multi_seed_ablation.json).")
    return parser.parse_args()

def main():
    global NUM_EPOCHS
    args = parse_args()

    training_seeds = TRAINING_SEEDS
    if args.seeds:
        if "," in args.seeds:
            training_seeds = [int(s) for s in args.seeds.split(",")]
        else:
            training_seeds = TRAINING_SEEDS[:int(args.seeds)]
    if args.epochs:
        NUM_EPOCHS = args.epochs

    output_path = args.output or os.path.join(RESULTS_DIR, "multi_seed_ablation.json")

    print(f"Config: {len(training_seeds)} seed(s), {NUM_EPOCHS} epoch(s)")

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    # Embedding features occupy indices 29+ and their count depends on --embed-dims, so
    # the ablation configs are resolved against the actual data rather than hardcoded.
    dim = _resolve_embedding_group(DATA_PATH)
    print(f"Feature dim: {dim} (embedding group: {len(FEATURE_GROUPS['embedding'])} dims)")

    train_keys, val_keys, test_keys = get_fixed_split_keys(DATA_PATH, SPLIT_SEED)

    results = {}
    for config_name, feature_indices in ABLATION_CONFIGS.items():
        results[config_name] = run_ablation(config_name, feature_indices, DATA_PATH, train_keys, val_keys, test_keys, training_seeds)

    # --- Is the Lean Router actually as good as the full model? ---
    # The paper claims retrieval-only (d=4) MATCHES or BEATS all-features (now d=29). Treat
    # that claim as suspect until this run confirms it: features 22-25 USED to be computed
    # from MRR (the reward itself), so pre-fix "4 features suffice" partly meant "the leaked
    # columns suffice." Post-fix they are raw score statistics, and retrieval_plus_cross
    # (d=7) is the config that now carries the compact-state claim. With per-config mean±std
    # that overlap heavily, none of this is testable by inspection.
    # A paired bootstrap on the per-query scores answers it directly. Note the expected
    # (and perfectly publishable) outcome is "no significant difference" — that is what
    # "matches the full model" should mean, and stating it as a tested null is far stronger
    # than implying the +0.7% is a real improvement.
    baseline_cfg = "all_features"
    if baseline_cfg in results and results[baseline_cfg].get("mean_per_query"):
        base_scores = results[baseline_cfg]["mean_per_query"]
        comparisons = {}
        print(f"\n{'='*60}\nSIGNIFICANCE vs. {baseline_cfg} (paired bootstrap, 10k resamples)\n{'='*60}")
        for cfg_name, cfg in results.items():
            if cfg_name == baseline_cfg or not cfg.get("mean_per_query"):
                continue
            delta, p_two_sided = paired_bootstrap_two_sided(cfg["mean_per_query"], base_scores)
            comparisons[cfg_name] = {
                "delta_vs_all_features": float(delta),
                "p_value_two_sided": float(p_two_sided),
                "significant_at_0.05": bool(p_two_sided < 0.05),
            }
            verdict = "SIGNIFICANT" if p_two_sided < 0.05 else "not significant (i.e. matches)"
            print(f"  {cfg_name:<24s} delta={delta:+.4f}  p={p_two_sided:.4f}  -> {verdict}")
        results["_significance_vs_all_features"] = comparisons

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results to {output_path}")

if __name__ == "__main__":
    main()
