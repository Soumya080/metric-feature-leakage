import os
import sys
import json
import random
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_val_test_split
from src.router.train_reinforce import ReinforceTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configuration ----
SPLIT_SEED = 42
TRAINING_SEEDS = [42, 123, 456, 789, 1024]
NUM_EPOCHS = 300
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
ENTROPY_COEFF = 0.15
REWARD_TYPE = "mrr"

DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}

def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Test if system A is significantly better than system B."""
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    delta_observed = np.mean(scores_a) - np.mean(scores_b)
    
    count_better = 0
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        delta_boot = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if delta_boot <= 0:
            count_better += 1
            
    p_value = count_better / n_bootstrap
    return delta_observed, p_value

def prepare_xy(env):
    X, y = [], []
    for qid in env.query_pool:
        entry = env.master_table[qid]
        X.append(env.get_state(qid))
        best_action = max(
            range(3),
            key=lambda a: entry["retrievers"][["bm25", "dense", "hybrid"][a]].get(REWARD_TYPE, 0.0)
        )
        y.append(best_action)
    return np.array(X), np.array(y)

def evaluate_gbt(env_train, env_test):
    X_train, y_train = prepare_xy(env_train)
    X_test, _ = prepare_xy(env_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = GradientBoostingClassifier(n_estimators=100, random_state=SPLIT_SEED, max_depth=5)
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    
    retriever_names = ["bm25", "dense", "hybrid"]
    scores = []
    for pred, qid in zip(predictions, env_test.query_pool):
        chosen_ret = retriever_names[pred]
        scores.append(env_test.master_table[qid]["retrievers"][chosen_ret].get(REWARD_TYPE, 0.0))
        
    return scores

def run_dataset(dataset_name, data_path):
    print(f"\n{'='*60}\nRunning dataset: {dataset_name}\n{'='*60}")
    
    # 1. Create one fixed train/val/test split. Using a held-out val set (rather than
    # train/test only) so that best-checkpoint selection below never touches the test
    # set — matching the methodology already used in canonical_evaluation.py.
    env_train, env_val, env_test = create_train_val_test_split(data_path, split_seed=SPLIT_SEED)
    test_keys = env_test.query_pool
    
    # 2. Compute fixed baselines on test set
    baselines = {}
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        baselines[ret_name] = [env_test.step(qid, action, target_metric=REWARD_TYPE) for qid in test_keys]
        
    bm25_mrr = np.mean(baselines["bm25"])
    dense_mrr = np.mean(baselines["dense"])
    hybrid_mrr = np.mean(baselines["hybrid"])
    
    best_fixed_scores = baselines["dense"]
    if bm25_mrr > np.mean(best_fixed_scores): best_fixed_scores = baselines["bm25"]
    if hybrid_mrr > np.mean(best_fixed_scores): best_fixed_scores = baselines["hybrid"]
    best_fixed_mrr = np.mean(best_fixed_scores)
    
    oracle_scores = [env_test.step(qid, env_test.get_oracle_action(qid, target_metric=REWARD_TYPE), target_metric=REWARD_TYPE) for qid in test_keys]
    oracle_mrr = np.mean(oracle_scores)
    
    # 3. GBT Baseline on same split
    gbt_scores = evaluate_gbt(env_train, env_test)
    gbt_mrr = np.mean(gbt_scores)
    
    print(f"Fixed Baselines -> Best Fixed: {best_fixed_mrr:.4f} | Oracle: {oracle_mrr:.4f} | GBT: {gbt_mrr:.4f}")
    
    # 4. Train across seeds
    all_rl_scores = []
    seed_results = []
    
    for seed in TRAINING_SEEDS:
        print(f"\nTraining seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        agent = PolicyAgent(
            input_dim=env_train.state_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=RouterEnvironment.NUM_ACTIONS,
            learningrate=LEARNING_RATE
        )
        
        trainer = ReinforceTrainer(
            env_train=env_train, env_test=env_test, agent=agent,
            reward_type=REWARD_TYPE, entropy_coeff=ENTROPY_COEFF, seed=seed,
            env_val=env_val
        )

        # Don't print output during training for brevity
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        trainer.train(num_epochs=NUM_EPOCHS, log_interval=NUM_EPOCHS)
        sys.stdout.close()
        sys.stdout = original_stdout
        
        # Evaluate
        rl_scores = []
        for qid in test_keys:
            raw_state = env_test.get_state(qid)
            state = trainer._normalize_state(raw_state)
            action, _ = agent.select_action(state, evaluate=True)
            rl_scores.append(env_test.step(qid, action, target_metric=REWARD_TYPE))
            
        test_mrr = np.mean(rl_scores)
        all_rl_scores.append(rl_scores)
        
        seed_results.append({
            "seed": seed,
            "test_mrr": float(test_mrr)
        })
        print(f"Seed {seed} MRR: {test_mrr:.4f}")
        
    # Aggregate RL results
    rl_means = [np.mean(scores) for scores in all_rl_scores]
    rl_mean = float(np.mean(rl_means))
    rl_std = float(np.std(rl_means))
    
    print(f"\nAggregated RL-Route MRR: {rl_mean:.4f} ± {rl_std:.4f}")
    
    # 5. Significance Tests
    median_idx = np.argsort(rl_means)[len(rl_means) // 2]
    median_rl_scores = all_rl_scores[median_idx]
    
    delta_fixed, p_val_fixed = paired_bootstrap_test(median_rl_scores, best_fixed_scores)
    delta_gbt, p_val_gbt = paired_bootstrap_test(median_rl_scores, gbt_scores)
    
    return {
        "dataset": dataset_name,
        "split_seed": SPLIT_SEED,
        "training_seeds": TRAINING_SEEDS,
        "fixed_baselines": {
            "bm25": float(bm25_mrr),
            "dense": float(dense_mrr),
            "hybrid": float(hybrid_mrr),
            "best_fixed": float(best_fixed_mrr),
            "oracle": float(oracle_mrr),
            "gbt_classifier": float(gbt_mrr)
        },
        "rl_route": {
            "mean": rl_mean,
            "std": rl_std,
            "per_seed": seed_results
        },
        "significance": {
            "rl_vs_best_fixed": {"delta": float(delta_fixed), "p_value": float(p_val_fixed)},
            "rl_vs_gbt": {"delta": float(delta_gbt), "p_value": float(p_val_gbt)}
        }
    }

def main():
    final_results = {}
    for ds_name, ds_path in DATASETS.items():
        if os.path.exists(ds_path):
            final_results[ds_name] = run_dataset(ds_name, ds_path)
            
    output_path = os.path.join(RESULTS_DIR, "fixed_split_multi_seed.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved all results to {output_path}")

if __name__ == "__main__":
    main()
