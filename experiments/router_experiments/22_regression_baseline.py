import os
import sys
import json
import random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.routing_env import create_train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPLIT_SEED = 42
REWARD_TYPE = "mrr"

DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}

def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
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

def prepare_xy_regression(env):
    X = []
    Y = [] # shape: (N, 3) for 3 retrievers
    for qid in env.query_pool:
        entry = env.master_table[qid]
        X.append(env.get_state(qid))
        
        y_row = [
            entry["retrievers"]["bm25"].get(REWARD_TYPE, 0.0),
            entry["retrievers"]["dense"].get(REWARD_TYPE, 0.0),
            entry["retrievers"]["hybrid"].get(REWARD_TYPE, 0.0)
        ]
        Y.append(y_row)
    return np.array(X), np.array(Y)

def evaluate_regression(models, X_test, env_test):
    retriever_names = ["bm25", "dense", "hybrid"]
    
    if isinstance(models, list):
        # One model per retriever
        predictions = []
        for j in range(3):
            predictions.append(models[j].predict(X_test))
        predictions = np.column_stack(predictions)
    else:
        # Single multi-output model
        predictions = models.predict(X_test)
        
    scores = []
    for i, qid in enumerate(env_test.query_pool):
        best_action = np.argmax(predictions[i])
        chosen_ret = retriever_names[best_action]
        scores.append(env_test.master_table[qid]["retrievers"][chosen_ret].get(REWARD_TYPE, 0.0))
        
    return scores

def run_dataset(dataset_name, data_path):
    print(f"\n{'='*60}\nRunning dataset: {dataset_name}\n{'='*60}")
    
    env_train, env_test = create_train_test_split(data_path, train_ratio=0.8, seed=SPLIT_SEED)
    
    X_train, Y_train = prepare_xy_regression(env_train)
    X_test, Y_test = prepare_xy_regression(env_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train 3 Ridge regressors
    ridge_models = []
    for j in range(3):
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, Y_train[:, j])
        ridge_models.append(model)
        
    ridge_scores = evaluate_regression(ridge_models, X_test_scaled, env_test)
    ridge_mrr = np.mean(ridge_scores)
    
    # Train single MLP for multi-output
    mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=SPLIT_SEED, max_iter=500)
    mlp_model.fit(X_train_scaled, Y_train)
    
    mlp_scores = evaluate_regression(mlp_model, X_test_scaled, env_test)
    mlp_mrr = np.mean(mlp_scores)
    
    print(f"Ridge Regression MRR: {ridge_mrr:.4f}")
    print(f"MLP Regression MRR: {mlp_mrr:.4f}")
    
    # Compare with best fixed
    baselines = {}
    test_keys = env_test.query_pool
    for action in range(3):
        ret_name = ["bm25", "dense", "hybrid"][action]
        baselines[ret_name] = [env_test.step(qid, action, target_metric=REWARD_TYPE) for qid in test_keys]
        
    bm25_mrr = np.mean(baselines["bm25"])
    dense_mrr = np.mean(baselines["dense"])
    hybrid_mrr = np.mean(baselines["hybrid"])
    
    best_fixed_scores = baselines["dense"]
    if bm25_mrr > np.mean(best_fixed_scores): best_fixed_scores = baselines["bm25"]
    if hybrid_mrr > np.mean(best_fixed_scores): best_fixed_scores = baselines["hybrid"]
    best_fixed_mrr = np.mean(best_fixed_scores)
    
    print(f"Best Fixed MRR: {best_fixed_mrr:.4f}")
    
    # Load multi-seed RL results to compare if available, else we just test vs best fixed
    # We will compute significance vs best fixed
    delta_ridge, p_val_ridge = paired_bootstrap_test(ridge_scores, best_fixed_scores)
    delta_mlp, p_val_mlp = paired_bootstrap_test(mlp_scores, best_fixed_scores)
    
    return {
        "dataset": dataset_name,
        "ridge_regression": {
            "mrr": float(ridge_mrr),
            "improvement_over_best_fixed": float(ridge_mrr - best_fixed_mrr),
            "p_value_vs_best_fixed": float(p_val_ridge)
        },
        "mlp_regression": {
            "mrr": float(mlp_mrr),
            "improvement_over_best_fixed": float(mlp_mrr - best_fixed_mrr),
            "p_value_vs_best_fixed": float(p_val_mlp)
        },
        "best_fixed_mrr": float(best_fixed_mrr)
    }

def main():
    final_results = {}
    for ds_name, ds_path in DATASETS.items():
        if os.path.exists(ds_path):
            final_results[ds_name] = run_dataset(ds_name, ds_path)
            
    output_path = os.path.join(RESULTS_DIR, "regression_baselines.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved all results to {output_path}")

if __name__ == "__main__":
    main()
