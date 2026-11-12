"""
04_classifier_baseline.py — Train supervised classifier baselines for routing.

This is the MOST IMPORTANT missing baseline. Adaptive-RAG and similar papers
use a classifier (not RL) to predict the best retriever. If a simple logistic
regression matches your RL router, a reviewer will ask "why use RL at all?"

You MUST show RL adds value beyond a classifier.

Usage:
    cd <repo>
    python experiments/router_experiments/04_classifier_baseline.py
"""

import json
import os
import sys
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.routing_env import deterministic_tiebreak

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}


def load_and_split(data_path, seed=42, train_ratio=0.8):
    """Load env data and create train/test split matching RL experiments."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)
    split = int(len(keys) * train_ratio)
    
    return data, keys[:split], keys[split:]


def prepare_xy(data, keys, metric="mrr"):
    """Extract features (X) and oracle labels (y) from env data."""
    X, y = [], []
    for k in keys:
        entry = data[k]
        X.append(entry["features"])
        
        # Oracle label: which retriever has the highest metric for this query?
        # Tie-breaking deterministically based on query ID hash to avoid BM25 bias
        scores = [entry["retrievers"][["bm25", "dense", "hybrid"][a]].get(metric, 0.0) for a in range(3)]
        max_score = max(scores)
        tied_actions = [a for a in range(3) if scores[a] == max_score]
        best_action = tied_actions[deterministic_tiebreak(k, len(tied_actions))]
        y.append(best_action)
    
    return np.array(X), np.array(y)


def evaluate_classifier(clf, X_test, y_test, data, test_keys, metric="mrr"):
    """Evaluate a trained classifier: predict best retriever, compute actual MRR."""
    predictions = clf.predict(X_test)
    
    # Compute actual MRR achieved by following classifier's predictions
    total_mrr = 0.0
    retriever_names = ["bm25", "dense", "hybrid"]
    for pred, k in zip(predictions, test_keys):
        chosen_ret = retriever_names[pred]
        total_mrr += data[k]["retrievers"][chosen_ret].get(metric, 0.0)
    
    mean_mrr = total_mrr / len(test_keys)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        "mrr": round(mean_mrr, 4),
        "oracle_accuracy": round(accuracy, 4),
        "predictions_distribution": {
            "bm25": int(np.sum(predictions == 0)),
            "dense": int(np.sum(predictions == 1)),
            "hybrid": int(np.sum(predictions == 2)),
        }
    }


def main():
    all_results = {}
    
    from sklearn.model_selection import RandomizedSearchCV
    
    gbt_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_leaf': [1, 5, 10],
    }
    
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=10),
        # Tuned GBT is initialized later per-dataset
    }
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        data, train_keys, test_keys = load_and_split(ds_path, seed=SEED)
        X_train, y_train = prepare_xy(data, train_keys)
        X_test, y_test = prepare_xy(data, test_keys)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Train: {len(train_keys)} queries, Test: {len(test_keys)} queries")
        print(f"Feature dim: {X_train.shape[1]}")
        print(f"Label distribution (train): BM25={np.sum(y_train==0)}, Dense={np.sum(y_train==1)}, Hybrid={np.sum(y_train==2)}")
        
        ds_results = {}
        
        # Train generic classifiers
        for clf_name, clf in classifiers.items():
            print(f"\n--- {clf_name} ---")
            clf.fit(X_train_scaled, y_train)
            result = evaluate_classifier(clf, X_test_scaled, y_test, data, test_keys)
            ds_results[clf_name] = result
            print(f"  MRR:       {result['mrr']:.4f}")
            print(f"  Oracle Acc: {result['oracle_accuracy']:.4f}")
            print(f"  Pred Dist:  {result['predictions_distribution']}")
            
            if hasattr(clf, "feature_importances_"):
                from src.router.feature_extractor import QueryFeatureExtractor
                feature_names = QueryFeatureExtractor.get_feature_order()
                importances = clf.feature_importances_
                top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
                ds_results[clf_name]["top_features"] = [{"name": name, "importance": round(float(imp), 4)} for name, imp in top_features]
        
        # Train and tune Gradient Boosting
        print(f"\n--- gradient_boosting (tuned) ---")
        gbt_search = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=SEED), 
            gbt_param_dist, 
            n_iter=20, 
            cv=3, 
            random_state=SEED, 
            n_jobs=-1
        )
        gbt_search.fit(X_train_scaled, y_train)
        best_gbt = gbt_search.best_estimator_
        
        result = evaluate_classifier(best_gbt, X_test_scaled, y_test, data, test_keys)
        ds_results["gradient_boosting_tuned"] = result
        print(f"  MRR:       {result['mrr']:.4f}")
        print(f"  Oracle Acc: {result['oracle_accuracy']:.4f}")
        print(f"  Pred Dist:  {result['predictions_distribution']}")
        print(f"  Best Params: {gbt_search.best_params_}")

        
        all_results[ds_name] = ds_results
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "classifier_baselines.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
