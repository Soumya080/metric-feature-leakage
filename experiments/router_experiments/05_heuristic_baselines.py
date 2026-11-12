"""
05_heuristic_baselines.py — Simple rule-based routing baselines.

These are "sanity check" baselines that any reviewer expects.
If RL can't beat simple rules, the RL is unjustified.

Usage:
    cd <repo>
    python experiments/router_experiments/05_heuristic_baselines.py
"""

import json
import os
import sys
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}

# Feature indices (matching QueryFeatureExtractor.get_feature_order())
FEAT_QUERY_LENGTH = 0
FEAT_STOPWORD_RATIO = 3
FEAT_HAS_QUOTES = 5
FEAT_NUM_ENTITIES = 6
FEAT_Q_TYPE_WHO = 13
FEAT_Q_TYPE_WHAT = 14
FEAT_Q_TYPE_WHEN = 15
FEAT_Q_TYPE_WHERE = 16
FEAT_CONTAINS_NUMBERS = 20
FEAT_CONTAINS_PROPER_NOUNS = 21
FEAT_AVG_BM25 = 22
FEAT_AVG_DENSE = 23
FEAT_SCORE_GAP = 24


def heuristic_query_length(features):
    """Short queries → BM25 (keyword match), Long queries → Dense (semantic)."""
    return 0 if features[FEAT_QUERY_LENGTH] <= 6 else 1


def heuristic_entity_presence(features):
    """Has named entities → BM25 (exact entity match), No entities → Dense."""
    return 0 if features[FEAT_NUM_ENTITIES] > 0 else 1


def heuristic_score_gap(features):
    """Pick whichever retriever scored higher on training data averages."""
    # score_gap = avg_dense - avg_bm25 (positive = dense better)
    return 1 if features[FEAT_SCORE_GAP] >= 0 else 0


def heuristic_question_type(features):
    """who/when/where → BM25 (factoid), what/how/why → Dense (semantic)."""
    if features[FEAT_Q_TYPE_WHO] or features[FEAT_Q_TYPE_WHEN] or features[FEAT_Q_TYPE_WHERE]:
        return 0  # BM25
    return 1  # Dense


def heuristic_combined(features):
    """Multi-rule: combine entity + length + question type signals."""
    votes = [0, 0, 0]  # BM25, Dense, Hybrid
    
    # Entity presence → BM25
    if features[FEAT_NUM_ENTITIES] > 0:
        votes[0] += 1
    else:
        votes[1] += 1
    
    # Query length
    if features[FEAT_QUERY_LENGTH] <= 5:
        votes[0] += 1
    elif features[FEAT_QUERY_LENGTH] >= 10:
        votes[1] += 1
    else:
        votes[2] += 1  # Medium length → Hybrid
    
    # Question type
    if features[FEAT_Q_TYPE_WHO] or features[FEAT_Q_TYPE_WHEN] or features[FEAT_Q_TYPE_WHERE]:
        votes[0] += 1
    elif features[FEAT_Q_TYPE_WHAT] or features[FEAT_CONTAINS_PROPER_NOUNS]:
        votes[0] += 1
    else:
        votes[1] += 1
    
    return int(np.argmax(votes))


def evaluate_heuristic(data, test_keys, heuristic_fn, metric="mrr"):
    """Evaluate a heuristic routing function."""
    retriever_names = ["bm25", "dense", "hybrid"]
    total = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for k in test_keys:
        features = data[k]["features"]
        action = heuristic_fn(features)
        chosen_ret = retriever_names[action]
        total += data[k]["retrievers"][chosen_ret].get(metric, 0.0)
        action_counts[action] += 1
    
    n = len(test_keys)
    return {
        "mrr": round(total / n, 4),
        "action_distribution": {retriever_names[a]: c for a, c in action_counts.items()},
    }


def main():
    heuristics = {
        "query_length": heuristic_query_length,
        "entity_presence": heuristic_entity_presence,
        "score_gap": heuristic_score_gap,
        "question_type": heuristic_question_type,
        "combined_rules": heuristic_combined,
    }
    
    all_results = {}
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        with open(ds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        keys = list(data.keys())
        random.seed(SEED)
        random.shuffle(keys)
        split = int(len(keys) * 0.8)
        test_keys = keys[split:]
        
        ds_results = {}
        for h_name, h_fn in heuristics.items():
            result = evaluate_heuristic(data, test_keys, h_fn)
            ds_results[h_name] = result
            print(f"  {h_name:<20s}  MRR={result['mrr']:.4f}  {result['action_distribution']}")
        
        all_results[ds_name] = ds_results
    
    output_path = os.path.join(RESULTS_DIR, "heuristic_baselines.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
