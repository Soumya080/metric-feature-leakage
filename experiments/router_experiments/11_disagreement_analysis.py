"""
11_disagreement_analysis.py — Analyze WHERE routing helps vs doesn't.

This is the heart of your paper's analysis section. It answers:
"On which queries does adaptive routing actually matter?"

Usage:
    cd <repo>
    python experiments/router_experiments/11_disagreement_analysis.py
"""
import json, os, sys, random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


def main():
    with open(NQ_PATH) as f:
        data = json.load(f)
    
    keys = list(data.keys())
    random.seed(SEED)
    random.shuffle(keys)
    split = int(len(keys) * 0.8)
    test_keys = keys[split:]
    
    # Load trained model
    model_path = os.path.join(RESULTS_DIR, "best_reinforce_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "best_reinforce_model.pt")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dim = len(data[test_keys[0]]["features"])
    agent = PolicyAgent(input_dim=state_dim, hidden_dim=128, output_dim=3)
    agent.policynetwork.load_state_dict(checkpoint["model_state_dict"])
    
    # Compute normalization stats from training set
    train_features = np.array([data[k]["features"] for k in keys[:split]])
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    
    # Categorize each test query
    categories = {
        "all_agree_nonzero": [],    # All retrievers hit (routing doesn't matter)
        "all_agree_zero": [],       # No retriever hits (routing can't help)
        "disagree_bm25_wins": [],   # BM25 is best
        "disagree_dense_wins": [],  # Dense is best
        "disagree_hybrid_wins": [], # Hybrid is best
    }
    
    rl_correct_on_disagree = 0
    rl_total_disagree = 0
    dense_correct_on_disagree = 0
    
    per_query_analysis = []
    
    for qid in test_keys:
        entry = data[qid]
        rets = entry["retrievers"]
        
        bm25_mrr = rets["bm25"]["mrr"]
        dense_mrr = rets["dense"]["mrr"]
        hybrid_mrr = rets["hybrid"]["mrr"]
        
        mrrs = {"bm25": bm25_mrr, "dense": dense_mrr, "hybrid": hybrid_mrr}
        oracle_ret = max(mrrs, key=mrrs.get)
        oracle_mrr = mrrs[oracle_ret]
        
        # Get RL router's choice
        raw_features = entry["features"]
        norm_features = ((np.array(raw_features) - feat_mean) / feat_std).tolist()
        action, _ = agent.select_action(norm_features, evaluate=True)
        rl_ret = ["bm25", "dense", "hybrid"][action]
        rl_mrr = mrrs[rl_ret]
        
        # Categorize
        all_same = (bm25_mrr == dense_mrr == hybrid_mrr)
        
        if all_same and bm25_mrr > 0:
            categories["all_agree_nonzero"].append(qid)
        elif all_same and bm25_mrr == 0:
            categories["all_agree_zero"].append(qid)
        else:
            # Retrievers disagree
            rl_total_disagree += 1
            if rl_ret == oracle_ret:
                rl_correct_on_disagree += 1
            if "dense" == oracle_ret:
                dense_correct_on_disagree += 1
            
            if oracle_ret == "bm25":
                categories["disagree_bm25_wins"].append(qid)
            elif oracle_ret == "dense":
                categories["disagree_dense_wins"].append(qid)
            else:
                categories["disagree_hybrid_wins"].append(qid)
        
        per_query_analysis.append({
            "qid": qid,
            "question": entry.get("question", ""),
            "bm25_mrr": bm25_mrr,
            "dense_mrr": dense_mrr,
            "hybrid_mrr": hybrid_mrr,
            "oracle_choice": oracle_ret,
            "rl_choice": rl_ret,
            "rl_correct": rl_ret == oracle_ret,
            "category": "agree" if all_same else "disagree",
        })
    
    # Print analysis
    n = len(test_keys)
    total_disagree = sum(len(v) for k, v in categories.items() if "disagree" in k)
    total_agree = sum(len(v) for k, v in categories.items() if "agree" in k)
    
    print(f"\n{'='*60}")
    print(f"DISAGREEMENT ANALYSIS (NQ, n={n})")
    print(f"{'='*60}")
    print(f"\nQuery Categories:")
    print(f"  All retrievers agree (same MRR):     {total_agree:3d} ({total_agree/n:.1%})")
    print(f"    ├── All hit (nonzero MRR):          {len(categories['all_agree_nonzero']):3d} ({len(categories['all_agree_nonzero'])/n:.1%})")
    print(f"    └── All miss (zero MRR):            {len(categories['all_agree_zero']):3d} ({len(categories['all_agree_zero'])/n:.1%})")
    print(f"  Retrievers DISAGREE:                  {total_disagree:3d} ({total_disagree/n:.1%})")
    print(f"    ├── BM25 wins:                      {len(categories['disagree_bm25_wins']):3d}")
    print(f"    ├── Dense wins:                     {len(categories['disagree_dense_wins']):3d}")
    print(f"    └── Hybrid wins:                    {len(categories['disagree_hybrid_wins']):3d}")
    
    print(f"\nRouting Value Analysis (on {total_disagree} disagreeing queries):")
    if rl_total_disagree > 0:
        print(f"  RL router correct on disagree queries: {rl_correct_on_disagree}/{rl_total_disagree} ({rl_correct_on_disagree/rl_total_disagree:.1%})")
        print(f"  Always-Dense correct on disagree:      {dense_correct_on_disagree}/{rl_total_disagree} ({dense_correct_on_disagree/rl_total_disagree:.1%})")
    
    print(f"\nKey Insight:")
    print(f"  On the {total_agree/n:.0%} of queries where retrievers agree,")
    print(f"  routing adds ZERO value (any choice gives the same MRR).")
    print(f"  Routing only matters on the {total_disagree/n:.0%} where they disagree.")
    
    # Save
    output = {
        "test_size": n,
        "categories": {k: len(v) for k, v in categories.items()},
        "total_agree": total_agree,
        "total_disagree": total_disagree,
        "agree_pct": round(total_agree/n, 3),
        "disagree_pct": round(total_disagree/n, 3),
        "rl_accuracy_on_disagree": round(rl_correct_on_disagree/max(rl_total_disagree,1), 3),
        "dense_accuracy_on_disagree": round(dense_correct_on_disagree/max(rl_total_disagree,1), 3),
        "per_query": per_query_analysis,
    }
    
    out_path = os.path.join(RESULTS_DIR, "disagreement_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
