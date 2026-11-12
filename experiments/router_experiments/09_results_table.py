"""
09_results_table.py — Generate the main comparison table for the paper.

Outputs both LaTeX (for paper) and Markdown (for README/blog) formats.

Usage:
    cd <repo>
    python experiments/router_experiments/09_results_table.py
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    baselines = load_json(os.path.join(RESULTS_DIR, "baselines.json"))
    classifiers = load_json(os.path.join(RESULTS_DIR, "classifier_baselines.json"))
    heuristics = load_json(os.path.join(RESULTS_DIR, "heuristic_baselines.json"))
    multi_seed = load_json(os.path.join(RESULTS_DIR, "multi_seed_nq.json"))
    cross_ds = load_json(os.path.join(RESULTS_DIR, "cross_dataset.json"))
    
    # Collect all methods with their MRR on NQ
    rows = []
    
    if baselines and "nq" in baselines:
        nq = baselines["nq"]
        rows.append(("Random", nq.get("random", {}).get("mrr", "—"), "—"))
        rows.append(("Always BM25", nq.get("always_bm25", {}).get("mrr", "—"), "—"))
        rows.append(("Always Dense", nq.get("always_dense", {}).get("mrr", "—"), "—"))
        rows.append(("Always Hybrid", nq.get("always_hybrid", {}).get("mrr", "—"), "—"))
    
    if heuristics and "nq" in heuristics:
        for h_name, h_data in heuristics["nq"].items():
            rows.append((f"Heuristic ({h_name})", h_data.get("mrr", "—"), "—"))
    
    if classifiers and "nq" in classifiers:
        for c_name, c_data in classifiers["nq"].items():
            rows.append((f"Classifier ({c_name.replace('_', ' ')})", c_data.get("mrr", "—"), "—"))
    
    if multi_seed:
        mean = multi_seed.get("test_mrr_mean", "—")
        std = multi_seed.get("test_mrr_std", "—")
        rows.append(("**REINFORCE Router (Ours)**", mean, f"± {std}"))
    else:
        # Fallback to single-run history
        history_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "reinforce_training_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                h = json.load(f)
            rows.append(("**REINFORCE Router (Ours)**", round(max(h["test_mean_reward"]), 4), "single run"))
    
    if baselines and "nq" in baselines:
        rows.append(("Oracle (Upper Bound)", baselines["nq"].get("oracle", {}).get("mrr", "—"), "—"))
    
    # Print Markdown table
    print("\n## Main Results: Routing Method Comparison on NQ (k=5)\n")
    print(f"| {'Method':<40s} | {'MRR':>8s} | {'Std':>10s} |")
    print(f"|{'-'*42}|{'-'*10}|{'-'*12}|")
    for method, mrr, std in rows:
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
        print(f"| {method:<40s} | {mrr_str:>8s} | {str(std):>10s} |")
    
    # Print LaTeX table
    print("\n\n% LaTeX table for paper:")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison of routing methods on Natural Questions (k=5).}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Method & MRR \\\\")
    print("\\midrule")
    for method, mrr, std in rows:
        method_clean = method.replace("**", "").replace("_", "\\_")
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
        if "Ours" in method:
            if isinstance(std, str) and "±" in std:
                print(f"\\textbf{{{method_clean}}} & \\textbf{{{mrr_str} {std}}} \\\\")
            else:
                print(f"\\textbf{{{method_clean}}} & \\textbf{{{mrr_str}}} \\\\")
        elif "Oracle" in method:
            print(f"\\midrule")
            print(f"{method_clean} & {mrr_str} \\\\")
        else:
            print(f"{method_clean} & {mrr_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Cross-dataset results if available
    if cross_ds:
        print("\n\n## Cross-Dataset Generalization\n")
        print(f"| {'Train -> Test':<25s} | {'Router MRR':>10s} | {'Best Fixed':>10s} | {'Delta':>8s} |")
        print(f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*10}|")
        for name, r in cross_ds.items():
            print(f"| {name:<25s} | {r['final_test_mrr']:>10.4f} | {r['best_fixed_mrr']:>10.4f} | {r['improvement']:>+8.4f} |")


if __name__ == "__main__":
    main()
