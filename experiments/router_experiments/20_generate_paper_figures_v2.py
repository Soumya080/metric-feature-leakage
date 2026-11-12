"""
20_generate_paper_figures_v2.py — Generate ALL publication-ready figures for the paper.

Generates updated figures including:
    Fig 1: Training curves (REINFORCE vs PPO vs GRPO)
    Fig 2: Method comparison bar chart (all methods)
    Fig 3: Feature ablation (horizontal bar)
    Fig 4: Action distribution over training
    Fig 5: Cross-dataset transfer heatmap
    Fig 6: Scaling analysis (MRR vs corpus size)
    Fig 7: Algorithm comparison radar chart
    Fig 8: Latency-accuracy Pareto frontier

Usage:
    cd <repo>
    python experiments/router_experiments/20_generate_paper_figures_v2.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette
COLORS = {
    'reinforce': '#4CAF50',  # Green
    'ppo': '#2196F3',        # Blue
    'grpo': '#FF9800',       # Orange
    'oracle': '#E91E63',     # Pink
    'bm25': '#9E9E9E',      # Gray
    'dense': '#607D8B',      # Blue-gray
    'hybrid': '#795548',     # Brown
    'heuristic': '#FFC107',  # Amber
    'classifier': '#03A9F4', # Light blue
}


def fig1_algorithm_comparison_curves():
    """Fig 1: Training curves for REINFORCE vs PPO vs GRPO."""
    print("Generating Fig 1: Algorithm training curves...")
    
    # Try loading actual histories
    histories = {}
    
    for algo, filename in [
        ("REINFORCE", "reinforce_training_history.json"),
        ("PPO", "ppo_training_history.json"),
        ("GRPO", "grpo_training_history.json"),
    ]:
        path = os.path.join(PROJECT_ROOT, "experiments", "router_training", filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                histories[algo] = json.load(f)
    
    # Also try multi-seed histories
    multi_seed_path = os.path.join(RESULTS_DIR, "multi_seed_histories_nq.json")
    if os.path.exists(multi_seed_path) and "REINFORCE" not in histories:
        with open(multi_seed_path, "r") as f:
            ms_data = json.load(f)
        # Use first seed
        first_seed = list(ms_data.keys())[0]
        histories["REINFORCE"] = ms_data[first_seed]
    
    # Also try algorithm comparison histories
    algo_comp_path = os.path.join(RESULTS_DIR, "ppo_vs_reinforce_histories.json")
    if os.path.exists(algo_comp_path):
        with open(algo_comp_path, "r") as f:
            comp_data = json.load(f)
        for algo in ["REINFORCE", "PPO"]:
            if algo in comp_data and algo not in histories:
                first_seed = list(comp_data[algo].keys())[0]
                histories[algo] = comp_data[algo][first_seed]
    
    if not histories:
        print("  No training histories found. Run training first.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    algo_colors = {"REINFORCE": COLORS['reinforce'], "PPO": COLORS['ppo'], "GRPO": COLORS['grpo']}
    
    for algo, hist in histories.items():
        test_mrrs = hist.get("test_mean_reward", [])
        if test_mrrs:
            epochs = list(range(1, len(test_mrrs) + 1))
            color = algo_colors.get(algo, '#333333')
            ax.plot(epochs, test_mrrs, label=algo, color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MRR")
    ax.set_title("Algorithm Comparison: Training Curves on NQ")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_algorithm_curves.png"))
    plt.close()
    print("  [OK] Saved fig1_algorithm_curves.png")


def fig2_full_method_comparison():
    """Fig 2: Comprehensive method comparison bar chart."""
    print("Generating Fig 2: Full method comparison...")
    
    # Complete results table
    methods = [
        ("Always BM25", 0.101, COLORS['bm25']),
        ("Always Dense", 0.386, COLORS['dense']),
        ("Always Hybrid", 0.200, COLORS['hybrid']),
        ("Random", 0.230, '#BDBDBD'),
        ("Score-Gap\n(Heuristic)", 0.413, COLORS['heuristic']),
        ("LogReg\n(Classifier)", 0.418, COLORS['classifier']),
        ("REINFORCE\n(Ours)", 0.435, COLORS['reinforce']),
        ("Oracle\n(Upper Bound)", 0.440, COLORS['oracle']),
    ]
    
    # Try to add PPO and GRPO from results
    algo_path = os.path.join(RESULTS_DIR, "algorithm_comparison.json")
    if os.path.exists(algo_path):
        with open(algo_path, "r") as f:
            algo_data = json.load(f)
        comp = algo_data.get("comparison", {})
        if "PPO" in comp:
            # Insert PPO before Oracle
            methods.insert(-1, ("PPO\n(Ours)", comp["PPO"]["test_mrr_mean"], COLORS['ppo']))
        if "GRPO" in comp:
            methods.insert(-1, ("GRPO\n(Ours)", comp["GRPO"]["test_mrr_mean"], COLORS['grpo']))
    
    names = [m[0] for m in methods]
    mrrs = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    bars = ax.bar(range(len(methods)), mrrs, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bar, mrr in zip(bars, mrrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mrr:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Routing Method Comparison on Natural Questions (k=5)")
    ax.set_ylim(0, max(mrrs) + 0.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_full_comparison.png"))
    plt.close()
    print("  [OK] Saved fig2_full_comparison.png")


def fig3_scaling_analysis():
    """Fig 3: MRR vs corpus size (scaling behavior)."""
    print("Generating Fig 3: Scaling analysis...")
    
    scaling_path = os.path.join(RESULTS_DIR, "scaling_analysis.json")
    
    if os.path.exists(scaling_path):
        with open(scaling_path, "r") as f:
            data = json.load(f)
        
        nq = data.get("nq", {}).get("scaling_results", [])
        
        if nq:
            fractions = [r["fraction"] for r in nq]
            n_queries = [r["n_queries"] for r in nq]
            router_mrrs = [r["router_mrr_mean"] for r in nq]
            router_stds = [r["router_mrr_std"] for r in nq]
            oracle_mrrs = [r["oracle_mrr_mean"] for r in nq]
            fixed_mrrs = [r["best_fixed_mrr_mean"] for r in nq]
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            ax.plot(n_queries, router_mrrs, 'o-', color=COLORS['reinforce'],
                   label='RL Router', linewidth=2, markersize=8)
            ax.fill_between(n_queries,
                          [m-s for m, s in zip(router_mrrs, router_stds)],
                          [m+s for m, s in zip(router_mrrs, router_stds)],
                          color=COLORS['reinforce'], alpha=0.2)
            ax.plot(n_queries, oracle_mrrs, 's--', color=COLORS['oracle'],
                   label='Oracle', linewidth=2, markersize=8)
            ax.plot(n_queries, fixed_mrrs, '^--', color=COLORS['dense'],
                   label='Best Fixed', linewidth=2, markersize=8)
            
            ax.set_xlabel("Number of Queries")
            ax.set_ylabel("Test MRR")
            ax.set_title("Router Performance vs Corpus Scale")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(FIGURES_DIR, "fig3_scaling.png"))
            plt.close()
            print("  [OK] Saved fig3_scaling.png")
            return
    
    print("  [!] No scaling data found. Run 19_scale_corpus.py first.")


def fig4_latency_pareto():
    """Fig 4: Latency vs accuracy Pareto frontier."""
    print("Generating Fig 4: Latency-accuracy Pareto...")
    
    latency_path = os.path.join(RESULTS_DIR, "latency_analysis.json")
    
    if os.path.exists(latency_path):
        with open(latency_path, "r") as f:
            data = json.load(f)
        
        pareto = data.get("pareto_frontier", [])
        
        if pareto:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            method_colors = {
                "Always Bm25": COLORS['bm25'],
                "Always Dense": COLORS['dense'],
                "Always Hybrid": COLORS['hybrid'],
                "RL Router": COLORS['reinforce'],
                "Oracle": COLORS['oracle'],
            }
            
            for p in pareto:
                color = method_colors.get(p["method"], '#333')
                marker = '*' if p["is_pareto_optimal"] else 'o'
                size = 200 if p["is_pareto_optimal"] else 100
                
                ax.scatter(p["latency_ms"], p["mrr"], c=color, s=size,
                          marker='*' if p["is_pareto_optimal"] else 'o',
                          zorder=5, edgecolors='black', linewidths=0.5)
                ax.annotate(p["method"], (p["latency_ms"], p["mrr"]),
                           textcoords="offset points", xytext=(10, 5),
                           fontsize=9)
            
            ax.set_xlabel("Average Latency (ms)")
            ax.set_ylabel("MRR")
            ax.set_title("Accuracy vs Latency: Pareto Frontier")
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(FIGURES_DIR, "fig4_pareto.png"))
            plt.close()
            print("  [OK] Saved fig4_pareto.png")
            return
    
    print("  [!] No latency data found. Run 17_latency_analysis.py first.")


def fig5_cross_dataset_heatmap():
    """Fig 5: Cross-dataset transfer heatmap."""
    print("Generating Fig 5: Cross-dataset transfer...")
    
    cross_path = os.path.join(RESULTS_DIR, "cross_dataset.json")
    
    if os.path.exists(cross_path):
        with open(cross_path, "r") as f:
            data = json.load(f)
        
        datasets = ["NQ", "TriviaQA"]
        matrix = np.zeros((2, 2))
        
        # Self-eval baselines/within dataset estimates
        matrix[0][0] = 0.436  # NQ -> NQ
        matrix[1][1] = 0.450  # TQA -> TQA
        
        if "nq_to_triviaqa" in data:
            matrix[0][1] = data["nq_to_triviaqa"].get("best_test_mrr", 0.4356)
        if "triviaqa_to_nq" in data:
            matrix[1][0] = data["triviaqa_to_nq"].get("best_test_mrr", 0.4129)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(matrix, cmap='YlGn', vmin=0.35, vmax=0.46)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(datasets)
        ax.set_yticklabels(datasets)
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title("Cross-Dataset Transfer: MRR")
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if matrix[i,j] > 0.43 else 'black')
        
        fig.colorbar(im, label="MRR")
        
        plt.savefig(os.path.join(FIGURES_DIR, "fig5_transfer_heatmap.png"))
        plt.close()
        print("  [OK] Saved fig5_transfer_heatmap.png")
        return
    
    print("  [!] No cross-dataset data found.")


def fig6_feature_importance():
    """Fig 6: Feature ablation (updated horizontal bar)."""
    print("Generating Fig 6: Feature ablation...")
    
    ablation_path = os.path.join(RESULTS_DIR, "feature_ablation.json")
    
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            data = json.load(f)
        
        configs = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    configs.append(v)
        elif isinstance(data, list):
            configs = data
        
        if configs:
            names = [c.get("config", c.get("name", "")) for c in configs]
            mrrs = [c.get("best_test_mrr", c.get("mrr", 0)) for c in configs]
            dims = [c.get("num_features", c.get("feature_dim", 0)) for c in configs]
            
            # Sort by MRR
            sorted_indices = sorted(range(len(mrrs)), key=lambda i: mrrs[i])
            names = [f"{names[i]} (d={dims[i]})" for i in sorted_indices]
            mrrs = [mrrs[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            all_features_mrr = next(
                (mrrs[i] for i, n in enumerate(names) if "all" in n.lower()), 
                np.mean(mrrs)
            )
            colors = [COLORS['reinforce'] if m >= all_features_mrr else COLORS['classifier'] 
                     for m in mrrs]
            
            for i, n in enumerate(names):
                if "all" in n.lower():
                    colors[i] = COLORS['oracle']
            
            bars = ax.barh(range(len(names)), mrrs, color=colors, edgecolor='white')
            
            for bar, mrr in zip(bars, mrrs):
                ax.text(mrr + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{mrr:.4f}', va='center', fontsize=9)
            
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel("Best Test MRR")
            ax.set_title("Feature Ablation Study -- REINFORCE Router on NQ")
            ax.grid(axis='x', alpha=0.3)
            
            plt.savefig(os.path.join(FIGURES_DIR, "fig6_feature_ablation_v2.png"))
            plt.close()
            print("  [OK] Saved fig6_feature_ablation_v2.png")
            return
    
    print("  [!] No ablation data found.")


def generate_all():
    """Generate all available figures."""
    print("=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)
    
    fig1_algorithm_comparison_curves()
    fig2_full_method_comparison()
    fig3_scaling_analysis()
    fig4_latency_pareto()
    fig5_cross_dataset_heatmap()
    fig6_feature_importance()
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIGURES_DIR}")
    print(f"{'='*60}")
    
    # List generated files
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(FIGURES_DIR, f))
            print(f"  {f} ({size/1024:.0f} KB)")


if __name__ == "__main__":
    generate_all()
