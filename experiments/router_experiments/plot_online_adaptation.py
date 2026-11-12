#!/usr/bin/env python3
"""
plot_online_adaptation.py
Generates a publication-ready PDF figure showing how RL adapts to distribution shifts
while static supervised models (Gradient Boosting) fail.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "online_adaptation_results.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "online_adaptation_plot.pdf")

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found. Run online_adaptation.py first.")
        return

    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    rolling = data.get("rolling_averages", {})
    boundaries = data.get("boundaries", {})

    if not rolling:
        print("Error: No rolling averages found in the results file.")
        return

    # Style configuration for publication
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6)
    })

    fig, ax = plt.subplots()

    # X-axis is simply the query index stream
    x = list(range(len(rolling["gbt"])))

    # Plot the lines
    ax.plot(x, rolling["gbt"], label="Frozen GBT", color="#d62728", linewidth=2.5, linestyle="--", alpha=0.8)
    ax.plot(x, rolling["rl_frozen"], label="Frozen RL", color="#7f7f7f", linewidth=2.0, linestyle=":", alpha=0.8)
    ax.plot(x, rolling["rl_online"], label="Online RL-Route (Ours)", color="#1f77b4", linewidth=3.0)

    # Draw vertical lines indicating dataset distribution shifts
    colors = {"nq": "#e1f5fe", "triviaqa": "#f3e5f5", "hotpotqa": "#fff3e0"}
    labels = {"nq": "Natural Questions\n(Source Domain)", 
              "triviaqa": "TriviaQA\n(Shift 1)", 
              "hotpotqa": "HotpotQA\n(Shift 2)"}
              
    y_min, y_max = ax.get_ylim()
    
    for ds_name, bounds in boundaries.items():
        start = bounds["start"]
        end = bounds["end"]
        # Add background span for each dataset
        ax.axvspan(start, end, color=colors.get(ds_name, "#ffffff"), alpha=0.5, lw=0)
        
        # Add a text label near the top of each section
        mid_point = (start + end) / 2
        ax.text(mid_point, y_max * 0.95, labels.get(ds_name, ds_name), 
                ha='center', va='top', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Draw a vertical line at the start of a shift
        if start > 0:
            ax.axvline(x=start, color='black', linestyle='-.', linewidth=1.5, alpha=0.6)

    # Labels and Legend
    ax.set_xlabel("Continuous Query Stream (Time $\\rightarrow$)")
    ax.set_ylabel("Rolling Average MRR (Window=100)")
    ax.set_title("Resilience to Distribution Shift: Online RL vs Static Classifiers")
    
    # Position legend in the bottom left or right to avoid overlapping data
    ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)

    # Clean up layout
    plt.tight_layout()

    # Save as high-res PDF for LaTeX
    plt.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Publication-ready plot saved to: {OUTPUT_PATH}")
    
    # Also save a PNG for quick viewing
    png_path = OUTPUT_PATH.replace(".pdf", ".png")
    plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    print(f"Quick preview PNG saved to: {png_path}")

if __name__ == "__main__":
    main()
