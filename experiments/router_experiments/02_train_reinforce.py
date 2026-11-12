"""
02_train_reinforce.py — Run REINFORCE training with clean output saving.

This is a thin wrapper around src.router.train_reinforce that saves results
to the experiment directory with proper naming.

Usage:
    cd <repo>
    python experiments/router_experiments/02_train_reinforce.py
"""

import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import and run the existing trainer
from src.router.train_reinforce import main as train_main

if __name__ == "__main__":
    print("=" * 60)
    print("Running REINFORCE Training (Single Seed)")
    print("=" * 60)
    
    # Run training (uses settings from train_reinforce.py main())
    train_main()
    
    # Copy results to experiment directory
    src_dir = os.path.join(PROJECT_ROOT, "experiments", "router_training")
    for fname in ["best_reinforce_model.pt", "reinforce_training_history.json"]:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {fname} -> results/")
    
    print("\nDone! Results saved to experiments/router_experiments/results/")
