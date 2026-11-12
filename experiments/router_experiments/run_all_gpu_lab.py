#!/usr/bin/env python3
"""
run_all_gpu_lab.py - Master execution script for RL-Route experiments.
Run this on your GPU lab machine to execute the full evaluation pipeline unattended.

Usage:
    cd <repo>
    python experiments/router_experiments/run_all_gpu_lab.py
"""
import os
import sys
import time
import subprocess
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONPATH'] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)

def run_script(name, script_path):
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"Script: {script_path}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    start = time.time()
    
    # We use subprocess.run so the user can see standard output in real-time
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        env={**os.environ, 'PYTHONPATH': PROJECT_ROOT}
    )
    
    elapsed = time.time() - start
    status = 'SUCCESS' if result.returncode == 0 else f'FAILED (code {result.returncode})'
    print(f"\n{'='*70}")
    print(f"FINISHED: {name} — {status} in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}\n")
    return result.returncode == 0, elapsed

def main():
    print(f"RL-Route Master Execution Pipeline")
    print(f"Project Root: {PROJECT_ROOT}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected. Running on CPU (this is fine for lookup-based training, but will be slower).")
    except ImportError:
        print("WARNING: PyTorch not found in current environment.")
        
    start_time = time.time()
    
    scripts = [
        ("Canonical Evaluation", os.path.join(PROJECT_ROOT, "experiments", "router_experiments", "canonical_evaluation.py")),
        ("Online Adaptation", os.path.join(PROJECT_ROOT, "experiments", "router_experiments", "online_adaptation.py")),
        ("Generate LaTeX Tables", os.path.join(PROJECT_ROOT, "experiments", "router_experiments", "generate_latex_tables.py")),
        ("Plot Online Adaptation", os.path.join(PROJECT_ROOT, "experiments", "router_experiments", "plot_online_adaptation.py")),
    ]
    
    success = True
    for name, path in scripts:
        if not os.path.exists(path):
            print(f"ERROR: Script not found: {path}")
            success = False
            continue
            
        script_success, _ = run_script(name, path)
        if not script_success:
            success = False
            print(f"Pipeline stopped due to error in {name}.")
            break
            
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("\nGenerated Results Files:")
        print(f"- {os.path.join(PROJECT_ROOT, 'results', 'canonical_results.json')}")
        print(f"- {os.path.join(PROJECT_ROOT, 'results', 'online_adaptation_results.json')}")
        print(f"- {os.path.join(PROJECT_ROOT, 'results', 'main_results_table.tex')}")
        print(f"- {os.path.join(PROJECT_ROOT, 'results', 'online_adaptation_plot.pdf')}")
        print("\nAll paper tables and figures have been regenerated from scratch.")

if __name__ == "__main__":
    main()
