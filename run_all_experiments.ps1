# RL-Route: Complete Experiment Pipeline
# Run this on the GPU lab server

$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Data Preparation ==="
python experiments/baseline_v2/01_prepare_data.py

Write-Host "=== Step 2: Run Baselines ==="
python experiments/baseline_v2/02_run_baselines.py --dataset all

Write-Host "=== Step 3: Prepare RL Env Data ==="
python -m src.router.prepare_env_data

Write-Host "=== Step 4: Train RL Router (Multi-seed) ==="
python experiments/router_experiments/03_train_multi_seed.py

Write-Host "=== Step 5: PPO vs REINFORCE ==="
python experiments/router_experiments/14_ppo_vs_reinforce.py

Write-Host "=== Step 6: GRPO Comparison ==="
python experiments/router_experiments/15_grpo_comparison.py

Write-Host "=== Step 7: Feature Ablation ==="
python experiments/router_experiments/06_feature_ablation.py

Write-Host "=== Step 8: Cross-Dataset Transfer ==="
python experiments/router_experiments/07_cross_dataset.py

Write-Host "=== Step 9: End-to-End Eval (vLLM) ==="
python experiments/router_experiments/18_end_to_end_eval.py --backend vllm --dataset all

Write-Host "=== Step 10: Generate Paper Figures ==="
python experiments/router_experiments/20_generate_paper_figures_v2.py

Write-Host "=== ALL DONE ==="
