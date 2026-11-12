# src/router — RL-Based Multi-Retriever Routing (Core Research Contribution)
#
# Components:
#   feature_extractor  — Query feature extraction for policy state
#   policy_network     — RouterPolicy MLP + PolicyAgent wrapper
#   routing_env        — Contextual bandit environment (lookup-table based)
#   compute_reward     — Reward functions for retriever quality
#   train_reinforce    — REINFORCE (policy gradient) training loop
#   train_ppo          — PPO (Proximal Policy Optimization) training loop
#   train_grpo         — GRPO (Group Relative Policy Optimization) training loop
#   prepare_env_data   — Pipeline to convert baseline_v2 results → env data

from src.router.feature_extractor import QueryFeatureExtractor
from src.router.policy_network import RouterPolicy, PolicyAgent
from src.router.routing_env import RouterEnvironment
from src.router.compute_reward import compute_reward
