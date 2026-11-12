"""
PPO Training Loop for the RL Retriever Router.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent, RouterPolicy
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.compute_reward import compute_reward, compute_reward_from_metrics


class PPOTrainer:
    RETRIEVER_NAMES = {0: "BM25", 1: "Dense", 2: "Hybrid"}
    
    def __init__(
        self,
        env_train: RouterEnvironment,
        env_test: RouterEnvironment,
        agent: PolicyAgent,
        reward_type: str = "mrr",
        entropy_coeff: float = 0.1,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        seed: int = 42,
        env_val: Optional[RouterEnvironment] = None,
    ):
        self.env_train = env_train
        self.env_test = env_test
        self.env_val = env_val  # if provided, best-checkpoint selection uses this, not test
        self.agent = agent
        self.reward_type = reward_type
        self.entropy_coeff = entropy_coeff
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self._feature_mean, self._feature_std = self._compute_norm_stats(env_train)
        
        # Value network for advantage estimation
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(self.env_train.state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        self.history = {
            "epoch": [],
            "train_mean_reward": [],
            "train_loss": [],
            "test_mean_reward": [],
            "test_oracle_reward": [],
            "action_distribution": [],
            "oracle_accuracy": [],
            "entropy": [],
        }
    
    def _compute_norm_stats(self, env: RouterEnvironment):
        all_features = []
        for qid in env.query_pool:
            all_features.append(env.get_state(qid))
        all_features = np.array(all_features)
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        std[std < 1e-8] = 1.0
        return mean, std
    
    def _normalize_state(self, state: List[float]) -> List[float]:
        return ((np.array(state) - self._feature_mean) / self._feature_std).tolist()
    
    def train(self, num_epochs: int = 200, log_interval: int = 10) -> Dict:
        print(f"\n{'='*60}")
        print(f"PPO Training")
        print(f"{'='*60}")
        print(f"Train queries: {len(self.env_train)}")
        print(f"Test queries:  {len(self.env_test)}")
        print(f"State dim:     {self.env_train.state_dim}")
        print(f"Reward type:   {self.reward_type}")
        print(f"PPO Epochs:    {self.ppo_epochs}")
        print(f"Epochs:        {num_epochs}")
        print(f"{'='*60}\n")
        
        best_test_reward = -float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_reward, train_loss, action_counts, mean_entropy = self._train_epoch()
            
            test_reward, oracle_reward, oracle_acc = self._evaluate()
            
            self.history["epoch"].append(epoch)
            self.history["train_mean_reward"].append(train_reward)
            self.history["train_loss"].append(train_loss)
            self.history["test_mean_reward"].append(test_reward)
            self.history["test_oracle_reward"].append(oracle_reward)
            self.history["action_distribution"].append(action_counts.copy())
            self.history["oracle_accuracy"].append(oracle_acc)
            self.history["entropy"].append(mean_entropy)

            # Model selection on the VALIDATION set when available. Selecting the best
            # checkpoint by test-set reward leaks the test set into training and inflates
            # reported numbers; env_val=None preserves the old behaviour only for callers
            # that haven't been updated yet.
            if self.env_val is not None:
                val_reward, _, _ = self._evaluate(use_val=True)
            else:
                val_reward = test_reward

            if val_reward > best_test_reward:
                best_test_reward = val_reward
                self._save_checkpoint(epoch, val_reward)
            
            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                total_actions = sum(action_counts.values())
                action_pcts = {k: f"{v/total_actions:.0%}" for k, v in action_counts.items()} if total_actions > 0 else {}
                
                print(
                    f"Epoch {epoch:4d}/{num_epochs} | "
                    f"Train R={train_reward:.4f} | "
                    f"Test R={test_reward:.4f} | "
                    f"Oracle={oracle_reward:.4f} | "
                    f"OracleAcc={oracle_acc:.1%} | "
                    f"Actions={action_pcts} | "
                    f"Loss={train_loss:.4f} | "
                    f"{elapsed:.0f}s"
                )
        
        return self.history

    def _train_epoch(self) -> Tuple[float, float, Dict[int, int], float]:
        query_ids = self.env_train.reset()
        
        epoch_states = []
        epoch_actions = []
        epoch_old_log_probs = []
        epoch_rewards = []
        action_counts = {0: 0, 1: 0, 2: 0}
        
        # 1. Collect trajectories
        for qid in query_ids:
            raw_state = self.env_train.get_state(qid)
            state = self._normalize_state(raw_state)
            action, log_prob = self.agent.select_action(state, evaluate=False)
            reward = self.env_train.step(qid, action, target_metric=self.reward_type)
            
            epoch_states.append(state)
            epoch_actions.append(action)
            epoch_old_log_probs.append(log_prob.item())
            epoch_rewards.append(reward)
            action_counts[action] += 1
            
        mean_reward = np.mean(epoch_rewards)
        
        states_t = torch.tensor(epoch_states, dtype=torch.float32)
        actions_t = torch.tensor(epoch_actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(epoch_old_log_probs, dtype=torch.float32)
        rewards_t = torch.tensor(epoch_rewards, dtype=torch.float32)
        
        # 2. Advantage Estimation (GAE for 1-step episode)
        # Since contextual bandit has episode length 1, return is just the reward
        with torch.no_grad():
            values = self.value_net(states_t).squeeze(-1)
        advantages_t = rewards_t - values
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        dataset_size = len(query_ids)
        batch_size = 64
        
        mean_loss = 0.0
        mean_entropy = 0.0
        updates = 0
        
        # 3. PPO Optimization
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = indices[start:start+batch_size]
                
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                batch_old_log_probs = old_log_probs_t[idx]
                batch_rewards = rewards_t[idx]
                batch_advantages = advantages_t[idx]
                
                # New policy probs
                action_probs = self.agent.policynetwork(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO Clipped Surrogate Objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value network loss
                values_pred = self.value_net(batch_states).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(values_pred, batch_rewards)
                
                # Total loss
                loss = policy_loss - self.entropy_coeff * entropy
                
                # Policy step
                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policynetwork.parameters(), max_norm=1.0)
                self.agent.optimizer.step()
                
                # Value step
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
                self.value_optimizer.step()
                
                mean_loss += loss.item()
                mean_entropy += entropy.item()
                updates += 1
                
        return mean_reward, mean_loss / max(1, updates), action_counts, mean_entropy / max(1, updates)
        
    def _evaluate(self, use_val: bool = False) -> Tuple[float, float, float]:
        env = self.env_val if (use_val and self.env_val is not None) else self.env_test
        query_ids = env.query_pool

        total_reward = 0.0
        total_oracle_reward = 0.0
        oracle_matches = 0

        for qid in query_ids:
            raw_state = env.get_state(qid)
            state = self._normalize_state(raw_state)

            action, _ = self.agent.select_action(state, evaluate=True)

            reward = env.step(qid, action, target_metric=self.reward_type)
            total_reward += reward

            oracle_action = env.get_oracle_action(qid, target_metric=self.reward_type)
            oracle_reward = env.step(qid, oracle_action, target_metric=self.reward_type)
            total_oracle_reward += oracle_reward

            if action == oracle_action:
                oracle_matches += 1
        
        n = len(query_ids)
        return (
            total_reward / n if n > 0 else 0.0,
            total_oracle_reward / n if n > 0 else 0.0,
            oracle_matches / n if n > 0 else 0.0,
        )

    def _save_checkpoint(self, epoch: int, test_reward: float):
        checkpoint_dir = os.path.join(PROJECT_ROOT, "experiments", "router_training")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_ppo_model.pt")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.agent.policynetwork.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "test_reward": test_reward,
            "reward_type": self.reward_type
        }, checkpoint_path)