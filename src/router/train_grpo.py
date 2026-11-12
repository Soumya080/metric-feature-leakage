"""
GRPO (Group Relative Policy Optimization) Training Loop for the RL Retriever Router.

Inspired by DeepSeek-R1 (arXiv:2501.12948).

Core algorithm:
    For each epoch:
        For each query:
            1. Sample K candidate actions from the policy
            2. Compute rewards for ALL K actions
            3. Normalize rewards within the group: advantage = (r - mean(R)) / (std(R) + eps)
            4. Update policy: loss = -mean(log_prob * group_advantage)

Key differences from REINFORCE and PPO:
    - NO critic/value network needed (critic-free)
    - Uses group-relative normalization instead of a baseline
    - More stable than REINFORCE (reduced variance through group statistics)
    - Simpler than PPO (no clipping, no value network, fewer hyperparameters)
    - Inspired by DeepSeek-R1's approach to training reasoning models

Usage:
    python -m src.router.train_grpo
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import RouterPolicy, PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.compute_reward import COST_PENALTIES


# =====================================================================
# GRPO TRAINER
# =====================================================================

class GRPOTrainer:
    """Group Relative Policy Optimization trainer for the retriever router.
    
    Following the DeepSeek-R1 paradigm:
    - For each query, sample K actions from the current policy
    - Compute the reward for each sampled action
    - Normalize rewards within the group (zero mean, unit variance)
    - Use group-normalized advantages for policy gradient update
    
    This eliminates the need for a critic network entirely while providing
    much lower variance than vanilla REINFORCE with a running baseline.
    """
    
    RETRIEVER_NAMES = {0: "BM25", 1: "Dense", 2: "Hybrid"}
    
    def __init__(
        self,
        env_train: RouterEnvironment,
        env_test: RouterEnvironment,
        agent: PolicyAgent,
        reward_type: str = "mrr",
        cost_aware: bool = False,
        entropy_coeff: float = 0.1,
        group_size: int = 8,
        kl_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        env_val: Optional[RouterEnvironment] = None,
    ):
        """
        Args:
            env_train: Training environment.
            env_test: Test environment.
            agent: PolicyAgent wrapping the RouterPolicy network.
            reward_type: Metric to use as reward ("hit_at_5" or "mrr").
            cost_aware: Whether to subtract cost penalties.
            entropy_coeff: Weight for entropy bonus.
            group_size: Number of actions to sample per query (K). 
                        For 3-action space, 8 gives good coverage with repeats.
            kl_coeff: KL divergence penalty coefficient (prevents policy from 
                      diverging too far from the reference/old policy).
            max_grad_norm: Maximum gradient norm for clipping.
            seed: Random seed.
        """
        self.env_train = env_train
        self.env_test = env_test
        self.env_val = env_val  # if provided, best-checkpoint selection uses this, not test
        self.agent = agent
        self.reward_type = reward_type
        self.cost_aware = cost_aware
        self.entropy_coeff = entropy_coeff
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Feature normalization
        self._feature_mean, self._feature_std = self._compute_norm_stats(env_train)
        
        # Store reference policy for KL penalty (updated periodically)
        self._update_reference_policy()
        
        # Training history
        self.history = {
            "epoch": [],
            "train_mean_reward": [],
            "train_loss": [],
            "test_mean_reward": [],
            "test_oracle_reward": [],
            "action_distribution": [],
            "oracle_accuracy": [],
            "entropy": [],
            "kl_divergence": [],
        }
    
    def _compute_norm_stats(self, env: RouterEnvironment):
        """Compute mean/std for z-score normalization."""
        all_features = [env.get_state(qid) for qid in env.query_pool]
        all_features = np.array(all_features)
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        std[std < 1e-8] = 1.0
        return mean, std
    
    def _normalize_state(self, state: List[float]) -> List[float]:
        return ((np.array(state) - self._feature_mean) / self._feature_std).tolist()
    
    def _update_reference_policy(self):
        """Snapshot the current policy as the reference for KL penalty."""
        self.ref_policy = RouterPolicy(
            input_dim=self.agent.policynetwork.fc1.in_features,
            hidden_dim=self.agent.policynetwork.fc1.out_features,
            output_dim=self.agent.policynetwork.fc3.out_features,
        )
        self.ref_policy.load_state_dict(self.agent.policynetwork.state_dict())
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    def train(self, num_epochs: int = 300, log_interval: int = 10,
              ref_update_interval: int = 50) -> Dict:
        """Run the full GRPO training loop.
        
        Args:
            num_epochs: Number of training epochs.
            log_interval: Print progress every N epochs.
            ref_update_interval: Update reference policy every N epochs.
            
        Returns:
            Training history dict.
        """
        print(f"\n{'='*60}")
        print(f"GRPO Training (DeepSeek-R1 Style)")
        print(f"{'='*60}")
        print(f"Train queries:  {len(self.env_train)}")
        print(f"Test queries:   {len(self.env_test)}")
        print(f"State dim:      {self.env_train.state_dim}")
        print(f"Reward type:    {self.reward_type}")
        print(f"Group size (K): {self.group_size}")
        print(f"KL coeff:       {self.kl_coeff}")
        print(f"Entropy coeff:  {self.entropy_coeff}")
        print(f"{'='*60}\n")
        
        best_test_reward = -float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train one epoch with GRPO
            train_reward, train_loss, action_counts, mean_entropy, mean_kl = (
                self._train_epoch()
            )
            
            # Evaluate on test set
            test_reward, oracle_reward, oracle_acc = self._evaluate()
            
            # Periodically update reference policy
            if epoch % ref_update_interval == 0:
                self._update_reference_policy()
            
            # Store history
            self.history["epoch"].append(epoch)
            self.history["train_mean_reward"].append(train_reward)
            self.history["train_loss"].append(train_loss)
            self.history["test_mean_reward"].append(test_reward)
            self.history["test_oracle_reward"].append(oracle_reward)
            self.history["action_distribution"].append(action_counts.copy())
            self.history["oracle_accuracy"].append(oracle_acc)
            self.history["entropy"].append(mean_entropy)
            self.history["kl_divergence"].append(mean_kl)
            
            # Track best model — on the VALIDATION set when available. Selecting the best
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
            
            # Log
            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                total_a = sum(action_counts.values())
                a_pcts = {k: f"{v/total_a:.0%}" for k, v in action_counts.items()}
                
                print(
                    f"Epoch {epoch:4d}/{num_epochs} | "
                    f"Train R={train_reward:.4f} | "
                    f"Test R={test_reward:.4f} | "
                    f"Oracle={oracle_reward:.4f} | "
                    f"OracleAcc={oracle_acc:.1%} | "
                    f"KL={mean_kl:.4f} | "
                    f"Actions={a_pcts} | "
                    f"Loss={train_loss:.4f} | "
                    f"{elapsed:.0f}s"
                )
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"GRPO Training complete in {total_time:.1f}s")
        print(f"Best test reward: {best_test_reward:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def _train_epoch(self) -> Tuple[float, float, Dict[int, int], float, float]:
        """Run one GRPO training epoch.
        
        For each query:
            1. Sample K actions from current policy
            2. Evaluate rewards for all K samples
            3. Compute group-relative advantages
            4. Accumulate policy gradient
        
        Returns:
            (mean_reward, loss, action_counts, mean_entropy, mean_kl)
        """
        query_ids = self.env_train.reset()
        action_counts = {0: 0, 1: 0, 2: 0}
        
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        all_kl_divs = []
        epoch_rewards = []
        
        self.agent.policynetwork.train()
        
        for qid in query_ids:
            raw_state = self.env_train.get_state(qid)
            state = self._normalize_state(raw_state)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Get current policy probabilities
            action_probs = self.agent.policynetwork(state_tensor)  # (1, 3)
            dist = torch.distributions.Categorical(action_probs)
            
            # Get reference policy probabilities for KL
            with torch.no_grad():
                ref_probs = self.ref_policy(state_tensor)
            
            # Compute KL divergence: KL(π_new || π_ref)
            kl_div = torch.distributions.kl_divergence(
                torch.distributions.Categorical(action_probs.detach()),
                torch.distributions.Categorical(ref_probs)
            ).item()
            all_kl_divs.append(kl_div)
            
            # Entropy of current policy
            entropy = dist.entropy()
            all_entropies.append(entropy.item())
            
            # === GRPO Core: Sample K actions and compute group-relative advantages ===
            
            # Sample K actions from the policy
            sampled_actions = dist.sample((self.group_size,))  # (K,)
            sampled_log_probs = dist.log_prob(sampled_actions)  # (K,)
            
            # Compute reward for each sampled action
            group_rewards = []
            for a in sampled_actions:
                a_int = a.item()
                r = self.env_train.step(qid, a_int, target_metric=self.reward_type)
                if self.cost_aware:
                    r -= COST_PENALTIES.get(a_int, 0.0)
                group_rewards.append(r)
            
            group_rewards = np.array(group_rewards)
            
            # Group-relative advantage normalization
            # A_i = (r_i - mean(R)) / (std(R) + eps)
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            group_advantages = (group_rewards - group_mean) / group_std
            
            # Store for batch update
            for k in range(self.group_size):
                all_log_probs.append(sampled_log_probs[k])
                all_advantages.append(group_advantages[k])
            
            # Track the greedy action for action distribution
            greedy_action = torch.argmax(action_probs, dim=-1).item()
            action_counts[greedy_action] += 1
            
            # Track reward of greedy action
            greedy_reward = self.env_train.step(qid, greedy_action, target_metric=self.reward_type)
            epoch_rewards.append(greedy_reward)
        
        # === Batch policy gradient update ===
        self.agent.optimizer.zero_grad()
        
        # GRPO loss: -mean(log_prob * advantage) - entropy_bonus + kl_penalty
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        
        policy_loss = torch.tensor(0.0, requires_grad=True)
        for log_prob, advantage in zip(all_log_probs, advantages_tensor):
            policy_loss = policy_loss + (-log_prob * advantage)
        policy_loss = policy_loss / len(all_log_probs)
        
        # Entropy bonus (recompute with gradients on a subset)
        entropy_bonus = torch.tensor(0.0, requires_grad=True)
        sample_qids = random.sample(query_ids, min(50, len(query_ids)))
        for qid in sample_qids:
            raw_s = self.env_train.get_state(qid)
            s = self._normalize_state(raw_s)
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            probs = self.agent.policynetwork(s_t)
            ent = -(probs * torch.log(probs + 1e-8)).sum()
            entropy_bonus = entropy_bonus + ent
        entropy_bonus = entropy_bonus / len(sample_qids)
        
        # KL penalty (prevents diverging too far from reference)
        mean_kl = np.mean(all_kl_divs)
        kl_penalty = self.kl_coeff * mean_kl
        
        total_loss = policy_loss - self.entropy_coeff * entropy_bonus + kl_penalty
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.agent.policynetwork.parameters(), self.max_grad_norm
        )
        self.agent.optimizer.step()
        
        mean_reward = np.mean(epoch_rewards)
        mean_entropy = np.mean(all_entropies)
        
        return mean_reward, total_loss.item(), action_counts, mean_entropy, mean_kl
    
    def _evaluate(self, use_val: bool = False) -> Tuple[float, float, float]:
        """Evaluate on the validation or test set (greedy)."""
        env = self.env_val if (use_val and self.env_val is not None) else self.env_test
        query_ids = env.query_pool
        total_reward = 0.0
        total_oracle_reward = 0.0
        oracle_matches = 0

        self.agent.policynetwork.eval()

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
        
        path = os.path.join(checkpoint_dir, "best_grpo_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.agent.policynetwork.state_dict(),
            "test_reward": test_reward,
            "reward_type": self.reward_type,
        }, path)
    
    def save_history(self, path: Optional[str] = None):
        if path is None:
            results_dir = os.path.join(PROJECT_ROOT, "experiments", "router_training")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, "grpo_training_history.json")
        
        serializable = {}
        for key, values in self.history.items():
            if isinstance(values, list):
                serializable[key] = [
                    float(v) if isinstance(v, (np.floating, float)) else v
                    for v in values
                ]
            else:
                serializable[key] = values
        
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"GRPO training history saved to: {path}")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    """End-to-end GRPO training pipeline."""
    
    REWARD_TYPE = "mrr"
    NUM_EPOCHS = 300
    HIDDEN_DIM = 128
    LEARNING_RATE = 3e-3
    ENTROPY_COEFF = 0.15
    GROUP_SIZE = 8  # K: number of samples per query
    KL_COEFF = 0.01
    SEED = 42
    LOG_INTERVAL = 20
    
    nq_env_path = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
    
    if not os.path.exists(nq_env_path):
        print(f"ERROR: Environment data not found at {nq_env_path}")
        print("Run 'python -m src.router.prepare_env_data' first.")
        sys.exit(1)
    
    print("Loading environment data...")
    env_train, env_test = create_train_test_split(nq_env_path, train_ratio=0.8, seed=SEED)
    
    state_dim = env_train.state_dim
    num_actions = RouterEnvironment.NUM_ACTIONS
    
    agent = PolicyAgent(
        input_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=num_actions,
        learningrate=LEARNING_RATE
    )
    
    total_params = sum(p.numel() for p in agent.policynetwork.parameters())
    print(f"Policy parameters: {total_params:,}")
    print(f"No value network needed (critic-free!)")
    
    trainer = GRPOTrainer(
        env_train=env_train,
        env_test=env_test,
        agent=agent,
        reward_type=REWARD_TYPE,
        entropy_coeff=ENTROPY_COEFF,
        group_size=GROUP_SIZE,
        kl_coeff=KL_COEFF,
        seed=SEED,
    )
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=LOG_INTERVAL)
    trainer.save_history()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL GRPO RESULTS")
    print(f"{'='*60}")
    
    best_fixed_rewards = {}
    for action in range(num_actions):
        total = sum(
            env_test.step(qid, action, target_metric=REWARD_TYPE)
            for qid in env_test.query_pool
        )
        best_fixed_rewards[action] = total / len(env_test)
    
    best_fixed_action = max(best_fixed_rewards, key=best_fixed_rewards.get)
    best_fixed_reward = best_fixed_rewards[best_fixed_action]
    
    final_test = history["test_mean_reward"][-1]
    final_oracle = history["test_oracle_reward"][-1]
    
    print(f"Best Fixed ({GRPOTrainer.RETRIEVER_NAMES[best_fixed_action]:>6s}): {best_fixed_reward:.4f}")
    print(f"GRPO Router:      {final_test:.4f}")
    print(f"Oracle:           {final_oracle:.4f}")
    print(f"Improvement:      {final_test - best_fixed_reward:+.4f}")


if __name__ == "__main__":
    main()
