"""
REINFORCE (Policy Gradient) Training Loop for the RL Retriever Router.

Core algorithm:
    For each episode (= one pass through all training queries):
        For each query in the shuffled training set:
            1. Get state (feature vector) from environment
            2. Sample action from policy network
            3. Get reward from environment lookup table
            4. Store (log_prob, reward) pair
        
        Compute advantage = reward - baseline (running mean)
        Policy gradient: loss = -mean(log_prob * advantage)
        Update policy network



Usage:
    python -m src.router.train_reinforce
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# Project root setup for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.compute_reward import compute_reward, compute_reward_from_metrics


# =====================================================================
# REINFORCE TRAINER
# =====================================================================

class ReinforceTrainer:
    """REINFORCE (vanilla policy gradient) trainer with baseline subtraction.
    
    Key design decisions (from the plan):
    - Baseline: running mean of rewards → reduces variance dramatically
    - Discount γ = 1.0 (each query is independent, no sequential dependency)
    - Action space: {BM25=0, Dense=1, Hybrid=2}
    - Reward: Hit@5 (binary) by default, MRR as ablation option
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
        reward_shaping: str = "absolute",
        contrastive_coeff: float = 0.0,
        seed: int = 42,
        env_val: Optional[RouterEnvironment] = None,
        advantage_mode: str = "per_query",
    ):
        """
        Args:
            env_train: Training environment (80% of queries).
            env_test: Test environment (20% of queries).
            agent: PolicyAgent wrapping the RouterPolicy network.
            reward_type: Metric to use as reward ("hit_at_5" or "mrr").
            cost_aware: Whether to subtract cost penalties from rewards.
            entropy_coeff: Weight for entropy bonus in loss (prevents action collapse).
                           Higher = more exploration. 0.05-0.2 is typical range.
            seed: Random seed for reproducibility.
            advantage_mode: How to turn rewards into advantages.

                "per_query" (default): A(q,a) = r(q,a) - mean over the query's OWN actions.
                    Retriever routing is a full-information offline bandit — the reward of
                    every action is known for every training query — so the per-query mean
                    is available for free and is the correct baseline.

                "global_ema": A(q,a) = r(q,a) - running mean over all queries. The original
                    implementation, kept so the ablation can quantify what it costs.

                Why this matters here specifically: on this data 70-83% of queries have all
                three retrievers scoring 0, so every action is tied. The correct advantage
                for a tied query is exactly 0 (no action was better). "global_ema" instead
                yields (0 - baseline) < 0, punishing whichever action was randomly sampled.
                That is a noise gradient on the large majority of queries, and because the
                sampled action is uniform the punishment is symmetric — it drives the policy
                toward uniform. Measured: RL router earned at or below a uniform random
                policy on all three datasets before this fix.
        """
        if advantage_mode not in ("per_query", "global_ema"):
            raise ValueError(f"advantage_mode must be 'per_query' or 'global_ema', "
                             f"got {advantage_mode!r}")
        self.env_train = env_train
        self.env_test = env_test
        self.env_val = env_val  # If provided, use for model selection instead of test
        self.agent = agent
        self.reward_type = reward_type
        self.cost_aware = cost_aware
        self.entropy_coeff = entropy_coeff
        self.reward_shaping = reward_shaping  # "absolute" or "pairwise"
        self.contrastive_coeff = contrastive_coeff  # 0.0 = disabled, 0.1 = typical
        self.advantage_mode = advantage_mode
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = 0.99  # Exponential moving average decay
        
        # Compute feature normalization stats from training data
        self._feature_mean, self._feature_std = self._compute_norm_stats(env_train)
        
        # Training history for plotting learning curves
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
        """Compute mean/std from training features for z-score normalization.
        
        Without normalization, features like query_length (1-20) dominate over
        binary features like has_quotes (0/1), causing the network to ignore
        important signals. This is critical for learning per-query routing.
        """
        all_features = []
        for qid in env.query_pool:
            all_features.append(env.get_state(qid))
        all_features = np.array(all_features)
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        # Avoid division by zero for constant features
        std[std < 1e-8] = 1.0
        return mean, std
    
    def _normalize_state(self, state: List[float]) -> List[float]:
        """Z-score normalize a feature vector using training set statistics.

        Handles dimension mismatches between datasets (e.g. NQ has 46 features,
        TriviaQA has 57) by padding with zeros or truncating to the training
        set's width.  Extra zero-padded columns are normalised to 0 (since
        mean=0 and std is set to 1 for constant features).
        """
        s = np.array(state, dtype=float)
        target_dim = len(self._feature_mean)
        if len(s) < target_dim:
            s = np.concatenate([s, np.zeros(target_dim - len(s))])
        elif len(s) > target_dim:
            s = s[:target_dim]
        return ((s - self._feature_mean) / self._feature_std).tolist()
    
    def train(self, num_epochs: int = 200, log_interval: int = 10) -> Dict:
        """Run the full REINFORCE training loop.
        
        Args:
            num_epochs: Number of training epochs (each epoch = full pass through training queries).
            log_interval: Print progress every N epochs.
            
        Returns:
            Training history dict with per-epoch metrics.
        """
        print(f"\n{'='*60}")
        print(f"REINFORCE Training")
        print(f"{'='*60}")
        print(f"Train queries: {len(self.env_train)}")
        print(f"Test queries:  {len(self.env_test)}")
        print(f"State dim:     {self.env_train.state_dim}")
        print(f"Reward type:   {self.reward_type}")
        print(f"Cost aware:    {self.cost_aware}")
        print(f"Epochs:        {num_epochs}")
        print(f"{'='*60}\n")
        
        best_test_reward = -float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # --- Training pass ---
            train_reward, train_loss, action_counts = self._train_epoch()
            
            # --- Evaluation pass (greedy, no exploration) ---
            test_reward, oracle_reward, oracle_acc = self._evaluate()
            
            # Store history
            self.history["epoch"].append(epoch)
            self.history["train_mean_reward"].append(train_reward)
            self.history["train_loss"].append(train_loss)
            self.history["test_mean_reward"].append(test_reward)
            self.history["test_oracle_reward"].append(oracle_reward)
            self.history["action_distribution"].append(action_counts.copy())
            self.history["oracle_accuracy"].append(oracle_acc)
            
            # --- Validation pass (for model selection) ---
            if self.env_val is not None:
                val_reward, _, _ = self._evaluate(use_val=True)
            else:
                val_reward = test_reward  # fallback: use test (backward compat)
            
            # Track best model on VALIDATION set (not test)
            if val_reward > best_test_reward:
                best_test_reward = val_reward
                self._save_checkpoint(epoch, val_reward)
            
            # Log progress
            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                total_actions = sum(action_counts.values())
                action_pcts = {k: f"{v/total_actions:.0%}" for k, v in action_counts.items()}
                
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
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Best test reward: {best_test_reward:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def _train_epoch(self) -> Tuple[float, float, Dict[int, int]]:
        """Run one training epoch (one full pass through training queries).
        
        Returns:
            (mean_reward, mean_loss, action_count_dict)
        """
        query_ids = self.env_train.reset()  # Shuffle query order
        
        epoch_log_probs = []
        epoch_entropies_grad = []  # Entropy WITH gradients for backprop
        epoch_rewards = []
        epoch_query_means = []  # per-query baselines, only filled when advantage_mode=per_query
        action_counts = {0: 0, 1: 0, 2: 0}
        
        for qid in query_ids:
            # 1. Get state (feature vector) — normalized
            raw_state = self.env_train.get_state(qid)
            state = self._normalize_state(raw_state)
            
            # 2. Sample action from policy (stochastic for exploration)
            action, log_prob = self.agent.select_action(state, evaluate=False)
            
            # 3. Compute entropy WITH gradients for regularization (full batch)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = self.agent.policynetwork(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy()
            epoch_entropies_grad.append(entropy)
            
            # 4. Get reward — use pairwise reward if enabled
            if self.reward_shaping == "pairwise":
                all_rewards = self.env_train.get_all_rewards(qid, target_metric=self.reward_type)
                from src.router.compute_reward import compute_pairwise_reward
                reward = compute_pairwise_reward(all_rewards, action, cost_aware=self.cost_aware)
            else:
                reward = self.env_train.step(qid, action, target_metric=self.reward_type)
                if self.cost_aware:
                    from src.router.compute_reward import COST_PENALTIES
                    reward -= COST_PENALTIES.get(action, 0.0)
            
            epoch_log_probs.append(log_prob)
            epoch_rewards.append(reward)
            # Per-query baseline: the mean reward over this query's OWN actions. Available
            # for free because every action's reward is known offline for training queries.
            if self.advantage_mode == "per_query":
                q_rewards = self.env_train.get_all_rewards(qid, target_metric=self.reward_type)
                if self.cost_aware:
                    from src.router.compute_reward import COST_PENALTIES
                    q_rewards = {a: r - COST_PENALTIES.get(a, 0.0) for a, r in q_rewards.items()}
                epoch_query_means.append(float(np.mean(list(q_rewards.values()))))
            action_counts[action] += 1

        # Compute advantages with baseline subtraction
        mean_reward = np.mean(epoch_rewards)

        # Update running baseline (exponential moving average). Kept current in both modes
        # so switching advantage_mode does not change any logged diagnostic.
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward

        # Policy gradient update: loss = -mean(log_prob * advantage) - entropy_coeff * entropy
        if self.advantage_mode == "per_query":
            # Ties (all actions equal) give advantage exactly 0 -> no gradient, which is
            # correct: no action was better. See advantage_mode in __init__ for why the
            # global baseline actively injects noise on this data.
            advantages = [r - m for r, m in zip(epoch_rewards, epoch_query_means)]
        else:
            advantages = [r - self.baseline for r in epoch_rewards]
        
        # Batch update (more stable than per-query updates)
        self.agent.optimizer.zero_grad()
        
        policy_loss = torch.tensor(0.0, requires_grad=True)
        for log_prob, advantage in zip(epoch_log_probs, advantages):
            policy_loss = policy_loss + (-log_prob.squeeze() * advantage)
        
        policy_loss = policy_loss / len(epoch_log_probs)
        
        # Entropy bonus from full batch (no subset sampling)
        entropy_bonus = torch.stack(epoch_entropies_grad).mean()
        mean_entropy = entropy_bonus.item()
        
        # Contrastive loss for feature space separation
        contrastive_loss = torch.tensor(0.0)
        if hasattr(self, 'contrastive_coeff') and self.contrastive_coeff > 0:
            contrastive_loss = self._compute_contrastive_loss(query_ids)
        
        # Total loss = policy gradient loss - entropy bonus + contrastive loss
        total_loss = policy_loss - self.entropy_coeff * entropy_bonus + getattr(self, 'contrastive_coeff', 0.0) * contrastive_loss
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.agent.policynetwork.parameters(), max_norm=1.0)
        
        self.agent.optimizer.step()
        
        self.history["entropy"].append(mean_entropy)
        
        return mean_reward, total_loss.item(), action_counts
    
    def _evaluate(self, use_val: bool = False) -> Tuple[float, float, float]:
        """Evaluate the current policy on the validation or test set (greedy).
        
        Args:
            use_val: If True, evaluate on self.env_val. Otherwise, self.env_test.
        
        Returns:
            (mean_reward, oracle_reward, oracle_accuracy)
        """
        env = self.env_val if (use_val and self.env_val is not None) else self.env_test
        query_ids = env.query_pool  # Don't shuffle eval set
        
        total_reward = 0.0
        total_oracle_reward = 0.0
        oracle_matches = 0
        
        for qid in query_ids:
            raw_state = env.get_state(qid)
            state = self._normalize_state(raw_state)
            
            # Greedy action (argmax, no sampling)
            action, _ = self.agent.select_action(state, evaluate=True)
            
            # Get reward for chosen action
            reward = env.step(qid, action, target_metric=self.reward_type)
            total_reward += reward
            
            # Get oracle reward for comparison
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

    def _compute_contrastive_loss(self, query_ids, num_pairs=50):
        """Compute contrastive loss on query pairs with different oracle actions.
        
        Pushes apart hidden representations of queries that need different retrievers.
        Pulls together representations of queries that need the same retriever.
        
        Uses the penultimate hidden layer of the policy network as the representation.
        """
        # Collect (hidden_repr, oracle_action) pairs
        reprs = []
        oracle_actions = []
        
        sampled_ids = random.sample(query_ids, min(num_pairs * 2, len(query_ids)))
        
        for qid in sampled_ids:
            raw_state = self.env_train.get_state(qid)
            state = self._normalize_state(raw_state)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Get penultimate layer representation (before output layer)
            x = torch.relu(self.agent.policynetwork.fc1(state_t))
            x = torch.relu(self.agent.policynetwork.fc2(x))
            reprs.append(x.squeeze(0))
            
            oracle_action = self.env_train.get_oracle_action(qid, target_metric=self.reward_type)
            oracle_actions.append(oracle_action)
        
        if len(reprs) < 4:
            return torch.tensor(0.0, requires_grad=True)
        
        reprs = torch.stack(reprs)  # (N, hidden_dim)
        
        # Compute pairwise cosine similarity
        reprs_norm = torch.nn.functional.normalize(reprs, dim=1)
        sim_matrix = torch.mm(reprs_norm, reprs_norm.t())  # (N, N)
        
        # Create label matrix: 1 if same oracle action, 0 if different
        oracle_t = torch.tensor(oracle_actions)
        label_matrix = (oracle_t.unsqueeze(0) == oracle_t.unsqueeze(1)).float()
        
        # Contrastive loss: pull same-class pairs together, push different apart
        # Using NT-Xent style loss
        temperature = 0.5
        sim_matrix = sim_matrix / temperature
        
        # Mask diagonal (self-similarity)
        mask = torch.eye(len(reprs)).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # For each anchor, positive = same oracle action, negative = different
        loss = torch.tensor(0.0, requires_grad=True)
        count = 0
        
        for i in range(len(reprs)):
            positives = (label_matrix[i] == 1) & ~mask[i]
            negatives = (label_matrix[i] == 0) & ~mask[i]
            
            if positives.sum() == 0 or negatives.sum() == 0:
                continue
            
            pos_sim = sim_matrix[i][positives].mean()
            neg_sim = sim_matrix[i][negatives].mean()
            
            # Margin loss: want pos_sim > neg_sim + margin
            margin = 0.5
            pair_loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
            loss = loss + pair_loss
            count += 1
        
        return loss / max(count, 1)
    
    def _save_checkpoint(self, epoch: int, test_reward: float):
        """Save the best model checkpoint."""
        checkpoint_dir = os.path.join(PROJECT_ROOT, "experiments", "router_training")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_reinforce_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.agent.policynetwork.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "test_reward": test_reward,
            "reward_type": self.reward_type,
            "baseline": self.baseline,
        }, checkpoint_path)
    
    def save_history(self, path: Optional[str] = None):
        """Save training history to JSON for plotting learning curves."""
        if path is None:
            results_dir = os.path.join(PROJECT_ROOT, "experiments", "router_training")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, "reinforce_training_history.json")
        
        # Convert numpy types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            if isinstance(values, list):
                serializable_history[key] = [
                    float(v) if isinstance(v, (np.floating, float)) else v
                    for v in values
                ]
            else:
                serializable_history[key] = values
        
        with open(path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training history saved to: {path}")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    """End-to-end REINFORCE training pipeline."""
    
    # --- Configuration ---
    REWARD_TYPE = "mrr"          # MRR gives richer continuous gradient signal vs binary hit
    COST_AWARE = False           # Set True for cost-aware ablation
    NUM_EPOCHS = 300             # Training epochs (more room for exploration)
    HIDDEN_DIM = 128             # Policy network hidden layer size
    LEARNING_RATE = 3e-3         # Slightly higher LR to escape local optima faster
    ENTROPY_COEFF = 0.15         # Entropy bonus weight (prevents action collapse)
    SEED = 42                    # Random seed
    LOG_INTERVAL = 20            # Print every N epochs
    
    # --- Data paths ---
    nq_env_path = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")
    
    if not os.path.exists(nq_env_path):
        print(f"ERROR: Environment data not found at {nq_env_path}")
        print("Run 'python -m src.router.prepare_env_data' first to prepare the data.")
        sys.exit(1)
    
    # --- Create train/test environments ---
    print("Loading environment data...")
    env_train, env_test = create_train_test_split(
        data_path=nq_env_path,
        train_ratio=0.8,
        seed=SEED
    )
    
    state_dim = env_train.state_dim
    num_actions = RouterEnvironment.NUM_ACTIONS
    
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {num_actions}")
    
    # --- Initialize policy agent ---
    agent = PolicyAgent(
        input_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=num_actions,
        learningrate=LEARNING_RATE
    )
    
    # Print model size
    total_params = sum(p.numel() for p in agent.policynetwork.parameters())
    print(f"Policy network parameters: {total_params:,}")
    
    # --- Create trainer and run ---
    trainer = ReinforceTrainer(
        env_train=env_train,
        env_test=env_test,
        agent=agent,
        reward_type=REWARD_TYPE,
        cost_aware=COST_AWARE,
        entropy_coeff=ENTROPY_COEFF,
        seed=SEED,
    )
    
    history = trainer.train(num_epochs=NUM_EPOCHS, log_interval=LOG_INTERVAL)
    
    # --- Save results ---
    trainer.save_history()
    
    # --- Print final summary ---
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    # Random baseline comparison
    random_reward = sum(
        env_test.step(qid, random.randint(0, 2), target_metric=REWARD_TYPE)
        for qid in env_test.query_pool
    ) / len(env_test)
    
    # Best fixed retriever baseline
    best_fixed_rewards = {}
    for action in range(num_actions):
        total = sum(
            env_test.step(qid, action, target_metric=REWARD_TYPE)
            for qid in env_test.query_pool
        )
        best_fixed_rewards[action] = total / len(env_test)
    
    best_fixed_action = max(best_fixed_rewards, key=best_fixed_rewards.get)
    best_fixed_reward = best_fixed_rewards[best_fixed_action]
    
    final_test_reward = history["test_mean_reward"][-1]
    final_oracle_reward = history["test_oracle_reward"][-1]
    
    print(f"Random Baseline:     {random_reward:.4f}")
    print(f"Best Fixed ({ReinforceTrainer.RETRIEVER_NAMES[best_fixed_action]:>6s}): {best_fixed_reward:.4f}")
    print(f"REINFORCE Router:    {final_test_reward:.4f}")
    print(f"Oracle Upper Bound:  {final_oracle_reward:.4f}")
    print(f"{'='*60}")
    
    gap_from_random = final_test_reward - random_reward
    gap_from_fixed = final_test_reward - best_fixed_reward
    oracle_gap_captured = ((final_test_reward - best_fixed_reward) / 
                          (final_oracle_reward - best_fixed_reward) * 100
                          if final_oracle_reward > best_fixed_reward else 0)
    
    print(f"\nImprovement over Random:     {gap_from_random:+.4f}")
    print(f"Improvement over Best-Fixed: {gap_from_fixed:+.4f}")
    print(f"Oracle headroom captured:    {oracle_gap_captured:.1f}%")


if __name__ == "__main__":
    main()