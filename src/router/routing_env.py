import json
import random
import zlib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def deterministic_tiebreak(query_id: Any, num_candidates: int) -> int:
    """Pick one of `num_candidates` options for `query_id`, reproducibly across processes.

    Python's built-in hash() is randomized per-process for str (PEP 456) unless
    PYTHONHASHSEED is pinned, so `tied[hash(qid) % len(tied)]` silently produces DIFFERENT
    oracle labels on every run. That makes every supervised baseline (Logistic Regression /
    Random Forest / Ridge / Gradient Boosting) irreproducible run-to-run, which is fatal for
    a paper's main results table. crc32 is a stable checksum — same input, same output,
    forever, on any machine.
    """
    if num_candidates <= 1:
        return 0
    return zlib.crc32(str(query_id).encode("utf-8")) % num_candidates

class RouterEnvironment:
    """
    Simulated Reinforcement Learning Environment for an LLM/RAG Router.
    Uses pre-computed query evaluations for sub-microsecond state-action steps.
    
    Supports two data formats:
      1. Pre-processed env data (from prepare_env_data.py) — fast, recommended
      2. Raw baseline_v2 per-query results JSON — auto-converted on load
    """

    # Action space: 0 → BM25, 1 → Dense, 2 → Hybrid
    ACTION_TO_RETRIEVER = {0: "bm25", 1: "dense", 2: "hybrid"}
    NUM_ACTIONS = 3

    def __init__(self, data_path: str, query_ids_subset: Optional[List[str]] = None):
        """
        1. __init__(data_path) - Loads the per-query JSON database.
        
        Args:
            data_path: Path to the structured query profile JSON file.
                       Supports both pre-processed env data and raw v2 results.
            query_ids_subset: Optional list of query IDs to restrict this environment to
                              (used to enforce clean Train/Test splits).
        """
        self.data_path = data_path
        
        # Load the master JSON lookup structure from storage
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        # Auto-detect format and normalize to the env lookup-table format
        self.master_table = self._normalize_data(raw_data)
            
        # Determine the active query pool for this split instance
        if query_ids_subset is not None:
            self.query_pool = [qid for qid in query_ids_subset if qid in self.master_table]
        else:
            self.query_pool = list(self.master_table.keys())
            
        # Track the active sequence index pointer inside the evaluation episode
        self.current_pointer = 0
    
    def _normalize_data(self, raw_data: Any) -> Dict[str, Dict[str, Any]]:
        """Auto-detect and normalize data format to the env lookup-table schema.
        
        Expected env format per query:
        {
            "query_id": {
                "features": [float, ...],
                "retrievers": {
                    "bm25":   {"hit_at_5": 0/1, "mrr": float, ...},
                    "dense":  {"hit_at_5": 0/1, "mrr": float, ...},
                    "hybrid": {"hit_at_5": 0/1, "mrr": float, ...}
                }
            }
        }
        """
        # Case 1: Already in env format (dict with string keys → dicts with "features")
        if isinstance(raw_data, dict):
            first_key = next(iter(raw_data), None)
            if first_key and isinstance(raw_data[first_key], dict):
                if "features" in raw_data[first_key]:
                    return raw_data
                    
        # Case 2: Raw baseline_v2 format (list of dicts with "question", "retrievers.*.metrics_at_k")
        if isinstance(raw_data, list) and len(raw_data) > 0 and "question" in raw_data[0]:
            return self._convert_v2_results(raw_data)
        
        # Case 3: Already dict format but without features (mock format from original code)
        if isinstance(raw_data, dict):
            return raw_data
            
        raise ValueError(
            f"Unrecognized data format in {self.data_path}. "
            "Expected either pre-processed env data (dict) or raw v2 results (list)."
        )
    
    def _convert_v2_results(self, results_list: list, k: str = "5") -> Dict[str, Dict[str, Any]]:
        """Convert raw baseline_v2 per-query results to env lookup-table format.
        
        This allows loading nq_per_query_results.json directly without
        running prepare_env_data.py first (useful for quick testing).
        Features are NOT populated here — use prepare_env_data.py for full features.
        """
        env_data = {}
        for entry in results_list:
            qid = str(entry["query_id"])
            retrievers = {}
            for ret_name in ["bm25", "dense", "hybrid"]:
                if ret_name in entry.get("retrievers", {}):
                    metrics_at_k = entry["retrievers"][ret_name].get("metrics_at_k", {})
                    # Extract metrics at the target k value
                    k_metrics = metrics_at_k.get(k, {})
                    retrievers[ret_name] = {
                        "hit_at_5": 1.0 if k_metrics.get("hit", False) else 0.0,
                        "mrr": float(k_metrics.get("mrr", 0.0)),
                        "f1": float(k_metrics.get("f1", 0.0)),
                        "recall": float(k_metrics.get("recall", 0.0)),
                        "em": float(k_metrics.get("em", 0.0)),
                    }
                    
            # Placeholder features (just query_id as float) — use prepare_env_data for real features
            env_data[qid] = {
                "question": entry.get("question", ""),
                "features": [],  # Empty until prepare_env_data fills them
                "retrievers": retrievers,
            }
        return env_data

    def reset(self) -> List[str]:
        """
        5. reset() - Shuffles the pool query sequence order to start a new training episode.
        
        Returns:
            The list of shuffled query IDs for the upcoming training episode.
        """
        random.shuffle(self.query_pool)
        self.current_pointer = 0
        return self.query_pool

    def get_state(self, query_id: str) -> List[float]:
        """
        2. get_state(query_id) - Fetches the numerical feature space vector for a query.
        
        Returns:
            A 1D numerical array list ready for neural policy network observation layers.
        """
        if query_id not in self.master_table:
            raise KeyError(f"Query ID '{query_id}' not found in loaded environment file.")
            
        features = self.master_table[query_id].get("features", [])
        if not features:
            raise ValueError(
                f"Query '{query_id}' has no features. Run prepare_env_data.py first "
                "to extract features from raw per-query results."
            )
        return features

    def step(self, query_id: str, action: int, target_metric: str = "hit_at_5") -> float:
        """
        3. step(query_id, action) - Evaluates a routing choice and returns its performance reward.
        
        Args:
            query_id: The query identifier.
            action: The retriever action (0=BM25, 1=Dense, 2=Hybrid).
            target_metric: The metric to use as reward. Options:
                - "hit_at_5": Binary hit/miss at k=5 (default, matches plan)
                - "mrr": Mean Reciprocal Rank (continuous, richer signal)
                - "f1": Token-level F1
        
        Returns:
            The float reward metric generated by the chosen retriever strategy.
        """
        if action not in self.ACTION_TO_RETRIEVER:
            raise ValueError(f"Action {action} is invalid. Choose 0 (BM25), 1 (Dense), or 2 (Hybrid).")
            
        retriever_name = self.ACTION_TO_RETRIEVER[action]
        
        # Step lookup execution tracking
        retriever_scores = self.master_table[query_id]["retrievers"][retriever_name]
        reward = float(retriever_scores.get(target_metric, 0.0))
        
        # Increment tracking pointer state
        self.current_pointer += 1
        
        return reward

    def get_oracle_action(self, query_id: str, target_metric: str = "hit_at_5") -> int:
        """
        4. get_oracle_action(query_id) - Evaluates the performance ceiling for analytical reporting.
        
        Returns:
            The integer action index of the absolute best-performing strategy on this query.

        Ties go through deterministic_tiebreak rather than first-index-wins. The previous
        `best_score = -1.0` with a strict `>` over {0: bm25, 1: dense, 2: hybrid} returned
        action 0 on every tie, and ties are not rare — on a sparse reward a large share of
        queries have every retriever equal. That biased the oracle_accuracy diagnostic in
        train_reinforce.py and its contrastive labels toward BM25, and disagreed with
        canonical_evaluation.py, which already tie-breaks this way when building supervised
        labels — two tie-break policies in one codebase.

        The oracle *value* is unaffected: max over rewards does not depend on which tied
        arm is named. Only the attribution changes.
        """
        retrievers_data = self.master_table[query_id]["retrievers"]

        scores = {action_idx: retrievers_data[retriever_name].get(target_metric, 0.0)
                  for action_idx, retriever_name in self.ACTION_TO_RETRIEVER.items()}
        best_score = max(scores.values())
        tied = sorted(action_idx for action_idx, s in scores.items() if s == best_score)

        if len(tied) == 1:
            return tied[0]
        return tied[deterministic_tiebreak(query_id, len(tied))]

    def get_all_rewards(self, query_id: str, target_metric: str = "hit_at_5") -> Dict[int, float]:
        """Returns rewards for ALL actions on a query (for oracle/analysis).
        
        Useful for computing the oracle upper bound and analyzing per-query behavior.
        """
        retrievers_data = self.master_table[query_id]["retrievers"]
        return {
            action_idx: float(retrievers_data[ret_name].get(target_metric, 0.0))
            for action_idx, ret_name in self.ACTION_TO_RETRIEVER.items()
        }

    def get_gold_type(self, query_id: str) -> str:
        """Whether this query's gold passage is a real Wikipedia snippet or a
        template-generated one ("real_wikipedia" / "constructed" / "unknown").

        Env data produced before gold_type was carried through prepare_env_data.py won't
        have the field inline; in that case this falls back to joining it from the
        baseline_v2 per-query results on query_id. Returns "unknown" if neither source
        has it, so callers can degrade gracefully rather than crash.
        """
        inline = self.master_table.get(query_id, {}).get("gold_type")
        if inline:
            return inline
        return self._gold_type_fallback().get(str(query_id), "unknown")

    def _gold_type_fallback(self) -> Dict[str, str]:
        """Lazily build a {query_id: gold_type} map from the baseline_v2 per-query results.

        Only used for env data generated before gold_type was propagated inline. Inferred
        from this env file's own name (e.g. nq_env_data.json -> nq_per_query_results.json)
        so it stays correct across datasets.
        """
        if hasattr(self, "_gold_type_map"):
            return self._gold_type_map

        self._gold_type_map: Dict[str, str] = {}
        try:
            import os
            base = os.path.basename(self.data_path).replace("_env_data.json", "")
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            results_path = os.path.join(
                project_root, "experiments", "baseline_v2", "results",
                f"{base}_per_query_results.json",
            )
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    for entry in json.load(f):
                        self._gold_type_map[str(entry.get("query_id"))] = entry.get("gold_type", "unknown")
        except Exception:
            pass  # best-effort only; callers handle "unknown"
        return self._gold_type_map

    @property
    def state_dim(self) -> int:
        """Returns the dimensionality of the feature vectors in this environment."""
        for qid in self.query_pool:
            features = self.master_table[qid].get("features", [])
            if features:
                return len(features)
        return 0

    def __len__(self) -> int:
        return len(self.query_pool)


# =====================================================================
# EXPERIMENTAL TRAIN / TEST SPLIT ORCHESTRATION PIPELINES (THE PAPERS SETUP)
# =====================================================================

def create_train_test_split(
    data_path: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[RouterEnvironment, RouterEnvironment]:
    """
    Setup 1: Within-Dataset Split.
    Splits a single dataset into train/test with reproducible randomization.
    
    Args:
        data_path: Path to the env data JSON.
        train_ratio: Fraction of queries for training (default 0.8).
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_env, test_env).
    """
    # Pre-load all available keys from target file to determine split boundary
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict):
        all_keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        all_keys = [str(entry.get("query_id", i)) for i, entry in enumerate(raw_data)]
    else:
        raise ValueError("Unsupported data format")
        
    random.seed(seed)
    random.shuffle(all_keys)
    
    split_idx = int(len(all_keys) * train_ratio)
    train_keys = all_keys[:split_idx]
    test_keys = all_keys[split_idx:]
    
    env_train = RouterEnvironment(data_path=data_path, query_ids_subset=train_keys)
    env_test = RouterEnvironment(data_path=data_path, query_ids_subset=test_keys)
    
    return env_train, env_test


def create_cross_dataset_envs(
    train_data_path: str,
    test_data_path: str
) -> Tuple[RouterEnvironment, RouterEnvironment]:
    """
    Setup 2: Cross-Dataset Transfer.
    Train on 100% of Dataset A, test on 100% of Dataset B.
    
    This is the generalization experiment that makes the paper strong.
    """
    env_train = RouterEnvironment(data_path=train_data_path)
    env_test = RouterEnvironment(data_path=test_data_path)
    return env_train, env_test


def split_query_ids(
    all_keys,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    split_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """The one true train/val/test partition of query ids.

    Extracted from create_train_val_test_split so prepare_env_data.py can fit feature
    transforms (StandardScaler, PCA) on the training split alone, using exactly the
    partition the environment will later use. If the two ever computed the split
    differently, the transform would be fitted on rows that end up in test -- which is the
    transductive leakage this exists to prevent.

    Uses a local random.Random rather than random.seed(): seeding the global RNG reaches
    outside this function and perturbs any later use of the module-level random stream.
    Both produce the same Mersenne Twister sequence, so the partition is unchanged.
    """
    keys = sorted(str(k) for k in all_keys)   # sorted() for determinism
    rng = random.Random(split_seed)
    rng.shuffle(keys)

    n = len(keys)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return keys[:train_end], keys[train_end:val_end], keys[val_end:]


def create_train_val_test_split(
    data_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    split_seed: int = 42
) -> Tuple[RouterEnvironment, RouterEnvironment, RouterEnvironment]:
    """Split into train/val/test with a FIXED split seed.
    
    The split_seed controls ONLY the data partitioning.
    Training seeds (for network init, action sampling) are set separately.
    This ensures all methods (RL, GBT, Ridge, etc.) are compared on the
    exact same test set regardless of how many seeds are used for training.
    
    Args:
        data_path: Path to the env data JSON.
        train_ratio: Fraction of queries for training (default 0.70).
        val_ratio: Fraction of queries for validation (default 0.15).
                   Test ratio = 1.0 - train_ratio - val_ratio.
        split_seed: Fixed seed for the data split (default 42).
    
    Returns:
        Tuple of (train_env, val_env, test_env).
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict):
        all_keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        all_keys = [str(entry.get("query_id", i)) for i, entry in enumerate(raw_data)]
    else:
        raise ValueError("Unsupported data format")

    train_keys, val_keys, test_keys = split_query_ids(
        all_keys, train_ratio=train_ratio, val_ratio=val_ratio, split_seed=split_seed)

    env_train = RouterEnvironment(data_path=data_path, query_ids_subset=train_keys)
    env_val = RouterEnvironment(data_path=data_path, query_ids_subset=val_keys)
    env_test = RouterEnvironment(data_path=data_path, query_ids_subset=test_keys)
    
    return env_train, env_val, env_test


if __name__ == "__main__":
    print("RouterEnvironment module loaded successfully.")
    print(f"Action space: {RouterEnvironment.ACTION_TO_RETRIEVER}")
    print(f"Number of actions: {RouterEnvironment.NUM_ACTIONS}")
