"""
17_latency_analysis.py — Measure and compare latency/cost per retriever.

Produces a table for the paper showing:
    - Mean latency (ms) per retriever
    - Router overhead (feature extraction + policy forward pass)
    - Total pipeline latency with/without routing
    - Cost-quality Pareto frontier analysis

Usage:
    cd <repo>
    python experiments/router_experiments/17_latency_analysis.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent, RouterPolicy
from src.router.routing_env import RouterEnvironment, create_train_test_split
from src.router.prepare_env_data import extract_lightweight_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NQ_PATH = os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json")


def measure_router_overhead(state_dim: int, hidden_dim: int = 128, num_trials: int = 1000):
    """Measure the wall-clock time for a single router decision.
    
    This includes:
    1. Feature extraction (lightweight, no spaCy)
    2. Feature normalization
    3. Policy network forward pass
    
    Returns:
        Dict with timing statistics in milliseconds.
    """
    print("Measuring router overhead...")
    
    # Create a policy network
    policy = RouterPolicy(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=3)
    policy.eval()
    
    # Pre-generate random states
    states = [torch.randn(1, state_dim) for _ in range(num_trials)]
    
    # Warm up
    for i in range(100):
        with torch.no_grad():
            _ = policy(states[i % num_trials])
    
    # Measure forward pass only
    forward_times = []
    for i in range(num_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = policy(states[i])
            _ = torch.argmax(probs, dim=-1).item()
        t1 = time.perf_counter()
        forward_times.append((t1 - t0) * 1000)  # Convert to ms
    
    # Measure feature extraction
    sample_query = "what is the capital of France and when was it founded"
    # Raw retrieval-score schema (see 02_run_baselines.py / prepare_env_data.py) — NOT
    # metrics_at_k/mrr, which would be the label-dependent (leaky) schema.
    mock_bm25 = {"top1_score": 7.2, "mean_score_top5": 4.5, "std_score_top5": 1.8}
    mock_dense = {"top1_score": 0.61, "mean_score_top5": 0.52, "std_score_top5": 0.05}
    
    feature_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        features = extract_lightweight_features(sample_query, mock_bm25, mock_dense, k="5")
        t1 = time.perf_counter()
        feature_times.append((t1 - t0) * 1000)
    
    # Measure full pipeline (feature extraction + normalization + forward pass)
    mean = np.random.randn(state_dim)
    std = np.abs(np.random.randn(state_dim)) + 0.01
    
    full_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        features = extract_lightweight_features(sample_query, mock_bm25, mock_dense, k="5")
        # Pad with zeros if features are shorter than state_dim (e.g. missing PCA
        # embedding dims).  This is a latency benchmark, so the padding values
        # don't affect correctness — only the wall-clock cost matters.
        feat_arr = np.array(features)
        if len(feat_arr) < state_dim:
            feat_arr = np.pad(feat_arr, (0, state_dim - len(feat_arr)))
        normed = ((feat_arr - mean) / std).tolist()
        state_t = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_t)
            action = torch.argmax(probs, dim=-1).item()
        t1 = time.perf_counter()
        full_times.append((t1 - t0) * 1000)
    
    return {
        "forward_pass_ms": {
            "mean": round(np.mean(forward_times), 4),
            "std": round(np.std(forward_times), 4),
            "p50": round(np.percentile(forward_times, 50), 4),
            "p95": round(np.percentile(forward_times, 95), 4),
            "p99": round(np.percentile(forward_times, 99), 4),
        },
        "feature_extraction_ms": {
            "mean": round(np.mean(feature_times), 4),
            "std": round(np.std(feature_times), 4),
            "p50": round(np.percentile(feature_times, 50), 4),
            "p95": round(np.percentile(feature_times, 95), 4),
        },
        "full_routing_decision_ms": {
            "mean": round(np.mean(full_times), 4),
            "std": round(np.std(full_times), 4),
            "p50": round(np.percentile(full_times, 50), 4),
            "p95": round(np.percentile(full_times, 95), 4),
        },
        "num_trials": num_trials,
    }


def measure_retriever_latencies(env_path: str):
    """Extract MEASURED per-retriever latency from the env data.

    Note: these are measured on the controlled 500-passage-per-query pool, NOT a
    production-scale index, so absolute values are not representative of a real
    deployment — only the router-overhead-vs-retriever-cost ratio is meaningful, and even
    that shifts at scale. Any latency claim in the paper must be stated against these
    measured numbers rather than round-number estimates.
    """
    with open(env_path, "r") as f:
        env_data = json.load(f)
    
    latencies = {"bm25": [], "dense": [], "hybrid": []}
    
    for qid, entry in env_data.items():
        for ret_name in ["bm25", "dense", "hybrid"]:
            lat = entry.get("retrievers", {}).get(ret_name, {}).get("latency_ms", None)
            if lat and lat > 0:
                latencies[ret_name].append(lat)
    
    results = {}
    for ret_name, lats in latencies.items():
        if lats:
            results[ret_name] = {
                "mean_ms": round(np.mean(lats), 2),
                "std_ms": round(np.std(lats), 2),
                "p50_ms": round(np.percentile(lats, 50), 2),
                "p95_ms": round(np.percentile(lats, 95), 2),
                "num_samples": len(lats),
            }
        else:
            # Typical latencies for RAG systems
            typical = {"bm25": 5.0, "dense": 45.0, "hybrid": 55.0}
            results[ret_name] = {
                "mean_ms": typical[ret_name],
                "std_ms": 0.0,
                "note": "estimated (no recorded latencies in env data)",
            }
    
    return results


def _measured_mean_latencies(env_path: str):
    """Mean measured per-retriever latency (ms) from the env data.

    Previously three different sets of retriever latencies existed in this codebase: the
    round-number literals {bm25: 5, dense: 45, hybrid: 55} hardcoded here, the values in
    the committed latency_analysis.json (bm25 4.78 / dense 1.11 / hybrid 1.06, computed on
    an older 496-query corpus), and what the current env data actually contains. Any paper
    claim of the form "router overhead is X% of retriever latency" is only checkable if
    there is exactly one source of truth — this function is it.
    """
    fallback = {"bm25": 5.0, "dense": 45.0, "hybrid": 55.0}
    if not os.path.exists(env_path):
        return fallback

    with open(env_path, "r") as f:
        env_data = json.load(f)

    out = {}
    for ret in ["bm25", "dense", "hybrid"]:
        vals = [e.get("retrievers", {}).get(ret, {}).get("latency_ms", 0.0)
                for e in env_data.values()]
        vals = [v for v in vals if v and v > 0]
        out[ret] = float(np.mean(vals)) if vals else fallback[ret]
    return out


def _measure_router_mrr(env_path: str, hidden_dim: int = 128):
    """Measure the trained router's actual test-set MRR (rather than hardcoding it).

    Falls back to the best-fixed retriever's MRR if no trained checkpoint exists, so the
    Pareto plot degrades gracefully instead of silently using a wrong literal.
    """
    env_train, env_test = create_train_test_split(env_path, train_ratio=0.8, seed=42)

    model_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "best_reinforce_model.pt")
    if not os.path.exists(model_path):
        print(f"  WARNING: no trained checkpoint at {model_path}; "
              "falling back to best-fixed MRR for the Pareto point.")
        best = max(
            sum(env_test.step(q, a, target_metric="mrr") for q in env_test.query_pool) / len(env_test)
            for a in range(3)
        )
        return best

    agent = PolicyAgent(input_dim=env_train.state_dim, hidden_dim=hidden_dim,
                        output_dim=3, learningrate=3e-3)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    # Same stale-checkpoint guard as estimate_router_latency() — see the note there.
    ckpt_in_dim = checkpoint["model_state_dict"]["fc1.weight"].shape[1]
    if ckpt_in_dim != env_train.state_dim:
        print(f"  WARNING: checkpoint input_dim={ckpt_in_dim} != current state_dim="
              f"{env_train.state_dim}; falling back to best-fixed MRR for the Pareto point.")
        return max(
            sum(env_test.step(q, a, target_metric="mrr") for q in env_test.query_pool) / len(env_test)
            for a in range(3)
        )
    agent.policynetwork.load_state_dict(checkpoint["model_state_dict"])

    all_features = [env_train.get_state(q) for q in env_train.query_pool]
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std[std < 1e-8] = 1.0

    total = 0.0
    for qid in env_test.query_pool:
        normed = ((np.array(env_test.get_state(qid)) - mean) / std).tolist()
        action, _ = agent.select_action(normed, evaluate=True)
        total += env_test.step(qid, action, target_metric="mrr")
    return total / len(env_test)


def compute_pareto_frontier(env_path: str, router_mrr: float, router_latency: float):
    """Compute the accuracy-latency Pareto frontier.
    
    Shows that the router achieves near-oracle accuracy with
    lower average latency than always using the most expensive retriever.
    """
    env = RouterEnvironment(data_path=env_path)
    
    typical_latencies = _measured_mean_latencies(env_path)
    
    points = []
    
    # Fixed retrievers
    for action in range(3):
        ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
        total_mrr = sum(env.step(qid, action, target_metric="mrr") for qid in env.query_pool)
        mean_mrr = total_mrr / len(env)
        
        points.append({
            "method": f"Always {ret_name.title()}",
            "mrr": round(mean_mrr, 4),
            "latency_ms": typical_latencies[ret_name],
            "is_pareto_optimal": False,  # Will be computed below
        })
    
    # Router
    points.append({
        "method": "RL Router",
        "mrr": round(router_mrr, 4),
        "latency_ms": round(router_latency, 2),
        "is_pareto_optimal": False,
    })
    
    # Oracle
    oracle_mrr_total = sum(
        env.step(qid, env.get_oracle_action(qid, "mrr"), target_metric="mrr")
        for qid in env.query_pool
    )
    oracle_mrr = oracle_mrr_total / len(env)
    
    # Oracle latency = weighted average based on oracle action distribution
    oracle_actions = [env.get_oracle_action(qid, "mrr") for qid in env.query_pool]
    oracle_latency = np.mean([
        typical_latencies[RouterEnvironment.ACTION_TO_RETRIEVER[a]] for a in oracle_actions
    ])
    
    points.append({
        "method": "Oracle",
        "mrr": round(oracle_mrr, 4),
        "latency_ms": round(oracle_latency, 2),
        "is_pareto_optimal": False,
    })
    
    # Compute Pareto optimality
    for i, p in enumerate(points):
        is_dominated = False
        for j, q in enumerate(points):
            if i != j:
                if q["mrr"] >= p["mrr"] and q["latency_ms"] <= p["latency_ms"]:
                    if q["mrr"] > p["mrr"] or q["latency_ms"] < p["latency_ms"]:
                        is_dominated = True
                        break
        points[i]["is_pareto_optimal"] = not is_dominated
    
    return points


def estimate_router_latency(env_path: str, router_overhead_ms: float):
    """Estimate the average latency of the router's chosen retriever.
    
    Uses the action distribution from training to compute weighted average.
    """
    env_train, env_test = create_train_test_split(env_path, train_ratio=0.8, seed=42)
    
    agent = PolicyAgent(
        input_dim=env_train.state_dim, hidden_dim=128,
        output_dim=3, learningrate=3e-3
    )
    
    # Load best model if available.
    # Guarded: a checkpoint left over from an earlier run can have a different input_dim
    # (e.g. a 10-feature prototype vs. today's 26-feature vector). Loading it blindly
    # raises a size-mismatch RuntimeError that kills this script — and, because this is a
    # stage in run_full_pipeline.py, would abort the whole multi-hour pipeline at the very
    # end. Fall back to an untrained policy instead; this function only needs an action
    # DISTRIBUTION to weight latencies, so degrading here is acceptable and self-reporting.
    model_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "best_reinforce_model.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        ckpt_in_dim = checkpoint["model_state_dict"]["fc1.weight"].shape[1]
        if ckpt_in_dim == env_train.state_dim:
            agent.policynetwork.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"  WARNING: checkpoint at {model_path} has input_dim={ckpt_in_dim} but the "
                  f"current env data has state_dim={env_train.state_dim} — it predates the current "
                  f"feature set. Ignoring it and using an untrained policy for the action "
                  f"distribution. Re-run training to refresh the checkpoint.")
    else:
        print("No trained model found, using random policy for latency estimation")
    
    # Compute action distribution on test set
    typical_latencies = _measured_mean_latencies(env_path)
    
    # Compute feature normalization stats
    all_features = [env_train.get_state(qid) for qid in env_train.query_pool]
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std[std < 1e-8] = 1.0
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    agent.policynetwork.eval()
    for qid in env_test.query_pool:
        state = env_test.get_state(qid)
        normed = ((np.array(state) - mean) / std).tolist()
        action, _ = agent.select_action(normed, evaluate=True)
        action_counts[action] += 1
    
    total = sum(action_counts.values())
    action_dist = {k: v/total for k, v in action_counts.items()}
    
    weighted_latency = sum(
        action_dist[a] * typical_latencies[RouterEnvironment.ACTION_TO_RETRIEVER[a]]
        for a in range(3)
    )
    
    return {
        "router_overhead_ms": router_overhead_ms,
        "weighted_retriever_latency_ms": round(weighted_latency, 2),
        "total_routed_latency_ms": round(router_overhead_ms + weighted_latency, 2),
        "action_distribution": {RouterEnvironment.ACTION_TO_RETRIEVER[k]: round(v, 3) 
                                for k, v in action_dist.items()},
    }


def main():
    print("="*60)
    print("LATENCY / COST ANALYSIS")
    print("="*60)
    
    # 1. Router overhead
    print("\n--- Router Decision Overhead ---")
    state_dim = 26  # Default feature dimension
    
    if os.path.exists(NQ_PATH):
        env = RouterEnvironment(data_path=NQ_PATH)
        state_dim = env.state_dim
    
    overhead = measure_router_overhead(state_dim)
    
    print(f"  Forward pass:   {overhead['forward_pass_ms']['mean']:.4f} ms "
          f"(p95: {overhead['forward_pass_ms']['p95']:.4f} ms)")
    print(f"  Features:       {overhead['feature_extraction_ms']['mean']:.4f} ms")
    print(f"  Full routing:   {overhead['full_routing_decision_ms']['mean']:.4f} ms "
          f"(p95: {overhead['full_routing_decision_ms']['p95']:.4f} ms)")
    
    # 2. Retriever latencies
    print("\n--- Retriever Latencies ---")
    if os.path.exists(NQ_PATH):
        ret_latencies = measure_retriever_latencies(NQ_PATH)
        for name, stats in ret_latencies.items():
            print(f"  {name:>8s}: {stats['mean_ms']:.2f} ms")
    else:
        ret_latencies = {}
        print("  No env data available for latency measurement")
    
    # 3. Router latency estimate
    print("\n--- Router Total Latency ---")
    if os.path.exists(NQ_PATH):
        router_latency = estimate_router_latency(
            NQ_PATH, overhead["full_routing_decision_ms"]["mean"]
        )
        print(f"  Router overhead:   {router_latency['router_overhead_ms']:.4f} ms")
        print(f"  Retriever (avg):   {router_latency['weighted_retriever_latency_ms']:.2f} ms")
        print(f"  Total:             {router_latency['total_routed_latency_ms']:.2f} ms")
        print(f"  Action dist:       {router_latency['action_distribution']}")
        
        # 4. Pareto frontier
        print("\n--- Pareto Frontier ---")
        # Router MRR is measured from the actual trained checkpoint here. It was
        # previously hardcoded to 0.435 ("from your results"), a stale literal that
        # matched no number in the current results and silently mis-placed the router on
        # the Pareto plot.
        measured_router_mrr = _measure_router_mrr(NQ_PATH)
        print(f"  (measured router MRR from checkpoint: {measured_router_mrr:.4f})")
        pareto = compute_pareto_frontier(
            NQ_PATH,
            router_mrr=measured_router_mrr,
            router_latency=router_latency["total_routed_latency_ms"]
        )
        
        for p in pareto:
            pareto_marker = "*" if p["is_pareto_optimal"] else " "
            print(f"  {pareto_marker} {p['method']:>20s}: MRR={p['mrr']:.4f}, "
                  f"Latency={p['latency_ms']:.1f}ms")
    else:
        router_latency = None
        pareto = None
    
    # 5. Model size comparison
    print("\n--- Model Size Comparison ---")
    model_sizes = {
        "REINFORCE Router (d=26)": sum(p.numel() for p in 
            RouterPolicy(26, 128, 3).parameters()),
        "Lean Router (d=4)": sum(p.numel() for p in 
            RouterPolicy(4, 128, 3).parameters()),
        "Logistic Regression": 26 * 3 + 3,  # weights + bias
        "Random Forest (100 trees)": "~5K nodes",
    }
    
    for name, size in model_sizes.items():
        if isinstance(size, int):
            print(f"  {name:>30s}: {size:,} params ({size*4/1024:.1f} KB)")
        else:
            print(f"  {name:>30s}: {size}")
    
    # --- Paper-ready overhead claim ---
    # The paper states router overhead is "< 2% of the fastest retriever's latency". That
    # number must be derived here, from measured values, not asserted — an earlier draft's
    # claim was off by ~11x against the actual measurements. This block computes the exact
    # sentence-ready figures so the claim is checkable against this JSON.
    measured = _measured_mean_latencies(NQ_PATH) if os.path.exists(NQ_PATH) else {}
    router_ms = overhead["full_routing_decision_ms"]["mean"]
    overhead_claim = {
        "router_overhead_ms": router_ms,
        "measured_retriever_mean_ms": {k: round(v, 3) for k, v in measured.items()},
        "overhead_as_pct_of_retriever": {
            k: round(router_ms / v * 100, 2) for k, v in measured.items() if v > 0
        },
    }
    if measured:
        fastest = min(measured, key=measured.get)
        pct_fastest = router_ms / measured[fastest] * 100
        overhead_claim["fastest_retriever"] = fastest
        overhead_claim["overhead_as_pct_of_fastest_retriever"] = round(pct_fastest, 2)
        overhead_claim["paper_sentence"] = (
            f"The router adds {router_ms:.3f}ms mean overhead, "
            f"{pct_fastest:.1f}% of the fastest retriever's measured latency "
            f"({fastest}, {measured[fastest]:.2f}ms) on the controlled 500-passage pool."
        )
        overhead_claim["caveat"] = (
            "Absolute retriever latencies are measured on a 500-passage-per-query pool, not "
            "a production-scale index; at real corpus scale retriever cost grows while router "
            "overhead stays constant, so this percentage is an UPPER bound on the router's "
            "relative cost. State it as such in the paper rather than quoting it bare."
        )
        print(f"\n--- Paper-ready overhead claim ---")
        print(f"  {overhead_claim['paper_sentence']}")

    # Save
    output = {
        "router_overhead": overhead,
        "retriever_latencies": ret_latencies,
        "router_total_latency": router_latency,
        "overhead_claim": overhead_claim,
        "pareto_frontier": pareto,
        "model_sizes": {k: v if isinstance(v, str) else v
                       for k, v in model_sizes.items()},
    }
    
    output_path = os.path.join(RESULTS_DIR, "latency_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
