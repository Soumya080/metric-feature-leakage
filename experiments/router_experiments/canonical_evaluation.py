#!/usr/bin/env python3
"""
Main routing evaluation (Table 3).

Fixed retrievers, heuristic routers, tuned supervised routers and a REINFORCE router (15
seeds) on one train/val/test split per dataset, with paired bootstrap tests against the best
fixed retriever. Writes results/canonical_results.json.
"""

import argparse
import os
import sys
import json
import random
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.routing_env import create_train_val_test_split, RouterEnvironment, deterministic_tiebreak
from src.router.policy_network import PolicyAgent
from src.router.train_reinforce import ReinforceTrainer
from src.router.provenance import build as build_provenance, summarize as summarize_provenance

DATASETS = {
    "nq": os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
    "triviaqa": os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
    "hotpotqa": os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
}


def get_dataset_paths(suffix: str = ""):
    """Dataset name -> env-data path, optionally targeting a suffixed variant produced by
    `prepare_env_data.py --suffix`, e.g. suffix='_leaky' ->
    src/router/nq_env_data_leaky.json. Used by leakage_ablation.py to evaluate the leaky/
    honest_minimal/honest_calibrated arms against the exact same pipeline. DATASETS itself
    is left untouched (failure_mode_analysis.py imports it directly)."""
    if not suffix:
        return dict(DATASETS)
    return {name: p.replace("_env_data.json", f"_env_data{suffix}.json")
            for name, p in DATASETS.items()}

SEEDS = [42, 123, 456, 789, 1024, 2025, 31337, 8675309, 90210, 1729,
         271828, 161803, 512, 65537, 20260829]  # 15 seeds — matches the table caption
SPLIT_SEED = 42
NUM_EPOCHS = 150
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
# Entropy bonus keeps the policy from collapsing to one action. It has to be small
# relative to the advantage signal, and on this data most queries are ties contributing
# advantage 0, so an entropy term that is too large simply holds the policy at uniform.
# Overridable via --entropy-coeff; tune on the validation split, never on test.
ENTROPY_COEFF = 0.01
REWARD_TYPE = 'mrr'
# "per_query" subtracts each query's own mean reward, so tied queries (all retrievers
# equal, ~70-83% of this corpus) contribute exactly zero gradient. "global_ema" is the
# original global-baseline behaviour, kept for the ablation that quantifies the difference.
ADVANTAGE_MODE = 'per_query'

def compute_static_baselines(env: RouterEnvironment, metric: str = 'mrr'):
    """Always BM25, Always Dense, Always Hybrid, and Oracle.

    Returns (means, per_query) — per_query holds the raw per-query score lists (aligned
    with env.query_pool) so downstream analyses (e.g. gold_type stratification, extra
    significance tests) can be run post-hoc without recomputing or retraining anything.
    """
    per_query = {"bm25": [], "dense": [], "hybrid": [], "oracle": []}
    for qid in env.query_pool:
        all_rewards = env.get_all_rewards(qid, target_metric=metric)
        per_query["bm25"].append(float(all_rewards[0]))
        per_query["dense"].append(float(all_rewards[1]))
        per_query["hybrid"].append(float(all_rewards[2]))

        oracle_action = env.get_oracle_action(qid, target_metric=metric)
        per_query["oracle"].append(float(all_rewards[oracle_action]))

    means = {k: float(np.mean(v)) for k, v in per_query.items()}
    return means, per_query

def best_fixed_per_query_scores(env_test: RouterEnvironment, static_res: dict, metric: str = 'mrr'):
    """Per-query scores for whichever of BM25/Dense/Hybrid has the best mean on the test
    set (per static_res, from compute_static_baselines). Needed for a paired significance
    test of RL router vs. Best-Fixed, not just RL router vs. tuned GBT."""
    best_action_name = max(("bm25", "dense", "hybrid"), key=lambda k: static_res[k])
    best_action = {"bm25": 0, "dense": 1, "hybrid": 2}[best_action_name]
    scores = [env_test.step(qid, best_action, target_metric=metric) for qid in env_test.query_pool]
    return best_action_name, scores

def compute_heuristic_baseline(env_train: RouterEnvironment, env_test: RouterEnvironment, metric: str = 'mrr'):
    """Score Gap Thresholding (Dense if score_gap > threshold else BM25). Feature 24 is score_gap."""
    def evaluate_threshold(env, threshold):
        total = 0.0
        for qid in env.query_pool:
            state = env.get_state(qid)
            score_gap = state[24]
            action = 1 if score_gap > threshold else 0
            total += env.step(qid, action, target_metric=metric)
        return total / len(env.query_pool)

    # Sweep the feature's own observed range, not a hard-coded interval. score_gap is a
    # difference of raw retriever scores on incommensurable scales (BM25 ~0-30 vs cosine
    # ~0.2-0.8), so its support depends entirely on the corpus and is nowhere near [-5, 5]:
    # on the 100K pool it lies in [-42.8, -2.96] with only 0-0.7% of values inside [-5, 5].
    # A sweep that misses the support cannot split the data, so `score_gap > threshold`
    # returns the same answer for nearly every query and this "heuristic" silently
    # degenerates into a fixed-arm baseline -- it was bit-identical to Always-BM25 on
    # HotpotQA. Quantiles of the training split are scale-free and always straddle the data.
    # Endpoints are included so the degenerate all-BM25 / all-Dense policies remain
    # reachable: if thresholding genuinely cannot beat committing to one arm, the sweep
    # should be free to say so rather than being unable to express it.
    train_gaps = np.array([env_train.get_state(qid)[24] for qid in env_train.query_pool],
                          dtype=float)
    candidates = np.unique(np.concatenate((
        np.quantile(train_gaps, np.linspace(0.0, 1.0, 99)),
        [float(train_gaps.min()) - 1.0, float(train_gaps.max()) + 1.0],
    )))

    best_thresh = float(candidates[0])
    best_train_score = -1.0
    for thresh in candidates:
        score = evaluate_threshold(env_train, float(thresh))
        if score > best_train_score:
            best_train_score = score
            best_thresh = float(thresh)

    per_query = []
    for qid in env_test.query_pool:
        action = 1 if env_test.get_state(qid)[24] > best_thresh else 0
        per_query.append(float(env_test.step(qid, action, target_metric=metric)))
    return float(np.mean(per_query)), per_query

def compute_supervised_baselines(env_train, env_test, metric='mrr'):
    def prepare_xy(env):
        X, y_cls, y_reg = [], [], []
        for qid in env.query_pool:
            X.append(env.get_state(qid))
            scores = [env.get_all_rewards(qid, metric)[a] for a in range(3)]
            y_reg.append(scores)
            
            max_score = max(scores)
            tied = [a for a in range(3) if scores[a] == max_score]
            y_cls.append(tied[deterministic_tiebreak(qid, len(tied))])
        return np.array(X), np.array(y_cls), np.array(y_reg)

    X_train, y_train_cls, y_train_reg = prepare_xy(env_train)
    X_test, y_test_cls, y_test_reg = prepare_xy(env_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = {}
    # Per-query scores for EVERY model (not just GBT) so any post-hoc analysis —
    # gold_type stratification, extra significance tests — needs no refit.
    raw_scores = {}

    def _record(name, chosen_actions):
        scores = [float(y_test_reg[i][chosen_actions[i]]) for i in range(len(chosen_actions))]
        raw_scores[name] = scores
        results[name] = float(np.mean(scores))

    # Every sklearn classifier raises when the training labels contain a single class, and
    # on a small run that is reachable: if all ~10 training queries share a best retriever,
    # the fit fails and takes the whole pipeline down with it. The honest answer in that
    # case is not an error -- it is that the only learnable policy IS the constant, which
    # is precisely what a degenerate routing problem means. Record it as such and move on.
    n_classes = len(np.unique(y_train_cls))
    if n_classes < 2:
        only = int(y_train_cls[0])
        print(f"  NOTE: training labels contain a single class "
              f"({['bm25', 'dense', 'hybrid'][only]}). No classifier can be fitted; every "
              f"supervised method is reported as that constant policy. Expected on small "
              f"runs, and a real finding if it ever happens at scale.")
        constant = [only] * len(y_test_cls)
        for name in ("logistic_regression", "random_forest", "ridge", "gradient_boosting"):
            _record(name, constant)
        return results, raw_scores

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_cls)
    _record("logistic_regression", lr.predict(X_test))

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train_cls)
    _record("random_forest", rf.predict(X_test))

    # 3. Ridge Regression (Multi-Output)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_reg)
    _record("ridge", np.argmax(ridge.predict(X_test), axis=1))

    # 4. Gradient Boosting (Tuned) — search depth matches 04_classifier_baseline.py
    # (n_iter=20, +min_samples_leaf) so "tuned GBT" means the same thing everywhere
    # it's reported in the paper.
    gbt_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_leaf': [1, 5, 10],
    }
    # RandomizedSearchCV stratifies internally, so it raises when any class has fewer
    # members than cv folds. That never happens at full scale, but a smoke or
    # calibration run can leave ~10 training queries across 3 classes, and the exception
    # aborts the whole pipeline before anything downstream gets to run. Fall back to an
    # untuned GBT and say so, rather than failing the stage over a sample-size limit that
    # has nothing to do with the code being tested.
    cls_counts = np.bincount(y_train_cls, minlength=3)
    n_folds = min(3, int(cls_counts[cls_counts > 0].min()) if (cls_counts > 0).any() else 0)
    if n_folds >= 2:
        gbt = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gbt_param_dist,
                                 n_iter=20, cv=n_folds, random_state=42, n_jobs=-1)
        if n_folds < 3:
            print(f"  NOTE: smallest class has {cls_counts[cls_counts > 0].min()} training "
                  f"queries; GBT search reduced to cv={n_folds}.")
    else:
        gbt = GradientBoostingClassifier(random_state=42)
        print(f"  NOTE: smallest class has {cls_counts[cls_counts > 0].min() if (cls_counts > 0).any() else 0} "
              f"training queries — too few to cross-validate. Using an UNTUNED GBT. "
              f"This number is not comparable to a full run's tuned GBT.")
    gbt.fit(X_train, y_train_cls)
    _record("gradient_boosting", gbt.predict(X_test))

    return results, raw_scores

def train_rl_single_seed(env_train, env_val, env_test, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    agent = PolicyAgent(input_dim=env_train.state_dim, hidden_dim=HIDDEN_DIM, output_dim=3, learningrate=LEARNING_RATE)
    trainer = ReinforceTrainer(
        env_train=env_train,
        env_test=env_test,
        agent=agent,
        reward_type=REWARD_TYPE,
        entropy_coeff=ENTROPY_COEFF,
        seed=seed,
        env_val=env_val,
        advantage_mode=ADVANTAGE_MODE,
    )
    
    trainer.train(num_epochs=NUM_EPOCHS, log_interval=300) # silent train
    
    # Evaluate best model on test
    test_mrr, _, _ = trainer._evaluate(use_val=False)
    
    # Get per-query scores for bootstrap
    per_query_scores = []
    for qid in env_test.query_pool:
        raw_state = env_test.get_state(qid)
        state = trainer._normalize_state(raw_state)
        action, _ = agent.select_action(state, evaluate=True)
        reward = env_test.step(qid, action, target_metric=REWARD_TYPE)
        per_query_scores.append(float(reward))
        
    return test_mrr, per_query_scores

def paired_bootstrap_test(scores_a, scores_b, n_resamples=10000, seed=12345):
    """One-sided paired bootstrap test of H0: mean(a - b) == 0.

    Uses an explicitly-seeded RandomState rather than the global numpy RNG: the global
    RNG's state here depends on how many RL seeds ran beforehand, so p-values were not
    reproducible across different --seeds/--datasets configurations of this same script.
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diffs = scores_a - scores_b
    observed_mean = np.mean(diffs)

    # Shift to satisfy the null hypothesis (mean = 0), then measure how often a resample
    # of the null-shifted differences reaches the observed effect.
    shifted_diffs = diffs - observed_mean

    count = 0
    for _ in range(n_resamples):
        resample = rng.choice(shifted_diffs, size=len(diffs), replace=True)
        if np.mean(resample) >= observed_mean:
            count += 1

    p_value = count / n_resamples
    return observed_mean, p_value

def parse_args():
    parser = argparse.ArgumentParser(description="Canonical RL router evaluation (static/heuristic/supervised/RL baselines + significance tests).")
    parser.add_argument("--seeds", type=str, default=None,
                        help=f"Comma-separated seed list (e.g. '42,123'), or a single int N to use the first N of the default {len(SEEDS)}-seed list. Default: all {len(SEEDS)} seeds.")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"REINFORCE training epochs per seed (default {NUM_EPOCHS}). Use a small value (e.g. 10-20) for a quick timing-calibration run.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of {nq,triviaqa,hotpotqa} (default: all three).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/canonical_results.json). Use a different path for quick-test runs so they don't overwrite the real results.")
    parser.add_argument("--env-suffix", type=str, default="",
                        help="Evaluate a suffixed env-data variant instead of the default "
                             "(e.g. '_leaky' -> nq_env_data_leaky.json). Used by "
                             "leakage_ablation.py; default '' preserves today's behavior.")
    parser.add_argument("--advantage-mode", type=str, default=ADVANTAGE_MODE,
                        choices=["per_query", "global_ema"],
                        help=f"Advantage estimator (default {ADVANTAGE_MODE}). 'per_query' "
                             f"centres each query on its own action rewards, so tied queries "
                             f"contribute zero gradient. 'global_ema' is the original global "
                             f"baseline — use it to reproduce the pre-fix behaviour.")
    parser.add_argument("--entropy-coeff", type=float, default=ENTROPY_COEFF,
                        help=f"Entropy bonus weight (default {ENTROPY_COEFF}). Tune on the "
                             f"validation split only.")
    # Recorded into the results file, not used in any computation. The env data does not
    # carry these facts, and without them a reader cannot tell which of several legitimate
    # variants a number came from -- see src/router/provenance.py for what that cost us.
    parser.add_argument("--gold-type", type=str, default=None,
                        help="Gold-passage filter the env data was built under "
                             "(real_wikipedia / constructed / all). Stamped into the "
                             "results file for provenance; does not affect computation.")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="WIKI_POOL_SIZE the corpora were mined from. Stamped for "
                             "provenance so runs at different pool sizes stay comparable.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Short label for this run, e.g. pool20k / pool100k. Stamped "
                             "for provenance.")
    return parser.parse_args()

def main():
    global NUM_EPOCHS, ADVANTAGE_MODE, ENTROPY_COEFF
    args = parse_args()
    # train_rl_single_seed reads these as module-level globals.
    ADVANTAGE_MODE = args.advantage_mode
    ENTROPY_COEFF = args.entropy_coeff

    seeds = SEEDS
    if args.seeds:
        if "," in args.seeds:
            seeds = [int(s) for s in args.seeds.split(",")]
        else:
            seeds = SEEDS[:int(args.seeds)]

    epochs = args.epochs or NUM_EPOCHS
    NUM_EPOCHS = epochs  # train_rl_single_seed reads this module-level global

    datasets = get_dataset_paths(args.env_suffix)
    if args.datasets:
        keep = {k.strip() for k in args.datasets.split(",")}
        datasets = {k: v for k, v in datasets.items() if k in keep}

    output_path = args.output or os.path.join(PROJECT_ROOT, "results", "canonical_results.json")

    print(f"Config: {len(seeds)} seed(s), {epochs} epoch(s), datasets={list(datasets.keys())}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_results = {}

    for ds_name, ds_path in datasets.items():
        print(f"\nEvaluating Dataset: {ds_name.upper()}")
        env_train, env_val, env_test = create_train_val_test_split(ds_path, split_seed=SPLIT_SEED)
        
        ds_res = {}
        
        # 1. Static & Heuristics
        print("Computing Static & Heuristic Baselines...")
        # compute_static_baselines returns (means, per_query). per_query holds the raw
        # per-query score lists aligned with env_test.query_pool, kept so post-hoc analyses
        # (e.g. gold_type stratification) can run without retraining.
        static_res, static_per_query = compute_static_baselines(env_test, REWARD_TYPE)
        ds_res.update(static_res)
        heuristic_mean, heuristic_per_query = compute_heuristic_baseline(env_train, env_test, REWARD_TYPE)
        ds_res["score_gap"] = heuristic_mean
        
        # 2. Supervised Baselines
        print("Computing Supervised Baselines (including tuned GBT)...")
        sup_res, raw_sup_scores = compute_supervised_baselines(env_train, env_test, REWARD_TYPE)
        ds_res.update(sup_res)
        
        # 3. RL Training (N seeds)
        print(f"Training RL ({len(seeds)} seed(s))...")
        rl_scores = []
        rl_per_query = []
        for seed in seeds:
            mrr, per_q = train_rl_single_seed(env_train, env_val, env_test, seed)
            rl_scores.append(mrr)
            if not rl_per_query:
                rl_per_query = list(per_q)
            else:
                rl_per_query = [a + b for a, b in zip(rl_per_query, per_q)] # sum across seeds

        # Average per_query scores across seeds for significance testing
        rl_per_query = [x / len(seeds) for x in rl_per_query]

        ds_res["rl_route_mean"] = float(np.mean(rl_scores))
        ds_res["rl_route_std"] = float(np.std(rl_scores))

        # 4. Statistical Significance (RL vs Tuned GBT, and RL vs Best-Fixed)
        print("Computing Bootstrap Significance...")
        delta, p_val = paired_bootstrap_test(rl_per_query, raw_sup_scores["gradient_boosting"])
        ds_res["significance_vs_gbt"] = {"delta": float(delta), "p_value": float(p_val)}

        best_fixed_name, best_fixed_scores = best_fixed_per_query_scores(env_test, static_res, REWARD_TYPE)
        delta_bf, p_val_bf = paired_bootstrap_test(rl_per_query, best_fixed_scores)
        ds_res["significance_vs_best_fixed"] = {
            "best_fixed_method": best_fixed_name,
            "delta": float(delta_bf),
            "p_value": float(p_val_bf),
        }

        # 5. Per-query dump — every method's raw per-query score, aligned with
        # env_test.query_pool, plus each query's gold_type. This makes any post-hoc
        # analysis (gold_type stratification, extra significance tests, failure-mode
        # slicing) a pure re-read of this JSON instead of a multi-hour retrain. Given the
        # full run costs ~7-12h, not saving this is how you end up re-running everything
        # for a single follow-up question.
        ds_res["per_query"] = {
            "query_ids": list(env_test.query_pool),
            "gold_type": [env_test.get_gold_type(q) for q in env_test.query_pool],
            "scores": {
                **{k: v for k, v in static_per_query.items()},
                "score_gap": heuristic_per_query,
                **raw_sup_scores,
                "rl_route": [float(x) for x in rl_per_query],
            },
        }

        all_results[ds_name] = ds_res

        print(f"Results for {ds_name.upper()}:")
        for k, v in ds_res.items():
            if k == "per_query":
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
                
    # Provenance is per-dataset because the gold-type filter changes n independently for
    # each one, and n is the fastest way to tell two otherwise-identical artifacts apart.
    # Consumers (generate_latex_tables, gold_type_analysis, cross_retriever_analysis,
    # leakage_ablation) all look datasets up by name rather than iterating the top level,
    # so this extra key is inert for them.
    per_dataset = {
        ds_name: build_provenance(env_path=ds_path, metric=REWARD_TYPE,
                                  split_seed=SPLIT_SEED, gold_type=args.gold_type,
                                  pool_size=args.pool_size, run_tag=args.run_tag)
        for ds_name, ds_path in datasets.items()
    }
    all_results["_provenance"] = {
        "reward_type": REWARD_TYPE,
        "split_seed": SPLIT_SEED,
        "seeds": seeds,
        "num_epochs": epochs,
        "advantage_mode": ADVANTAGE_MODE,
        "entropy_coeff": ENTROPY_COEFF,
        "env_suffix": args.env_suffix or "(none)",
        "gold_type": args.gold_type,
        "wiki_pool_size": args.pool_size,
        "run_tag": args.run_tag,
        "datasets": per_dataset,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_path}")
    for ds_name, block in per_dataset.items():
        print(f"  {ds_name}: {summarize_provenance(block)}")

if __name__ == "__main__":
    main()
