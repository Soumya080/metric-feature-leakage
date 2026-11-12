#!/usr/bin/env python3
"""
estimate_pipeline_time.py - Empirically measure retrieval + RL training throughput on
THIS machine, then extrapolate a real wall-clock estimate for the full run_full_pipeline.py
job, before you commit hours of cluster time to it.

Design goal: zero risk of touching any real/committed file. This script:
  - Times retrieval on SYNTHETIC queries/passages generated in-memory (never reads or
    writes experiments/baseline_v2/data/ or results/) — so it can't clobber the real
    per-query results the way a "quick" flag on 02_run_baselines.py could.
  - Times RL training by loading an EXISTING, real *_env_data.json READ-ONLY (never writes
    it) and training for a handful of epochs with a throwaway PolicyAgent, writing its
    (tiny) results only to a path you pass explicitly.

Usage:
    python experiments/router_experiments/estimate_pipeline_time.py
    python experiments/router_experiments/estimate_pipeline_time.py --num_queries 50 --corpus_size 500
"""
import argparse
import os
import random
import string
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.embeddings.encoder import EmbeddingEncoder
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_faiss import FAISSRetriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.router.policy_network import PolicyAgent
from src.router.routing_env import create_train_val_test_split
from src.router.train_reinforce import ReinforceTrainer

# Real target scale (matches 01_prepare_data.py's defaults / canonical_evaluation.py)
REAL_QUERIES_PER_DATASET = 2000
REAL_NUM_DATASETS = 3
REAL_CORPUS_SIZE = 500
REAL_SEEDS_CANONICAL = 15
REAL_EPOCHS_CANONICAL = 150
REAL_SEEDS_ABLATION = 15
REAL_EPOCHS_ABLATION = 200
REAL_ABLATION_CONFIGS = 4


def _random_passage(min_words=30, max_words=200):
    n = random.randint(min_words, max_words)
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9))) for _ in range(n)]
    return " ".join(words)


def measure_retrieval_throughput(num_queries: int, corpus_size: int):
    """Time BM25 + Dense(FAISS) + Hybrid(RRF) retrieval on synthetic in-memory data —
    same code path as 02_run_baselines.py's run_single_query_all_retrievers, but never
    touches any file on disk."""
    print(f"Loading embedding model (first call downloads/loads all-MiniLM-L6-v2)...")
    encoder = EmbeddingEncoder("all-MiniLM-L6-v2")

    print(f"Timing retrieval: {num_queries} synthetic queries x {corpus_size}-passage corpora...")
    t_start = time.time()

    for qi in range(num_queries):
        question = "synthetic query " + _random_passage(5, 12)
        corpus_texts = [_random_passage() for _ in range(corpus_size)]
        chunks = [{"content": t, "chunk_id": f"c{i}", "chunk_index": i} for i, t in enumerate(corpus_texts)]

        embeddings = encoder.encode_batch([c["content"] for c in chunks])
        for i, chunk in enumerate(chunks):
            chunk["chunk_embedding"] = embeddings[i]
        query_embedding = encoder.encode(question)

        bm25 = BM25Retriever()
        bm25.fit(chunks)
        bm25_results = bm25.retrieve(question, top_k=20)

        faiss_ret = FAISSRetriever(encoder.dimension)
        faiss_ret.add_chunks(chunks)
        faiss_results = faiss_ret.retrieve(query_embedding, top_k=20)

        dense_candidates = faiss_ret.retrieve(query_embedding, top_k=40)
        sparse_candidates = bm25.retrieve(question, top_k=40)
        _ = reciprocal_rank_fusion(dense_candidates, sparse_candidates, top_n=20)

        if (qi + 1) % max(1, num_queries // 5) == 0:
            print(f"  {qi+1}/{num_queries} queries done...")

    elapsed = time.time() - t_start
    sec_per_query = elapsed / num_queries
    print(f"Done: {elapsed:.2f}s total, {sec_per_query:.3f}s/query at corpus_size={corpus_size}")
    return sec_per_query


def measure_rl_training_throughput(env_data_path: str, epochs: int = 5, seed: int = 42):
    """Time N epochs of REINFORCE training on a REAL (but read-only) env_data.json."""
    print(f"Loading real env data (read-only): {env_data_path}")
    env_train, env_val, env_test = create_train_val_test_split(env_data_path, split_seed=42)
    print(f"  train={len(env_train)}  val={len(env_val)}  test={len(env_test)}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = PolicyAgent(input_dim=env_train.state_dim, hidden_dim=128, output_dim=3, learningrate=3e-3)
    trainer = ReinforceTrainer(
        env_train=env_train, env_test=env_test, agent=agent,
        reward_type="mrr", entropy_coeff=0.15, seed=seed, env_val=env_val,
    )

    print(f"Timing {epochs} training epochs...")
    t_start = time.time()
    trainer.train(num_epochs=epochs, log_interval=epochs + 1)
    elapsed = time.time() - t_start

    sec_per_epoch = elapsed / epochs
    print(f"Done: {elapsed:.2f}s total, {sec_per_epoch:.3f}s/epoch ({len(env_train)} train queries)")
    return sec_per_epoch


def fmt_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


def main():
    parser = argparse.ArgumentParser(description="Empirically calibrate pipeline runtime on this machine.")
    parser.add_argument("--num_queries", type=int, default=20, help="Synthetic queries for the retrieval timing test.")
    parser.add_argument("--corpus_size", type=int, default=REAL_CORPUS_SIZE, help="Passages per synthetic query (default matches the real 500-passage corpus).")
    parser.add_argument("--rl_epochs", type=int, default=5, help="Epochs for the RL training timing test.")
    parser.add_argument("--env_data", type=str, default=os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
                        help="Real env_data.json to read (read-only) for the RL timing test.")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip the retrieval timing test (e.g. if you already know it's slow and just want the RL number).")
    args = parser.parse_args()

    print("="*70)
    print("PIPELINE TIME CALIBRATION (read-only w.r.t. all real/committed data)")
    print("="*70)

    sec_per_query = None
    if not args.skip_retrieval:
        print("\n--- Stage 2 calibration: retrieval throughput ---")
        sec_per_query = measure_retrieval_throughput(args.num_queries, args.corpus_size)

    print("\n--- Stage 4/8 calibration: RL training throughput ---")
    if not os.path.exists(args.env_data):
        print(f"WARNING: {args.env_data} not found, skipping RL timing test.")
        sec_per_epoch = None
    else:
        sec_per_epoch = measure_rl_training_throughput(args.env_data, epochs=args.rl_epochs)

    print("\n" + "="*70)
    print("EXTRAPOLATED ESTIMATE FOR THE FULL PIPELINE")
    print("="*70)

    total_low, total_high = 5*60, 15*60  # stage 1 (data prep): 5-15 min, not measured here

    if sec_per_query is not None:
        real_queries = REAL_QUERIES_PER_DATASET * REAL_NUM_DATASETS
        stage2_est = sec_per_query * real_queries
        print(f"Stage 2 (retrieval, {real_queries} queries x {REAL_CORPUS_SIZE}-passage corpora): ~{fmt_hms(stage2_est)}")
        total_low += stage2_est * 0.8
        total_high += stage2_est * 1.3
    else:
        print("Stage 2 (retrieval): not measured (--skip_retrieval)")

    if sec_per_epoch is not None:
        stage4_epochs = REAL_SEEDS_CANONICAL * REAL_EPOCHS_CANONICAL * REAL_NUM_DATASETS
        stage4_est = sec_per_epoch * stage4_epochs
        print(f"Stage 4 (canonical_evaluation.py, {REAL_SEEDS_CANONICAL} seeds x {REAL_EPOCHS_CANONICAL} epochs x {REAL_NUM_DATASETS} datasets): ~{fmt_hms(stage4_est)}")

        stage8_epochs = REAL_SEEDS_ABLATION * REAL_EPOCHS_ABLATION * REAL_ABLATION_CONFIGS
        stage8_est = sec_per_epoch * stage8_epochs
        print(f"Stage 8 (23_multi_seed_ablation.py, {REAL_SEEDS_ABLATION} seeds x {REAL_EPOCHS_ABLATION} epochs x {REAL_ABLATION_CONFIGS} configs): ~{fmt_hms(stage8_est)}")

        total_low += (stage4_est + stage8_est) * 0.8
        total_high += (stage4_est + stage8_est) * 1.3
    else:
        print("Stage 4/8 (RL training): not measured")

    print(f"\nSTAGES 1+2+4+8 COMBINED (stages 3,5,6,7 are seconds, not counted): ~{fmt_hms(total_low)} - {fmt_hms(total_high)}")
    print("\nThis is a real, machine-measured estimate (not a guess) for stages 2/4/8;")
    print("stage 1 (data prep) is a rough guess since it's cheap and hasn't been timed here.")
    print("Note: this does NOT include failure_mode_analysis.py or 17_latency_analysis.py")
    print("(both single-seed / instant, a few minutes at most).")


if __name__ == "__main__":
    main()
