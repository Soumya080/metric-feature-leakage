"""
18_end_to_end_eval.py — End-to-end RAG evaluation: retrieval → generation → answer quality.

This script uses REAL retrieved passages from the baseline_v2 results and 
generates answers via a running vLLM OpenAI-compatible API server. It then 
computes actual Exact Match (EM) and Token F1 scores against gold answers.

Prerequisites:
    1. Start vLLM server (12GB GPU - use AWQ quantized model):
       vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --max-model-len 2048
    2. Install openai client: pip install openai

Usage:
    cd <repo>
    python experiments/router_experiments/18_end_to_end_eval.py --backend vllm --dataset all
    python experiments/router_experiments/18_end_to_end_eval.py --backend vllm --dataset nq --vllm_url http://localhost:8000
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.router.policy_network import PolicyAgent
from src.router.routing_env import RouterEnvironment, create_train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Configuration ----
SEED = 42
HIDDEN_DIM = 128
REWARD_TYPE = "mrr"

# =====================================================================
# GENERATORS
# =====================================================================

class VLLMGenerator:
    """Connects to a running vLLM OpenAI-compatible API server."""
    def __init__(self, base_url="http://localhost:8000", model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for vLLM API mode.\n"
                "Install with: pip install openai"
            )
        self.client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="EMPTY",  # vLLM doesn't require a real key
        )
        self.model_name = model_name
        # Verify connection
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            print(f"  Connected to vLLM at {base_url}")
            print(f"  Available models: {available}")
            if model_name not in available:
                print(f"  WARNING: Requested model '{model_name}' not in available models.")
                if available:
                    self.model_name = available[0]
                    print(f"  Falling back to: {self.model_name}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to vLLM server at {base_url}.\n"
                f"Make sure vLLM is running: vllm serve {model_name}\n"
                f"Error: {e}"
            )

    def generate(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        results = []
        # Process in batches to avoid overwhelming the server
        batch_size = 32
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            for prompt in batch:
                try:
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.0,
                    )
                    results.append(response.choices[0].text.strip())
                except Exception as e:
                    print(f"  Generation error: {e}")
                    results.append("")
            if i + batch_size < len(prompts):
                print(f"    Processed {min(i+batch_size, len(prompts))}/{len(prompts)} prompts...")
        return results

class FallbackGenerator:
    def __init__(self):
        print("Initializing Fallback Generator (no GPU).")

    def generate(self, prompts: List[str]) -> List[str]:
        # Simple heuristic mock answer for fallback
        return ["Fallback simulated answer" for _ in prompts]


# =====================================================================
# ANSWER QUALITY METRICS
# =====================================================================

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/extra whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_exact_match(prediction: str, gold: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0

def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    return 2 * precision * recall / (precision + recall)


# =====================================================================
# PROMPT BUILDING
# =====================================================================

def build_prompt(question: str, passages: List[str]) -> str:
    context = "\n\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))
    
    prompt = f"""Based on the following passages, answer the question concisely.
If the answer is not found in the passages, say "I don't know".
Give ONLY the answer, no explanation.

{context}

Question: {question}
Answer:"""
    return prompt


# =====================================================================
# EVALUATION LOGIC
# =====================================================================

def _get_router_action(agent, env, qid, feat_mean, feat_std):
    """Get the router's greedy action for a query."""
    if agent is None:
        return 1 # Fallback to dense if no model
    state = env.get_state(qid)
    normed = ((np.array(state) - feat_mean) / feat_std).tolist()
    agent.policynetwork.eval()
    action, _ = agent.select_action(normed, evaluate=True)
    return action


def run_end_to_end_eval_for_dataset(dataset_name: str, generator, num_queries: int):
    print(f"\n[{dataset_name.upper()}] Loading environments and baseline data...")
    
    env_path = os.path.join(PROJECT_ROOT, "src", "router", f"{dataset_name}_env_data.json")
    baseline_path = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "results", f"{dataset_name}_per_query_results.json")
    
    if not os.path.exists(env_path):
        print(f"Skipping {dataset_name}, missing env file: {env_path}")
        return None
    if not os.path.exists(baseline_path):
        print(f"Skipping {dataset_name}, missing baseline file: {baseline_path}")
        return None
        
    env = RouterEnvironment(data_path=env_path)
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
        
    # Standardize baseline_data format if it's a dict or list
    if isinstance(baseline_data, list):
        # Convert list to dict keyed by index or question
        baseline_dict = {}
        # Try to map by question
        for i, item in enumerate(baseline_data):
            q = item.get("question", str(i))
            baseline_dict[q] = item
    else:
        baseline_dict = baseline_data
        
    # Attempt to load trained router
    model_path = os.path.join(PROJECT_ROOT, "experiments", "router_training", "best_reinforce_model.pt")
    
    agent = None
    if os.path.exists(model_path):
        agent = PolicyAgent(
            input_dim=env.state_dim, hidden_dim=HIDDEN_DIM,
            output_dim=3, learningrate=3e-3
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        agent.policynetwork.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained router from {model_path}")
    else:
        print("WARNING: No trained model found, rl_router will fallback to dense.")
        
    all_features = [env.get_state(qid) for qid in env.query_pool]
    feat_mean = np.mean(all_features, axis=0)
    feat_std = np.std(all_features, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    
    random.seed(SEED)
    eval_qids = random.sample(env.query_pool, min(num_queries, len(env.query_pool)))
    
    methods = {
        "always_bm25": lambda qid: 0,
        "always_dense": lambda qid: 1,
        "always_hybrid": lambda qid: 2,
        "rl_router": lambda qid: _get_router_action(agent, env, qid, feat_mean, feat_std),
        "oracle": lambda qid: env.get_oracle_action(qid, REWARD_TYPE),
    }
    
    results = {method: {
        "answer_em": [],
        "answer_f1": [],
    } for method in methods}
    
    # Track the raw data needed for batch generation
    # method -> list of (qid, prompt, gold_answers)
    method_to_queries = {m: [] for m in methods}
    
    for qid in eval_qids:
        entry = env.master_table[qid]
        question = entry.get("question", "")
        
        # Try to find corresponding baseline entry
        b_entry = None
        if str(qid) in baseline_dict:
            b_entry = baseline_dict[str(qid)]
        elif question in baseline_dict:
            b_entry = baseline_dict[question]
        else:
            # Maybe list index? Or fall back to searching by question string
            for k, v in baseline_dict.items():
                if v.get("question") == question:
                    b_entry = v
                    break
                    
        if not b_entry:
            continue
            
        gold_answers = b_entry.get("answers", [])
        if not gold_answers and "answer" in b_entry:
            gold_answers = [b_entry["answer"]]
            
        for method_name, action_fn in methods.items():
            action = action_fn(qid)
            ret_name = RouterEnvironment.ACTION_TO_RETRIEVER[action]
            
            # Fetch real retrieved passages
            passages = []
            retrievers_data = b_entry.get("retrievers", {})
            if ret_name in retrievers_data:
                passages = retrievers_data[ret_name].get("retrieved", [])[:3]
            
            # If still no passages, use simulated empty
            if not passages:
                passages = ["No relevant information was found in the retrieved documents."]
                
            prompt = build_prompt(question, passages)
            method_to_queries[method_name].append((qid, prompt, gold_answers))
            
    # Now execute batch generation for each method
    summary = {}
    for method_name in methods.keys():
        queries_data = method_to_queries[method_name]
        if not queries_data:
            print(f"  No valid queries found for {method_name}")
            continue
            
        print(f"  Generating answers for {method_name} ({len(queries_data)} queries)...")
        
        prompts = [item[1] for item in queries_data]
        gold_answers_list = [item[2] for item in queries_data]
        
        # Generate in batch
        generated_answers = generator.generate(prompts)
        
        em_scores = []
        f1_scores = []
        
        for pred, golds in zip(generated_answers, gold_answers_list):
            if not golds:
                em_scores.append(0.0)
                f1_scores.append(0.0)
                continue
                
            best_em = max([compute_exact_match(pred, g) for g in golds])
            best_f1 = max([compute_f1(pred, g) for g in golds])
            
            em_scores.append(best_em)
            f1_scores.append(best_f1)
            
        summary[method_name] = {
            "answer_em": round(float(np.mean(em_scores)), 4),
            "answer_f1": round(float(np.mean(f1_scores)), 4),
            "num_queries": len(queries_data),
        }
        
    return summary


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nq", choices=["nq", "triviaqa", "hotpotqa", "all"])
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "fallback"])
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000",
                        help="URL of the running vLLM server (default: http://localhost:8000)")
    parser.add_argument("--model", type=str, default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                        help="Model name served by vLLM (default: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"END-TO-END RAG EVALUATION (Backend: {args.backend})")
    print("=" * 60)
    
    if args.backend == "vllm":
        generator = VLLMGenerator(base_url=args.vllm_url, model_name=args.model)
    else:
        generator = FallbackGenerator()
        
    datasets_to_run = ["nq", "triviaqa", "hotpotqa"] if args.dataset == "all" else [args.dataset]
    
    all_results = {}
    
    for ds in datasets_to_run:
        ds_summary = run_end_to_end_eval_for_dataset(ds, generator, args.num_queries)
        if ds_summary:
            all_results[ds] = ds_summary
            
            print(f"\nResults for {ds.upper()}:")
            print(f"{'Method':<20s} {'Ans EM':>10s} {'Ans F1':>10s}")
            print("-" * 45)
            for method, metrics in ds_summary.items():
                print(f"{method:<20s} {metrics['answer_em']:>10.4f} {metrics['answer_f1']:>10.4f}")
    
    # Save
    output = {
        "end_to_end": all_results,
        "config": {
            "num_queries": args.num_queries,
            "backend": args.backend,
            "seed": SEED,
        }
    }
    
    output_path = os.path.join(RESULTS_DIR, "end_to_end_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
