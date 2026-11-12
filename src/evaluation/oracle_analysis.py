"""Oracle and winner analysis for retrieval routes."""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from src.evaluation.benchmark_datasets import DatasetExample
from src.evaluation.metrics import hit_rate_at_k, mrr_at_k, answer_in_context
from src.pipeline import RAGPipeline
from src.router.routing_env import deterministic_tiebreak

RETRIEVAL_MODES = ["sparse", "faiss", "hybrid"]


def _normalize_output_path(output_path: Optional[str]) -> Optional[str]:
    if not output_path:
        return None

    normalized = str(output_path).strip().strip('"').strip("'")
    if not normalized:
        return None

    if normalized.endswith(":"):
        normalized = normalized[:-1]

    if re.search(r"[A-Za-z]:$", normalized):
        normalized = normalized[:-2]

    return normalized


def _evaluate_single_query(
    pipeline: RAGPipeline,
    example: DatasetExample,
    mode: str,
    top_k: int = 5,
) -> Dict[str, float]:
    pipeline.retrieval_mode = mode

    prepared_corpus = example.metadata.get("corpus") or []
    if prepared_corpus:
        pipeline.ingest(prepared_corpus, verbose=False)

    result = pipeline.query(
        question=example.question,
        top_k=top_k,
        use_llm=False,
        use_expansion=False,
        verbose=False,
    )
    retrieved = result.get("retrieved", [])
    hit = hit_rate_at_k(retrieved, example.answers, k=top_k)
    mrr = mrr_at_k(retrieved, example.answers, k=top_k)
    return {"hit": hit, "mrr": mrr, "retrieved": retrieved}


def evaluate_retrieval_modes(
    pipeline: RAGPipeline,
    examples: List[DatasetExample],
    modes: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, object]:
    if modes is None:
        modes = RETRIEVAL_MODES

    summary = {
        "dataset_size": len(examples),
        "top_k": top_k,
        "modes": {mode: {"hits": [], "mrrs": []} for mode in modes},
        "oracle": {"hits": [], "mrrs": []},
        "per_query_winners": []
    }

    for example in examples:
        query = example.question
        query_metrics = {}

        for mode in modes:
            mode_result = _evaluate_single_query(pipeline, example, mode, top_k=top_k)
            query_metrics[mode] = mode_result
            summary["modes"][mode]["hits"].append(mode_result["hit"])
            summary["modes"][mode]["mrrs"].append(mode_result["mrr"])

        oracle_hit = int(any(query_metrics[mode]["hit"] for mode in modes))
        oracle_mrr = max(query_metrics[mode]["mrr"] for mode in modes)
        summary["oracle"]["hits"].append(oracle_hit)
        summary["oracle"]["mrrs"].append(oracle_mrr)

        # Use hit first, then MRR as tie breaker.
        winner = None
        winning_modes = [mode for mode in modes if query_metrics[mode]["hit"]]
        if len(winning_modes) == 1:
            winner = winning_modes[0]
        elif len(winning_modes) > 1:
            best_mrr = max(query_metrics[mode]["mrr"] for mode in winning_modes)
            winner_candidates = [mode for mode in winning_modes if query_metrics[mode]["mrr"] == best_mrr]
            winner = winner_candidates[deterministic_tiebreak(example.id, len(winner_candidates))] if winner_candidates else "tie"
        else:
            # No hit by any mode: choose the mode with highest MRR or mark tie.
            best_mrr = max(query_metrics[mode]["mrr"] for mode in modes)
            best_modes = [mode for mode in modes if query_metrics[mode]["mrr"] == best_mrr]
            winner = best_modes[deterministic_tiebreak(example.id, len(best_modes))] if len(best_modes) >= 1 else "tie"

        summary["per_query_winners"].append({
            "query_id": example.id,
            "query": query,
            "winner": winner,
            "mode_metrics": {
                mode: {
                    "hit": query_metrics[mode]["hit"],
                    "mrr": query_metrics[mode]["mrr"]
                }
                for mode in modes
            }
        })

    return summary


def summarize_results(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    report = {}
    for mode, data in summary["modes"].items():
        hits = data["hits"]
        mrrs = data["mrrs"]
        report[mode] = {
            "hit_rate": float(sum(hits) / len(hits)) if hits else 0.0,
            "mrr": float(sum(mrrs) / len(mrrs)) if mrrs else 0.0,
        }

    oracle_hits = summary["oracle"]["hits"]
    oracle_mrrs = summary["oracle"]["mrrs"]
    report["oracle"] = {
        "hit_rate": float(sum(oracle_hits) / len(oracle_hits)) if oracle_hits else 0.0,
        "mrr": float(sum(oracle_mrrs) / len(oracle_mrrs)) if oracle_mrrs else 0.0,
    }

    return report


def compute_winner_counts(summary: Dict[str, object]) -> Dict[str, int]:
    counter = Counter(item["winner"] for item in summary["per_query_winners"])
    return dict(counter)


def run_oracle_analysis(
    examples: List[DatasetExample],
    pipeline: RAGPipeline,
    modes: Optional[List[str]] = None,
    top_k: int = 5,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    if modes is None:
        modes = RETRIEVAL_MODES

    analysis = evaluate_retrieval_modes(pipeline, examples, modes=modes, top_k=top_k)
    report = summarize_results(analysis)
    winners = compute_winner_counts(analysis)
    report["winner_counts"] = winners
    report["dataset_size"] = len(examples)
    report["top_k"] = top_k
    report["retrieval_modes"] = modes

    normalized_output_path = _normalize_output_path(output_path)
    if normalized_output_path:
        output_dir = os.path.dirname(normalized_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(normalized_output_path, "w", encoding="utf-8") as handle:
            json.dump({"report": report, "raw_summary": analysis}, handle, indent=2)

    return report
