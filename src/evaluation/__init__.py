"""Evaluation helpers: benchmark datasets and retrieval metrics."""

from .benchmark_datasets import load_dataset_examples, load_local_fallback_dataset, DatasetExample
from .metrics import (
    exact_match,
    f1_score,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
    bootstrap_confidence_interval,
)

__all__ = [
    "DatasetExample",
    "load_dataset_examples",
    "load_local_fallback_dataset",
    "exact_match",
    "f1_score",
    "hit_rate_at_k",
    "mrr_at_k",
    "recall_at_k",
    "bootstrap_confidence_interval",
]
