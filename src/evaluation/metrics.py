"""Retrieval and QA evaluation metrics for benchmark pipelines."""

import re
from typing import List, Sequence, Tuple, Union

import numpy as np


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    return re.findall(r"\w+", normalized)


def exact_match(prediction: str, ground_truth: str) -> int:
    """Exact string match after light normalization."""
    return int(_normalize_text(prediction) == _normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    common = 0
    for token in truth_tokens:
        if pred_counts.get(token, 0) > 0:
            common += 1
            pred_counts[token] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _answer_matches_context(answer: str, context: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_context = _normalize_text(context)
    if not normalized_answer:
        return False
    # Word-boundary matching to prevent false positives on short answers
    pattern = r'(?<!\w)' + re.escape(normalized_answer) + r'(?!\w)'
    return bool(re.search(pattern, normalized_context))


def answer_in_context(answer_list: Sequence[str], context: str) -> bool:
    """Check whether any gold answer appears in retrieved context."""
    for answer in answer_list:
        if _answer_matches_context(answer, context):
            return True
    return False


def hit_rate_at_k(retrieved: List[dict], gold_answers: Sequence[str], k: int = 5) -> int:
    """Return 1 if any gold answer appears in the top-k retrieved chunks."""
    if not retrieved:
        return 0

    top_k = retrieved[:k]
    context = " ".join(hit.get("content", "") for hit in top_k)
    return int(answer_in_context(gold_answers, context))


def mrr_at_k(retrieved: List[dict], gold_answers: Sequence[str], k: int = 5) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    if not retrieved:
        return 0.0

    for rank, hit in enumerate(retrieved[:k], start=1):
        if answer_in_context(gold_answers, hit.get("content", "")):
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: List[dict], gold_answers: Sequence[str], k: int = 5) -> float:
    """Compute recall of unique gold answers retrieved in top-k chunks."""
    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    context = " ".join(hit.get("content", "") for hit in top_k)
    unique_answers = set(_normalize_text(a) for a in gold_answers if a)
    if not unique_answers:
        return 0.0

    found = 0
    normalized_ctx = _normalize_text(context)
    for gold in unique_answers:
        if not gold:
            continue
        pattern = r'(?<!\w)' + re.escape(gold) + r'(?!\w)'
        if re.search(pattern, normalized_ctx):
            found += 1
    return found / len(unique_answers)


def bootstrap_confidence_interval(
    values: Sequence[float], num_samples: int = 1000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean of a metric."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    sample_means = []
    n = len(values)
    values_array = np.array(values, dtype=float)
    for _ in range(num_samples):
        sample = np.random.choice(values_array, size=n, replace=True)
        sample_means.append(np.mean(sample))

    lower = float(np.percentile(sample_means, 100 * (alpha / 2)))
    upper = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(values_array))
    return mean, lower, upper
