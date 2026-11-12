"""Dataset loaders for standard QA benchmarks used in retrieval evaluation."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass
class DatasetExample:
    question: str
    answers: List[str]
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_answer_text(value: str) -> str:
    if value is None:
        return ""
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_answers(item: dict) -> List[str]:
    if item is None:
        return []

    if "answers" in item:
        answers = item["answers"]
        if isinstance(answers, dict) and "text" in answers:
            return [_normalize_answer_text(a) for a in answers.get("text", []) if a]
        if isinstance(answers, list):
            return [_normalize_answer_text(a) for a in answers if a]

    if "answer" in item:
        answer = item["answer"]
        if isinstance(answer, list):
            return [_normalize_answer_text(a) for a in answer if a]
        if isinstance(answer, str):
            return [_normalize_answer_text(answer)]

    # Natural Questions style short answers
    if "short_answers" in item:
        answers = item["short_answers"]
        if isinstance(answers, list):
            collected = []
            for block in answers:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("value")
                    if text:
                        collected.append(_normalize_answer_text(text))
            return collected

    # HotpotQA / other QA formats
    if "answer_text" in item:
        return [_normalize_answer_text(item["answer_text"])]

    return []


def _extract_question(item: dict) -> Optional[str]:
    if item is None:
        return None
    for field in ["question", "question_text", "query"]:
        if field in item and isinstance(item[field], str):
            return _normalize_answer_text(item[field])
    return None


def _load_dataset_safe(dataset_name: str, config_name: Optional[str] = None, split: str = "validation"):
    if load_dataset is None:
        raise ImportError(
            "datasets is required for benchmark dataset loading. Install via `pip install datasets`."
        )

    kwargs = {"split": split}
    if config_name is not None:
        kwargs["name"] = config_name

    return load_dataset(dataset_name, **kwargs)


def _select_subset(dataset, subset: Optional[int]):
    if subset is None:
        return dataset
    if subset <= 0:
        return dataset

    subset_size = min(subset, len(dataset))
    if hasattr(dataset, "select"):
        return dataset.select(range(subset_size))
    return dataset[:subset_size]


def load_nq_open(subset: Optional[int] = None) -> List[DatasetExample]:
    """Load Natural Questions / NQ Open examples."""
    if load_dataset is None:
        return load_local_fallback_dataset()

    try:
        dataset = _load_dataset_safe("natural_questions", split="validation")
    except Exception:
        try:
            dataset = _load_dataset_safe("natural_questions", config_name="default", split="validation")
        except Exception:
            return load_local_fallback_dataset()

    dataset = _select_subset(dataset, subset)
    examples = []
    for i, item in enumerate(dataset):
        question = _extract_question(item)
        answers = _extract_answers(item)
        if not question or not answers:
            continue
        examples.append(
            DatasetExample(
                question=question,
                answers=answers,
                id=str(item.get("example_id", item.get("id", i))),
                metadata={"source": "nq_open"},
            )
        )
    return examples


def load_trivia_qa(subset: Optional[int] = None) -> List[DatasetExample]:
    """Load TriviaQA examples."""
    if load_dataset is None:
        return load_local_fallback_dataset()

    try:
        dataset = _load_dataset_safe("trivia_qa", split="validation")
    except Exception:
        return load_local_fallback_dataset()

    dataset = _select_subset(dataset, subset)
    examples = []
    for i, item in enumerate(dataset):
        question = _extract_question(item)
        answers = _extract_answers(item)
        if not question or not answers:
            continue
        examples.append(
            DatasetExample(
                question=question,
                answers=answers,
                id=str(item.get("id", i)),
                metadata={"source": "trivia_qa"},
            )
        )
    return examples


def load_hotpot_qa(subset: Optional[int] = None) -> List[DatasetExample]:
    """Load HotpotQA examples."""
    if load_dataset is None:
        return load_local_fallback_dataset()

    try:
        dataset = _load_dataset_safe("hotpot_qa", split="validation")
    except Exception:
        try:
            dataset = _load_dataset_safe("hotpot_qa", config_name="fullwiki", split="validation")
        except Exception:
            return load_local_fallback_dataset()

    dataset = _select_subset(dataset, subset)
    examples = []
    for i, item in enumerate(dataset):
        question = _extract_question(item)
        answers = _extract_answers(item)
        if not question or not answers:
            continue
        examples.append(
            DatasetExample(
                question=question,
                answers=answers,
                id=str(item.get("id", i)),
                metadata={
                    "source": "hotpot_qa",
                    "supporting_facts": item.get("supporting_facts", []),
                },
            )
        )
    return examples


def load_local_fallback_dataset() -> List[DatasetExample]:
    """Fallback small dataset when standard benchmark datasets cannot be loaded."""
    return [
        DatasetExample(
            question="What is the self-attention mechanism used in?",
            answers=["transformer architecture", "transformers"],
            metadata={"source": "local_fallback"},
        ),
        DatasetExample(
            question="Which optimizer is commonly used in deep learning?",
            answers=["adam", "sgd", "adam and sgd"],
            metadata={"source": "local_fallback"},
        ),
        DatasetExample(
            question="What does FAISS stand for?",
            answers=["facebook ai similarity search"],
            metadata={"source": "local_fallback"},
        ),
        DatasetExample(
            question="How does BM25 score keywords in documents?",
            answers=["term frequency saturation", "document length normalization"],
            metadata={"source": "local_fallback"},
        ),
    ]


def _resolve_data_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def load_prepared_dataset(path: str, subset: Optional[int] = None) -> List[DatasetExample]:
    """Load benchmark examples from a prepared JSON file."""
    resolved_path = _resolve_data_path(path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Prepared dataset file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "examples" in payload:
        payload = payload["examples"]

    if not isinstance(payload, list):
        raise ValueError(f"Prepared dataset file must contain a list of examples: {resolved_path}")

    selected = _select_subset(payload, subset)
    examples = []
    for i, item in enumerate(selected):
        if not isinstance(item, dict):
            continue
        question = _extract_question(item)
        answers = _extract_answers(item)
        if not question or not answers:
            continue
        examples.append(
            DatasetExample(
                question=question,
                answers=answers,
                id=str(item.get("id", item.get("question_id", i))),
                metadata={
                    "source": "prepared",
                    "path": resolved_path,
                    "gold_passage": item.get("gold_passage"),
                    "corpus_size": item.get("corpus_size"),
                    "corpus": item.get("corpus") or [],
                },
            )
        )
    return examples


def load_dataset_examples(dataset_name: str, subset: Optional[int] = None) -> List[DatasetExample]:
    """Load a dataset by its alias."""
    loader_map = {
        "nq_open": load_nq_open,
        "trivia_qa": load_trivia_qa,
        "hotpot_qa": load_hotpot_qa,
        "local": lambda subset=None: load_local_fallback_dataset(),
        "prepared_nq": lambda subset=None: load_prepared_dataset(
            "experiments/baseline_v1/data/nq_prepared.json",
            subset=subset,
        ),
        "prepared_triviaqa": lambda subset=None: load_prepared_dataset(
            "experiments/baseline_v1/data/triviaqa_prepared.json",
            subset=subset,
        ),
        "prepared_both": lambda subset=None: (
            load_prepared_dataset(
                "experiments/baseline_v1/data/nq_prepared.json",
                subset=subset,
            )
            + load_prepared_dataset(
                "experiments/baseline_v1/data/triviaqa_prepared.json",
                subset=subset,
            )
        ),
    }

    loader = loader_map.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loader(subset=subset)
