"""Shared plumbing for the corpus_v3 (shared-index) pipeline.

Why this pipeline exists
------------------------
`experiments/baseline_v2/` gives every query its own private 500-passage corpus
(1 gold + 499 hard negatives). That is fine for measuring routing behaviour but it
cannot answer "does metric-as-feature contamination get worse as the corpus grows?",
because there is no corpus to grow — only a negative-sampling pool.

corpus_v3 builds ONE shared index that every query retrieves from, at four nested
scales (5K subset of 20K subset of 50K subset of 100K), so leak severity can be
plotted against corpus size with only the corpus varying.

Two invariants make the resulting numbers comparable to the 500-passage results:

  1. Metrics are not reimplemented here. `load_metric_module()` imports the exact
     functions `02_run_baselines.py` uses (check_hit / compute_mrr / f1_score / ...),
     so a difference between the two settings can never be a difference in how Hit
     or MRR was computed.
  2. The per-query output schema is byte-compatible with
     `experiments/baseline_v2/results/{ds}_per_query_results.json`, so
     `src/router/prepare_env_data.py --feature-mode ...` consumes it unchanged and
     every leak arm is built by the same code that built the 500-passage arms.

Everything here is deterministic: the same seed produces the same corpus subsets,
the same query order, and the same shard files on any machine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORPUS_V3_DIR = os.path.join(PROJECT_ROOT, "experiments", "corpus_v3")
ARTIFACT_DIR = os.path.join(CORPUS_V3_DIR, "artifacts")
RESULTS_DIR = os.path.join(CORPUS_V3_DIR, "results")

POOL_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "baseline_v2", "cache", "wiki_passage_pool.json"
)
BASELINE_RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments", "baseline_v2", "results"
)
RUN_BASELINES_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "baseline_v2", "02_run_baselines.py"
)

CORPUS_PATH = os.path.join(ARTIFACT_DIR, "corpus.jsonl")
CORPUS_MANIFEST_PATH = os.path.join(ARTIFACT_DIR, "corpus_manifest.json")
EMBEDDINGS_PATH = os.path.join(ARTIFACT_DIR, "corpus_embeddings.npy")
EMBEDDINGS_MANIFEST_PATH = os.path.join(ARTIFACT_DIR, "corpus_embeddings_manifest.json")
QUERY_EMBEDDINGS_PATH = os.path.join(ARTIFACT_DIR, "query_embeddings_{dataset}.npy")

# --- Experiment constants --------------------------------------------------

DATASETS = ("nq", "triviaqa", "hotpotqa")

# Nested: SCALES[i] documents are always a prefix of SCALES[i+1] documents, so the
# leak-vs-scale curve varies corpus size and nothing else. See build_scale_order().
SCALES = (5_000, 20_000, 50_000, 100_000)
SMOKE_SCALES = (2_000, 5_000)

# Must match 02_run_baselines.TOP_K_VALUES — prepare_env_data reads k in {1,3,5,10}
# for the leak arms' variance feature, and the reward is MRR@5.
TOP_K_VALUES = (1, 3, 5, 10, 20)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVERS = ("bm25", "dense", "hybrid")

# One seed governs corpus ordering everywhere. Changing it changes which documents
# land in the 5K subset, so it is recorded in every manifest.
CORPUS_SEED = 42

# Sanity gate for 04_run_retrieval, expressed against chance rather than as an
# absolute band.
#
# An absolute floor would be miscalibrated here. In the 500-passage setting a gold
# passage was INSERTED into every query's corpus, so Hit@5 started high by
# construction. A shared corpus guarantees nothing: the answer may be absent
# entirely, and when present it competes with N-1 real distractors instead of 499.
# Absolute Hit@5 is therefore legitimately much lower and falls as N grows — which
# is the axis under study, not a defect.
#
# What a broken pipeline actually looks like is retrieval performing at chance. For
# top-5 over N documents chance is 5/N, so the gate asks for a large multiple of it.
# The absolute figure is still printed, and anything below WARN_HIT5_FLOOR is flagged
# for a look without failing the run.
SANITY_CHANCE_MULTIPLE = 20.0
WARN_HIT5_FLOOR = 0.10


# --- Determinism helpers ---------------------------------------------------

def build_scale_order(n_docs: int, seed: int = CORPUS_SEED) -> np.ndarray:
    """A single seeded permutation of document indices.

    Every scale takes a PREFIX of this one ordering, which is what makes the
    subsets nested (5K subset of 20K subset of 50K subset of 100K). Drawing an
    independent sample per scale would confound "bigger corpus" with "different
    documents", and the leak-vs-scale curve would be uninterpretable.
    """
    rng = np.random.RandomState(seed)
    order = np.arange(n_docs)
    rng.shuffle(order)
    return order


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


# --- Artifact IO -----------------------------------------------------------

def ensure_dirs() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_manifest(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus() -> List[str]:
    """Read corpus.jsonl in file order (== index order used by every index)."""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"{CORPUS_PATH} not found. Run 01_build_shared_corpus.py first."
        )
    texts: List[str] = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def index_dir(scale: int) -> str:
    return os.path.join(ARTIFACT_DIR, f"scale_{scale}")


def shard_path(scale: int, dataset: str) -> str:
    """Per-(scale, dataset) results shard.

    Sharding this way is deliberate: if HotpotQA at 50K fails, the other eleven
    cells are still on disk and the leak-vs-scale figure is still plottable from
    whatever completed. Nothing in this pipeline requires all cells to exist.
    """
    return os.path.join(RESULTS_DIR, f"scale_{scale}", f"{dataset}_per_query_results.json")


# --- Query loading ---------------------------------------------------------

def load_queries(dataset: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Questions + answers + query_id, taken from the 500-passage per-query results.

    Using that file rather than `{ds}_prepared.json` is intentional on two counts:
    it is 35 MB instead of 575 MB (the prepared files inline all 500 passages per
    query), and it fixes `query_id` to the SAME ids as the 500-passage setting — so
    §3b and §3 can be compared per query, not just in aggregate.
    """
    path = os.path.join(BASELINE_RESULTS_DIR, f"{dataset}_per_query_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — corpus_v3 takes its query list from the "
            "baseline_v2 per-query results so that query_ids line up."
        )
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    queries = [
        {
            "query_id": r["query_id"],
            "question": r["question"],
            "answers": r["answers"],
            # Carried through only so downstream stratification can exclude the
            # template-generated gold of the 500-passage setting. corpus_v3 itself
            # never synthesises a passage — see 01_build_shared_corpus.py.
            "gold_type_v2": r.get("gold_type", "unknown"),
        }
        for r in rows
    ]
    return queries[:limit] if limit else queries


# --- Metric functions, borrowed rather than reimplemented ------------------

_METRIC_MODULE = None


def load_metric_module():
    """Import `02_run_baselines.py` for its metric functions.

    Loaded by path because the filename starts with a digit and is not a valid
    module identifier. The point is that corpus_v3 must NOT have its own copy of
    check_hit / compute_mrr / f1_score: if the 500-passage and shared-index
    settings disagreed because their metric code differed, the leak-vs-scale
    comparison would be measuring the reimplementation.

    Safe to import — that module guards execution behind `if __name__ == "__main__"`.
    """
    global _METRIC_MODULE
    if _METRIC_MODULE is not None:
        return _METRIC_MODULE
    spec = importlib.util.spec_from_file_location("_v2_baselines", RUN_BASELINES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {RUN_BASELINES_PATH}")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        # That module does `sys.path.insert(0, os.path.abspath("."))` at import.
        os.chdir(PROJECT_ROOT)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    _METRIC_MODULE = module
    return module


# --- Misc ------------------------------------------------------------------

def resolve_scales(smoke: bool, only: Tuple[int, ...] | None = None) -> Tuple[int, ...]:
    if only:
        return tuple(only)
    return SMOKE_SCALES if smoke else SCALES


def fmt_int(n: int) -> str:
    return f"{n:,}"
