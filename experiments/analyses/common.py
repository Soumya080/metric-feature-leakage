"""Shared paths and helpers for the scripts in experiments/analyses/.

Everything here is read-only with respect to existing artifacts: env files under src/router/
and results under results/ are only ever opened for reading. New outputs go under
results/analyses/<experiment>/, and every writer refuses a non-empty output directory rather
than overwriting -- there is deliberately no --overwrite flag. Re-running means choosing a
new --out-dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROUTER_DIR = os.path.join(PROJECT_ROOT, "src", "router")
ROUTER_EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "router_experiments")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_RESULTS = os.path.join(RESULTS_DIR, "analyses")
ANALYSIS_ENV_DIR = os.path.join(ANALYSIS_RESULTS, "env_data")

for p in (PROJECT_ROOT, ROUTER_EXP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DATASETS = ("nq", "triviaqa", "hotpotqa")
ACTIONS = ("bm25", "dense", "hybrid")
LEAK_SLOTS = (22, 23, 24, 25)

# Existing Table 2 rungs, read in place from src/router (never written).
ANCHOR_SUFFIXES = {
    "honest_minimal": "",
    "leak_one_arm": "_leak1",
    "leak_hit": "_leakhit",
    "leak_noisy": "_leaknoisy",
    "leaky": "_leaky",
}

# Noise half-widths on the reward scale [0, 1]. 0.15 is the paper's LEAK_NOISE_SCALE.
NOISE_SCALES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
FAMILIES = ("mrr20", "rankreward")
CLIPPED_SCALE = 0.15


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    return env.get("queries", env) if isinstance(env, dict) else env


def anchor_path(dataset: str, suffix: str) -> str:
    return os.path.join(ROUTER_DIR, f"{dataset}_env_data{suffix}.json")


def variant_tag(family: str, scale: float, clipped: bool = False) -> str:
    tag = f"{family}_s{int(round(scale * 100)):03d}"
    return tag + "_clip" if clipped else tag


def variant_path(dataset: str, tag: str, env_dir: str = ANALYSIS_ENV_DIR) -> str:
    return os.path.join(env_dir, f"{dataset}_env_data_crc_{tag}.json")


def all_variant_tags() -> List[str]:
    tags = [variant_tag(f, s) for f in FAMILIES for s in NOISE_SCALES]
    tags += [variant_tag(f, CLIPPED_SCALE, clipped=True) for f in FAMILIES]
    return tags


def jitter(value: float, salt: str, scale: float) -> float:
    """src.router.prepare_env_data._jitter generalised to any half-width.

    Same crc32 hash of f"{salt}:{float(value)!r}", so at scale == LEAK_NOISE_SCALE (0.15)
    it is bit-identical to the repo's leak_noisy rule (checked at build time and in tests).
    Deterministic, no RNG, unclipped.
    """
    if scale == 0.0:
        return float(value)
    h = zlib.crc32(f"{salt}:{float(value)!r}".encode("utf-8"))
    return float(value) + ((h / 2 ** 32) - 0.5) * 2.0 * scale


def reward_matrix(queries: Dict[str, dict], qids: Sequence[str], metric: str = "rank_reward") -> np.ndarray:
    return np.array([[float(queries[q]["retrievers"][a][metric]) for a in ACTIONS] for q in qids])


def feature_matrix(queries: Dict[str, dict], qids: Sequence[str]) -> np.ndarray:
    return np.array([queries[q]["features"] for q in qids], dtype=float)


def pooled(paths_by_dataset: Dict[str, str], metric: str = "mrr") -> Tuple[np.ndarray, np.ndarray]:
    """Pooled (X, R) in DATASETS order, query order as stored. Mirrors leak_gradient.load_arm."""
    Xs, Rs = [], []
    for ds in DATASETS:
        q = load_env(paths_by_dataset[ds])
        qids = list(q)
        Xs.append(feature_matrix(q, qids))
        Rs.append(reward_matrix(q, qids, metric))
    return np.vstack(Xs), np.vstack(Rs)


def prepare_out_dir(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.listdir(path):
        raise SystemExit(f"{path} is non-empty. Choose a new --out-dir (no overwrite by design).")
    for protected in (ROUTER_DIR, RESULTS_DIR):
        if os.path.normcase(path) == os.path.normcase(protected):
            raise SystemExit(f"Refusing to write directly into {protected}.")
    os.makedirs(path, exist_ok=True)
    return path


def run_header(args) -> dict:
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                 for k, v in vars(args).items()},
    }


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
