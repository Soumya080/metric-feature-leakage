"""
Data Pipeline: Convert baseline_v2 per-query results → RL environment data.

This script bridges the gap between the evaluation results (experiments/baseline_v2/results/)
and the RL training environment (routing_env.py). It:
  1. Reads raw per-query results JSON (nq_per_query_results.json)
  2. Extracts query features using QueryFeatureExtractor
  3. Computes per-retriever metrics in the env lookup-table format
  4. Outputs router_env_data.json ready for RouterEnvironment

Usage:
    python -m src.router.prepare_env_data
"""

import json
import os
import sys
import time
import re
import zlib
import numpy as np
from typing import Dict, Any, List

# Project root setup for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.router.feature_extractor import QueryFeatureExtractor


# =====================================================================
# LIGHTWEIGHT FEATURE EXTRACTION (No spaCy/sentence-transformers needed)
# =====================================================================

# Standard stopwords (same as in feature_extractor.py)
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "did", 
    "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", 
    "itself", "just", "me", "more", "most", "myself", "no", "nor", "not", "now", 
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", 
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", 
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "with", "won't", "would", "you", "your", "yours", "yourself", "yourselves"
}

QUESTION_WORDS = ["who", "what", "when", "where", "why", "how"]

# Regex patterns for lightweight NER (no spaCy dependency for data preparation)
CAPITALIZED_WORD_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
NUMBER_PATTERN = re.compile(r'\d+')

# =====================================================================
# RETRIEVAL-SCORE FEATURE MODES (see PLAN_leakage_harness.md)
#
# The last 4 of 26 features (avg_bm25, avg_dense, score_gap, bm25_variance) are meant to
# be retrieval CONFIDENCE signals. Three ways to compute them, for the leakage ablation:
#
#   honest_minimal   - raw top-5 score mean/std (the shipped fix). Label-free, but BM25
#                       (~0-30) and cosine similarity (~0.2-0.8) are on incommensurable
#                       scales, so score_gap is mostly "how big are BM25's raw numbers."
#   honest_calibrated - unitless within-retriever signals (margin, normalized entropy),
#                       so the cross-retriever comparison is actually meaningful.
#   leaky            - MRR-derived. Reproduces the ORIGINAL BUG on purpose, to quantify how
#                       much it inflates every downstream router. avg_bm25/avg_dense under
#                       this mode are EXACTLY the reward for actions 0/1 — see
#                       src/router/routing_env.py. Never use this mode for a reported
#                       headline result; it exists only for the ablation comparison.
#
# The three modes below turn "a leak inflates results" from a binary into a dose-response
# curve. honest vs leaky answers whether contamination matters; these answer HOW MUCH
# contamination it takes, which is the more useful question for anyone auditing their own
# router. A reader who concludes "I did not copy the MRR column, so I am fine" should be
# able to see what a partial, coarser, or degraded leak still buys.
#
#   leak_one_arm     - only BM25's MRR leaks; Dense keeps its honest raw score. Models the
#                       realistic case of contaminating one arm rather than all of them.
#   leak_hit         - Hit@k instead of MRR: a coarser, binary label-derived signal. Tests
#                       whether the leak needs the reward's exact value or just its sign.
#   leak_noisy       - MRR plus deterministic jitter. Tests whether a leak survives being
#                       an imprecise proxy — which is what an accidental leak usually is.
# =====================================================================
FEATURE_MODES = ("honest_minimal", "honest_calibrated", "leaky",
                 "leak_one_arm", "leak_hit", "leak_noisy")
DEFAULT_FEATURE_MODE = "honest_minimal"

# Jitter half-width for leak_noisy, on the MRR scale (0, 1]. Large enough that the feature
# is no longer the reward — exact-equality leak detectors miss it entirely — while still
# ranking queries roughly as the reward does.
LEAK_NOISE_SCALE = 0.15


def _mrr_at(m: Dict[str, Any], k: str) -> float:
    """The metrics_at_k MRR — i.e. the reward. Only the leak modes may call this."""
    return float(m.get("metrics_at_k", {}).get(k, {}).get("mrr", 0.0))


def _jitter(value: float, salt: str) -> float:
    """Deterministic pseudo-random offset in [-LEAK_NOISE_SCALE, +LEAK_NOISE_SCALE].

    Derived from the value via crc32 rather than drawn from an RNG, for the same reason
    deterministic_tiebreak exists in routing_env.py: the leak_noisy arm has to produce
    byte-identical env data on every machine and every run, or the ablation comparing it
    against the other arms is measuring the seed.

    Salted per feature so that BM25 and Dense do not receive the same offset when they
    happen to share an MRR value, which would leave their difference uncorrupted.
    """
    h = zlib.crc32(f"{salt}:{float(value)!r}".encode("utf-8"))
    return float(value) + ((h / 2 ** 32) - 0.5) * 2.0 * LEAK_NOISE_SCALE


def _margin(m: Dict[str, Any]) -> float:
    """(top1 - mean(top5)) / (std(top5) + eps): unitless peakedness of this retriever's
    own result set, relative to its own spread. Falls back to the summary fields
    (top1_score/mean_score_top5/std_score_top5) if scores_top10 isn't present, so this
    still works on env data built before 02_run_baselines.py started persisting it."""
    s = m.get("scores_top10") or []
    if len(s) >= 2:
        top5 = np.asarray(s[:5], dtype=float)
        return float((top5[0] - top5.mean()) / (top5.std() + 1e-8))
    top1 = float(m.get("top1_score", 0.0))
    mean5 = float(m.get("mean_score_top5", 0.0))
    std5 = float(m.get("std_score_top5", 0.0))
    return float((top1 - mean5) / (std5 + 1e-8))


def _norm_entropy(m: Dict[str, Any]) -> float:
    """Normalized entropy in [0, 1] of the min-max-scaled top-10 score distribution.
    Min-max scaling first is what makes this comparable across retrievers whose raw scores
    live on different scales (BM25 ~0-30 vs. cosine ~0-1). 1.0 = flat/unconfident,
    0.0 = all mass on one passage."""
    s = np.asarray(m.get("scores_top10") or [], dtype=float)
    if s.size < 2:
        return 0.0
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo + 1e-8)
    p = s / (s.sum() + 1e-8)
    p = p[p > 0]
    if p.size < 2:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def _retrieval_features(bm25_m: Dict[str, Any], dense_m: Dict[str, Any],
                         mode: str = DEFAULT_FEATURE_MODE, k: str = "5") -> List[float]:
    """Compute the 4 retrieval-score features [avg_bm25, avg_dense, score_gap,
    bm25_variance] under the given mode. See FEATURE_MODES docstring above."""
    if mode == "leaky":
        # Reproduces the original bug ON PURPOSE, to quantify it in the ablation.
        f22 = float(bm25_m.get("metrics_at_k", {}).get(k, {}).get("mrr", 0.0))
        f23 = float(dense_m.get("metrics_at_k", {}).get(k, {}).get("mrr", 0.0))
        f24 = f23 - f22
        vals = [float(bm25_m.get("metrics_at_k", {}).get(kv, {}).get("mrr", 0.0))
                for kv in ("1", "3", "5", "10")]
        f25 = float(np.var(vals))
        return [f22, f23, f24, f25]

    if mode == "honest_minimal":
        f22 = float(bm25_m.get("mean_score_top5", bm25_m.get("top1_score", 0.0)))
        f23 = float(dense_m.get("mean_score_top5", dense_m.get("top1_score", 0.0)))
        f24 = f23 - f22
        f25 = float(bm25_m.get("std_score_top5", 0.0)) ** 2
        return [f22, f23, f24, f25]

    if mode == "honest_calibrated":
        f22 = _margin(bm25_m)
        f23 = _margin(dense_m)
        f24 = f23 - f22
        f25 = _norm_entropy(bm25_m)
        return [f22, f23, f24, f25]

    # --- Graded leak variants: see FEATURE_MODES above for what each one asks. ---

    if mode == "leak_one_arm":
        # BM25 contaminated, Dense honest. The two are then on incommensurable scales
        # (MRR in (0,1] against a raw BM25 score in ~0-30), which is precisely what a
        # half-finished fix looks like in practice.
        f22 = _mrr_at(bm25_m, k)
        f23 = float(dense_m.get("mean_score_top5", dense_m.get("top1_score", 0.0)))
        f24 = f23 - f22
        f25 = float(bm25_m.get("std_score_top5", 0.0)) ** 2
        return [f22, f23, f24, f25]

    if mode == "leak_hit":
        f22 = float(bm25_m.get("metrics_at_k", {}).get(k, {}).get("hit", 0.0))
        f23 = float(dense_m.get("metrics_at_k", {}).get(k, {}).get("hit", 0.0))
        f24 = f23 - f22
        vals = [float(bm25_m.get("metrics_at_k", {}).get(kv, {}).get("hit", 0.0))
                for kv in ("1", "3", "5", "10")]
        f25 = float(np.var(vals))
        return [f22, f23, f24, f25]

    if mode == "leak_noisy":
        f22 = _jitter(_mrr_at(bm25_m, k), "bm25")
        f23 = _jitter(_mrr_at(dense_m, k), "dense")
        f24 = f23 - f22
        vals = [_jitter(float(bm25_m.get("metrics_at_k", {}).get(kv, {}).get("mrr", 0.0)),
                        f"bm25@{kv}") for kv in ("1", "3", "5", "10")]
        f25 = float(np.var(vals))
        return [f22, f23, f24, f25]

    raise ValueError(f"unknown feature mode: {mode!r} (expected one of {FEATURE_MODES})")


# Set once per conversion so a missing-IDs warning prints a single line, not 2000.
_MISSING_IDS_WARNED = False

# spaCy pipeline for named-entity features, loaded lazily and only once.
# _NLP: None = not yet attempted, False = unavailable, otherwise the loaded pipeline.
_NLP = None
USE_SPACY_NER = True


def _spacy_entities(query: str):
    """Named-entity counts/flags for a query, or None if spaCy is unavailable.

    Entity structure is the signal most likely to separate the retrievers — a query
    pivoting on a rare named entity favours BM25's exact lexical match, while a
    paraphrastic one favours Dense. The previous regex heuristic could not express this at
    all (it hardcoded all four entity_present_* flags to 0.0), so this feature group has
    never actually been tested.

    Caveat, measured: spaCy's NER degrades on lowercase text. On the NQ-style query
    "who wrote the karate kid 2010 screenplay" it recovers only DATE, whereas on a
    capitalized TriviaQA query it recovers WORK_OF_ART and DATE. So this mainly helps
    TriviaQA and HotpotQA (85%% and 97%% of their queries contain capitalization) and
    barely helps NQ (0%%).
    """
    global _NLP
    if not USE_SPACY_NER:
        return None
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
            print("  spaCy en_core_web_sm loaded for entity features.")
        except Exception as exc:
            print(f"  WARNING: spaCy unavailable ({type(exc).__name__}) — falling back to the "
                  f"regex heuristic, which leaves the 4 entity_present_* features constant "
                  f"at 0. Install with: python -m spacy download en_core_web_sm")
            _NLP = False
    if _NLP is False:
        return None

    doc = _NLP(query)
    labels = {e.label_ for e in doc.ents}
    return {
        "count": len(doc.ents),
        "PERSON": 1.0 if "PERSON" in labels else 0.0,
        "ORG": 1.0 if "ORG" in labels else 0.0,
        # GPE (countries/cities) and LOC (non-GPE places) are distinct spaCy labels.
        "GPE": 1.0 if "GPE" in labels else 0.0,
        "LOC": 1.0 if "LOC" in labels else 0.0,
    }


def _cross_retriever_features(bm25_m: Dict[str, Any], dense_m: Dict[str, Any]) -> List[float]:
    """Compute the 3 cross-retriever agreement features [overlap@5, overlap@10, top1_agree].

    The other feature groups describe the query, or how confident a SINGLE retriever is.
    These describe the relationship BETWEEN retrievers: how much do BM25's and Dense's
    ranked lists actually overlap? Where they agree, every action earns a similar reward
    and routing has nothing to gain; where they diverge, the choice matters. That makes
    disagreement the natural state variable for a routing policy.

    BM25-vs-Dense only: Hybrid is reciprocal-rank fusion OF those two, so its overlap with
    either parent is a function of the fusion, not independent evidence.

    Label-free by construction — this compares the two retrievers' outputs to each other
    and never consults the gold answer, so unlike the MRR-derived features these cannot
    leak reward into the policy's input.
    """
    global _MISSING_IDS_WARNED

    bm25_ids = bm25_m.get("retrieved_ids_top10")
    dense_ids = dense_m.get("retrieved_ids_top10")

    if not bm25_ids or not dense_ids:
        # Pre-fix per-query result files predate retrieved_ids_top10. Returning silent
        # zeros is how the MRR-leak hid for so long, so make the degradation loud.
        if not _MISSING_IDS_WARNED:
            print("  WARNING: 'retrieved_ids_top10' missing from per-query results — the 3 "
                  "cross-retriever features will be 0.0 for every query. Re-run "
                  "02_run_baselines.py (post-2026-08-02) to populate them.")
            _MISSING_IDS_WARNED = True
        return [0.0, 0.0, 0.0]

    def overlap_at(n: int) -> float:
        a, b = set(bm25_ids[:n]), set(dense_ids[:n])
        denom = min(n, len(bm25_ids), len(dense_ids))
        return len(a & b) / denom if denom else 0.0

    top1_agree = 1.0 if bm25_ids[0] == dense_ids[0] else 0.0
    return [overlap_at(5), overlap_at(10), top1_agree]


def extract_lightweight_features(
    query: str,
    bm25_metrics: Dict[str, Any],
    dense_metrics: Dict[str, Any],
    k: str = "5",
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> List[float]:
    """Extract features without heavy NLP models (spaCy/sentence-transformers).

    This produces the same 26-dim feature vector as QueryFeatureExtractor.to_feature_vector()
    but uses regex heuristics instead of spaCy NER, making data preparation fast.

    Feature order matches QueryFeatureExtractor.get_feature_order() exactly.

    feature_mode controls how the LAST 4 features (avg_bm25/avg_dense/score_gap/
    bm25_variance) are computed — see FEATURE_MODES above. Features 0-21 are identical
    across all modes; only the retrieval-score block differs. Default is
    "honest_minimal" (the shipped, label-free fix) — passing "leaky" reproduces the
    original bug on purpose for the leakage ablation and must never be used to generate a
    headline result.
    """
    import math
    
    words = query.strip().split() if query.strip() else []
    query_length = len(words)
    
    # Lexical features (6)
    log_query_length = math.log1p(query_length)
    total_word_len = sum(len(w) for w in words)
    avg_word_length = total_word_len / query_length if query_length > 0 else 0.0
    stopword_count = sum(1 for w in words if w.lower() in STANDARD_STOPWORDS)
    stopword_ratio = stopword_count / query_length if query_length > 0 else 0.0
    unique_words = len(set(w.lower() for w in words))
    unique_word_ratio = unique_words / query_length if query_length > 0 else 0.0
    has_quotes = 1.0 if ('"' in query or "'" in query) else 0.0
    
    # Entity features (7)
    ner = _spacy_entities(query)
    if ner is not None:
        num_entities = float(ner["count"])
        entity_present_person = ner["PERSON"]
        entity_present_org = ner["ORG"]
        entity_present_gpe = ner["GPE"]
        entity_present_loc = ner["LOC"]
        entities = ner["count"] > 0
    else:
        # Regex fallback when spaCy is unavailable. Note this is badly degraded, which is
        # why spaCy is the default: it can only spot capitalized words, and NQ queries are
        # 100%% lowercase ("where does the karate kid 2010 take place"), so num_entities
        # and contains_proper_nouns are 0 for every NQ query under this path. The four
        # entity_present_* columns are 0 for EVERY dataset — a regex cannot classify entity
        # types — so 5-7 of the 29 features were constant before spaCy was wired in.
        capitalized_words = CAPITALIZED_WORD_PATTERN.findall(query)
        entities = [w for w in capitalized_words if w.lower() not in STANDARD_STOPWORDS
                    and w.lower() not in QUESTION_WORDS]
        num_entities = float(len(entities))
        entity_present_person = 0.0
        entity_present_org = 0.0
        entity_present_gpe = 0.0
        entity_present_loc = 0.0
    entity_present_date = 1.0 if NUMBER_PATTERN.search(query) and any(
        m in query.lower() for m in ["january", "february", "march", "april", "may", 
        "june", "july", "august", "september", "october", "november", "december",
        "19", "20"]  # year prefixes
    ) else 0.0
    entity_present_cardinal = 1.0 if NUMBER_PATTERN.search(query) else 0.0
    
    # Structural features (9)
    cleaned = query.lower().strip()
    q_type_vec = {w: 0.0 for w in QUESTION_WORDS}
    q_type_other = 1.0
    for w in QUESTION_WORDS:
        if cleaned.startswith(w):
            q_type_vec[w] = 1.0
            q_type_other = 0.0
            break
    contains_numbers = 1.0 if any(c.isdigit() for c in query) else 0.0
    contains_proper_nouns = 1.0 if entities else 0.0
    
    # Retrieval score features (4) — see FEATURE_MODES / _retrieval_features above.
    avg_bm25, avg_dense, score_gap, bm25_variance = _retrieval_features(
        bm25_metrics, dense_metrics, mode=feature_mode, k=k
    )
    # Cross-retriever agreement features (3). Independent of feature_mode: these are
    # identical across the leaky/honest arms, which leakage_ablation.py asserts.
    xr_overlap5, xr_overlap10, xr_top1 = _cross_retriever_features(
        bm25_metrics, dense_metrics
    )

    # Assemble in canonical order (matches QueryFeatureExtractor.get_feature_order())
    return [
        float(query_length), log_query_length, avg_word_length,
        stopword_ratio, unique_word_ratio, has_quotes,
        num_entities,
        entity_present_person, entity_present_org, entity_present_gpe,
        entity_present_loc, entity_present_date, entity_present_cardinal,
        q_type_vec["who"], q_type_vec["what"], q_type_vec["when"], q_type_vec["where"],
        q_type_vec["why"], q_type_vec["how"], q_type_other,
        contains_numbers, contains_proper_nouns,
        avg_bm25, avg_dense, score_gap, bm25_variance,
        xr_overlap5, xr_overlap10, xr_top1,
    ]


def _append_cross_score_features(env_data: Dict[str, Any], raw_results) -> int:
    """Append 6 cross-scoring features: each retriever's output judged by the OTHER scorer.

    Every score feature above is a retriever's opinion of its own output, which is
    circular — a retriever is always confident about what it just ranked first. Measured
    on real data, both BM25 and Dense score their own top-5 near their own corpus maximum
    on the same query, so self-confidence cannot distinguish the one that succeeded from
    the one that failed.

    The cross terms can. If BM25's top passages are semantically far from the query while
    Dense's are close, Dense probably found the answer. This is the only label-free signal
    available that reflects the query-PASSAGE interaction rather than the query alone —
    and that interaction is precisely what determines which retriever wins, which is why
    query-side features plateau at the majority baseline.

    Everything is normalised by the query's own corpus maximum for that scorer, because
    raw BM25 scores are unbounded and depend on the query's IDF profile; without this the
    features are incomparable across queries.

    The two `_gap` features are the most direct form of the signal: how much better or
    worse one retriever's picks look under the other's scoring function than that
    retriever's own picks do. They are scale-free and signed, so a linear model can use
    them directly.

    Returns the number of features appended (0 if the source data lacks cross scores).
    """
    qids = list(env_data)
    by_qid = {str(e["query_id"]): e for e in raw_results}

    missing = [q for q in qids
               if not by_qid.get(q, {}).get("retrievers", {})
               .get("bm25", {}).get("cross_scores")]
    if missing:
        # Same policy as the embedding features: degrade loudly and completely rather than
        # emitting real values for some queries and zeros for others, which would look like
        # signal and silently corrupt every downstream comparison.
        print(f"  WARNING: {len(missing)}/{len(qids)} queries have no 'cross_scores' — "
              f"skipping cross-scoring features entirely. Re-run 02_run_baselines.py "
              f"(post-2026-08-03) to populate them.")
        return 0

    for qid in qids:
        rec = by_qid[qid]
        norm = rec.get("cross_score_norm", {}) or {}
        b_max = float(norm.get("bm25_max", 0.0)) or 1e-8
        d_max = float(norm.get("dense_max", 0.0)) or 1e-8

        def cs(ret: str, scorer: str) -> float:
            block = rec["retrievers"][ret].get("cross_scores", {}).get(scorer, {})
            return float(block.get("mean", 0.0))

        # Foreign judgements: each retriever's picks under the scorer that did not choose
        # them. Hybrid gets both because RRF fuses both parents, so neither is "foreign".
        bm25_under_dense = cs("bm25", "dense") / d_max
        dense_under_bm25 = cs("dense", "bm25") / b_max
        hybrid_under_bm25 = cs("hybrid", "bm25") / b_max
        hybrid_under_dense = cs("hybrid", "dense") / d_max

        # Self judgements, used only to form the contrasts below.
        bm25_self = cs("bm25", "bm25") / b_max
        dense_self = cs("dense", "dense") / d_max

        # Signed disagreement. Negative dense_gap: BM25 picked passages Dense considers
        # much worse than its own — evidence BM25 is the weaker choice on this query.
        dense_gap = bm25_under_dense - dense_self
        bm25_gap = dense_under_bm25 - bm25_self

        env_data[qid]["features"] = env_data[qid]["features"] + [
            bm25_under_dense, dense_under_bm25,
            hybrid_under_bm25, hybrid_under_dense,
            dense_gap, bm25_gap,
        ]

    print("  Cross-scoring features: 6 appended "
          "(bm25_under_dense, dense_under_bm25, hybrid_under_bm25, hybrid_under_dense, "
          "dense_gap, bm25_gap)")
    return 6


def _append_embedding_features(env_data: Dict[str, Any], raw_results, embed_dims: int,
                               split_seed: int = 42):
    """PCA-reduce the persisted query embeddings and append them to each feature vector.

    The 22 handcrafted query features are surface statistics — length, stopword ratio,
    question word. Whether Dense beats BM25 plausibly depends on what the query *means*,
    which none of them encode. 02_run_baselines.py already computes a 384-dim MiniLM
    embedding per query for dense retrieval; this makes it available to the router.

    Reduced rather than used raw, because 384 dimensions against ~10^3 training queries
    overfits badly — the classifier memorises queries instead of learning a routing rule.

    The scaler and PCA are fitted on the TRAINING SPLIT ONLY, then applied to val and test.
    An earlier version fitted over all queries and argued that PCA is unsupervised, so it
    never sees a reward and therefore cannot leak labels. That argument is correct about
    *label* leakage and beside the point about *transductive* leakage: the test set's
    feature distribution still shapes the components, and with --embed-dims 32 that is 32
    of 61 dimensions carrying information from rows the model is about to be scored on.
    The honest answer to "were the PCA components fit on the test split?" has to be no —
    especially in a paper whose contribution is that leakage inflates routing results.

    The per-component renormalisation below is fitted on train statistics for the same
    reason; dividing by all-query std would put test rows back into the transform.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  WARNING: scikit-learn unavailable — skipping embedding features.")
        return

    qids = list(env_data)
    by_qid = {str(e["query_id"]): e.get("query_embedding") for e in raw_results}
    embs = [by_qid.get(q) for q in qids]
    if any(e is None or len(e) == 0 for e in embs):
        n_missing = sum(1 for e in embs if not e)
        print(f"  WARNING: {n_missing}/{len(qids)} queries have no 'query_embedding' — "
              f"skipping embedding features entirely rather than emitting zero rows for "
              f"some queries and real ones for others. Re-run 02_run_baselines.py "
              f"(post-2026-08-02) to populate them.")
        return

    X = np.asarray(embs, dtype=float)
    n_comp = int(min(embed_dims, X.shape[0] - 1, X.shape[1]))
    if n_comp < 1:
        print(f"  WARNING: too few queries ({X.shape[0]}) for PCA — skipping embeddings.")
        return

    # Fit on train ids only, using the same partition RouterEnvironment will use later.
    # split_query_ids is imported from routing_env rather than reimplemented so the two
    # cannot drift -- a different partition here would fit the transform on rows that end
    # up in test, which is exactly the leak being removed.
    from src.router.routing_env import split_query_ids
    train_ids, _, _ = split_query_ids(qids, split_seed=split_seed)
    train_pos = [i for i, q in enumerate(qids) if str(q) in set(train_ids)]
    if len(train_pos) < 2:
        print(f"  WARNING: only {len(train_pos)} training queries — skipping embeddings.")
        return
    n_comp = int(min(n_comp, len(train_pos) - 1))

    scaler = StandardScaler().fit(X[train_pos])
    Z = scaler.transform(X)
    pca = PCA(n_components=n_comp, random_state=42).fit(Z[train_pos])
    P = pca.transform(Z)
    # Scale to unit variance per component so the policy's z-scoring, and any linear model,
    # see embedding dims on the same footing as the handcrafted features. Train std, not
    # all-query std, or the test rows re-enter the transform through the denominator.
    P = P / (P[train_pos].std(axis=0, keepdims=True) + 1e-8)

    ev = float(pca.explained_variance_ratio_.sum())
    print(f"  Embedding features: {X.shape[1]}-dim -> {n_comp} PCA components "
          f"({ev*100:.1f}% variance retained; scaler+PCA fitted on "
          f"{len(train_pos)}/{len(qids)} training queries only)")

    for qid, row in zip(qids, P):
        env_data[qid]["features"] = env_data[qid]["features"] + [float(v) for v in row]


def convert_per_query_results(
    results_path: str,
    output_path: str,
    k: str = "5",
    use_spacy: bool = False,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    gold_type: str = "all",
    reward: str = "rank",
    embed_dims: int = 0,
    cross_scores: bool = False,
) -> Dict[str, Any]:
    """Convert raw baseline_v2 per-query results to env lookup-table format.

    Args:
        results_path: Path to nq_per_query_results.json or triviaqa_per_query_results.json.
        output_path: Where to save the processed env data JSON.
        k: Which k value to extract metrics at (default "5" for Hit@5).
        use_spacy: If True, use QueryFeatureExtractor with spaCy (slower but more accurate).
                   If False, use lightweight regex-based features (fast, no dependencies).
        feature_mode: How to compute the 4 retrieval-score features — one of
                      FEATURE_MODES. Default "honest_minimal". Features 0-21 are
                      IDENTICAL across modes regardless of this setting.
        gold_type: Keep only queries whose gold passage has this provenance —
                   "real_wikipedia", "constructed", or "all" (no filtering).
                   01_prepare_data.py falls back to a generated template passage
                   (construct_gold()) when no real Wikipedia passage containing the
                   answer is found in the pool. That template repeats the answer but
                   shares no terms with the question, so BM25 systematically fails on
                   it while Dense does not — a data-generation artifact, not a real
                   retrieval difference. Pass "real_wikipedia" for the headline
                   results; "all" reproduces the unfiltered numbers.

    Returns:
        The env data dict (also saved to output_path).
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")

    print(f"\n{'='*60}")
    print(f"Converting: {os.path.basename(results_path)}")
    print(f"Output:     {os.path.basename(output_path)}")
    print(f"Metric k:   {k}")
    print(f"Feature mode: {feature_mode}")
    print(f"{'='*60}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)
    
    print(f"Loaded {len(raw_results)} queries from {os.path.basename(results_path)}")

    if gold_type != "all":
        before = len(raw_results)
        raw_results = [e for e in raw_results if e.get("gold_type") == gold_type]
        print(f"Gold-type filter {gold_type!r}: kept {len(raw_results)}/{before} queries")
        if not raw_results:
            raise ValueError(
                f"No queries left after filtering {results_path} to gold_type={gold_type!r}. "
                f"Check that 02_run_baselines.py wrote a 'gold_type' field (it does as of the "
                f"post-fix version); older per-query result files predate it."
            )

    # Optional: use full spaCy-based feature extractor
    extractor = None
    if use_spacy:
        print("Initializing spaCy-based feature extractor (slower but more accurate)...")
        extractor = QueryFeatureExtractor(include_embeddings=False)
    
    env_data = {}
    start_time = time.time()
    
    for i, entry in enumerate(raw_results):
        qid = str(entry["query_id"])
        question = entry.get("question", "")
        
        # Extract retriever metrics at target k
        retrievers_env = {}
        n_corpus = int(entry.get("n_corpus", 500))
        for ret_name in ["bm25", "dense", "hybrid"]:
            ret_data = entry.get("retrievers", {}).get(ret_name, {})
            k_metrics = ret_data.get("metrics_at_k", {}).get(k, {})

            mrr_at_k = float(k_metrics.get("mrr", 0.0))
            # Dense reward: reciprocal rank of the first answer-bearing passage over the
            # WHOLE corpus. MRR@k truncates to 0 beyond rank k, which on this data leaves
            # 70-83% of queries with every retriever at 0 — all actions tied, no gradient
            # for a policy, and argmax labels that are pure tie-break artifact. The
            # full-corpus rank keeps the signal graded (rank 40 really is better than rank
            # 300) and is non-zero for essentially every query.
            gold_rank = ret_data.get("gold_rank")
            if gold_rank is None:
                # Retriever never ranked an answer-bearing passage (BM25 drops zero-score
                # passages). Treat as strictly worse than any achievable rank rather than
                # dropping the query or inventing a finite rank.
                gold_rank = n_corpus + 1
            rank_reward = 1.0 / float(gold_rank)

            retrievers_env[ret_name] = {
                "hit_at_5": 1.0 if k_metrics.get("hit", False) else 0.0,
                # "mrr" is what routing_env serves as the reward. Which quantity it holds
                # depends on --reward; both are always stored so evaluation can report
                # MRR@k regardless of what training optimised.
                "mrr": rank_reward if reward == "rank" else mrr_at_k,
                f"mrr_at_{k}": mrr_at_k,
                "rank_reward": rank_reward,
                "gold_rank": int(gold_rank),
                "f1": float(k_metrics.get("f1", 0.0)),
                "recall": float(k_metrics.get("recall", 0.0)),
                "em": float(k_metrics.get("em", 0.0)),
                "latency_ms": float(ret_data.get("latency_ms", 0.0)),
            }
        
        # Extract features
        if use_spacy and extractor is not None:
            bm25_data = entry.get("retrievers", {}).get("bm25", {})
            dense_data = entry.get("retrievers", {}).get("dense", {})

            # Same _retrieval_features() as the non-spacy path below, so both paths agree
            # under every feature_mode — see FEATURE_MODES above for why mrr (leaky mode
            # excepted, on purpose) is not used here.
            avg_bm25, avg_dense, _score_gap, bm25_variance = _retrieval_features(
                bm25_data, dense_data, mode=feature_mode, k=k
            )

            flat_record = extractor.extract_query_features(
                query_id=qid, query=question,
                avg_bm25=avg_bm25,
                avg_dense=avg_dense,
                bm25_variance=bm25_variance
            )
            # extract_query_features() knows nothing about cross-retriever agreement, and
            # to_feature_vector() defaults unknown keys to 0.0 — so without this the spaCy
            # path would silently emit three zeros where the lightweight path emits real
            # values. Inject them explicitly to keep the two paths identical.
            xr5, xr10, xr1 = _cross_retriever_features(bm25_data, dense_data)
            flat_record["cross_ret_overlap_5"] = xr5
            flat_record["cross_ret_overlap_10"] = xr10
            flat_record["cross_ret_top1_agree"] = xr1
            features = QueryFeatureExtractor.to_feature_vector(flat_record)
        else:
            bm25_data = entry.get("retrievers", {}).get("bm25", {})
            dense_data = entry.get("retrievers", {}).get("dense", {})
            features = extract_lightweight_features(question, bm25_data, dense_data, k=k,
                                                      feature_mode=feature_mode)
        
        # Determine oracle best retriever for this query
        best_ret = max(retrievers_env.keys(), 
                       key=lambda r: retrievers_env[r]["hit_at_5"])
        
        env_data[qid] = {
            "question": question,
            "features": features,
            "retrievers": retrievers_env,
            "oracle_best": best_ret,
            # Carried through from 01_prepare_data.py so results can be stratified by
            # whether the gold passage is a REAL Wikipedia snippet or a template-generated
            # one (construct_gold()). ~56% of NQ gold passages are "constructed", and
            # retrievers behave measurably differently on them, so any headline number
            # needs to be checkable on the real_wikipedia subset alone. Purely additive
            # metadata — does not affect features or rewards, so it cannot change training.
            "gold_type": entry.get("gold_type", "unknown"),
        }
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i+1}/{len(raw_results)} queries ({elapsed:.1f}s)")

    base_dim = len(next(iter(env_data.values()))["features"]) if env_data else 0
    n_embed = 0
    if embed_dims > 0:
        before = base_dim
        _append_embedding_features(env_data, raw_results, embed_dims)
        after = len(next(iter(env_data.values()))["features"]) if env_data else 0
        n_embed = after - before

    # Appended last, so the embedding block keeps the contiguous range [29, 29+n_embed)
    # that existing consumers assume. Off by default: turning it on changes the feature
    # dimension, which invalidates saved checkpoints and makes results incomparable, so it
    # must be an explicit choice that produces its own env suffix.
    n_cross = _append_cross_score_features(env_data, raw_results) if cross_scores else 0

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)

    # Feature-group boundaries as a SIDECAR, not a key inside env_data: RouterEnvironment
    # treats every top-level key as a query id, so a metadata key would be parsed as a
    # malformed query. Consumers that need group ranges (ablations, per-retriever routing)
    # should read this instead of hardcoding index arithmetic.
    layout = {
        "handcrafted": [0, 29],
        "embedding": [29, 29 + n_embed],
        "cross_score": [29 + n_embed, 29 + n_embed + n_cross],
        "total_dim": 29 + n_embed + n_cross,
        "cross_score_names": ["bm25_under_dense", "dense_under_bm25", "hybrid_under_bm25",
                              "hybrid_under_dense", "dense_gap", "bm25_gap"][:n_cross],
    }
    with open(os.path.splitext(output_path)[0] + "_layout.json", "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nDone! Processed {len(env_data)} queries in {elapsed:.1f}s")
    print(f"Saved to: {output_path}")
    
    # Print summary statistics
    _print_env_summary(env_data)
    
    return env_data


def _print_env_summary(env_data: Dict[str, Any]):
    """Print summary statistics of the converted environment data."""
    total = len(env_data)
    if total == 0:
        print("  No queries in dataset!")
        return
    
    # Feature dimensionality
    first_qid = next(iter(env_data))
    feat_dim = len(env_data[first_qid]["features"])
    
    # Oracle action distribution
    oracle_counts = {"bm25": 0, "dense": 0, "hybrid": 0}
    hit_counts = {"bm25": 0, "dense": 0, "hybrid": 0}
    
    for qid, entry in env_data.items():
        oracle_counts[entry["oracle_best"]] += 1
        for ret_name in ["bm25", "dense", "hybrid"]:
            if entry["retrievers"][ret_name]["hit_at_5"] > 0:
                hit_counts[ret_name] += 1
    
    print(f"\n  Summary:")
    print(f"  |-- Total queries:    {total}")
    print(f"  |-- Feature dim:      {feat_dim}")
    print(f"  |-- Oracle best dist: BM25={oracle_counts['bm25']} | Dense={oracle_counts['dense']} | Hybrid={oracle_counts['hybrid']}")
    print(f"  +-- Hit@5 rates:      BM25={hit_counts['bm25']/total:.1%} | Dense={hit_counts['dense']/total:.1%} | Hybrid={hit_counts['hybrid']/total:.1%}")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert baseline_v2 per-query results to RL env lookup tables."
    )
    parser.add_argument(
        "--feature-mode", type=str, default=DEFAULT_FEATURE_MODE, choices=FEATURE_MODES,
        help=f"How to compute the 4 retrieval-score features (default: {DEFAULT_FEATURE_MODE}). "
             "'leaky' reproduces the original MRR-leakage bug ON PURPOSE for the ablation "
             "in leakage_ablation.py — never use it for a reported headline result.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix for output filenames, e.g. '_leaky' -> nq_env_data_leaky.json. "
             "Default '' preserves today's exact filenames/behavior.",
    )
    parser.add_argument(
        "--k", type=str, default="5", choices=["1", "3", "5", "10", "20"],
        help="Cutoff k for the reward metric, i.e. the reward becomes MRR@k (default 5). "
             "Already computed by 02_run_baselines.py at every k, so changing this costs "
             "no extra compute. It matters because the reward is extremely sparse at k=5: "
             "on real-Wikipedia queries 70-83%% have all three retrievers scoring 0, which "
             "leaves nothing to learn from. Raising k to 20 roughly doubles the share of "
             "queries with a strictly-best action (NQ 25%%->46%%, HotpotQA 14%%->35%%) while "
             "leaving oracle headroom unchanged or slightly better.",
    )
    parser.add_argument(
        "--reward", type=str, default="rank", choices=["rank", "mrr"],
        help="Training reward. 'rank' (default) = 1/rank of the first answer-bearing "
             "passage over the WHOLE corpus; 'mrr' = MRR@k, the original. MRR@k is 0 "
             "beyond rank k, which leaves 70-83%% of queries with every retriever tied at "
             "0 and nothing to learn from; the rank reward is non-zero for essentially "
             "every query and raises the share with a strictly-best action from ~33%% to "
             "~92%%. Both are always stored, so evaluation still reports MRR@k.",
    )
    parser.add_argument(
        "--embed-dims", type=int, default=32,
        help="PCA components of the query embedding to append as features (default 32; "
             "0 disables). Raw 384-dim embeddings overfit at this sample size, so they "
             "are reduced. Tune on the validation split, never on test.",
    )
    parser.add_argument(
        "--cross-scores", action="store_true",
        help="Append 6 cross-scoring features: each retriever's top-5 scored under the "
             "OTHER retriever's scoring function. Off by default because it changes the "
             "feature dimension — always pair it with a --suffix (e.g. _xscore) so the "
             "result is a separate, comparable arm rather than an overwrite. This is the "
             "only feature group that measures the query-PASSAGE interaction, which is "
             "what actually determines which retriever wins; every other group describes "
             "the query alone or a retriever's confidence in its own output.",
    )
    parser.add_argument(
        "--no-spacy-ner", action="store_true",
        help="Disable spaCy entity features and fall back to the regex heuristic. The "
             "fallback leaves the 4 entity_present_* features constant at 0 (and, on "
             "NQ's all-lowercase queries, num_entities too), so this is for debugging "
             "or environments without the spaCy model.",
    )
    parser.add_argument(
        "--gold-type", type=str, default="all",
        choices=["all", "real_wikipedia", "constructed"],
        help="Keep only queries whose gold passage has this provenance. Use "
             "'real_wikipedia' (with --suffix _real) for the headline results: it drops "
             "queries whose gold passage is a generated template, which BM25 cannot match "
             "by construction. Default 'all' reproduces the unfiltered numbers.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Where to read *_per_query_results.json from (default: "
             "experiments/baseline_v2/results). Use a scratch dir for smoke tests.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write *_env_data*.json (default: src/router).",
    )
    cli_args = parser.parse_args()
    USE_SPACY_NER = not cli_args.no_spacy_ner

    # Paths relative to project root
    RESULTS_DIR = cli_args.results_dir or os.path.join(
        PROJECT_ROOT, "experiments", "baseline_v2", "results")
    OUTPUT_DIR = cli_args.output_dir or os.path.join(PROJECT_ROOT, "src", "router")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("RL Router Data Preparation Pipeline")
    print(f"Feature mode: {cli_args.feature_mode} | Suffix: {cli_args.suffix!r} | "
          f"Gold type: {cli_args.gold_type} | Reward: {cli_args.reward} (k={cli_args.k}) | "
          f"Embed dims: {cli_args.embed_dims} | spaCy NER: {USE_SPACY_NER}")
    print("=" * 60)

    datasets = [
        ("NQ", "nq_per_query_results.json", f"nq_env_data{cli_args.suffix}.json"),
        ("TriviaQA", "triviaqa_per_query_results.json", f"triviaqa_env_data{cli_args.suffix}.json"),
        ("HotpotQA", "hotpotqa_per_query_results.json", f"hotpotqa_env_data{cli_args.suffix}.json"),
    ]

    for name, input_name, output_name in datasets:
        input_path = os.path.join(RESULTS_DIR, input_name)
        output_path = os.path.join(OUTPUT_DIR, output_name)

        if os.path.exists(input_path):
            convert_per_query_results(input_path, output_path, k=cli_args.k, use_spacy=False,
                                       feature_mode=cli_args.feature_mode,
                                       gold_type=cli_args.gold_type,
                                       reward=cli_args.reward,
                                       embed_dims=cli_args.embed_dims,
                                       cross_scores=cli_args.cross_scores)
        else:
            print(f"WARNING: {name} results not found at {input_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Environment data files saved to: {OUTPUT_DIR}")
    print("=" * 60)
