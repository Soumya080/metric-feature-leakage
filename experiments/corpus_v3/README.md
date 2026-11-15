# corpus_v3: leak severity vs. corpus scale

Builds one shared retrieval index that every query searches, at four nested
scales (5K ⊂ 20K ⊂ 50K ⊂ 100K), and rebuilds the six leak-gradient arms at each scale.

The question, *how much does metric-as-feature contamination inflate a
router, as a function of corpus size?*, is not answerable in `baseline_v2`, because
there every query gets its own private 500-passage corpus. There is no corpus to grow,
only a negative-sampling pool.

## Running it

```bash
# 0. Optional: a CUDA build of torch makes stage 02 about 3 minutes instead of 20. Everything else is CPU regardless.
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

# 1. Smoke test first: 5K corpus, 200 queries/dataset, ~8 minutes end to end.
#    --verify asserts the two fast paths against the reference implementations.
python experiments/corpus_v3/01_build_shared_corpus.py --smoke
python experiments/corpus_v3/02_embed_corpus.py        --smoke
python experiments/corpus_v3/03_build_indexes.py       --smoke
python experiments/corpus_v3/04_run_retrieval.py       --smoke --verify
python experiments/corpus_v3/05_build_arms.py          --smoke

# 2. Full run. Stage 02 wants the GPU; the rest is CPU.
python experiments/corpus_v3/01_build_shared_corpus.py --force
python experiments/corpus_v3/02_embed_corpus.py
python experiments/corpus_v3/03_build_indexes.py
python experiments/corpus_v3/04_run_retrieval.py
python experiments/corpus_v3/05_build_arms.py
```

`--force` on stage 01 is needed once because the smoke run leaves a 5K corpus on disk
and every stage is idempotent, it skips work whose output already exists.

### Expected cost

| Stage | Hardware | 100K corpus, ~6,000 queries × 3 datasets |
|---|---|---|
| 01 build corpus | CPU | < 1 min |
| 02 embed (once, cached) | **GPU** | 3–5 min GPU / ~20 min CPU |
| 03 build 4 index pairs | CPU | ~3 min |
| 04 retrieval, 4 scales × 3 datasets | CPU | ~1–3 h |
| 05 build 6 arms × 4 scales | CPU | ~30 min |

Stage 04 dominates. It is sharded per `(scale, dataset)` and resumable, so it can be
stopped and restarted, and a failure in one cell leaves the other eleven on disk.

## What it produces

```
artifacts/
  corpus.jsonl                  the shared corpus, in index order
  corpus_manifest.json          sha256, size, seed, source, counts
  corpus_embeddings.npy         embedded ONCE; every scale slices this
  query_embeddings_{ds}.npy     also feeds the embedding-state control
  scale_{N}/dense.faiss         IndexFlatIP over the first N docs of the seeded order
  scale_{N}/bm25.pkl            precomputed CSR Okapi weights
  scale_{N}/doc_ids.npy         which corpus rows this scale holds
  scale_{N}/env/                the six arms' env data for this scale
results/
  scale_{N}/{ds}_per_query_results.json    same schema as baseline_v2
  retrieval_summary.json                   per-shard Hit@5, answerable share, timing
  arms_report.json                         per-scale integrity verdicts
```

## Design decisions that the result depends on

Nested subsets. One seeded permutation of document ids is computed once; scale *N*
takes its first *N* entries. So the smaller corpora are literal subsets of the larger
and the curve varies corpus size and nothing else. Verified on disk by
`03_build_indexes.py`, which fails if the nesting is violated.

Same ranking function as the 500-passage setting. `bm25_sparse.SparseBM25` is a
vectorised Okapi BM25 with the same constants, tokeniser and IDF formula as
`src/retrieval/sparse.BM25Retriever`. `tests/test_bm25_parity.py` asserts identical
scores and ordering (19 tests). Without this, a change in the leak-vs-scale curve could
be the retriever changing rather than the corpus growing.

Metrics are imported, not reimplemented. `common.load_metric_module()` loads
`02_run_baselines.py` and uses its `check_hit` / `compute_mrr` / `f1_score` /
`compute_recall_at_k` / `substring_match_score` directly. A difference between the two
settings can therefore never be a difference in how Hit or MRR was computed.

Arms are imported, not restated. `05_build_arms.py` reads `ARMS` and `ARM_ORDER`
from `experiments/router_experiments/leakage_ablation.py`, and builds each arm by
calling `src/router/prepare_env_data.py --feature-mode ...`, the same code that built
the 500-passage arms. Nothing about feature construction lives in this directory.

No synthetic passages. `baseline_v2/01_prepare_data.py` fabricates a gold passage
when no real one is found; 41.8% of NQ's gold is such template filler
(`results/corpus_census.json`). corpus_v3 has no equivalent and never synthesises text,
which makes it the *cleaner* of the two settings.

Two Hit@5 numbers, and the difference matters. In the 500-passage setting a gold
passage was inserted into every query's corpus, so every query was answerable by
construction. A shared corpus promises nothing: the answer may simply be absent. Overall
Hit@5 mixes "the retriever failed" with "there was nothing to find", and that mixture
shifts with corpus size, the very axis under study. `04_run_retrieval.py` reports both
and gates on the answerable subset. The unanswerable share is itself a result worth
reporting, not a nuisance.

The sanity gate is against chance, not an absolute band. Chance for top-5 over *N*
documents is 5/N; the gate asks for 20× that. An absolute floor would be miscalibrated,
because absolute Hit@5 legitimately falls as *N* grows. Cells above chance but low in
absolute terms are printed as `warn`, not `FAIL`.

## Integrity checks that must pass before any number is reported

| Check | Where | Why |
|---|---|---|
| BM25 parity with `sparse.py` | `pytest experiments/corpus_v3/tests/` | ranking function is unchanged |
| Nested subsets | `03_build_indexes.py` | only corpus size varies across scales |
| RRF matches reference | `04_run_retrieval.py --verify` | vectorised fusion is not a new system |
| `gold_rank` matches reference | `04_run_retrieval.py --verify` | inverted-index lookup ≡ linear scan |
| Retrieval above chance | `04_run_retrieval.py` | pipeline is functioning at each scale |
| Reward tables identical across arms | `05_build_arms.py` | **only features differ**, else an accuracy gap could be the reward moving |
| features[0:22] and [26:] identical across arms | `05_build_arms.py` | contamination is confined to features 22–25 |
| features[22:26] differ between every arm pair | `05_build_arms.py` | no arm is being compared against itself |
| `leaky` reproduces the bug exactly | `05_build_arms.py` | f22 == BM25 MRR@k, f23 == Dense MRR@k |

## Smoke-run status on the dev machine

Full chain green at 5K/200 queries: parity 19/19, nesting verified, `--verify` clean,
sanity gate passed, all six arms built, integrity clean on all three datasets.

`hotpotqa` at 5K came back `warn` (Hit@5 0.087 among answerable = 87× chance). Expected
for multi-hop questions against a small shared corpus, worth re-checking at 100K, not a
blocker.
