# Data licences and attribution

The MIT licence in `LICENSE` covers the **code** in this repository. It does not and cannot
cover the data, because the sources this work builds on are distributed under share-alike
terms that MIT is incompatible with. This file records what comes from where, under what
terms, and what that means for anyone redistributing the artifacts here.

## Sources

| Source | Licence | Used for |
|---|---|---|
| [Wikipedia](https://en.wikipedia.org) via [`wiki_snippets`](https://huggingface.co/datasets/community-datasets/wiki_snippets) (`wiki40b_en_100_0`) | **CC BY-SA** | The passage pool every candidate set and shared index is drawn from |
| [Natural Questions (NQ-Open)](https://huggingface.co/datasets/google-research-datasets/nq_open) | **CC BY-SA 3.0** | Questions and answers |
| [HotpotQA](https://hotpotqa.github.io/) | **CC BY-SA 4.0** | Questions and answers |
| [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (`rc.nocontext`) | Apache 2.0 | Questions and answers |
| [RouterBench](https://github.com/withmartian/routerbench) | see upstream repository | External audit in the detector section only; no RouterBench data is redistributed here |

Text drawn from Wikipedia remains under CC BY-SA and is attributed to Wikipedia and its
contributors. Question and answer text from NQ and HotpotQA remains under the CC BY-SA
licence of those datasets.

## What this repository does and does not redistribute

**Not redistributed.** The Wikipedia passage pool is no longer tracked in git. It is a
verbatim copy of CC BY-SA text, and shipping it inside an MIT-licensed repository would
have been a licence violation regardless of intent. It is regenerated deterministically:

```bash
python experiments/baseline_v2/01_prepare_data.py
```

That downloads `wiki_snippets` from its original source under its own licence and rebuilds
the identical pool, because the sampling seed is fixed (`RANDOM_SEED = 42`). Everything
downstream — the 500-passage candidate sets, the `corpus_v3` shared index — derives from it
the same way. Nothing is lost by not shipping it; the artifact is reproducible, not unique.

**Redistributed.** `experiments/baseline_v2/results/*_per_query_results.json` and
`src/router/*_env_data*.json` contain question text, answer strings, retrieval scores,
evaluation metrics and feature vectors. They contain **no Wikipedia passage text** — this was
checked, not assumed. The question and answer strings are CC BY-SA (NQ, HotpotQA) or Apache
2.0 (TriviaQA), so **these files are distributed under CC BY-SA 4.0**, not MIT.

## If you redistribute anything from here

- Code: MIT, as in `LICENSE`.
- Any file containing question, answer, or passage text: **CC BY-SA 4.0**. Attribute the
  original sources above, and license your derivative under the same terms.
- If you regenerate the passage pool yourself, you are obtaining it directly from Wikipedia
  under CC BY-SA and the same obligations attach.

## Why this file exists

The repository was initially released MIT-only, with 56 MB of CC BY-SA Wikipedia text
committed and no attribution anywhere. That is a licence violation, and it was found by
audit rather than noticed. Recording it here rather than quietly correcting it is
consistent with how the rest of this project handles its own errors.
