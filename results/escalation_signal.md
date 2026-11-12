# Escalation Signal: is a retriever's failure detectable from its own output?

Binary probe per retriever — did it fail to rank an answer-bearing passage within
k? — using only evidence available after running THAT retriever. ROC-AUC under
repeated stratified CV; 0.5 is chance.

`pre_retrieval` is the control (query features only, what a pre-retrieval router
sees). `+cross_score` adds the retriever's top-5 judged by the other retriever's
scoring function. The gap between them is the value of looking after retrieving,
and it is the number that decides whether a cascade is worth building.

| dataset | retriever | fail rate | pre-retrieval AUC | +self AUC | +cross AUC | post-pre |
|---|---|---:|---:|---:|---:|---:|
| nq | bm25 | 70.2% | 0.533±0.038 | 0.541±0.038 | 0.553±0.034 | +0.020 |
| nq | dense | 58.6% | 0.572±0.029 | 0.602±0.034 | 0.599±0.030 | +0.028 |
| triviaqa | bm25 | 66.1% | 0.519±0.037 | 0.568±0.031 | 0.582±0.031 | +0.062 |
| triviaqa | dense | 55.2% | 0.519±0.027 | 0.549±0.031 | 0.556±0.023 | +0.036 |
| hotpotqa | bm25 | 73.7% | 0.542±0.042 | 0.540±0.038 | 0.536±0.032 | -0.006 |
| hotpotqa | dense | 69.9% | 0.555±0.038 | 0.553±0.040 | 0.538±0.050 | -0.017 |

## Verdict

A cascade is worth building if `+cross_score` AUC clears ~0.55 and improves materially on `pre_retrieval`. That means the cheap retriever's own output tells you when it failed, so escalation can be targeted rather than random.

If every column sits near 0.5, failure is not detectable, escalation would be arbitrary, and the cascade should be dropped BEFORE the policy is built — which is the point of running this first.

Note the asymmetry this whole line of work rests on: predicting which retriever will win requires knowing the query-passage interaction in advance, while detecting that a retriever just failed only requires looking at what it returned. The second is strictly easier, and only the second is needed.

