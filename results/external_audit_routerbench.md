# External Audit — RouterBench

_36,497 queries x 11 language models, 86 benchmarks. Source: `routerbench_0shot.pkl`._


Two questions: can published routing work be audited for metric-as-feature leakage at all, and do this repository's structural findings reproduce in a different domain with different arms, a different metric, and independent authors?


## Q1 — Is the leak auditable from published artifacts?

| shipped by RouterBench | present? | lets a reader check for the leak? |
|---|---|---|
| raw prompt text | yes | partly -- features must be re-derived, so a paper's own matrix is still unavailable |
| per-model performance (the reward table) | yes | no -- this is the reward, not the router's input |
| per-model cost | yes | no |
| per-model responses | yes | no |
| oracle routing label | yes | no |
| the feature matrix a router was trained on | no | this is the only artifact that would make the leak checkable |

**No. RouterBench ships no feature matrix, so metric-as-feature leakage is unauditable from this release -- and it is the most widely used public artifact in LLM routing. A reader cannot distinguish a router trained on label-free query features from one trained on features derived from the evaluation metric.**


This is not a criticism of RouterBench, which is a performance benchmark and never claimed to ship router inputs. It is a statement about what published artifacts make checkable: a paper reporting that its router uses "query score statistics" as features cannot be verified or falsified by a reader, because the feature matrix is not part of the release. The hazard is therefore a property of reporting conventions, not of any one paper.


## Q2 — Do the structural findings reproduce here?

| quantity | RouterBench (11 LLMs) | RL-Route (3 retrievers, pooled) |
|---|---:|---:|
| queries | 36,497 | 3,612 |
| arms | 11 | 3 |
| best fixed arm | gpt-4-1106-preview 0.7814 | dense 0.1682 |
| oracle | 0.9121 | 0.2245 |
| oracle headroom | **+0.1307** (17%) | **+0.0564** (34%) |
| all-arms-tied rate | 93.0% | 12.1% |
| majority-class share | **30.1%** (gpt-4-1106-preview) | **60.9%** (dense) |

The majority-class share is the number that matters: it is the accuracy a router achieves by ignoring its input entirely and always naming the same arm. A learned router has to beat it to be worth anything.


## Q2b — Oracle fragility, same two diagnostics as `oracle_inflation.py`

| injected noise | +0% sd | +5% sd | +10% sd | +25% sd |
|---|---:|---:|---:|---:|
| headroom | 0.1307 | 0.1569 | 0.1831 | 0.2620 |

| headroom from margins | <= 1% | <= 5% | <= 10% | <= 25% |
|---|---:|---:|---:|---:|
| share | 69.9% | 69.9% | 69.9% | 70.0% |

As in `oracle_inflation.md`: a max over arms inflates under any added noise, so the sweep is a sensitivity and not proof of inflation. Quote the +10% column.


## Q1b — leakcheck on prompt-only features

Audited 12 label-free features derived from the prompt text alone against the 11-arm reward table.

```
leakcheck 0.1.0: 36497 rows, 12 features, 11 arms
  [SUSPECT] monotone_derived  n_question_marks vs meta/llama-2-70b-chat: Spearman rho = 0.881 (strong) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.014 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs mistralai/mistral-7b-chat: Spearman rho = 0.867 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.014 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.852 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.029 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs claude-instant-v1: Spearman rho = 0.827 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.054 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs claude-v2: Spearman rho = 0.821 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.059 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.819 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.061 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs claude-v1: Spearman rho = 0.817 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.063 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs gpt-3.5-turbo-1106: Spearman rho = 0.817 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.064 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.812 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.001 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs meta/llama-2-70b-chat: Spearman rho = 0.811 (strong) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.001 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.808 (strong) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.072 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.804 (strong) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.076 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs claude-instant-v1: Spearman rho = 0.795 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.017 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs claude-v1: Spearman rho = 0.791 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.021 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs gpt-3.5-turbo-1106: Spearman rho = 0.788 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.024 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.788 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.024 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs claude-v2: Spearman rho = 0.787 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.025 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs mistralai/mistral-7b-chat: Spearman rho = 0.785 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.027 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_question_marks vs gpt-4-1106-preview: Spearman rho = 0.784 (strong) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.096 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.777 (strong) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.035 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs meta/llama-2-70b-chat: Spearman rho = 0.755 (strong) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.029 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs gpt-4-1106-preview: Spearman rho = 0.750 (strong) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.062 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.726 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.029 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs mistralai/mistral-7b-chat: Spearman rho = 0.720 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.035 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs claude-instant-v1: Spearman rho = 0.701 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.054 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs claude-v1: Spearman rho = 0.700 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.055 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs claude-v2: Spearman rho = 0.693 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.062 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs gpt-3.5-turbo-1106: Spearman rho = 0.690 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.065 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.689 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.065 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.683 (strong) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.072 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_digits vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.683 (strong) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.129 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs gpt-4-1106-preview: Spearman rho = 0.661 (strong) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.094 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_upper vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.647 (strong) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.108 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.626 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.031 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.606 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.020 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  longest_word vs meta/llama-2-70b-chat: Spearman rho = 0.599 (strong) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.179 - arm-specific, which is what a real leak looks like
  [SUSPECT] monotone_derived  n_words vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.596 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.031 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.587 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.020 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs mistralai/mistral-7b-chat: Spearman rho = 0.586 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.041 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.585 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.023 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.581 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.022 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs gpt-3.5-turbo-1106: Spearman rho = 0.567 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.059 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs claude-v1: Spearman rho = 0.564 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.062 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.562 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.023 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs claude-instant-v1: Spearman rho = 0.562 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.064 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.559 (strong) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.022 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs mistralai/mistral-7b-chat: Spearman rho = 0.559 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.048 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs claude-v2: Spearman rho = 0.551 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.075 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  type_token_ratio vs claude-v1: Spearman rho = 0.549 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.052 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs mistralai/mistral-7b-chat: Spearman rho = 0.539 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.046 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs gpt-3.5-turbo-1106: Spearman rho = 0.536 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.070 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs mistralai/mistral-7b-chat: Spearman rho = 0.531 (strong) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.049 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs gpt-3.5-turbo-1106: Spearman rho = 0.531 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.050 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs claude-v1: Spearman rho = 0.530 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.076 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs gpt-3.5-turbo-1106: Spearman rho = 0.528 (strong) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.056 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs claude-instant-v1: Spearman rho = 0.528 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.078 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs claude-v1: Spearman rho = 0.523 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.058 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs claude-v1: Spearman rho = 0.522 (strong) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.063 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.521 (strong) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.106 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs claude-instant-v1: Spearman rho = 0.518 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.062 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs meta/llama-2-70b-chat: Spearman rho = 0.518 (strong) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.108 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs claude-instant-v1: Spearman rho = 0.517 (strong) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.068 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs gpt-4-1106-preview: Spearman rho = 0.516 (strong) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.110 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs claude-v2: Spearman rho = 0.515 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.066 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_words vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.514 (strong) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.112 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  type_token_ratio vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.514 (strong) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.087 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs claude-v2: Spearman rho = 0.508 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.077 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs claude-v2: Spearman rho = 0.505 (strong) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.101 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.502 (strong) on 19499 non-zero-reward rows with only 0.0% exact agreement; arm selectivity +0.015 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  type_token_ratio vs gpt-4-1106-preview: Spearman rho = 0.498 (moderate) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.103 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.487 (moderate) on 23631 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.015 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs gpt-4-1106-preview: Spearman rho = 0.484 (moderate) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.097 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.477 (moderate) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.103 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs gpt-4-1106-preview: Spearman rho = 0.477 (moderate) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.108 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs mistralai/mistral-7b-chat: Spearman rho = 0.471 (moderate) on 15649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.031 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs gpt-4-1106-preview: Spearman rho = 0.467 (moderate) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.139 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.464 (moderate) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.121 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs claude-v1: Spearman rho = 0.460 (moderate) on 25709 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.042 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.460 (moderate) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.146 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs meta/llama-2-70b-chat: Spearman rho = 0.458 (moderate) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.123 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_chars vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.451 (moderate) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.130 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.444 (moderate) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.141 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs claude-instant-v1: Spearman rho = 0.441 (moderate) on 24711 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.061 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs gpt-3.5-turbo-1106: Spearman rho = 0.436 (moderate) on 25649 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.066 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs claude-v2: Spearman rho = 0.429 (moderate) on 25812 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.073 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_unique_words vs meta/llama-2-70b-chat: Spearman rho = 0.428 (moderate) on 15647 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.157 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  longest_word vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.419 (moderate) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.179 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs zero-one-ai/Yi-34B-Chat: Spearman rho = 0.415 (moderate) on 27097 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.087 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  mean_word_len vs gpt-4-1106-preview: Spearman rho = 0.389 (moderate) on 31165 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.113 - spread across arms, which is what a confound looks like
  [SUSPECT] monotone_derived  n_punct vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.372 (moderate) on 11537 non-zero-reward rows with only 0.0% exact agreement; arm selectivity -0.234 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs WizardLM/WizardLM-13B-V1.2: Spearman rho = 0.601 (strong) on 19499 non-zero-reward rows with only 0.6% exact agreement; arm selectivity +0.028 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs mistralai/mixtral-8x7b-chat: Spearman rho = 0.573 (strong) on 23631 non-zero-reward rows with only 0.6% exact agreement; arm selectivity -0.028 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs mistralai/mistral-7b-chat: Spearman rho = 0.571 (strong) on 15649 non-zero-reward rows with only 0.8% exact agreement; arm selectivity -0.030 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs gpt-3.5-turbo-1106: Spearman rho = 0.551 (strong) on 25649 non-zero-reward rows with only 0.7% exact agreement; arm selectivity -0.050 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs claude-instant-v1: Spearman rho = 0.544 (strong) on 24711 non-zero-reward rows with only 0.7% exact agreement; arm selectivity -0.057 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs claude-v2: Spearman rho = 0.539 (strong) on 25812 non-zero-reward rows with only 0.7% exact agreement; arm selectivity -0.062 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs meta/llama-2-70b-chat: Spearman rho = 0.528 (strong) on 15647 non-zero-reward rows with only 0.7% exact agreement; arm selectivity -0.073 - spread across arms, which is what a confound looks like
  [SUSPECT] jittered          type_token_ratio vs meta/code-llama-instruct-34b-chat: Spearman rho = 0.498 (moderate) on 11537 non-zero-reward rows with only 0.9% exact agreement; arm selectivity -0.103 - spread across arms, which is what a confound looks like
  98 SUSPECT finding(s) on column(s) [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]. These are NOT verdicts: a confound through a shared upstream cause produces the same signature as a monotone leak. Check the selectivity figures, then resolve by ablation - retrain without these columns and see if the result survives.
```


**No verdicts, as it must be: features built from the prompt alone cannot encode a model's score. But 98 SUSPICIONS, and those are the interesting result. Prompt-length and punctuation-count features reach |rho| = 0.88 against a model's reward, with zero exact agreement and near-zero arm selectivity. The association is real (Pearson agrees), and its cause is that RouterBench pools 86 benchmarks whose prompts differ in format: a feature like 'number of question marks' is effectively a benchmark identifier, and benchmark identity predicts which model wins.

This is a confound, not a leak, and it is what forced leakcheck's central design decision. An earlier version called a verdict on strong rank correlation and would have reported all 98 of these as leakage. A confound through a shared upstream cause is indistinguishable from a monotone leak by correlation alone at any threshold, so only exact agreement can be a verdict. This dataset is the falsifying case that establishes it.**

