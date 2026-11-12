# Online Adaptation — Segment Averages

Auto-generated from results/online_adaptation_results.json. Paste the relevant
numbers into the Results section text — right now this effect only appears in
Figure 1, which requires the reader to eyeball a plot.

| Segment | Frozen GBT | Frozen RL | Online RL | Online RL vs. Frozen GBT |
|---|---|---|---|---|
| NQ (source) | 0.1559 | 0.1461 | 0.1446 | -0.0113 (-7.2%) |
| TriviaQA (shift 1) | 0.1936 | 0.1964 | 0.1771 | -0.0165 (-8.5%) |
| HotpotQA (shift 2) | 0.0779 | 0.0579 | 0.0586 | -0.0193 (-24.8%) |

## Suggested prose

"On the final HotpotQA segment, online RL-Route reaches 0.0586 MRR against the frozen Gradient Boosting model's 0.0779 — -24.8%. Online policy-gradient updates on the observed reward did not close the gap to the frozen supervised baseline on this segment."

**Scoreboard.** Online RL-Route beats frozen GBT on 0 of 3 segments. It is beaten by the untrained frozen network on 2 of 3 (NQ (source), TriviaQA (shift 1)).

> Winning a minority of segments does not support an adaptation claim. State the per-segment numbers rather than the aggregate, or drop the experiment.

## LaTeX table

\begin{table}[h]
\centering
\caption{Online Adaptation: Segment-Averaged MRR@5 as the query stream shifts NQ $\to$ TriviaQA $\to$ HotpotQA.}
\begin{tabular}{lccc}
\toprule
\textbf{Segment} & \textbf{Frozen GBT} & \textbf{Frozen RL} & \textbf{Online RL (Ours)} \\
\midrule
NQ (source) & 0.1559 & 0.1461 & 0.1446 \\
TriviaQA (shift 1) & 0.1936 & 0.1964 & 0.1771 \\
HotpotQA (shift 2) & 0.0779 & 0.0579 & 0.0586 \\
\bottomrule
\end{tabular}
\end{table}