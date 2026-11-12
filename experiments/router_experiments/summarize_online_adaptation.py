#!/usr/bin/env python3
"""
summarize_online_adaptation.py - Turn online_adaptation_results.json's segment_averages
into paper-ready prose and a LaTeX table.

Right now the entire quantitative payoff of the online-adaptation experiment (the paper's
headline differentiator) lives only inside Figure 1 — a reviewer has to eyeball a plot to
see the effect size. This script surfaces the actual per-segment numbers (frozen GBT vs.
frozen RL vs. online RL, on each of the NQ/TriviaQA/HotpotQA stream segments) as text you
can paste directly into the Results section.

Usage:
    python experiments/router_experiments/summarize_online_adaptation.py
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "online_adaptation_results.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "online_adaptation_summary.md")


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found. Run online_adaptation.py first.")
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    seg = data["segment_averages"]
    datasets = ["nq", "triviaqa", "hotpotqa"]
    labels = {"nq": "NQ (source)", "triviaqa": "TriviaQA (shift 1)", "hotpotqa": "HotpotQA (shift 2)"}

    lines = []
    lines.append("# Online Adaptation — Segment Averages\n")
    lines.append("Auto-generated from results/online_adaptation_results.json. Paste the relevant")
    lines.append("numbers into the Results section text — right now this effect only appears in")
    lines.append("Figure 1, which requires the reader to eyeball a plot.\n")

    lines.append("| Segment | Frozen GBT | Frozen RL | Online RL | Online RL vs. Frozen GBT |")
    lines.append("|---|---|---|---|---|")
    for ds in datasets:
        if ds not in seg:
            continue
        gbt, rl_f, rl_o = seg[ds]["gbt"], seg[ds]["rl_frozen"], seg[ds]["rl_online"]
        delta = rl_o - gbt
        rel = (delta / gbt * 100) if gbt else float("nan")
        lines.append(f"| {labels[ds]} | {gbt:.4f} | {rl_f:.4f} | {rl_o:.4f} | {delta:+.4f} ({rel:+.1f}%) |")

    # This block previously emitted a fixed sentence describing a win -- "recovered
    # entirely through online policy-gradient updates" -- with the numbers substituted in.
    # On the 2026-08-04 run that produced paper-ready prose narrating a 12.2% LOSS as a
    # recovery. The template assumed its hypothesis and reported it regardless of the data,
    # which is the same failure as the feature leak one layer up: the pipeline was built
    # expecting a positive result and confirmed one at every layer.
    #
    # The prose now branches on what actually happened, and the scoreboard below it is
    # unconditional so a single cherry-picked segment cannot stand in for the result.
    lines.append("\n## Suggested prose\n")

    scored = [ds for ds in datasets if ds in seg]
    wins = [ds for ds in scored if seg[ds]["rl_online"] > seg[ds]["gbt"]]
    beaten_by_untrained = [ds for ds in scored
                           if seg[ds]["rl_frozen"] > seg[ds]["rl_online"]]

    if "hotpotqa" in seg:
        gbt_h = seg["hotpotqa"]["gbt"]
        rl_o_h = seg["hotpotqa"]["rl_online"]
        rl_f_h = seg["hotpotqa"]["rl_frozen"]
        delta_h = rl_o_h - gbt_h
        rel_h = (delta_h / gbt_h * 100) if gbt_h else float("nan")

        if delta_h > 0:
            lines.append(
                f'"On the final HotpotQA segment, after the query distribution has fully '
                f'shifted away from the training distribution, online RL-Route reaches '
                f'{rl_o_h:.4f} MRR versus the frozen Gradient Boosting model\'s {gbt_h:.4f} '
                f'— {rel_h:+.1f}% — obtained through online policy-gradient updates on the '
                f'observed reward, with no labeled oracle targets."'
            )
        else:
            lines.append(
                f'"On the final HotpotQA segment, online RL-Route reaches {rl_o_h:.4f} MRR '
                f'against the frozen Gradient Boosting model\'s {gbt_h:.4f} — {rel_h:+.1f}%. '
                f'Online policy-gradient updates on the observed reward did not close the '
                f'gap to the frozen supervised baseline on this segment."'
            )

        if rl_f_h > rl_o_h:
            lines.append(
                f'\n**Report alongside it:** the untrained "Frozen RL" network scores '
                f'{rl_f_h:.4f} on this segment, above online RL-Route\'s {rl_o_h:.4f}. '
                f'A randomly-initialised policy outperforming the learning one is not a '
                f'result that survives review unexplained.'
            )

    lines.append(
        f"\n**Scoreboard.** Online RL-Route beats frozen GBT on "
        f"{len(wins)} of {len(scored)} segments"
        + (f" ({', '.join(labels.get(d, d) for d in wins)})" if wins else "")
        + "."
        + (f" It is beaten by the untrained frozen network on "
           f"{len(beaten_by_untrained)} of {len(scored)} "
           f"({', '.join(labels.get(d, d) for d in beaten_by_untrained)})."
           if beaten_by_untrained else "")
    )
    if len(wins) * 2 <= len(scored):
        lines.append(
            "\n> Winning a minority of segments does not support an adaptation claim. "
            "State the per-segment numbers rather than the aggregate, or drop the "
            "experiment."
        )

    lines.append("\n## LaTeX table\n")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Online Adaptation: Segment-Averaged MRR@5 as the query stream shifts NQ $\to$ TriviaQA $\to$ HotpotQA.}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Segment} & \textbf{Frozen GBT} & \textbf{Frozen RL} & \textbf{Online RL (Ours)} \\")
    lines.append(r"\midrule")
    for ds in datasets:
        if ds not in seg:
            continue
        gbt, rl_f, rl_o = seg[ds]["gbt"], seg[ds]["rl_frozen"], seg[ds]["rl_online"]
        bold = r"\textbf{%.4f}" % rl_o if rl_o >= max(gbt, rl_f) else f"{rl_o:.4f}"
        lines.append(f"{labels[ds]} & {gbt:.4f} & {rl_f:.4f} & {bold} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output = "\n".join(lines)
    # encoding is explicit because open() defaults to the locale codepage, which is cp1252
    # on the lab machine. The em-dashes in this report were being written as \x97 rather
    # than UTF-8, so the file rendered as mojibake anywhere it was read as UTF-8 -- and any
    # character outside cp1252 would have raised UnicodeEncodeError mid-pipeline.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
