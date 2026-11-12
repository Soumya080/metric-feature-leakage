"""
learnability_analysis.py — Paper-ready evidence for the three core claims.

Produces the tables and figure behind the paper's central argument, in one pass, at
whatever scale the env data happens to be (small pilot corpus or the full run):

  Claim 1 — Routing headroom is real.
      An oracle that picks the best retriever per query beats the best single retriever by
      a wide margin. This is what motivates routing at all.

  Claim 2 — The reward is sparse, which bounds what any router can learn.
      Most queries have every retriever scoring 0, so all actions tie. Ties carry no
      routing signal: they contribute no gradient to a policy, and they degrade
      argmax-derived supervised labels into a tie-break artifact (numpy resolves ties
      toward index 0, i.e. always BM25). Only queries with a strictly-best action are
      informative, and that share is the real ceiling on learnability.

  Claim 3 — With label-free features, the best retriever is not predictable.
      Trained directly as a classification problem (predict the strictly-best retriever),
      no honest feature set beats the majority-class baseline. Reproducing the original
      MRR-derived features recovers high accuracy, which quantifies exactly how much of
      the original result was reward leakage rather than routing skill.

Claim 3 is deliberately framed as a *classification* problem rather than an RL result: if
a tuned classifier with full supervision cannot predict the target, no policy-gradient
method can either, so this isolates "is there signal in the features" from "is the RL
setup working". It is the cleanest possible statement of the negative result.

Usage:
    python experiments/router_experiments/learnability_analysis.py
    python experiments/router_experiments/learnability_analysis.py \
        --env-suffix _smoke --leaky-suffix _leaky --tag "400 queries/dataset pilot"
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]
PRETTY = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
N_REPEATS = 5      # repeated stratified CV — single-split accuracy is too noisy to report
N_SPLITS = 5


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    return env.get("queries", env)


def reward_matrix(queries, metric="mrr"):
    qids = list(queries)
    R = np.array([[float(queries[q]["retrievers"][a][metric]) for a in ACTIONS] for q in qids])
    X = np.array([queries[q]["features"] for q in qids], dtype=float)
    return qids, X, R


def characterize(R):
    """Claim 1 + 2: headroom, and how much of the corpus carries any routing signal."""
    n = len(R)
    means = R.mean(axis=0)
    best_idx = int(means.argmax())
    best_fixed = float(means[best_idx])
    oracle = float(R.max(axis=1).mean())
    strictly_best = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    all_zero = (R.max(axis=1) == 0)
    return {
        "n": n,
        "per_retriever": {a: float(m) for a, m in zip(ACTIONS, means)},
        "best_fixed_name": ACTIONS[best_idx],
        "best_fixed": best_fixed,
        "oracle": oracle,
        "headroom": oracle - best_fixed,
        "headroom_rel": (oracle - best_fixed) / best_fixed if best_fixed > 0 else float("nan"),
        "all_zero_rate": float(all_zero.mean()),
        "learnable_rate": float(strictly_best.mean()),
        "n_learnable": int(strictly_best.sum()),
    }


def predictability(X, R, seed=0):
    """Claim 3: can a supervised classifier predict the strictly-best retriever?

    Restricted to queries that HAVE a strictly-best action — on ties there is no correct
    label, so including them would score a model on an arbitrary tie-break.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    if len(yi) < N_SPLITS * 2:
        return None

    counts = np.bincount(yi, minlength=len(ACTIONS))
    # Stratified CV needs at least N_SPLITS members per class.
    if (counts[counts > 0] < N_SPLITS).any():
        return None

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    majority = float(counts.max() / len(yi))

    def score(model, cols=None):
        Xs = Xi if cols is None else Xi[:, cols]
        s = cross_val_score(model, Xs, yi, cv=cv, scoring="accuracy")
        return float(s.mean()), float(s.std())

    gbt_m, gbt_s = score(GradientBoostingClassifier(random_state=seed))
    lr_m, lr_s = score(LogisticRegression(max_iter=3000))
    out = {
        "n_informative": int(len(yi)),
        "class_counts": {a: int(c) for a, c in zip(ACTIONS, counts)},
        "majority": majority,
        "gbt": gbt_m, "gbt_std": gbt_s,
        "logreg": lr_m, "logreg_std": lr_s,
        "gbt_lift": gbt_m - majority,
    }

    # Which representation, if any, carries routing signal? Cumulative groups isolate the
    # contribution of each block, and the "-only" rows show whether a block stands alone.
    dim = Xi.shape[1]
    groups = {
        "query_only (0-21)": list(range(0, 22)),
        "+retrieval_scores (0-25)": list(range(0, 26)),
        "+cross_retriever (0-28)": list(range(0, 29)),
        "retrieval_only (22-25)": list(range(22, 26)),
        "cross_only (26-28)": list(range(26, 29)),
    }
    if dim > 29:
        groups["+embeddings (all)"] = list(range(0, dim))
        groups["embeddings_only (29+)"] = list(range(29, dim))
    out["groups"] = {}
    for label, cols in groups.items():
        cols = [c for c in cols if c < dim]
        if not cols:
            continue
        m, s = score(GradientBoostingClassifier(random_state=seed), cols=cols)
        out["groups"][label] = {"acc": m, "std": s, "lift": m - majority, "n_features": len(cols)}
    return out


def analyze(paths, leaky_paths=None, metric="mrr"):
    per_ds, pooled_X, pooled_R, pooled_leakyX = {}, [], [], []
    for ds, path in paths.items():
        if not os.path.exists(path):
            print(f"  {ds}: env data missing at {path} — skipped.")
            continue
        q = load_queries(path)
        _, X, R = reward_matrix(q, metric)
        entry = {"corpus": characterize(R), "honest": predictability(X, R)}

        if leaky_paths and ds in leaky_paths and os.path.exists(leaky_paths[ds]):
            ql = load_queries(leaky_paths[ds])
            _, Xl, Rl = reward_matrix(ql, metric)
            if len(Xl) == len(X):
                entry["leaky"] = predictability(Xl, Rl)
                pooled_leakyX.append(Xl)
            else:
                print(f"  {ds}: leaky env has {len(Xl)} queries vs honest {len(X)} — "
                      f"not comparable, skipping leaky arm.")
        per_ds[ds] = entry
        pooled_X.append(X)
        pooled_R.append(R)

    pooled = None
    if pooled_X:
        # Datasets can end up with different feature widths, because the PCA component
        # count is capped by sample size: min(embed_dims, n_train - 1, 384). At full scale
        # every dataset clears embed_dims and all widths agree, but on a small run the cap
        # binds at a different value per dataset (e.g. 17 / 28 / 9 components from 18 / 29
        # / 10 training queries) and np.vstack raises. Pooling matrices of different widths
        # is meaningless at any scale, so report and skip rather than crash the stage --
        # the per-dataset results above are unaffected and are what a small run is for.
        widths = {X.shape[1] for X in pooled_X}
        if len(widths) > 1:
            detail = ", ".join(f"{ds}={X.shape[1]}"
                               for ds, X in zip(per_ds, pooled_X))
            print(f"  Pooled analysis skipped: feature widths differ ({detail}). "
                  f"PCA components are capped by training-set size, so this happens when "
                  f"a dataset has fewer than embed_dims+1 training queries. Per-dataset "
                  f"results above are unaffected.")
        else:
            X = np.vstack(pooled_X)
            R = np.vstack(pooled_R)
            pooled = {"corpus": characterize(R), "honest": predictability(X, R)}
            if (pooled_leakyX and len(pooled_leakyX) == len(pooled_X)
                    and len({Xl.shape[1] for Xl in pooled_leakyX}) == 1):
                pooled["leaky"] = predictability(np.vstack(pooled_leakyX), R)
    return per_ds, pooled


def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}\\%"


def render_markdown(per_ds, pooled, tag):
    L = [f"# Routing Learnability Analysis\n", f"_Scale: {tag}_\n"]

    L += ["\n## Table 1 — Routing headroom and reward sparsity\n",
          "| dataset | n | best fixed | oracle | headroom | all-zero | learnable |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    rows = list(per_ds.items()) + ([("**pooled**", pooled)] if pooled else [])
    for ds, e in rows:
        c = e["corpus"]
        L.append(f"| {ds} | {c['n']} | {PRETTY.get(c['best_fixed_name'], c['best_fixed_name'])} "
                 f"{c['best_fixed']:.4f} | {c['oracle']:.4f} | **+{c['headroom']:.4f}** "
                 f"({c['headroom_rel']*100:.0f}%) | {c['all_zero_rate']*100:.1f}% | "
                 f"{c['learnable_rate']*100:.1f}% |")
    L += ["\nHeadroom is the gain an oracle would get from per-query routing. "
          "`all-zero` is the share of queries where every retriever scores 0; `learnable` "
          "is the share with a strictly-best action — the only queries that carry routing "
          "signal.\n"]

    L += ["\n## Table 2 — Can the best retriever be predicted from label-free features?\n",
          "| dataset | informative n | majority | GBT (honest) | LogReg (honest) | lift | GBT (leaky) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, e in rows:
        h, lk = e.get("honest"), e.get("leaky")
        if not h:
            L.append(f"| {ds} | too few | — | — | — | — | — |")
            continue
        leaky_txt = f"{lk['gbt']*100:.1f}%" if lk else "—"
        L.append(f"| {ds} | {h['n_informative']} | {h['majority']*100:.1f}% | "
                 f"{h['gbt']*100:.1f}% ± {h['gbt_std']*100:.1f} | "
                 f"{h['logreg']*100:.1f}% ± {h['logreg_std']*100:.1f} | "
                 f"**{h['gbt_lift']*100:+.1f}%** | {leaky_txt} |")
    L += ["\nAccuracy predicting the strictly-best retriever "
          f"({N_SPLITS}-fold stratified CV × {N_REPEATS} repeats). `majority` is the "
          "always-predict-the-most-common-retriever baseline. A negative lift means the "
          "classifier does **worse than a constant policy**. The leaky column reproduces "
          "the original MRR-derived features, which are the reward itself for 2 of the 3 "
          "actions — the gap between it and the honest columns is the size of the leak.\n"]

    if pooled and pooled.get("honest") and pooled["honest"].get("groups"):
        h = pooled["honest"]
        L += ["\n## Table 3 — Which feature group carries the signal? (pooled)\n",
              f"Majority baseline: **{h['majority']*100:.1f}%**. Lift is accuracy minus that "
              "baseline; a negative lift means the group is worse than a constant policy.\n",
              "| feature group | # features | accuracy | lift |",
              "|---|---:|---:|---:|"]
        for label, g in h["groups"].items():
            L.append(f"| {label} | {g['n_features']} | {g['acc']*100:.1f}% ± {g['std']*100:.1f} | "
                     f"**{g['lift']*100:+.1f}%** |")
        if pooled.get("leaky"):
            L.append(f"| _leaky reference (all)_ | — | {pooled['leaky']['gbt']*100:.1f}% | "
                     f"{(pooled['leaky']['gbt']-h['majority'])*100:+.1f}% |")
        L.append("")

    return "\n".join(L) + "\n"


def render_latex(per_ds, pooled, tag):
    rows = list(per_ds.items()) + ([("Pooled", pooled)] if pooled else [])
    L = ["% Auto-generated by learnability_analysis.py — do not edit by hand.",
         f"% Scale: {tag}",
         "\\begin{table}[t]", "\\centering", "\\small",
         "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Dataset & $N$ & Best Fixed & Oracle & Headroom & Learnable \\\\", "\\midrule"]
    for ds, e in rows:
        c = e["corpus"]
        L.append(f"{ds} & {c['n']} & {c['best_fixed']:.4f} & {c['oracle']:.4f} & "
                 f"+{c['headroom']:.4f} ({c['headroom_rel']*100:.0f}\\%) & "
                 f"{fmt_pct(c['learnable_rate'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Oracle routing headroom is substantial, but most queries carry no "
          "routing signal: `Learnable' is the fraction of queries with a strictly-best "
          "retriever. On the remainder every retriever ties (almost always at zero), so no "
          "router can differentiate them.}",
          "\\label{tab:headroom}", "\\end{table}", "",
          "\\begin{table}[t]", "\\centering", "\\small",
          "\\begin{tabular}{lrrrrr}", "\\toprule",
          "Dataset & $N_{\\text{inf}}$ & Majority & GBT & Logistic & GBT (leaky) \\\\",
          "\\midrule"]
    for ds, e in rows:
        h, lk = e.get("honest"), e.get("leaky")
        if not h:
            continue
        L.append(f"{ds} & {h['n_informative']} & {fmt_pct(h['majority'])} & "
                 f"{fmt_pct(h['gbt'])} & {fmt_pct(h['logreg'])} & "
                 f"{fmt_pct(lk['gbt']) if lk else '---'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Accuracy predicting the best retriever from label-free features. No "
          "honest feature set exceeds the majority-class baseline. Restoring the original "
          "MRR-derived features (`leaky') recovers high accuracy, showing that the "
          "previously reported routing gains came from reward leakage.}",
          "\\label{tab:predictability}", "\\end{table}"]
    return "\n".join(L) + "\n"


def render_pdf(per_ds, pooled, tag, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping figure.")
        return False

    rows = list(per_ds.items()) + ([("pooled", pooled)] if pooled else [])
    rows = [(d, e) for d, e in rows if e.get("honest")]
    if not rows:
        return False
    labels = [d for d, _ in rows]
    xs = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.bar(xs - 0.2, [e["corpus"]["all_zero_rate"] * 100 for _, e in rows], 0.4,
           label="all retrievers score 0", color="#C44E52")
    ax.bar(xs + 0.2, [e["corpus"]["learnable_rate"] * 100 for _, e in rows], 0.4,
           label="strictly-best action exists", color="#4C72B0")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("% of queries"); ax.set_title("Most queries carry no routing signal")
    ax.legend(fontsize=8)

    ax = axes[1]
    w = 0.26
    ax.bar(xs - w, [e["honest"]["majority"] * 100 for _, e in rows], w,
           label="majority baseline", color="#8C8C8C")
    ax.bar(xs, [e["honest"]["gbt"] * 100 for _, e in rows], w,
           label="GBT (honest features)", color="#4C72B0")
    leaky = [e.get("leaky") for _, e in rows]
    if any(leaky):
        ax.bar(xs + w, [(l["gbt"] * 100 if l else 0) for l in leaky], w,
               label="GBT (leaky features)", color="#DD8452")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Best retriever is not predictable without leakage")
    ax.legend(fontsize=8)

    fig.suptitle(f"Routing learnability — {tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="", help="Honest env-data suffix.")
    p.add_argument("--leaky-suffix", type=str, default=None,
                   help="Leaky env-data suffix (same queries, leaky feature mode). "
                        "Omit to skip the leakage column.")
    p.add_argument("--env-dir", type=str, default=None,
                   help="Directory holding the env data (default: src/router).")
    p.add_argument("--tag", type=str, default="full run", help="Scale label for captions.")
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    args = p.parse_args()

    paths = get_dataset_paths(args.env_suffix)
    leaky = get_dataset_paths(args.leaky_suffix) if args.leaky_suffix is not None else None
    if args.env_dir:
        paths = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in paths.items()}
        if leaky:
            leaky = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in leaky.items()}

    print(f"Analysing ({args.tag})...")
    per_ds, pooled = analyze(paths, leaky)
    if not per_ds:
        print("No env data found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    js = os.path.join(args.outdir, "learnability_analysis.json")
    md = os.path.join(args.outdir, "learnability_analysis.md")
    tex = os.path.join(args.outdir, "learnability_tables.tex")
    pdf = os.path.join(args.outdir, "learnability_analysis.pdf")

    with open(js, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "per_dataset": per_ds, "pooled": pooled}, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_markdown(per_ds, pooled, args.tag))
    with open(tex, "w", encoding="utf-8") as f:
        f.write(render_latex(per_ds, pooled, args.tag))
    wrote = render_pdf(per_ds, pooled, args.tag, pdf)

    print(render_markdown(per_ds, pooled, args.tag))
    print(f"Saved: {js}\nSaved: {md}\nSaved: {tex}" + (f"\nSaved: {pdf}" if wrote else ""))


if __name__ == "__main__":
    main()
