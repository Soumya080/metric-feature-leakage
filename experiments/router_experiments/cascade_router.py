"""
cascade_router.py — A learned post-retrieval cascade: run cheap, look, escalate if needed.

The learnability analysis shows that which retriever wins cannot be predicted from the
query: no feature set beats the majority baseline. That is a statement about PREDICTION.
This module does not predict. It runs the cheap retriever, looks at what came back, and
decides whether to pay for more — replacing an unlearnable task with a much easier one.

The cascade is defined so that escalation is exactly what it costs:

    ACCEPT    -> keep the cheap retriever's result       cost = c(probe)
    ESCALATE  -> use Hybrid (RRF of both retrievers)     cost = c(probe) + c(other)

Escalating means running the second retriever, and once both ranked lists exist Hybrid is
free — RRF is a dict merge. So the escalation branch needs no oracle to pick a winner,
which is the trap a naive "escalate then choose the better one" formulation falls into: it
would silently reintroduce the very label-dependent choice this whole line of work shows is
not learnable.

Baselines, in increasing order of how much they matter:

    always-accept       never escalate (= the cheap retriever alone)
    always-escalate     always escalate (= Hybrid alone)
    oracle cascade      escalate iff it actually helps; the ceiling, needs labels
    RANDOM AT MATCHED   escalate at random, at the SAME rate the learned policy chose
      RATE

The last one is the comparison that decides whether this works. A learned policy that
escalates 40% of the time will beat always-accept simply by escalating, so any gain must be
measured against escalating 40% of the time *at random*. Beating that, and only that, shows
the post-retrieval evidence carried information.

Two policies are fitted:
    threshold  GBT failure probability, threshold swept to trace a quality/cost Pareto
               curve. Robust, interpretable, and the primary result.
    reinforce  a small REINFORCE policy over the same state, for the budget-constrained
               formulation where the cost penalty is inside the reward. Guarded: a failure
               here reports and continues rather than losing the threshold results.

The state respects the same causal constraint as escalation_signal_check, and imports it
from there rather than restating it, so the two cannot drift: deciding whether to run Dense
must never use Dense's output.

Usage:
    python experiments/router_experiments/cascade_router.py --env-suffix _xscore
    python experiments/router_experiments/cascade_router.py --env-suffix _xscore --no-reinforce
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canonical_evaluation import get_dataset_paths, paired_bootstrap_test  # noqa: E402
from escalation_signal_check import feature_sets, load_layout  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]
SPLIT_SEED = 42
# Same structural costs as cost_aware_analysis: Dense pays a neural encoding plus ANN
# search where BM25 pays an inverted-index lookup. Escalation cost is probe + other, since
# that is literally what running Hybrid requires.
DEFAULT_COSTS = {"bm25": 1.0, "dense": 3.0}


def load_split(path):
    """Train/val/test with features, per-action MRR@k, and full-corpus gold ranks."""
    envs = create_train_val_test_split(path, split_seed=SPLIT_SEED)
    out, mrr_key = [], None
    for env in envs:
        qids = list(env.query_pool)
        X, M, R = [], [], []
        for q in qids:
            rec = env.master_table[q]
            rets = rec["retrievers"]
            if mrr_key is None:
                mrr_key = next((kk for kk in rets["bm25"] if kk.startswith("mrr_at_")), "mrr")
            X.append(rec["features"])
            M.append([float(rets[a][mrr_key]) for a in ACTIONS])
            R.append([float(rets[a].get("gold_rank") or 10**9) for a in ACTIONS])
        out.append({"qids": qids, "X": np.array(X, dtype=float),
                    "M": np.array(M, dtype=float), "R": np.array(R, dtype=float)})
    return out, mrr_key


def simulate(split, probe_idx, escalate, costs):
    """Realised (quality, cost) per query for a boolean escalation decision.

    Accepting keeps the probe's own result; escalating yields Hybrid, because having run
    both retrievers is exactly what Hybrid needs. No label is consulted here — `escalate`
    is whatever the policy decided.
    """
    probe = ACTIONS[probe_idx]
    other = "dense" if probe == "bm25" else "bm25"
    hyb = ACTIONS.index("hybrid")

    escalate = np.asarray(escalate, dtype=bool)
    quality = np.where(escalate, split["M"][:, hyb], split["M"][:, probe_idx])
    cost = np.where(escalate, costs[probe] + costs[other], costs[probe])
    return quality, cost


def fit_failure_model(train, probe, k_fail, layout, seed):
    """P(probe failed) from admissible post-retrieval evidence. None if unfittable."""
    from sklearn.ensemble import GradientBoostingClassifier

    dim = train["X"].shape[1]
    sets, _ = feature_sets(probe, dim, layout)
    cols = sets.get("+cross_score") or sets.get("+self_score") or sets["pre_retrieval"]
    j = ACTIONS.index(probe)
    y = (train["R"][:, j] > k_fail).astype(int)
    if len(np.unique(y)) < 2:
        return None, cols
    m = GradientBoostingClassifier(random_state=seed).fit(train["X"][:, cols], y)
    return m, cols


def pareto(model, cols, split, probe_idx, costs, thresholds):
    """Quality/cost at each escalation threshold. Lower threshold => escalate more."""
    p = model.predict_proba(split["X"][:, cols])[:, 1]
    rows = []
    for t in thresholds:
        esc = p >= t
        q, c = simulate(split, probe_idx, esc, costs)
        rows.append({"threshold": float(t), "escalation_rate": float(esc.mean()),
                     "quality": float(q.mean()), "cost": float(c.mean()),
                     "_q": q, "_c": c})
    return rows, p


def matched_random(split, probe_idx, rate, costs, seed, n_rep=1000):
    """Escalate at random at the given rate. THE control: isolates signal from spend.

    Returns the full distribution of realised means, not just its average, because the
    right question is "how often does random targeting do at least as well as the learned
    policy?" — a Monte-Carlo test against the null of untargeted escalation.

    Testing the learned policy against the AVERAGE random per-query vector instead would
    be wrong and anti-conservative: averaging over draws collapses the null's variance, so
    a policy with no signal at all still comes out significant. That is not a hypothetical
    — it produced p=0.009 on synthetic data built with no signal for that probe.
    """
    rng = np.random.default_rng(seed)
    n = split["X"].shape[0]
    means, costs_ = np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        esc = rng.random(n) < rate
        q, c = simulate(split, probe_idx, esc, costs)
        means[i] = q.mean(); costs_[i] = c.mean()
    return float(means.mean()), float(costs_.mean()), means


def monte_carlo_p(observed, null_means):
    """One-sided p: how often untargeted escalation matches or beats the policy.

    The +1 in numerator and denominator is the standard finite-sample correction; it keeps
    p strictly positive, so a policy is never reported as p=0 on the strength of a finite
    number of draws.
    """
    return float((1.0 + np.sum(null_means >= observed)) / (len(null_means) + 1.0))


def train_reinforce_cascade(train, val, test, probe, cols, probe_idx, costs, lam,
                            seed, epochs=60):
    """Small REINFORCE policy over the same state, with cost folded into the reward.

    Included because the budget-constrained formulation is the one place RL is genuinely
    required rather than decorative: the objective is a policy-level trade-off, not a
    per-query label. Kept deliberately small (one hidden layer, per-batch advantage
    centring) since the decision is binary and the sample size is ~10^3.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    mu = train["X"][:, cols].mean(axis=0)
    sd = train["X"][:, cols].std(axis=0) + 1e-8

    def prep(s):
        return torch.tensor((s["X"][:, cols] - mu) / sd, dtype=torch.float32)

    net = nn.Sequential(nn.Linear(len(cols), 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def reward_for(split, esc):
        q, c = simulate(split, probe_idx, esc, costs)
        return q - lam * c

    Xtr, Xva = prep(train), prep(val)
    best_val, best_state = -np.inf, None
    for _ in range(epochs):
        logits = net(Xtr)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        r = torch.tensor(reward_for(train, a.numpy().astype(bool)), dtype=torch.float32)
        adv = r - r.mean()
        loss = -(dist.log_prob(a) * adv).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            va = net(Xva).argmax(dim=1).numpy().astype(bool)
            v = reward_for(val, va).mean()
        if v > best_val:
            best_val = float(v)
            best_state = {k: t.clone() for k, t in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        esc = net(prep(test)).argmax(dim=1).numpy().astype(bool)
    q, c = simulate(test, probe_idx, esc, costs)
    return {"escalation_rate": float(esc.mean()), "quality": float(q.mean()),
            "cost": float(c.mean()), "lambda": float(lam), "_q": q}


def analyze_probe(name, train, val, test, probe, k_fail, layout, costs, seed,
                  use_reinforce, lam):
    probe_idx = ACTIONS.index(probe)
    hyb = ACTIONS.index("hybrid")
    res = {"probe": probe, "n_test": int(test["X"].shape[0])}

    # Reference points. always-escalate IS Hybrid, always-accept IS the probe alone.
    q_acc, c_acc = simulate(test, probe_idx, np.zeros(test["X"].shape[0], bool), costs)
    q_esc, c_esc = simulate(test, probe_idx, np.ones(test["X"].shape[0], bool), costs)
    helps = test["M"][:, hyb] > test["M"][:, probe_idx]
    q_or, c_or = simulate(test, probe_idx, helps, costs)
    res["baselines"] = {
        "always_accept": {"quality": float(q_acc.mean()), "cost": float(c_acc.mean()),
                          "escalation_rate": 0.0},
        "always_escalate": {"quality": float(q_esc.mean()), "cost": float(c_esc.mean()),
                            "escalation_rate": 1.0},
        "oracle_cascade": {"quality": float(q_or.mean()), "cost": float(c_or.mean()),
                           "escalation_rate": float(helps.mean())},
    }

    model, cols = fit_failure_model(train, probe, k_fail, layout, seed)
    if model is None:
        res["note"] = "failure label is single-class on train — cascade not fittable"
        return res
    res["n_features"] = len(cols)

    thresholds = np.linspace(0.05, 0.95, 19)
    rows, _ = pareto(model, cols, test, probe_idx, costs, thresholds)

    # Headline operating point: best test quality per unit cost is not the right criterion
    # (it would be selected on test). Pick the threshold on VALIDATION, then report it on
    # test — the same discipline the rest of the codebase uses for checkpoint selection.
    v_rows, _ = pareto(model, cols, val, probe_idx, costs, thresholds)
    lam_sel = lam
    best_v = max(v_rows, key=lambda r: r["quality"] - lam_sel * r["cost"])
    chosen = min(rows, key=lambda r: abs(r["threshold"] - best_v["threshold"]))

    rnd_q, rnd_c, rnd_null = matched_random(
        test, probe_idx, chosen["escalation_rate"], costs, seed)

    res["pareto"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    res["threshold_policy"] = {
        "threshold": chosen["threshold"],
        "escalation_rate": chosen["escalation_rate"],
        "quality": chosen["quality"],
        "cost": chosen["cost"],
        "vs_always_accept": chosen["quality"] - float(q_acc.mean()),
        "vs_always_escalate": chosen["quality"] - float(q_esc.mean()),
        "cost_saving_vs_always_escalate": float(c_esc.mean()) - chosen["cost"],
        # The comparison that matters: same spend, random targeting.
        "matched_random_quality": rnd_q,
        "matched_random_cost": rnd_c,
        "vs_matched_random": chosen["quality"] - rnd_q,
        "p_vs_matched_random": monte_carlo_p(chosen["quality"], rnd_null),
        # always-accept is a fixed alternative, not a random null, so a paired bootstrap
        # over per-query differences is the appropriate test there.
        "p_vs_always_accept": float(paired_bootstrap_test(chosen["_q"], q_acc)[1]),
    }

    if use_reinforce:
        try:
            res["reinforce"] = train_reinforce_cascade(
                train, val, test, probe, cols, probe_idx, costs, lam, seed)
            res["reinforce"].pop("_q", None)
            r_q, _, r_null = matched_random(
                test, probe_idx, res["reinforce"]["escalation_rate"], costs, seed)
            res["reinforce"]["matched_random_quality"] = r_q
            res["reinforce"]["vs_matched_random"] = res["reinforce"]["quality"] - r_q
            res["reinforce"]["p_vs_matched_random"] = monte_carlo_p(
                res["reinforce"]["quality"], r_null)
        except Exception as exc:
            # Never let the RL arm cost us the threshold results, which are the primary
            # finding and already computed by this point.
            res["reinforce_error"] = f"{type(exc).__name__}: {exc}"
            print(f"    WARNING: REINFORCE arm failed ({exc}); threshold results kept.")
    return res


def render(all_res):
    L = ["# Learned Cross-Scoring Cascade\n",
         "Run the cheap retriever, inspect what it returned, escalate to Hybrid only when",
         "the evidence says it failed. Quality is MRR@k; cost is in structural units",
         "(BM25=1, Dense=3), so escalating costs 4.\n",
         "`vs matched random` is the number that decides whether this works: the learned",
         "policy compared against escalating at random AT THE SAME RATE. Any policy that",
         "escalates more will look better than always-accept; only beating matched random",
         "shows the post-retrieval evidence carried information.\n",
         "| dataset | probe | esc. rate | quality | cost | vs always-accept | vs matched random | p |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds, probes in all_res.items():
        for probe, r in probes.items():
            t = r.get("threshold_policy")
            if not t:
                L.append(f"| {ds} | {probe} | — | — | — | — | {r.get('note', 'n/a')} | — |")
                continue
            L.append(f"| {ds} | {probe} | {t['escalation_rate']:.1%} | {t['quality']:.4f} | "
                     f"{t['cost']:.2f} | {t['vs_always_accept']:+.4f} | "
                     f"{t['vs_matched_random']:+.4f} | {t['p_vs_matched_random']:.3f} |")

    L += ["\n## Reference points\n",
          "| dataset | probe | always-accept | always-escalate (Hybrid) | oracle cascade |",
          "|---|---|---|---|---|"]
    for ds, probes in all_res.items():
        for probe, r in probes.items():
            b = r.get("baselines", {})
            def fmt(k):
                x = b.get(k)
                return f"{x['quality']:.4f} @ {x['cost']:.2f}" if x else "—"
            L.append(f"| {ds} | {probe} | {fmt('always_accept')} | "
                     f"{fmt('always_escalate')} | {fmt('oracle_cascade')} |")

    L += ["\n## Reading this\n",
          "A positive `vs matched random` with p < 0.05 means the cascade is targeting its "
          "escalations, not just buying quality with compute. That is the claim.\n",
          "`vs always-escalate` near zero at a much lower cost is also a win, and often the "
          "more useful one in practice: same quality as always running Hybrid, for a "
          "fraction of the compute.\n",
          "If `vs matched random` is ~0, the escalation signal is not informative and the "
          "cascade reduces to spending more — which should be reported as such rather than "
          "tuned until it looks positive.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="_xscore")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--probes", type=str, default="bm25,dense",
                   help="Which retriever(s) to use as the cheap first probe.")
    p.add_argument("--k", type=int, default=20, help="Failure cutoff for the probe label.")
    p.add_argument("--lam", type=float, default=0.02,
                   help="Cost weight for selecting the operating point and for the "
                        "REINFORCE reward. Quality is ~0-1 and cost is 1-4, so 0.02 trades "
                        "roughly one MRR point for one cost unit.")
    p.add_argument("--cost-bm25", type=float, default=DEFAULT_COSTS["bm25"])
    p.add_argument("--cost-dense", type=float, default=DEFAULT_COSTS["dense"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-reinforce", action="store_true",
                   help="Skip the RL arm (the threshold policy is the primary result).")
    p.add_argument("--output", type=str,
                   default=os.path.join(RESULTS_DIR, "cascade_router.json"))
    p.add_argument("--output-md", type=str,
                   default=os.path.join(RESULTS_DIR, "cascade_router.md"))
    args = p.parse_args()

    costs = {"bm25": args.cost_bm25, "dense": args.cost_dense}
    probes = [x.strip() for x in args.probes.split(",") if x.strip() in ("bm25", "dense")]
    if not probes:
        p.error("--probes must name at least one of bm25,dense")

    paths = get_dataset_paths(args.env_suffix)
    if args.env_dir:
        paths = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in paths.items()}
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        paths = {k: v for k, v in paths.items() if k in keep}

    all_res = {}
    for ds, path in paths.items():
        if not os.path.exists(path):
            print(f"  {ds}: env data missing ({path}) — skipped.")
            continue
        print(f"\n--- {ds} ---")
        try:
            (train, val, test), _ = load_split(path)
            layout = load_layout(path)
            if layout is None or layout.get("cross_score", [0, 0])[1] <= \
                    layout.get("cross_score", [0, 0])[0]:
                print("  NOTE: no cross-score features in this env — the cascade will run "
                      "on self-scores only. Rebuild with --cross-scores for the full method.")
            per_probe = {}
            for probe in probes:
                r = analyze_probe(ds, train, val, test, probe, args.k, layout, costs,
                                  args.seed, not args.no_reinforce, args.lam)
                per_probe[probe] = r
                t = r.get("threshold_policy")
                if t:
                    print(f"  {probe:5s} esc={t['escalation_rate']:.1%} "
                          f"q={t['quality']:.4f} cost={t['cost']:.2f}  "
                          f"vs_random={t['vs_matched_random']:+.4f} "
                          f"(p={t['p_vs_matched_random']:.3f})")
                else:
                    print(f"  {probe:5s} {r.get('note', 'not fittable')}")
            all_res[ds] = per_probe
        except Exception:
            # One bad dataset must not lose the others; the traceback is printed so a
            # failure is diagnosable from the unattended log rather than silent.
            print(f"  ERROR on {ds}:\n{traceback.format_exc()}")

    if not all_res:
        print("\nNo env data found. Build it first: "
              "python src/router/prepare_env_data.py --cross-scores --suffix _xscore")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render(all_res))
    print(f"\nSaved: {args.output}\nSaved: {args.output_md}")


if __name__ == "__main__":
    main()
