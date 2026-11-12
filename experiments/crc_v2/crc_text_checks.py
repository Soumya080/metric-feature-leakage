#!/usr/bin/env python3
"""Artifact for the numbers the camera-ready text adds or corrects (paper/main_v2.tex, CRC pass).

Each number below was first computed ad hoc during the 2026-10-31 audit
(paper/crc_audit/CRC_METHODS_CLAIMS_LITERATURE_AUDIT.md). This script recomputes all of them
from committed inputs, so that every sentence the CRC edit introduced traces to a file:

  leak_equality      share of rows where the MRR@20 leak (slots 22/23 of *_env_data_leaky.json)
                     equals the full-rank reward, and where it is 0      -> Table 2 "26--45%"
  scale_mrr5_zero    share of rows where the shared-index leak (MRR@5) is 0, at Figure 1's three
                     scales (experiments/corpus_v3/artifacts/scale_*/env) -> Figure 1 "76--95%"
  bootstrap_vs_dense one-sided paired bootstrap (canonical_evaluation.paired_bootstrap_test,
                     seed 12345, B 10,000) of every Table 3 method against Always-Dense
                                                                          -> caption "p = 0.094"
  train_selected     capture against the arm with the best TRAIN mean instead of the test-best
                     arm                                                  -> Sec. 5 "12--35%"
  rl_restored        Table 3 RL row with the best-validation checkpoint (results/crc_v2/
                     rl_restore/rows.jsonl): mean capture, per-seed range, bootstrap vs Dense
  gb_test_actions    tuned GB's test predictions, refit exactly as compute_supervised_baselines
                     does (split seed 42; RandomizedSearchCV n_iter 20, random_state 42)
                                                                          -> Sec. 5 "169/174, 231/231, 138/138"
                     This refit is the slow part (~10 min CPU); --skip-gb-refit omits it.

Reads only. Writes --out-dir (refuses non-empty, as every crc_v2 writer does).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import common as C
import canonical_evaluation as CE  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

CORPUS_V3_ENV = os.path.join(C.PROJECT_ROOT, "experiments", "corpus_v3", "artifacts")
RL_RESTORE_CANDIDATES = [os.path.join(C.CRC_RESULTS, "rl_restore", "rows.jsonl"),
                         os.path.join(C.CRC_RESULTS, "crc_v2", "rl_restore", "rows.jsonl")]
FIG1_SCALES = ("20000", "50000", "100000")  # the three sizes Figure 1 plots
METHODS = ("score_gap", "logistic_regression", "random_forest", "ridge", "gradient_boosting",
           "rl_route")


def leak_equality():
    out = {}
    for ds in C.DATASETS:
        q = C.load_env(C.anchor_path(ds, "_leaky"))
        row = {}
        for slot, arm in ((22, "bm25"), (23, "dense")):
            f = np.array([e["features"][slot] for e in q.values()], dtype=float)
            r = np.array([e["retrievers"][arm]["rank_reward"] for e in q.values()], dtype=float)
            m20 = np.array([e["retrievers"][arm]["mrr_at_20"] for e in q.values()], dtype=float)
            row[arm] = {"slot": slot, "n": int(len(f)),
                        "equals_mrr_at_20": float(np.mean(np.isclose(f, m20))),
                        "equals_reward": float(np.mean(np.isclose(f, r))),
                        "is_zero": float(np.mean(f == 0))}
        out[ds] = row
    return out


def scale_mrr5_zero():
    out = {}
    for d in sorted(glob.glob(os.path.join(CORPUS_V3_ENV, "scale_*", "env"))):
        scale = os.path.basename(os.path.dirname(d)).split("_", 1)[1]
        if scale not in FIG1_SCALES:
            continue
        per = {}
        for ds in C.DATASETS:
            p = os.path.join(d, f"{ds}_env_data_leaky.json")
            if not os.path.exists(p):
                continue
            q = C.load_env(p)
            f = np.array([[e["features"][22], e["features"][23]] for e in q.values()], dtype=float)
            per[ds] = {"n": int(len(f)), "bm25_zero": float(np.mean(f[:, 0] == 0)),
                       "dense_zero": float(np.mean(f[:, 1] == 0))}
        if per:
            out[scale] = per
    return out


def _capture(v, fixed, oracle):
    return (v - fixed) / (oracle - fixed)


def table3_checks(canon, rl_rows):
    boot, train_sel, rl = {}, {}, {}
    for ds in C.DATASETS:
        S = canon[ds]["per_query"]["scores"]
        qids = [str(x) for x in canon[ds]["per_query"]["query_ids"]]
        dense, oracle = np.array(S["dense"]), float(np.mean(S["oracle"]))
        boot[ds] = {}
        for m in METHODS:
            delta, p = CE.paired_bootstrap_test(S[m], S["dense"])
            boot[ds][m] = {"mean": float(np.mean(S[m])), "delta_vs_dense": float(delta), "p_value": float(p)}

        tr, _, _ = create_train_val_test_split(CE.DATASETS[ds], split_seed=CE.SPLIT_SEED)
        R = np.array([[tr.get_all_rewards(q, CE.REWARD_TYPE)[a] for a in range(3)] for q in tr.query_pool])
        arm = C.ACTIONS[int(R.mean(axis=0).argmax())]
        fixed = float(np.mean(S[arm]))
        train_sel[ds] = {"train_means": dict(zip(C.ACTIONS, map(float, R.mean(axis=0)))),
                         "train_selected_arm": arm, "test_mean_of_that_arm": fixed,
                         "capture": {m: float(_capture(np.mean(S[m]), fixed, oracle)) for m in METHODS}}

        if rl_rows:
            rows = [r for r in rl_rows if r["dataset"] == ds]
            M = np.array([[{str(e["qid"]): e["reward_restored"] for e in r["per_query"]}[q] for q in qids]
                          for r in rows])
            seed_caps = _capture(M.mean(axis=1), float(dense.mean()), oracle)
            delta, p = CE.paired_bootstrap_test(M.mean(axis=0), S["dense"])
            rl[ds] = {"n_seeds": int(len(rows)), "restored_mean": float(M.mean()),
                      "restored_sd_over_seeds": float(M.mean(axis=1).std()),
                      "mean_capture": float(_capture(M.mean(), float(dense.mean()), oracle)),
                      "seed_capture_min": float(seed_caps.min()), "seed_capture_max": float(seed_caps.max()),
                      "seeds_above_0.4pct": int((seed_caps > 0.004).sum()),
                      "seed_averaged_vs_dense": {"delta": float(delta), "p_value": float(p)}}
    return boot, train_sel, rl


def gb_test_actions():
    """Refit tuned GB exactly as compute_supervised_baselines does and capture its test actions.

    The function returns only rewards, so its RandomizedSearchCV.predict is wrapped for the
    duration of the call; nothing about the fit is changed.

    Runs under joblib's threading backend: loky worker processes re-import this module and with
    it torch (via canonical_evaluation), which fails on machines whose paging file cannot map
    torch's CUDA DLLs (BrokenProcessPool, the failure permutation_control.py documents). The
    backend changes only where the CV folds run, not the search or its result.
    """
    from joblib import parallel_backend
    from sklearn.model_selection import RandomizedSearchCV

    out, original = {}, RandomizedSearchCV.predict
    backend = parallel_backend("threading")
    backend.__enter__()
    try:
        for ds in C.DATASETS:
            captured = {}

            def wrapped(self, X, _orig=original):
                captured["pred"] = _orig(self, X)
                return captured["pred"]

            RandomizedSearchCV.predict = wrapped
            tr, _, te = create_train_val_test_split(CE.DATASETS[ds], split_seed=CE.SPLIT_SEED)
            res, _ = CE.compute_supervised_baselines(tr, te, CE.REWARD_TYPE)
            pred = np.asarray(captured["pred"])
            out[ds] = {"n_test": int(len(pred)),
                       "action_counts": dict(zip(C.ACTIONS, map(int, np.bincount(pred, minlength=3)))),
                       "same_as_always_dense": int(np.sum(pred == 1)),
                       "gb_reward": float(res["gradient_boosting"])}
    finally:
        RandomizedSearchCV.predict = original
        backend.__exit__(None, None, None)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "crc_text_checks"))
    ap.add_argument("--skip-gb-refit", action="store_true")
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)

    canon = json.load(open(os.path.join(C.RESULTS_DIR, "canonical_results.json"), encoding="utf-8"))
    rl_path = next((p for p in RL_RESTORE_CANDIDATES if os.path.exists(p)), None)
    rl_rows = [json.loads(line) for line in open(rl_path, encoding="utf-8")] if rl_path else []
    boot, train_sel, rl = table3_checks(canon, rl_rows)
    summary = {"header": C.run_header(args), "rl_restore_rows": rl_path,
               "leak_equality": leak_equality(), "scale_mrr5_zero": scale_mrr5_zero(),
               "bootstrap_vs_dense": boot, "train_selected_comparator": train_sel,
               "rl_restored": rl,
               "gb_test_actions": None if args.skip_gb_refit else gb_test_actions()}
    C.write_json(os.path.join(out_dir, "summary.json"), summary)

    eq = [v[a]["equals_reward"] for v in summary["leak_equality"].values() for a in ("bm25", "dense")]
    z = [v[k] for s in summary["scale_mrr5_zero"].values() for v in s.values() for k in ("bm25_zero", "dense_zero")]
    p_min = min(v["p_value"] for d in boot.values() for m, v in d.items() if m != "rl_route")
    caps = [c for ds in ("hotpotqa",) for c in train_sel[ds]["capture"].values()]
    print(f"MRR@20 leak equals reward on {min(eq)*100:.1f}-{max(eq)*100:.1f}% of rows")
    print(f"MRR@5 leak is zero on {min(z)*100:.1f}-{max(z)*100:.1f}% of rows (shared index)")
    print(f"smallest bootstrap p vs Dense (non-RL methods): {p_min:.3f}")
    print(f"HotpotQA capture vs train-selected {train_sel['hotpotqa']['train_selected_arm']}: "
          f"{min(caps)*100:.1f}-{max(caps)*100:.1f}%")
    for ds, r in rl.items():
        print(f"RL restored {ds}: mean capture {r['mean_capture']*100:+.1f}%, seeds "
              f"[{r['seed_capture_min']*100:+.1f}, {r['seed_capture_max']*100:+.1f}]%, p={r['seed_averaged_vs_dense']['p_value']:.3f}")
    if summary["gb_test_actions"]:
        print("GB identical to Always-Dense: " + ", ".join(
            f"{ds} {v['same_as_always_dense']}/{v['n_test']}" for ds, v in summary["gb_test_actions"].items()))
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
