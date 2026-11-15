#!/usr/bin/env python3
"""Audit a third party's released router features with leakcheck (QueryRoute, CIKM '26).

QueryRoute ("Route Me If You Can", Le et al., CIKM 2026) releases, on the Hugging Face Hub
(datasets/haisonle001/QueryRoute), per-query QPP feature caches for its QPP selector and the
per-query outcome of each of its 11 candidate query reformulations. This script joins them into
the two arrays leakcheck needs and audits them -- the check the paper's checklist asks readers to
run, applied to the first routing benchmark we found that ships both arrays.

  X: for each query, every numeric feature of every candidate (pre_qpp, post_qpp, bert_qpp,
     rv_pre files, candidates 0000-0010), as a selector scoring all candidates would see them.
  R: per-query outcome of each candidate (judge_labels/<backbone>/eval_<ds>_methods/eval_00XX.json,
     BM25), audited against nDCG10 (the benchmark's metric) and RR10, AP10, mAP, Recall1000.

Their data is NOT redistributed here (no licence is stated in the release): --download fetches it
into --data-dir, which should sit outside the repository. Only JSON is read; no pickles.

    python queryroute_audit.py --download --data-dir <dir outside repo> --out-dir <new dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import numpy as np

import common as C
import leakcheck

BASE = "https://huggingface.co/datasets/haisonle001/QueryRoute/resolve/main"
BACKBONE_QPP, BACKBONE_JUDGE = "qwen25_7b", "Qwen2.5-7B-Instruct"   # the paper's main backbone
DATASETS = {"scifact": "eval_scifact_methods", "fiqa": "eval_fiqa_methods",
            "arguana": "eval_arguana_methods", "trec2019": "eval19_methods"}
CANDS = [f"{i:04d}" for i in range(11)]
KINDS = ["pre_qpp", "post_qpp", "bert_qpp", "rv_pre"]
METRICS = ["nDCG10", "RR10", "AP10", "mAP", "Recall1000"]


def download(data_dir):
    for ds, ev in DATASETS.items():
        for c in CANDS:
            jobs = [(f"{BASE}/qpp_features/qpp_cache_{BACKBONE_QPP}/{ds}/{BACKBONE_JUDGE}/{k}_{c}.json",
                     os.path.join(data_dir, ds, "feat", f"{k}_{c}.json")) for k in KINDS]
            jobs.append((f"{BASE}/judge_labels/{BACKBONE_JUDGE}/{ev}/eval_{c}.json",
                         os.path.join(data_dir, ds, "rew", f"eval_{c}.json")))
            for url, path in jobs:
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(url, path)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build(data_dir, ds):
    D = os.path.join(data_dir, ds)
    rew = {c: load(os.path.join(D, "rew", f"eval_{c}.json")) for c in CANDS}
    feat = {(k, c): load(os.path.join(D, "feat", f"{k}_{c}.json")) for k in KINDS for c in CANDS}
    qids = sorted(set.intersection(*[set(v) for v in rew.values()], *[set(v) for v in feat.values()]))
    names, cols = [], []
    for c in CANDS:
        for k in KINDS:
            keys = sorted({kk for q in qids for kk, vv in feat[(k, c)][q].items()
                           if isinstance(vv, (int, float)) and not isinstance(vv, bool)})
            for kk in keys:
                names.append(f"{k}:{kk}@{c}")
                cols.append([feat[(k, c)][q].get(kk) for q in qids])
    X = np.nan_to_num(np.array([[np.nan if v is None else float(v) for v in col] for col in cols]).T, nan=0.0)
    Rs = {m: np.array([[float(rew[c][q].get(m) or 0.0) for c in CANDS] for q in qids]) for m in METRICS}
    return X, Rs, names


def max_exact_share(X, R):
    best = 0.0
    for a in range(R.shape[1]):
        m = R[:, a] != 0
        if m.any():
            share = np.isclose(X[m], R[m, a][:, None], rtol=1e-6, atol=1e-9).mean(0).max()
            best = max(best, float(share))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "queryroute_audit"))
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    if args.download:
        download(args.data_dir)
    res = {"header": {**C.run_header(args), "leakcheck_version": leakcheck.__version__,
                      "source": "huggingface.co/datasets/haisonle001/QueryRoute",
                      "backbone": BACKBONE_JUDGE, "reward_retriever": "BM25"}, "datasets": {}}
    for ds in DATASETS:
        X, Rs, names = build(args.data_dir, ds)
        entry = {"n_queries": int(X.shape[0]), "n_features": int(X.shape[1]), "n_arms": len(CANDS), "by_metric": {}}
        for m, R in Rs.items():
            rep = leakcheck.audit(X, R, feature_names=names, arm_names=CANDS)
            rhos = [abs(f.statistic) for f in rep.suspicions if f.rung in ("monotone_derived", "jittered")]
            entry["by_metric"][m] = {
                "n_verdicts": len(rep.verdicts), "n_suspicions": len(rep.suspicions),
                "max_exact_share_nonzero_rows": max_exact_share(X, R),
                "max_abs_rho_among_suspicions": max(rhos) if rhos else 0.0,
                "top_suspicions": [str(f) for f in sorted(rep.suspicions, key=lambda f: -abs(f.statistic))][:5]}
        t = time.time()
        rep2 = leakcheck.audit_v2(X, Rs["nDCG10"], feature_names=names, arm_names=CANDS, n_perm=args.n_perm)
        dec = [f for f in rep2.findings if f.rung == "decodable"]
        entry["audit_v2_nDCG10"] = {
            "n_verdicts": len(rep2.verdicts), "n_decodable_pairs": len(dec),
            "decodable_column_share": len({f.feature for f in dec}) / X.shape[1],
            "max_heldout_r2": max((f.statistic for f in dec), default=0.0),
            "top_decodable": [str(f) for f in sorted(dec, key=lambda f: -f.statistic)][:5],
            "seconds": round(time.time() - t, 1)}
        res["datasets"][ds] = entry
        print(ds, entry["n_queries"], "queries,", entry["n_features"], "features: verdicts",
              sum(v["n_verdicts"] for v in entry["by_metric"].values()),
              "| max exact share", max(v["max_exact_share_nonzero_rows"] for v in entry["by_metric"].values()),
              "| max|rho|", round(max(v["max_abs_rho_among_suspicions"] for v in entry["by_metric"].values()), 3),
              "| max decode R2", round(entry["audit_v2_nDCG10"]["max_heldout_r2"], 3), flush=True)
    C.write_json(os.path.join(out_dir, "summary.json"), res)


if __name__ == "__main__":
    main()
