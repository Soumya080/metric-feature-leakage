"""05 — Build the leak-gradient arms at each corpus scale.

For every scale, runs `src/router/prepare_env_data.py` once per feature mode, pointed
at that scale's retrieval shards. The arm definitions are IMPORTED from
`experiments/router_experiments/leakage_ablation.py` rather than restated, so the
scaled arms and the 500-passage arms can never drift apart in naming or in which
feature mode backs which arm — the leak-vs-scale curve compares like with like.

Nothing about feature construction is reimplemented here. `prepare_env_data.py` is the
only thing that has ever built these features; this script decides where it reads from
and writes to, and then checks its output.

Integrity checks mirror leakage_ablation.check_integrity(), applied per scale:

  1. every arm covers the same query-ID set
  2. reward tables are bitwise identical across arms — only features may differ
  3. features[0:22] and features[26:] are identical across arms
  4. features[22:26] actually differ between every pair of arms
  5. the `leaky` arm genuinely reproduces the bug (f22 == bm25 MRR@k, f23 == dense MRR@k)

Check 2 is the one that makes the whole experiment meaningful: if the reward tables
differed, an accuracy gap between arms could be the reward moving rather than the
feature construction.

Usage:
    python experiments/corpus_v3/05_build_arms.py
    python experiments/corpus_v3/05_build_arms.py --smoke
    python experiments/corpus_v3/05_build_arms.py --scale 100000 --arm leaky honest_minimal
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

LEAKAGE_ABLATION_PATH = os.path.join(
    common.PROJECT_ROOT, "experiments", "router_experiments", "leakage_ablation.py"
)

# k for the reward metric and for the leak arms' MRR@k copy. Must match whatever the
# 500-passage arms used, or the two dose-response curves are not comparable.
# CORRECTED NOTE 2026-10-31: it does NOT match. The 500-passage leak arms in src/router/ copy
# MRR@20 (slot 22 == mrr_at_20 on every row), while this builds MRR@5, which is zero on 76-95%
# of rows at 20K-100K. The value is left unchanged so the committed Figure 1 artifacts stay
# reproducible. The paper's Figure 1 caption states the difference and does not compare the
# two curves' heights.
REWARD_K = "5"


def load_arm_definitions():
    """ARMS / ARM_ORDER from leakage_ablation.py, by path (leading digit in siblings)."""
    spec = importlib.util.spec_from_file_location("_leakage_ablation", LEAKAGE_ABLATION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {LEAKAGE_ABLATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(common.PROJECT_ROOT)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module.ARMS, module.ARM_ORDER


def env_dir(scale: int) -> str:
    return os.path.join(common.ARTIFACT_DIR, f"scale_{scale}", "env")


def env_path(scale: int, dataset: str, suffix: str) -> str:
    return os.path.join(env_dir(scale), f"{dataset}_env_data{suffix}.json")


def build_arm(scale: int, arm: str, mode: str, suffix: str, *,
              gold_type: str = "all", force: bool = False) -> bool:
    """Run prepare_env_data.py for one (scale, arm). Returns True if data is on disk."""
    out_dir = env_dir(scale)
    os.makedirs(out_dir, exist_ok=True)

    expected = [env_path(scale, ds, suffix) for ds in common.DATASETS]
    have = [p for p in expected if os.path.exists(p)]
    if have and not force:
        print(f"    {arm:<18} already built ({len(have)}/{len(expected)} datasets)")
        return True

    results_dir = os.path.join(common.RESULTS_DIR, f"scale_{scale}")
    if not os.path.isdir(results_dir):
        print(f"    {arm:<18} SKIPPED — no retrieval shards at {results_dir}")
        return False

    cmd = [
        sys.executable, "-m", "src.router.prepare_env_data",
        "--feature-mode", mode,
        "--suffix", suffix,
        "--k", REWARD_K,
        "--gold-type", gold_type,
        "--results-dir", results_dir,
        "--output-dir", out_dir,
    ]
    print(f"    {arm:<18} building (--feature-mode {mode} --suffix {suffix!r})")
    proc = subprocess.run(cmd, cwd=common.PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        print(f"    {arm:<18} FAILED:\n      " + "\n      ".join(tail))
        return False
    return any(os.path.exists(p) for p in expected)


def _load_env(scale: int, dataset: str, suffix: str) -> dict | None:
    path = env_path(scale, dataset, suffix)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_integrity(scale: int, dataset: str, arms: dict, arm_order: list) -> list[str]:
    """Mirror of leakage_ablation.check_integrity for one (scale, dataset).

    Returns a list of violation strings; empty means the arms form a clean ablation.
    """
    envs = {}
    for arm in arm_order:
        if arm not in arms:
            continue
        data = _load_env(scale, dataset, arms[arm][1])
        if data is not None:
            envs[arm] = data

    if len(envs) < 2:
        return [f"only {len(envs)} arm(s) present — nothing to compare"]

    problems: list[str] = []
    order = [a for a in arm_order if a in envs]
    ref = order[0]
    qids_ref = set(envs[ref].keys())

    for arm in order[1:]:
        if set(envs[arm].keys()) != qids_ref:
            problems.append(
                f"query-ID sets differ between {ref!r} and {arm!r} "
                f"({len(qids_ref)} vs {len(envs[arm])})"
            )
    if problems:
        return problems  # later checks assume aligned ids

    qids = sorted(qids_ref)

    for arm in order[1:]:
        for qid in qids:
            for ret in ("bm25", "dense", "hybrid"):
                a = envs[ref][qid]["retrievers"][ret]["mrr"]
                b = envs[arm][qid]["retrievers"][ret]["mrr"]
                if a != b:
                    problems.append(
                        f"reward mismatch qid={qid} ret={ret}: {ref}={a} vs {arm}={b} — "
                        "rewards must be identical across arms"
                    )
                    break
            if problems:
                break
        if problems:
            break

    def outside(vec):
        return vec[0:22] + vec[26:]

    for arm in order[1:]:
        bad = [q for q in qids if outside(envs[ref][q]["features"]) != outside(envs[arm][q]["features"])]
        if bad:
            problems.append(
                f"features[0:22]+[26:] differ between {ref!r} and {arm!r} for "
                f"{len(bad)} queries (e.g. qid={bad[0]})"
            )

    for i, a in enumerate(order):
        for b in order[i + 1:]:
            n_diff = sum(1 for q in qids
                         if envs[a][q]["features"][22:26] != envs[b][q]["features"][22:26])
            if n_diff == 0:
                problems.append(f"features[22:26] identical between {a!r} and {b!r} "
                                "for every query — comparing an arm against itself")

    if "leaky" in envs:
        raw_path = common.shard_path(scale, dataset)
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = {str(r["query_id"]): r for r in json.load(f)}
            mismatched = 0
            for qid in qids:
                rec = raw.get(qid)
                if rec is None:
                    continue
                f22, f23 = envs["leaky"][qid]["features"][22:24]
                m = rec["retrievers"]
                exp22 = m["bm25"]["metrics_at_k"][REWARD_K]["mrr"]
                exp23 = m["dense"]["metrics_at_k"][REWARD_K]["mrr"]
                if abs(f22 - exp22) > 1e-12 or abs(f23 - exp23) > 1e-12:
                    mismatched += 1
            if mismatched:
                problems.append(
                    f"leaky arm does not reproduce the bug on {mismatched}/{len(qids)} "
                    f"queries (features[22:24] should equal MRR@{REWARD_K})"
                )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scale", type=int, nargs="+", default=None)
    parser.add_argument("--arm", type=str, nargs="+", default=None)
    parser.add_argument("--gold-type", type=str, default="all",
                        choices=["all", "real_wikipedia", "constructed"],
                        help="corpus_v3 has no constructed gold, so 'all' is the honest "
                             "default here — unlike the 500-passage setting.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    common.ensure_dirs()
    arms, arm_order = load_arm_definitions()
    if args.arm:
        unknown = [a for a in args.arm if a not in arms]
        if unknown:
            raise SystemExit(f"unknown arm(s): {unknown}. Known: {sorted(arms)}")
        arm_order = [a for a in arm_order if a in args.arm]

    scales = common.resolve_scales(args.smoke, tuple(args.scale) if args.scale else None)

    report = {"scales": {}, "arms": arm_order, "reward_k": REWARD_K,
              "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    any_problem = False

    for scale in scales:
        print(f"\nscale {common.fmt_int(scale)}")
        built = []
        for arm in arm_order:
            mode, suffix = arms[arm]
            if build_arm(scale, arm, mode, suffix,
                         gold_type=args.gold_type, force=args.force):
                built.append(arm)

        scale_report = {"built_arms": built, "integrity": {}}
        for dataset in common.DATASETS:
            if not os.path.exists(common.shard_path(scale, dataset)):
                continue
            problems = check_integrity(scale, dataset, arms, arm_order)
            scale_report["integrity"][dataset] = problems
            if problems:
                any_problem = True
                print(f"    integrity {dataset}: {len(problems)} PROBLEM(S)")
                for p in problems[:3]:
                    print(f"      - {p}")
            else:
                print(f"    integrity {dataset}: clean")

        report["scales"][str(scale)] = scale_report

    common.write_manifest(
        os.path.join(common.RESULTS_DIR, "arms_report.json"), report
    )
    print(f"\nWrote {os.path.join(common.RESULTS_DIR, 'arms_report.json')}")

    if any_problem:
        print("\nINTEGRITY PROBLEMS FOUND — the affected (scale, dataset) cells are not a "
              "clean ablation and must not be reported.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
