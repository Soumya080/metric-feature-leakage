#!/usr/bin/env python3
"""Table 3 RL row, re-run with the best-validation checkpoint actually restored.

Why: canonical_evaluation.train_rl_single_seed calls ReinforceTrainer.train(), whose
_save_checkpoint writes the best-validation weights to disk but never loads them back, so
Table 3's "Policy-gradient RL" row is scored with FINAL-epoch weights.

This script changes only that. Same 61-D honest env files, same split (split_seed 42,
70/15/15), same 15 seeds, epochs, hidden size, learning rate, entropy coefficient and
advantage mode -- all imported from canonical_evaluation, not restated -- and the same
ReinforceTrainer._train_epoch update. The epoch loop touches only train and validation;
test is evaluated after training, twice, and reported side by side:
  final_epoch   current weights at the last epoch  (the old protocol; should land near Table 3)
  restored      best-validation weights loaded back (the corrected protocol)
Neither test number feeds back into any choice.

src/router/train_reinforce.py is not modified, and nothing is written to its shared
checkpoint path. Outputs: results/crc_v2/rl_restore/ (rows.jsonl incremental, --resume).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch

import common as C
import canonical_evaluation as CE  # noqa: E402
from src.router.policy_network import PolicyAgent  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402
from src.router.train_reinforce import ReinforceTrainer  # noqa: E402

TABLE3_RL = {"nq": 0.1512, "triviaqa": 0.1881, "hotpotqa": 0.0683}


def test_pass(trainer, env, agent):
    actions, rewards = [], []
    for qid in env.query_pool:
        a, _ = agent.select_action(trainer._normalize_state(env.get_state(qid)), evaluate=True)
        actions.append(int(a))
        rewards.append(float(env.step(qid, a, target_metric=CE.REWARD_TYPE)))
    return actions, rewards


def run_seed(ds, path, seed, epochs, split_seed):
    tr, va, te = create_train_val_test_split(path, split_seed=split_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PolicyAgent(input_dim=tr.state_dim, hidden_dim=CE.HIDDEN_DIM, output_dim=3, learningrate=CE.LEARNING_RATE)
    trainer = ReinforceTrainer(env_train=tr, env_test=te, agent=agent, reward_type=CE.REWARD_TYPE,
                               entropy_coeff=CE.ENTROPY_COEFF, seed=seed, env_val=va,
                               advantage_mode=CE.ADVANTAGE_MODE)
    best_val, best_epoch, best_state, val_hist = -float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        trainer._train_epoch()
        v = trainer._evaluate(use_val=True)[0]
        val_hist.append(v)
        if v > best_val:
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(agent.policynetwork.state_dict())
    fin_actions, fin_rewards = test_pass(trainer, te, agent)
    agent.policynetwork.load_state_dict(best_state)
    res_actions, res_rewards = test_pass(trainer, te, agent)
    oracle = [max(te.get_all_rewards(q, target_metric=CE.REWARD_TYPE).values()) for q in te.query_pool]
    return {
        "dataset": ds, "seed": seed, "split_seed": split_seed, "epochs": epochs,
        "best_validation_epoch": best_epoch, "best_validation_reward": best_val, "validation_history": val_hist,
        "test_final_epoch": float(np.mean(fin_rewards)), "test_restored": float(np.mean(res_rewards)),
        "test_oracle": float(np.mean(oracle)), "n_test": len(te.query_pool),
        "per_query": [{"qid": q, "action_final": af, "reward_final": rf, "action_restored": ar, "reward_restored": rr}
                      for q, af, rf, ar, rr in zip(te.query_pool, fin_actions, fin_rewards, res_actions, res_rewards)],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "rl_restore"))
    ap.add_argument("--datasets", nargs="+", choices=C.DATASETS, default=list(C.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(CE.SEEDS))
    ap.add_argument("--epochs", type=int, default=CE.NUM_EPOCHS)
    ap.add_argument("--split-seed", type=int, default=CE.SPLIT_SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="time 5 epochs on NQ, one seed, then stop")
    args = ap.parse_args(argv)

    if args.calibrate:
        t = time.time()
        run_seed("nq", C.anchor_path("nq", ""), args.seeds[0], 5, args.split_seed)
        per_epoch = (time.time() - t) / 5
        n = len(args.datasets) * len(args.seeds) * args.epochs
        print(f"~{per_epoch:.2f}s per NQ epoch; full run ~ {per_epoch * n / 3600:.1f} h (NQ-sized estimate)")
        return

    if args.resume and os.path.isdir(args.out_dir):
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = C.prepare_out_dir(args.out_dir)
        C.write_json(os.path.join(out_dir, "run_header.json"),
                     {**C.run_header(args), "constants_from_canonical_evaluation": {
                         k: getattr(CE, k) for k in ("HIDDEN_DIM", "LEARNING_RATE", "ENTROPY_COEFF",
                                                     "REWARD_TYPE", "ADVANTAGE_MODE", "NUM_EPOCHS", "SPLIT_SEED")}})
    rows_path = os.path.join(out_dir, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            done = {(r["dataset"], r["seed"]) for r in map(json.loads, f)}

    for ds in args.datasets:
        path = C.anchor_path(ds, "")
        for seed in args.seeds:
            if (ds, seed) in done:
                continue
            t = time.time()
            r = run_seed(ds, path, seed, args.epochs, args.split_seed)
            r["source_sha256"] = C.sha256_file(path)
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  {ds} seed={seed} final={r['test_final_epoch']:.4f} restored={r['test_restored']:.4f} "
                  f"best_epoch={r['best_validation_epoch']} ({time.time() - t:.0f}s)", flush=True)

    with open(rows_path, encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]
    summary = {}
    for ds in args.datasets:
        rs = [r for r in runs if r["dataset"] == ds]
        if not rs:
            continue
        fin = np.array([r["test_final_epoch"] for r in rs])
        res = np.array([r["test_restored"] for r in rs])
        summary[ds] = {"n_seeds": len(rs), "final_epoch_mean": float(fin.mean()), "final_epoch_sd": float(fin.std(ddof=1)) if len(rs) > 1 else None,
                       "restored_mean": float(res.mean()), "restored_sd": float(res.std(ddof=1)) if len(rs) > 1 else None,
                       "table3_reported": TABLE3_RL[ds]}
    C.write_json(os.path.join(out_dir, "summary.json"), summary)
    with open(os.path.join(out_dir, "per_query.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "qid", "action_final", "reward_final", "action_restored", "reward_restored"])
        for r in runs:
            for p in r["per_query"]:
                w.writerow([r["dataset"], r["seed"], p["qid"], p["action_final"], p["reward_final"],
                            p["action_restored"], p["reward_restored"]])
    for ds, s in summary.items():
        print(f"  {ds}: final-epoch {s['final_epoch_mean']:.4f}  restored {s['restored_mean']:.4f}  (Table 3: {s['table3_reported']})")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
