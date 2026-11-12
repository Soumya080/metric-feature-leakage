#!/usr/bin/env python3
"""R1 (build step): noise-scale variants of the 61-D leak, at the paper's original scale.

Each variant is the honest env file (src/router/{ds}_env_data.json, honest_minimal) with
ONLY slots 22-24 replaced:

    slot 22 = jitter(v_bm25,  "bm25",  s)
    slot 23 = jitter(v_dense, "dense", s)
    slot 24 = slot 23 - slot 22
    slot 25 = honest value (unchanged)

  family "mrr20":      v_arm = retrievers[arm].mrr_at_20  (what the paper's leaky rung copies)
  family "rankreward": v_arm = retrievers[arm].rank_reward (the actual full-ranking reward)

s in NOISE_SCALES (0 = exact copy), plus a clipped-to-[0,1] copy at s = 0.15 for both
families. The jitter is the repo's crc32 rule generalised to any half-width; at s = 0.15 it
is asserted bit-identical to the existing leak_noisy file's slots 22-23.

Difference from the paper's leaky / leak_noisy rungs, stated on purpose: those also leak
through slot 25 (variance of BM25 MRR@k). These variants hold slot 25 honest so that one
thing varies (the noise on the per-arm metric), and so the mrr20 and rankreward families
have identical structure. The existing rungs are evaluated alongside as anchors.

Fails closed on any of: query IDs or rewards differing between honest/leaky/leak_noisy
files, non-leak features differing, mrr != rank_reward, leaky slots 22-23 not equal to
mrr_at_20, or the generalised jitter not reproducing leak_noisy at s = 0.15.

Reads src/router/*.json only; writes results/crc_v2/env_data/ (refuses if non-empty).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import common as C

TOL = 1e-12


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"INTEGRITY FAILURE: {msg}")


def check_sources(ds: str) -> dict:
    honest = C.load_env(C.anchor_path(ds, ""))
    leaky = C.load_env(C.anchor_path(ds, "_leaky"))
    noisy = C.load_env(C.anchor_path(ds, "_leaknoisy"))
    _assert(set(honest) == set(leaky) == set(noisy), f"{ds}: query IDs differ across source files")
    n_checked = 0
    for q, e in honest.items():
        for other, name in ((leaky, "leaky"), (noisy, "leak_noisy")):
            _assert(other[q]["retrievers"] == e["retrievers"], f"{ds}/{q}: rewards differ vs {name}")
            fo, fh = other[q]["features"], e["features"]
            _assert(len(fh) == 61 and len(fo) == 61, f"{ds}/{q}: feature width is not 61")
            _assert(fo[:22] == fh[:22] and fo[26:] == fh[26:], f"{ds}/{q}: non-leak features differ vs {name}")
        for a in C.ACTIONS:
            r = e["retrievers"][a]
            for key in ("mrr", "rank_reward", "mrr_at_20"):
                _assert(key in r and math.isfinite(float(r[key])), f"{ds}/{q}/{a}: missing or non-finite {key}")
            _assert(abs(float(r["mrr"]) - float(r["rank_reward"])) < TOL, f"{ds}/{q}/{a}: mrr != rank_reward")
        b20 = float(e["retrievers"]["bm25"]["mrr_at_20"])
        d20 = float(e["retrievers"]["dense"]["mrr_at_20"])
        lf = leaky[q]["features"]
        _assert(abs(lf[22] - b20) < TOL and abs(lf[23] - d20) < TOL,
                f"{ds}/{q}: leaky slots 22-23 are not mrr_at_20 (got {lf[22:24]}, want {[b20, d20]})")
        nf = noisy[q]["features"]
        _assert(abs(nf[22] - C.jitter(b20, "bm25", 0.15)) < TOL and abs(nf[23] - C.jitter(d20, "dense", 0.15)) < TOL,
                f"{ds}/{q}: generalised jitter does not reproduce leak_noisy at s=0.15")
        n_checked += 1
    return {"honest": honest, "n_queries": n_checked}


def build_entry(entry: dict, family: str, scale: float, clipped: bool) -> dict:
    key = "mrr_at_20" if family == "mrr20" else "rank_reward"
    vb = float(entry["retrievers"]["bm25"][key])
    vd = float(entry["retrievers"]["dense"][key])
    f22, f23 = C.jitter(vb, "bm25", scale), C.jitter(vd, "dense", scale)
    if clipped:
        f22, f23 = min(1.0, max(0.0, f22)), min(1.0, max(0.0, f23))
    out = dict(entry)
    feats = list(entry["features"])
    feats[22], feats[23], feats[24] = f22, f23, f23 - f22
    out["features"] = feats
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=C.CRC_ENV_DIR)
    ap.add_argument("--datasets", nargs="+", choices=C.DATASETS, default=list(C.DATASETS))
    ap.add_argument("--check-only", action="store_true", help="run the source integrity checks, write nothing")
    args = ap.parse_args(argv)

    checked = {ds: check_sources(ds) for ds in args.datasets}
    for ds, c in checked.items():
        print(f"  {ds}: {c['n_queries']} queries, source checks passed")
    if args.check_only:
        return

    out_dir = C.prepare_out_dir(args.out_dir)
    manifest = {"header": C.run_header(args), "rule": __doc__.split("\n\n")[1],
                "noise_scales": list(C.NOISE_SCALES), "clipped_scale": C.CLIPPED_SCALE,
                "source_sha256": {}, "outputs": {}}
    for ds in args.datasets:
        manifest["source_sha256"][ds] = {s or "honest": C.sha256_file(C.anchor_path(ds, s))
                                          for s in ("", "_leaky", "_leaknoisy")}
        honest = checked[ds]["honest"]
        specs = [(f, s, False) for f in C.FAMILIES for s in C.NOISE_SCALES]
        specs += [(f, C.CLIPPED_SCALE, True) for f in C.FAMILIES]
        for fam, s, clip in specs:
            tag = C.variant_tag(fam, s, clip)
            path = C.variant_path(ds, tag, out_dir)
            env = {q: build_entry(e, fam, s, clip) for q, e in honest.items()}
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(env, f)
            manifest["outputs"].setdefault(tag, {})[ds] = {"file": os.path.basename(path),
                                                            "sha256": C.sha256_file(path)}
        print(f"  {ds}: wrote {len(specs)} variants")
    C.write_json(os.path.join(out_dir, "generation_manifest.json"), manifest)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
