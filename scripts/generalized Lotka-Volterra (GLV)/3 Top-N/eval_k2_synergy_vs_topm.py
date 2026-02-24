#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pandas as pd


def pr_auc(scores: np.ndarray, ytrue: np.ndarray):
    order = np.argsort(-scores)
    y = ytrue[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1], 1)

    auc = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        auc += float(p) * float(r - prev_r)
        prev_r = float(r)
    return auc, precision, recall


def precision_at_topN(scores: np.ndarray, ytrue: np.ndarray, Ns):
    order = np.argsort(-scores)
    y = ytrue[order]
    total = len(ytrue)
    pos = int(ytrue.sum())
    baseline = pos / total if total else 0.0

    rows = []
    for N in Ns:
        N = min(int(N), total)
        prec = float(y[:N].mean()) if N > 0 else 0.0
        lift = (prec / baseline) if baseline > 0 else np.nan
        rows.append((N, prec, baseline, lift))
    return rows


def canonical_pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def build_topm_truth_sets(meta: dict, m: int):
    species = meta["species"]
    n = len(species)

    truth_by_site = {}
    stats_by_site = {}

    for site_key, obj in meta["sites"].items():
        A = np.array(obj["A"], dtype=float)  # A[target, source]
        S = set()
        total_targets = 0
        total_pos_sources = 0
        total_selected = 0
        total_pairs = 0

        for ti in range(n):
            tname = species[ti]
            total_targets += 1

            row = A[ti, :].copy()
            row[ti] = -np.inf  

            pos_idx = np.where(row > 0)[0]
            total_pos_sources += int(pos_idx.size)
            if pos_idx.size < 2:
                continue

            pos_vals = row[pos_idx]
            order = np.argsort(-pos_vals)
            chosen = pos_idx[order[: min(m, pos_idx.size)]]
            total_selected += int(chosen.size)

            if chosen.size < 2:
                continue

            for ii in range(chosen.size):
                for jj in range(ii + 1, chosen.size):
                    jname = species[chosen[ii]]
                    kname = species[chosen[jj]]
                    p1, p2 = canonical_pair(jname, kname)
                    S.add((tname, p1, p2))
                    total_pairs += 1

        truth_by_site[site_key] = S
        stats_by_site[site_key] = {
            "targets": total_targets,
            "sum_pos_sources": total_pos_sources,
            "sum_selected_sources": total_selected,
            "num_truth_triplets": len(S),
            "num_pairs_generated_raw": total_pairs,  # includes duplicates across targets? (target included in key, so no)
        }

    return truth_by_site, stats_by_site


def combine_truth(truth_by_site: dict, truth_mode: str):
    site_keys = list(truth_by_site.keys())
    if not site_keys:
        return set(), []

    if truth_mode == "site1":
        return set(truth_by_site[site_keys[0]]), site_keys

    sets = [truth_by_site[k] for k in site_keys]
    if truth_mode == "intersection":
        out = set.intersection(*sets) if sets else set()
    else:
        out = set.union(*sets) if sets else set()
    return out, site_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="glv200_alpha1.0_K2_all_targets_df.tsv")
    ap.add_argument("--glv_meta", required=True, help="glv_meta_alpha1.0.json (must contain sites->A, species)")
    ap.add_argument("--metric", default="synergy", choices=["synergy", "score"],
                    help="synergy or score=synergy-redundant")
    ap.add_argument("--truth_mode", default="union", choices=["union", "intersection", "site1"])
    ap.add_argument("--m", type=int, default=10, help="top-m positive sources per target")
    ap.add_argument("--Ns", default="50,100,200", help="Top-N list")
    ap.add_argument("--out_prefix", default="k2_topm")

    args = ap.parse_args()

    # ---- load SID ----
    sid = pd.read_csv(args.sid, sep="\t")
    sid.columns = [c.strip() for c in sid.columns]

    required = {"target_var", "order", "source_otu", "target", "synergy", "redundant"}
    if not required.issubset(set(sid.columns)):
        raise ValueError(f"SID file missing columns. Need {required}, got {set(sid.columns)}")

    sid = sid[sid["order"].astype(int) == 2].copy()


    a = sid["source_otu"].astype(str).to_numpy()
    b = sid["target"].astype(str).to_numpy()
    sid["p1"] = np.minimum(a, b)
    sid["p2"] = np.maximum(a, b)

    if args.metric == "synergy":
        sid["score"] = sid["synergy"].astype(float)
    else:
        sid["score"] = sid["synergy"].astype(float) - sid["redundant"].astype(float)


    with open(args.glv_meta, "r") as f:
        meta = json.load(f)

    truth_by_site, stats_by_site = build_topm_truth_sets(meta, m=args.m)
    truth_set, site_keys = combine_truth(truth_by_site, args.truth_mode)


    scores = []
    ytrue = []

    missing = 0
    for _, r in sid.iterrows():
        t = str(r["target_var"])
        p1 = str(r["p1"])
        p2 = str(r["p2"])
        if p1 == t or p2 == t or p1 == p2:
            continue
        key = (t, p1, p2)
        y = 1 if key in truth_set else 0
        ytrue.append(y)
        scores.append(float(r["score"]))

    scores = np.array(scores, dtype=float)
    ytrue = np.array(ytrue, dtype=int)

    auc, prec, rec = pr_auc(scores, ytrue)

    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    top_rows = precision_at_topN(scores, ytrue, Ns)

    # ---- outputs ----
    pr_df = pd.DataFrame({"precision": prec, "recall": rec})
    pr_df.to_csv(f"{args.out_prefix}_pr.tsv", sep="\t", index=False)

    top_df = pd.DataFrame(top_rows, columns=["N", "precision", "baseline", "lift"])
    top_df.insert(0, "metric", args.metric)
    top_df.to_csv(f"{args.out_prefix}_precision_at_topN.tsv", sep="\t", index=False)

    summary = {
        "sid": args.sid,
        "glv_meta": args.glv_meta,
        "metric": args.metric,
        "truth_def": f"topm_pos(m={args.m}) per target per site; pairs among selected sources",
        "truth_mode": args.truth_mode,
        "sites": site_keys,
        "n_triplets_scored": int(len(scores)),
        "positives": int(ytrue.sum()),
        "baseline": float(ytrue.mean()),
        "AUPRC": float(auc),
        "stats_by_site": stats_by_site,
    }
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] metric={args.metric} truth=TOPM-POS(m={args.m}) mode={args.truth_mode}")
    print(f"[OK] AUPRC={auc:.6f} baseline={ytrue.mean():.6f} positives={int(ytrue.sum())}/{len(ytrue)}")
    print(top_df)


if __name__ == "__main__":
    main()
