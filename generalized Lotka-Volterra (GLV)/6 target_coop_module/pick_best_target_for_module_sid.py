#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pandas as pd


def load_meta_A(meta_path: str, site: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    n = len(meta["species"])
    sites = meta.get("sites", {})
    if not sites:
        raise ValueError("glv_meta has no 'sites'.")
    if site not in sites:
        site = list(sites.keys())[0]
    A = np.array(sites[site]["A"], dtype=float)
    return meta, A, n, site


def parse_S_to_int(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.slice(1).astype(np.int32).to_numpy()


def topm_pos_sources(A: np.ndarray, t1: int, m: int) -> np.ndarray:
    t0 = t1 - 1
    row = A[t0].astype(float).copy()
    row[t0] = -np.inf
    pos = np.where(row > 0)[0]
    if pos.size == 0:
        return np.array([], dtype=np.int32)
    order = np.argsort(-row[pos])
    chosen = pos[order[: min(m, pos.size)]] + 1
    return chosen.astype(np.int32)


def encode_pair(p1: int, p2: int, base: int) -> int:
    if p1 > p2:
        p1, p2 = p2, p1
    return p1 * base + p2


def truth_pairs_from_sources(src: np.ndarray, base: int):
    if src.size < 2:
        return np.array([], dtype=np.int64)
    enc = []
    for i in range(src.size):
        for j in range(i + 1, src.size):
            enc.append(encode_pair(int(src[i]), int(src[j]), base))
    enc = np.array(enc, dtype=np.int64)
    enc.sort()
    return np.unique(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="*_K2_all_targets_df.tsv(.gz)")
    ap.add_argument("--glv_meta", required=True)
    ap.add_argument("--site", default="site1")
    ap.add_argument("--m", type=int, default=10, help="truth top-m positive sources in A")
    ap.add_argument("--topK", type=int, default=80, help="Top-K SID pairs per target (ranked by synergy)")
    ap.add_argument("--min_truth_pairs", type=int, default=5)
    ap.add_argument("--out", default="best_targets_sid.tsv")
    args = ap.parse_args()

    meta, A, n, site_used = load_meta_A(args.glv_meta, args.site)
    base = n + 1

    usecols = ["target_var", "order", "source_otu", "target", "synergy", "redundant"]
    df = pd.read_csv(args.sid, sep="\t", compression="infer", engine="c", usecols=usecols)
    df = df[df["order"].astype(np.int16) == 2].copy()
    if df.empty:
        raise RuntimeError("No order==2 rows found.")

    t = parse_S_to_int(df["target_var"])
    s1 = parse_S_to_int(df["source_otu"])
    s2 = parse_S_to_int(df["target"])
    p1 = np.minimum(s1, s2).astype(np.int32)
    p2 = np.maximum(s1, s2).astype(np.int32)

    valid = (p1 != p2) & (p1 != t) & (p2 != t)
    df = df.loc[valid].copy()
    t = t[valid]; p1 = p1[valid]; p2 = p2[valid]

    df2 = pd.DataFrame({
        "t": t.astype(np.int32),
        "p1": p1.astype(np.int32),
        "p2": p2.astype(np.int32),
        "syn": df["synergy"].to_numpy(dtype=np.float32),
    })

    results = []
    for t1, sub in df2.groupby("t", sort=True):
        src = topm_pos_sources(A, int(t1), args.m)
        truth = truth_pairs_from_sources(src, base)
        if truth.size < args.min_truth_pairs:
            continue

        sub = sub.sort_values("syn", ascending=False).head(args.topK)
        enc_pred = (sub["p1"].to_numpy(np.int64) * base + sub["p2"].to_numpy(np.int64))
        hits = int(np.isin(enc_pred, truth, assume_unique=False).sum())

        results.append({
            "target": f"S{int(t1)}",
            "site": site_used,
            "m": args.m,
            "topK": args.topK,
            "truth_sources": int(src.size),
            "truth_pairs": int(truth.size),
            "hits_in_topK": hits,
            "hit_rate": hits / args.topK,
            "topK_mean_synergy": float(sub["syn"].mean()),
            "topK_max_synergy": float(sub["syn"].max()),
        })

    if not results:
        raise RuntimeError("No targets found. Try increasing --m or lowering --min_truth_pairs.")

    out = pd.DataFrame(results).sort_values(
        ["hits_in_topK", "hit_rate", "topK_mean_synergy", "truth_pairs"],
        ascending=[False, False, False, False]
    )
    out.to_csv(args.out, sep="\t", index=False)

    best = out.iloc[0]
    print("[OK] wrote:", args.out)
    print("\n[RECOMMEND] best target by SID synergy topK hits:")
    print(best.to_string())
    print("\n[TOP 10]")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
