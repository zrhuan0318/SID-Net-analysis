#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_S_to_int(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.slice(1).astype(np.int32).to_numpy()


def build_truth_topm_pairs(meta: dict, m: int, truth_mode: str = "site1"):
    species = meta["species"]
    n = len(species)
    base = n + 1
    sites = meta.get("sites", {})
    site_keys = list(sites.keys())
    if not site_keys:
        raise ValueError("glv_meta has no sites.")

    def per_site_truth(A: np.ndarray):
        per_t = [set() for _ in range(n + 1)]
        for t in range(1, n + 1):
            row = A[t - 1, :].astype(float).copy()
            row[t - 1] = -np.inf
            pos = np.where(row > 0)[0]
            if pos.size < 2:
                continue
            order = np.argsort(-row[pos])
            chosen = (pos[order[: min(m, pos.size)]] + 1).astype(int)  
            if chosen.size < 2:
                continue
            for i in range(chosen.size):
                for j in range(i + 1, chosen.size):
                    p1, p2 = int(chosen[i]), int(chosen[j])
                    if p1 > p2:
                        p1, p2 = p2, p1
                    per_t[t].add(p1 * base + p2)
        return per_t

    truth_site = {}
    for sk in site_keys:
        A = np.array(sites[sk]["A"], dtype=float)
        truth_site[sk] = per_site_truth(A)

    if truth_mode == "site1":
        out = truth_site[site_keys[0]]
    elif truth_mode == "union":
        out = [set() for _ in range(n + 1)]
        for t in range(1, n + 1):
            s = set()
            for sk in site_keys:
                s |= truth_site[sk][t]
            out[t] = s
    elif truth_mode == "intersection":
        out = [set() for _ in range(n + 1)]
        for t in range(1, n + 1):
            s = None
            for sk in site_keys:
                s = truth_site[sk][t] if s is None else (s & truth_site[sk][t])
            out[t] = s if s is not None else set()
    else:
        raise ValueError("truth_mode must be site1/union/intersection")

    truth_arr_by_t = [np.array([], dtype=np.int32) for _ in range(n + 1)]
    for t in range(1, n + 1):
        if out[t]:
            truth_arr_by_t[t] = np.array(sorted(out[t]), dtype=np.int32)
    return truth_arr_by_t, base, n


def load_sid_with_labels(sid_path: str, truth_arr_by_t, base: int, metric: str):
    usecols = ["target_var", "order", "source_otu", "target", "synergy", "redundant"]
    df = pd.read_csv(sid_path, sep="\t", compression="infer", engine="c", usecols=usecols)
    df = df[df["order"].astype(np.int16) == 2].copy()
    if df.empty:
        raise RuntimeError("No order==2 rows found.")

    t = parse_S_to_int(df["target_var"])
    s1 = parse_S_to_int(df["source_otu"])
    s2 = parse_S_to_int(df["target"])

    p1 = np.minimum(s1, s2)
    p2 = np.maximum(s1, s2)
    valid = (p1 != p2) & (p1 != t) & (p2 != t)
    df = df.loc[valid].copy()
    t = t[valid]; p1 = p1[valid]; p2 = p2[valid]

    syn = df["synergy"].to_numpy(dtype=np.float32)
    red = df["redundant"].to_numpy(dtype=np.float32)
    score = syn if metric == "synergy" else (syn - red)

    enc = (p1 * base + p2).astype(np.int32)
    labels = np.zeros(enc.shape[0], dtype=bool)

    for tt in np.unique(t):
        arr = truth_arr_by_t[int(tt)]
        if arr.size == 0:
            continue
        idx = (t == tt)
        labels[idx] = np.isin(enc[idx], arr, assume_unique=False)

    out = pd.DataFrame({"target_int": t.astype(np.int32), "score": score.astype(np.float32), "label": labels.astype(np.int8)})
    return out


def make_N_grid(maxN: int, n_points: int = 60, include: list = None):
    xs = np.unique(np.round(np.logspace(np.log10(200), np.log10(maxN), n_points)).astype(int))
    xs = xs[(xs > 0) & (xs <= maxN)]
    if include:
        xs = np.unique(np.concatenate([xs, np.array(include, dtype=int)]))
        xs = xs[(xs > 0) & (xs <= maxN)]
    return xs


def precision_curve(scores: np.ndarray, labels: np.ndarray, Ns: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]
    c = np.cumsum(y)
    prec = c[Ns - 1] / Ns
    return prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--glv_meta", required=True)
    ap.add_argument("--metric", choices=["synergy", "score"], default="synergy")
    ap.add_argument("--truth_mode", choices=["site1", "union", "intersection"], default="site1")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--maxN", type=int, default=20000)
    ap.add_argument("--n_points", type=int, default=60)
    ap.add_argument("--include_Ns", default="1000,5000,10000,20000")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates")
    ap.add_argument("--subsample_frac", type=float, default=0.8, help="fraction of targets per replicate (w/o replacement)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    include = [int(x) for x in args.include_Ns.split(",") if x.strip()]
    with open(args.glv_meta, "r") as f:
        meta = json.load(f)

    truth_arr_by_t, base, n = build_truth_topm_pairs(meta, m=args.m, truth_mode=args.truth_mode)
    dat = load_sid_with_labels(args.sid, truth_arr_by_t, base, metric=args.metric)

    # baseline
    baseline = dat["label"].mean()
    print(f"[INFO] baseline={baseline:.6g}, rows={len(dat)}")

    # Ns
    maxN = min(args.maxN, len(dat))
    Ns = make_N_grid(maxN=maxN, n_points=args.n_points, include=include)

    # main curve (all targets)
    prec = precision_curve(dat["score"].to_numpy(float), dat["label"].to_numpy(int), Ns)
    lift = prec / baseline if baseline > 0 else np.full_like(prec, np.nan, dtype=float)
    curve = pd.DataFrame({"N": Ns, "precision": prec, "baseline": baseline, "lift": lift})
    curve.to_csv(f"{args.out_prefix}_curve.tsv", sep="\t", index=False)

    # bootstrap over targets 
    rng = np.random.default_rng(args.seed)
    targets = np.unique(dat["target_int"].to_numpy(np.int32))
    n_t = len(targets)
    k = max(2, int(round(n_t * args.subsample_frac)))
    lifts = np.zeros((args.boot, len(Ns)), dtype=float)

    scores_all = dat["score"].to_numpy(float)
    labels_all = dat["label"].to_numpy(int)
    t_all = dat["target_int"].to_numpy(np.int32)

    for b in range(args.boot):
        chosen = rng.choice(targets, size=k, replace=False)
        mask = np.isin(t_all, chosen)
        if mask.sum() < maxN:
            sb = scores_all[mask]; yb = labels_all[mask]
            base_b = yb.mean()
            if base_b <= 0:
                lifts[b, :] = np.nan
                continue
            Ns_b = np.minimum(Ns, len(sb))
            prec_b = precision_curve(sb, yb, Ns_b)
            lifts[b, :] = prec_b / base_b
        else:
            sb = scores_all[mask]; yb = labels_all[mask]
            base_b = yb.mean()
            if base_b <= 0:
                lifts[b, :] = np.nan
                continue
            prec_b = precision_curve(sb, yb, Ns)
            lifts[b, :] = prec_b / base_b

    lift_mean = np.nanmean(lifts, axis=0)
    lift_lo = np.nanpercentile(lifts, 2.5, axis=0)
    lift_hi = np.nanpercentile(lifts, 97.5, axis=0)

    ci = pd.DataFrame({"N": Ns, "lift_mean": lift_mean, "lift_lo": lift_lo, "lift_hi": lift_hi, "baseline": baseline})
    ci.to_csv(f"{args.out_prefix}_curve_ci.tsv", sep="\t", index=False)

    # plot
    plt.figure()
    plt.plot(Ns, lift, label=f"{args.metric} (all)")
    plt.fill_between(Ns, lift_lo, lift_hi, alpha=0.25, label=f"target-subsample {int(args.subsample_frac*100)}% CI")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Top-N")
    plt.ylabel("Lift over baseline")
    plt.title(f"K=2 {args.metric} vs TOPM-POS (m={args.m}, mode={args.truth_mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_curve.png", dpi=300)
    print(f"[OK] wrote {args.out_prefix}_curve.tsv, {args.out_prefix}_curve_ci.tsv, {args.out_prefix}_curve.png")


if __name__ == "__main__":
    main()
