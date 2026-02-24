#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_S_to_int(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.slice(1).astype(np.int32).to_numpy()


def read_matrix_species_by_samples(path: str, n_species_hint: int = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", engine="c")
    if df.shape[1] < 2:
        raise ValueError(f"{path}: seems to have <2 columns.")


    first = df.columns[0]
    col0 = df[first].astype(str)
    if (col0.str.match(r"^S\d+$").mean() > 0.9):
        df = df.set_index(first)


    if df.index.astype(str).str.match(r"^S\d+$").mean() > 0.9:
        return df


    if (pd.Series(df.columns).astype(str).str.match(r"^S\d+$").mean() > 0.5):
        maybe_id_col = df.columns[0]
        if not str(maybe_id_col).startswith("S"):
            if pd.to_numeric(df[maybe_id_col], errors="coerce").isna().mean() > 0.5:
                df = df.set_index(maybe_id_col)

        dft = df.T
        if dft.index.astype(str).str.match(r"^S\d+$").mean() > 0.9:
            return dft
        raise ValueError(f"{path}: columns looked like species, but transpose didn't yield species index.")

    if n_species_hint is not None:
        if df.shape[1] == n_species_hint:
            dft = df.T
            return dft
    raise ValueError(
        f"{path}: cannot detect species IDs. "
        f"Expected S1.. either in first column/index or in column names."
    )


def build_truth_topm_pairs(meta: dict, m: int, truth_mode: str = "site1"):
    n = len(meta["species"])
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


def mi_from_disc(x: np.ndarray, y: np.ndarray, nbins: int) -> float:
    joint = np.zeros((nbins, nbins), dtype=float)
    for a in range(nbins):
        for b in range(nbins):
            joint[a, b] = np.mean((x == a) & (y == b))
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    denom = (px @ py)
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def compute_mi_matrix_from_disc(X_disc: np.ndarray, nbins: int) -> np.ndarray:
    n, _ = X_disc.shape
    MI = np.zeros((n + 1, n + 1), dtype=np.float32)
    for t in range(1, n + 1):
        xt = X_disc[t - 1]
        for s in range(1, n + 1):
            if s == t:
                continue
            MI[t, s] = mi_from_disc(xt, X_disc[s - 1], nbins=nbins)
    return MI


def precision_curve(scores: np.ndarray, labels: np.ndarray, Ns: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]
    c = np.cumsum(y)
    return c[Ns - 1] / Ns


def make_N_grid(maxN: int, include: list):
    xs = np.unique(np.round(np.logspace(np.log10(200), np.log10(maxN), 60)).astype(int))
    xs = xs[(xs > 0) & (xs <= maxN)]
    xs = np.unique(np.concatenate([xs, np.array(include, dtype=int)]))
    xs = xs[(xs > 0) & (xs <= maxN)]
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="K=2 all_targets_df.tsv(.gz)")
    ap.add_argument("--glv_meta", required=True)
    ap.add_argument("--X_clr", required=True, help="sid_input_alpha1.0_clr.tsv")
    ap.add_argument("--X_disc", required=True, help="sid_input_alpha1.0_disc.tsv")
    ap.add_argument("--nbins", type=int, default=3, help="disc bins for MI baseline (should match disc file)")
    ap.add_argument("--sid_metric", choices=["synergy", "score"], default="synergy")
    ap.add_argument("--corr_posonly", action="store_true", help="if set, use max(rho,0) in Spearman-sum; else abs(rho)")
    ap.add_argument("--truth_mode", choices=["site1", "union", "intersection"], default="site1")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--maxN", type=int, default=30000)
    ap.add_argument("--include_Ns", default="1000,5000,10000,20000")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    include = [int(x) for x in args.include_Ns.split(",") if x.strip()]

    with open(args.glv_meta, "r") as f:
        meta = json.load(f)

    truth_arr_by_t, base, n = build_truth_topm_pairs(meta, m=args.m, truth_mode=args.truth_mode)

    # --- load SID rows ---
    usecols = ["target_var", "order", "source_otu", "target", "synergy", "redundant"]
    df = pd.read_csv(args.sid, sep="\t", compression="infer", engine="c", usecols=usecols)
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

    enc = (p1 * base + p2).astype(np.int32)
    labels = np.zeros(enc.shape[0], dtype=bool)
    for tt in np.unique(t):
        arr = truth_arr_by_t[int(tt)]
        if arr.size == 0:
            continue
        idx = (t == tt)
        labels[idx] = np.isin(enc[idx], arr, assume_unique=False)
    labels_i = labels.astype(np.int8)
    baseline = labels_i.mean()
    print(f"[INFO] baseline={baseline:.6g}, rows={len(labels_i)}")

    syn = df["synergy"].to_numpy(dtype=np.float32)
    red = df["redundant"].to_numpy(dtype=np.float32)
    sid_score = syn if args.sid_metric == "synergy" else (syn - red)

    # --- load X matrices ---
    Xclr = read_matrix_species_by_samples(args.X_clr, n_species_hint=n)
    Xdisc_df = read_matrix_species_by_samples(args.X_disc, n_species_hint=n)


    # align species order to S1..Sn
    wanted = [f"S{i}" for i in range(1, n + 1)]
    Xclr = Xclr.reindex(wanted)
    Xdisc_df = Xdisc_df.reindex(wanted)
    if Xclr.isna().any().any() or Xdisc_df.isna().any().any():
        raise ValueError("Species mismatch when aligning X matrices to S1..Sn")

    X = Xclr.to_numpy(dtype=float)           # species x samples
    Xd = Xdisc_df.to_numpy(dtype=np.int16)   # species x samples

    # --- Spearman correlation matrix ---
    ranks = np.apply_along_axis(lambda v: pd.Series(v).rank(method="average").to_numpy(), 1, X)
    rho = np.corrcoef(ranks)  # species x species
    rho_use = np.maximum(rho, 0.0) if args.corr_posonly else np.abs(rho)
    corr_sum = (rho_use[t - 1, p1 - 1] + rho_use[t - 1, p2 - 1]).astype(np.float32)

    # --- MI matrix from disc ---
    MI = compute_mi_matrix_from_disc(Xd.astype(np.int16), nbins=args.nbins)
    mi_sum = (MI[t, p1] + MI[t, p2]).astype(np.float32)

    # --- curves ---
    maxN = min(args.maxN, len(labels_i))
    Ns = make_N_grid(maxN=maxN, include=include)

    def curve(scores, name):
        prec = precision_curve(scores.astype(float), labels_i.astype(int), Ns)
        lift = prec / baseline if baseline > 0 else np.full_like(prec, np.nan, dtype=float)
        return pd.DataFrame({"method": name, "N": Ns, "precision": prec, "baseline": baseline, "lift": lift})

    out = pd.concat([
        curve(sid_score, f"SID-{args.sid_metric}"),
        curve(mi_sum, f"MI-sum(nbins={args.nbins})"),
        curve(corr_sum, "Spearman-sum" + ("(posonly)" if args.corr_posonly else "(abs)")),
    ], ignore_index=True)

    out.to_csv(f"{args.out_prefix}_curves.tsv", sep="\t", index=False)

    plt.figure()
    for name, sub in out.groupby("method"):
        plt.plot(sub["N"], sub["lift"], label=name)
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Top-N")
    plt.ylabel("Lift over baseline")
    plt.title(f"Baseline comparison vs TOPM-POS (m={args.m}, mode={args.truth_mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_baseline_compare.png", dpi=300)
    print(f"[OK] wrote {args.out_prefix}_curves.tsv and {args.out_prefix}_baseline_compare.png")


if __name__ == "__main__":
    main()
