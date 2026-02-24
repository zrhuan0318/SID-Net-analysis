#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_S_to_int(x: str) -> int:
    x = str(x)
    if not x.startswith("S"):
        raise ValueError(f"Expected species ID like S82, got {x}")
    return int(x[1:])


def load_meta_A(meta_path: str, site: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    n = len(meta["species"])
    sites = meta.get("sites", {})
    if not sites:
        raise ValueError("glv_meta has no 'sites' field.")
    if site not in sites:
        site = list(sites.keys())[0]
    A = np.array(sites[site]["A"], dtype=float)
    return meta, A, n, site


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


def circle_layout(nodes, radius=1.0, phase=0.35):
    pos = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        ang = phase + 2.0 * math.pi * i / n
        pos[node] = (radius * math.cos(ang), radius * math.sin(ang))
    return pos


def spring_layout(nodes, seed=1):
    rng = np.random.default_rng(seed)
    base = circle_layout(nodes, radius=1.0, phase=0.2)
    pos = {}
    for k, (x, y) in base.items():
        pos[k] = (x + rng.normal(0, 0.03), y + rng.normal(0, 0.03))
    return pos


def read_sid_edges_for_target(sid_path: str, target: str, metric: str):
    usecols = ["target_var", "order", "source_otu", "target", "synergy", "redundant"]
    df = pd.read_csv(sid_path, sep="\t", compression="infer", engine="c", usecols=usecols)
    df = df[df["order"].astype(np.int16) == 2].copy()
    df = df[df["target_var"].astype(str) == target].copy()
    if df.empty:
        raise RuntimeError(f"No K=2 rows found for target={target} in {sid_path}")

    s1 = df["source_otu"].astype(str)
    s2 = df["target"].astype(str)
    p1 = s1.str.slice(1).astype(np.int32).to_numpy()
    p2 = s2.str.slice(1).astype(np.int32).to_numpy()
    pmin = np.minimum(p1, p2).astype(np.int32)
    pmax = np.maximum(p1, p2).astype(np.int32)

    syn = df["synergy"].to_numpy(dtype=np.float32)
    red = df["redundant"].to_numpy(dtype=np.float32)
    if metric == "synergy":
        sc = syn
    elif metric == "score":
        sc = syn - red
    else:
        raise ValueError("metric must be synergy or score")

    out = pd.DataFrame({"p1": pmin, "p2": pmax, "synergy": syn, "redundant": red, "score": sc})
    t1 = parse_S_to_int(target)
    out = out[(out["p1"] != out["p2"]) & (out["p1"] != t1) & (out["p2"] != t1)].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="*_K2_all_targets_df.tsv(.gz)")
    ap.add_argument("--glv_meta", required=True, help="glv_meta_alpha*.json")
    ap.add_argument("--target", required=True, help="e.g., S82")
    ap.add_argument("--site", default="site1")
    ap.add_argument("--m", type=int, default=10, help="truth top-m positive sources in A")
    ap.add_argument("--metric", choices=["synergy", "score"], default="synergy")
    ap.add_argument("--topK", type=int, default=60, help="Top-K edges to draw (ranked by metric)")
    ap.add_argument(
        "--show_only_truth_sources",
        action="store_true",
        help="Restrict candidate pairs & nodes to truth sources only (clean module view).",
    )
    ap.add_argument(
        "--draw_truth_backbone",
        action="store_true",
        help="Draw all truth cooperative pairs among top-m sources as a thin background.",
    )
    ap.add_argument("--truth_backbone_width", type=float, default=0.6)
    ap.add_argument("--truth_backbone_alpha", type=float, default=0.18)
    ap.add_argument("--layout", choices=["circle", "spring"], default="circle")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out_prefix", default="fig_target_coop_module")
    ap.add_argument(
        "--fig_width",
        type=float,
        default=10.5,
        help="Figure width in inches (height kept fixed).",
    )
    ap.add_argument(
        "--fig_height",
        type=float,
        default=7.4,
        help="Figure height in inches.",
    )
    args = ap.parse_args()

    COL_TRUTH_BACKBONE = "#D0D0D0"   # light gray
    COL_HIT_EDGE = "#1F4E79"         # deep blue
    COL_NONHIT_EDGE = "#9E9E9E"      # medium gray
    COL_NODE = "#2F3E46"             # dark gray-blue
    COL_TARGET = "#D55E00"           # Nature-style orange

    target = str(args.target)
    t1 = parse_S_to_int(target)

    meta, A, n, site_used = load_meta_A(args.glv_meta, args.site)
    base = n + 1

    # truth sources and truth pair set
    src = topm_pos_sources(A, t1, args.m)
    if src.size < 2:
        raise RuntimeError(f"{target}: <2 positive sources in A at site={site_used} with m={args.m}.")
    truth_pairs = set(encode_pair(int(a), int(b), base) for a, b in combinations(src.tolist(), 2))

    edges = read_sid_edges_for_target(args.sid, target, args.metric)

    if args.show_only_truth_sources:
        src_set = set(int(x) for x in src.tolist())
        edges = edges[edges["p1"].isin(src_set) & edges["p2"].isin(src_set)].copy()

    if edges.empty:
        raise RuntimeError("No edges left after filtering. Try removing --show_only_truth_sources or increasing m.")

    # pick topK
    col = "synergy" if args.metric == "synergy" else "score"
    edges = edges.sort_values(col, ascending=False).head(args.topK).copy()

    # nodes to display
    if args.show_only_truth_sources:
        shown = sorted(int(x) for x in src.tolist())
        nodes = [f"S{i}" for i in shown]
    else:
        nodes_set = set(f"S{int(x)}" for x in src.tolist())
        for a, b in zip(edges["p1"].tolist(), edges["p2"].tolist()):
            nodes_set.add(f"S{int(a)}")
            nodes_set.add(f"S{int(b)}")
        nodes = sorted(nodes_set, key=lambda z: int(z[1:]))

    if args.layout == "circle":
        pos = circle_layout(nodes, radius=1.0, phase=0.35)
    else:
        pos = spring_layout(nodes, seed=args.seed)
    pos[target] = (0.0, 0.0)  # target at center

    scores = edges[col].to_numpy(dtype=float)
    smax = float(np.max(scores)) if scores.size else 1.0
    if not np.isfinite(smax) or smax <= 0:
        smax = 1.0
    widths = 0.8 + 3.6 * (scores / smax)

    plt.figure(figsize=(args.fig_width, args.fig_height))

    if args.draw_truth_backbone:
        displayed = set(nodes)
        for a, b in combinations(src.tolist(), 2):
            u, v = f"S{int(a)}", f"S{int(b)}"
            if (u not in displayed) or (v not in displayed):
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            plt.plot(
                [x1, x2],
                [y1, y2],
                linewidth=args.truth_backbone_width,
                color=COL_TRUTH_BACKBONE,
                alpha=args.truth_backbone_alpha,
                zorder=1,
            )

    displayed = set(nodes)
    hit_cnt = 0
    for (a, b, sc), w in zip(edges[["p1", "p2", col]].itertuples(index=False, name=None), widths):
        u, v = f"S{int(a)}", f"S{int(b)}"
        if (u not in displayed) or (v not in displayed):
            continue
        is_truth = encode_pair(int(a), int(b), base) in truth_pairs
        if is_truth:
            hit_cnt += 1

        if is_truth:
            color = COL_HIT_EDGE
            ls = "-"
            alpha = 0.95
            z = 3
        else:
            color = COL_NONHIT_EDGE
            ls = (0, (3, 3))
            alpha = 0.55
            z = 2

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot(
            [x1, x2],
            [y1, y2],
            linewidth=float(w),
            linestyle=ls,
            color=color,
            alpha=alpha,
            zorder=z,
        )

    # nodes
    xs = [pos[s][0] for s in nodes]
    ys = [pos[s][1] for s in nodes]
    plt.scatter(xs, ys, s=120, color=COL_NODE, zorder=4)
    for s in nodes:
        x, y = pos[s]
        plt.text(x * 1.08, y * 1.08, s, fontsize=9, ha="center", va="center")

    # target node
    plt.scatter(
        [0],
        [0],
        s=260,
        marker="s",
        color=COL_TARGET,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    plt.text(0, 0, target, fontsize=11, ha="center", va="center")

    plt.title(
        f"Target-centered cooperation module ({args.metric})\n"
        f"Target={target}; truth=top-m positive sources in A (m={args.m}, site={site_used}); "
        f"TopK={len(edges)}; hits={hit_cnt}/{len(edges)}"
    )
    plt.text(
        -1.35,
        -1.27,
        "thin background: all truth cooperative pairs (within top-m sources)\n"
        "solid (blue): predicted & truth; dashed (gray): predicted but not truth\n"
        f"edge width ∝ {args.metric}",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    plt.axis("off")
    plt.tight_layout()

    tag = f"{target}_m{args.m}_{site_used}_{args.metric}_top{len(edges)}"
    out_png = f"{args.out_prefix}_{tag}.png"
    out_pdf = f"{args.out_prefix}_{tag}.pdf"
    plt.savefig(out_png, dpi=args.dpi)
    plt.savefig(out_pdf)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()
