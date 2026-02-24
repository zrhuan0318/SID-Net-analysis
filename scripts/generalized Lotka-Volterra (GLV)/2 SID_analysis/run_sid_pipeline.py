#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import multiprocessing as mp

# ========= Defaults =========
DEF_INPUT    = "sid_input_alpha1.0_disc.tsv"
DEF_OUTDIR   = "sid_results_glv200_k2_alpha1.0"
DEF_BASENAME = "glv200_alpha1.0_K2"
DEF_NBINS    = 3
DEF_K        = 2
DEF_TOPVAR   = 0
DEF_WORKERS  = 64

# ========= Imports from sidnet package =========
def _try_imports():
    sid_decompose = None
    sid_to_network_df = None
    build_sid_network = None
    try:
        from sidnet.sid import sid_decompose as _sd
        sid_decompose = _sd
    except Exception as e:
        print(f"[ERROR] cannot import sidnet.sid.sid_decompose: {e}")
    try:
        from sidnet.sid import sid_to_network_df as _s2n
        sid_to_network_df = _s2n
    except Exception as e:
        print(f"[WARN] sid_to_network_df not available: {e}")
    try:
        from sidnet.sid_net import build_sid_network as _build
        build_sid_network = _build
    except Exception as e:
        print(f"[WARN] build_sid_network not available: {e}")
    return sid_decompose, sid_to_network_df, build_sid_network


def load_matrix(input_path: str, top_var: Optional[int]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(input_path, sep="\t", engine="c", low_memory=False, memory_map=True)
    species_cols = [c for c in df.columns if c.startswith("S")]
    if not species_cols:
        raise ValueError("No species columns starting with 'S' were found.")

    X = df[species_cols].to_numpy(dtype=float).T   # (n_species, n_samples)
    names = species_cols[:]

    if top_var and int(top_var) > 0 and X.shape[0] > int(top_var):
        var = X.var(axis=1)
        keep = np.argsort(-var)[:int(top_var)]
        X = X[keep, :]
        names = [names[i] for i in keep]
    return X, names


def _to_edges_from_dicts(target_name: str,
                         predictor_names: List[str],
                         I_S: Dict[tuple, float],
                         I_R: Dict[tuple, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    single_R: Dict[int, float] = {}

    for key, val in I_R.items():
        if isinstance(key, tuple) and len(key) == 1:
            single_R[key[0]] = float(val)
        elif not isinstance(key, tuple):
            single_R[int(key)] = float(val)

    for key, syn in I_S.items():
        if not key:
            continue
        if len(key) == 2:
            i, j = key  # 1..p（相对 Y）
            si = predictor_names[i-1]
            sj = predictor_names[j-1]
            r_i = single_R.get(i, 0.0)
            r_j = single_R.get(j, 0.0)
            rows.append({
                "target_var": target_name,
                "order": 2,
                "source_otu": si,
                "target": sj,
                "synergy": float(syn),
                "redundant": float((r_i + r_j) / 2.0),
            })
    return pd.DataFrame(rows)


def _run_one_target(args) -> pd.DataFrame:
    (t_idx, X, names, nbins, k, use_sid_to_df) = args

    from sidnet.sid import sid_decompose as _sid_decompose
    _sid_to_df = None
    if use_sid_to_df:
        try:
            from sidnet.sid import sid_to_network_df as _tmp
            _sid_to_df = _tmp
        except Exception:
            _sid_to_df = None

    target_name = names[t_idx]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[t_idx] = False
    predictors = X[mask, :]
    predictor_names = [names[i] for i in range(len(names)) if i != t_idx]
    Y = np.vstack([X[t_idx:t_idx+1, :], predictors])


    if _sid_to_df is not None:
        try:
            df = _sid_to_df(
                X=Y,
                names=[target_name] + predictor_names,
                k=k,
                nbins=nbins,
                topM=None,
                target_idx=0
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "target_var" not in df.columns:
                    df["target_var"] = target_name
                if "target" not in df.columns:
                    df["target"] = target_name
                return df
        except TypeError:
            pass
        except Exception:
            pass


    res = _sid_decompose(Y, nbins=nbins, max_combs=k)

    if isinstance(res, pd.DataFrame):
        df = res.copy()
        if "target_var" not in df.columns:
            df["target_var"] = target_name
        if "target" not in df.columns:
            df["target"] = target_name
        return df

    if isinstance(res, tuple) and len(res) >= 2:
        # sid_decompose returns (I_R, I_S, MI)
        I_R, I_S = res[0], res[1]
        return _to_edges_from_dicts(target_name, predictor_names, I_S, I_R)


    if isinstance(res, dict) and ("I_S" in res and "I_R" in res):
        return _to_edges_from_dicts(target_name, predictor_names, res["I_S"], res["I_R"])

    return pd.DataFrame([])

# ========= Main =========
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default=DEF_INPUT)
    p.add_argument("--outdir",   default=DEF_OUTDIR)
    p.add_argument("--basename", default=DEF_BASENAME)
    p.add_argument("--nbins",    type=int, default=DEF_NBINS)
    p.add_argument("--k",        type=int, default=DEF_K)
    p.add_argument("--topvar",   type=int, default=DEF_TOPVAR)
    p.add_argument("--workers",  type=int, default=DEF_WORKERS)
    args = p.parse_args()

    INPUT, OUTDIR, BASENAME, NBINS, K, TOPVAR, WORKERS = (
        args.input, args.outdir, args.basename, args.nbins, args.k, args.topvar, args.workers
    )

    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[INFO] INPUT={INPUT}  OUTDIR={OUTDIR}  BASENAME={BASENAME}  NBINS={NBINS}  K={K}  TOPVAR={TOPVAR}  WORKERS={WORKERS}")

    sid_decompose, sid_to_network_df, build_sid_network = _try_imports()
    if sid_decompose is None:
        sys.exit(1)

    X, names = load_matrix(INPUT, TOPVAR)
    print(f"[INFO] Matrix: {X.shape[0]} species × {X.shape[1]} samples")

    tasks = [(t_idx, X, names, NBINS, K, sid_to_network_df is not None) for t_idx in range(X.shape[0])]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=WORKERS) as pool:
        dfs = list(pool.imap_unordered(_run_one_target, tasks, chunksize=1))

    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        print("[ERROR] No edges produced; please check sid_decompose return format / mapping.")
        sys.exit(2)

    df_all = pd.concat(dfs, ignore_index=True)

    out_path = os.path.join(OUTDIR, f"{BASENAME}_all_targets_df.tsv")
    df_all.to_csv(out_path, sep="\t", index=False)
    print(f"[INFO] Combined network written: {out_path}")

    meta = {
        "input": INPUT, "outdir": OUTDIR, "basename": BASENAME,
        "nbins": NBINS, "k": K, "topvar": TOPVAR,
        "n_species": int(X.shape[0]), "n_samples": int(X.shape[1]),
        "workers": WORKERS
    }
    meta_path = os.path.join(OUTDIR, f"{BASENAME}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved meta: {meta_path}")


    try:
        if build_sid_network is not None and {"source_otu", "target"}.issubset(df_all.columns):
            viz_dir = os.path.join(OUTDIR, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            build_sid_network(df_all, output_dir=viz_dir, env_name=BASENAME)
            print(f"[INFO] Wrote viz CSVs to: {viz_dir}")
        else:
            print("[WARN] Skip viz: build_sid_network not available or missing source_otu/target.")
    except Exception as e:
        print(f"[WARN] Visualization export failed: {e}")

if __name__ == "__main__":
    main()
