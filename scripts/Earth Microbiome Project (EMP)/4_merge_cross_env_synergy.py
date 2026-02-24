#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import uuid
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

ROOT = "trim_results"
N_WORKERS = 64
OUT_FILE = "4_global_synergy_k2_k3.tsv"

def _pin_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def process_one_env(env_name: str, tmp_root: str):
    env_path = os.path.join(ROOT, env_name)
    if not os.path.isdir(env_path):
        return None

    cand = glob.glob(os.path.join(env_path, "*_sid_feature_summary.tsv"))
    if not cand:
        return None
    if len(cand) > 1:
        # 如果有多个，优先取精确匹配 <ENV>_sid_feature_summary.tsv；否则取第一个
        exact = [p for p in cand if os.path.basename(p) == f"{env_name}_sid_feature_summary.tsv"]
        f = exact[0] if exact else cand[0]
    else:
        f = cand[0]

    try:
        df = pd.read_csv(f, sep="\t")
    except Exception:
        return None

    if "k" not in df.columns or "Feature_norm" not in df.columns:
        return None
    sub = df[df["k"].isin([2, 3])][["Feature_norm", "k"]].dropna()
    if sub.empty:
        return None

    feats = sorted(set(map(str, sub["Feature_norm"].tolist())))
    if not feats:
        return None

    os.makedirs(tmp_root, exist_ok=True)
    tmp_path = os.path.join(tmp_root, f"{env_name}__{uuid.uuid4().hex}.tsv")
    with open(tmp_path, "w", encoding="utf-8") as w:
        for feat in feats:
            w.write(feat + "\n")
    return tmp_path

def main():
    _pin_threads()

    # 收集环境列表
    envs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    envs.sort()
    if not envs:
        print(f"No environments under {ROOT}")
        sys.exit(0)

    # 并行：按环境处理，子进程写临时小文件
    ctx = get_context("forkserver")
    tmp_root = os.path.join(".", ".cross_env_tmp")
    temp_files = []
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(envs)), mp_context=ctx) as ex:
        futs = {ex.submit(process_one_env, env, tmp_root): env for env in envs}
        for fut in as_completed(futs):
            p = fut.result()
            if p:
                temp_files.append(p)

    if not temp_files:
        print("No pair/triple synergy found in summaries.")
        sys.exit(0)

    feat_envs = {}
    for p in temp_files:
        env_name = os.path.basename(p).split("__", 1)[0]  # 文件名写了前缀 ENV__
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                feat = line.strip()
                if not feat:
                    continue
                s = feat_envs.setdefault(feat, set())
                s.add(env_name)

    # 整理输出
    rows = []
    for feat, senv in feat_envs.items():
        k = feat.count("||") + 1
        rows.append({
            "feature": feat,
            "k": k,
            "env_freq": len(senv),
            "env_list": ";".join(sorted(senv)),
        })

    out = pd.DataFrame(rows).sort_values(
        by=["k", "env_freq", "feature"], ascending=[True, False, True]
    )
    out.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"Wrote: {OUT_FILE}  (rows={len(out)})")

    # 清理临时
    for p in temp_files:
        try: os.remove(p)
        except: pass
    try:
        os.rmdir(tmp_root)
    except: pass

if __name__ == "__main__":
    main()
