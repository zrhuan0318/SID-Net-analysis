#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DEFAULT = "trim_results"

def clean_zero_rows(file: str, overwrite: bool = False) -> str:
    try:
        df = pd.read_csv(file, sep="\t")
    except Exception as e:
        return f"[✗] {file} 读取失败：{e}"

    value_cols = [c for c in df.columns if any(k in c.lower() for k in ["max", "min", "mean"])]
    if not value_cols:
        return f"[!] {file} 未找到数值列（包含 max/min/mean 的列），跳过"

    # 删除数值列全为 0 的行
    mask = (df[value_cols] != 0).any(axis=1)
    df_clean = df.loc[mask].copy()

    out_file = file if overwrite else file.replace("_sid_feature_summary.tsv", "_sid_feature_summary_clean.tsv")
    try:
        df_clean.to_csv(out_file, sep="\t", index=False)
    except Exception as e:
        return f"[✗] {file} 写出失败：{e}"

    return f"[✓] {file} -> {out_file}，保留 {df_clean.shape[0]} 行"

def main():
    ap = argparse.ArgumentParser(description="并行清理 *_sid_feature_summary.tsv：删掉 max/min/mean 全为 0 的行")
    ap.add_argument("--root", type=str, default=ROOT_DEFAULT, help="包含各环境子目录的父目录（默认 trim_results）")
    ap.add_argument("--workers", type=int, default=64, help="并行进程数（默认 64）")
    ap.add_argument("--overwrite", action="store_true", help="直接覆盖原文件（默认不覆盖，写 *_clean.tsv）")
    args = ap.parse_args()

    # 钳制底层线程，避免与多进程争核
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # 搜集所有目标文件
    pattern = os.path.join(args.root, "*", "*_sid_feature_summary.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"未找到 {pattern}")
        return

    print(f"发现 {len(files)} 个待处理文件，开始并行清理……")

    results = []
    workers = max(1, min(args.workers, os.cpu_count() or 1, len(files)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(clean_zero_rows, f, args.overwrite): f for f in files}
        for fut in as_completed(futs):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)

    print("=== 完成 ===")
    ok = sum(m.startswith("[✓]") for m in results)
    warn = sum(m.startswith("[!]") for m in results)
    fail = sum(m.startswith("[✗]") for m in results)
    print(f"成功 {ok}，警告 {warn}，失败 {fail}")

if __name__ == "__main__":
    main()
