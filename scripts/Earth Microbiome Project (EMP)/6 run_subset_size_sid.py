#!/usr/bin/env python3
import os
import argparse
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from sidnet import sid_decompose, sid_to_network_df, build_sid_network

# -------- 固定配置（100×100） ----------
INPUT_DIR = "trim_otu"                 
OUTPUT_DIR = "sensitivity/subset_size" 
NBINS = 8
K_VALUES = [2, 3, 4]                   
TARGET_SAMPLE_SIZE = 100
TARGET_OTU_SIZE    = 100

# -------- 小工具 ----------
@contextmanager
def pushd(newdir):
    prev = os.getcwd()
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prev)

def run_sid_all_targets_fixed(Y, species_names, output_dir, basename, nbins=8, max_combs=2):
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    n_otus = Y.shape[0]

    with pushd(output_dir):
        for i in range(n_otus):
            target_name = species_names[i]
            target_vec = Y[i]
            predictors = np.delete(Y, i, axis=0)
            names_with_target = [target_name] + species_names[:i] + species_names[i+1:]
            Y_sid = np.vstack([target_vec, predictors])

            I_R, I_S, MI = sid_decompose(
                Y_sid, nbins=nbins, max_combs=max_combs,
                species_names=names_with_target, input_file=None
            )
            df_net = sid_to_network_df(
                I_R, I_S, species_names=names_with_target,
                basename=f"{basename}_T{target_name}"
            )
            df_net["target"] = target_name
            all_rows.append(df_net)

        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df.to_csv(f"{basename}_all_targets_df.tsv", sep="\t", index=False)
        build_sid_network(combined_df, output_dir=".", env_name=basename)

# -------- main pipeline ----------
def process_environment(env_file):
    env_name = os.path.splitext(os.path.basename(env_file))[0]
    print(f"\n>>> Processing environment: {env_name}")

    try:
        # 读取（samples × OTUs），并尽量转为数值；非数值置 NaN 再填 0
        df = pd.read_csv(env_file, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # 严格检查：必须至少 (samples ≥ 100) 且 (OTUs ≥ 100)
        n_samples, n_otus = df.shape
        if n_samples < TARGET_SAMPLE_SIZE or n_otus < TARGET_OTU_SIZE:
            raise ValueError(
                f"{env_name} 数据不足以满足严格的 100×100 要求："
                f"samples={n_samples}, OTUs={n_otus}"
            )

        # 固定随机种子保证可复现：先抽 100 个样本
        df_100s = df.sample(n=TARGET_SAMPLE_SIZE, replace=False, random_state=42)

        # 在所选 100 个样本内，按丰度和选取前 100 个 OTU
        selected_otus = (
            df_100s.sum(axis=0)
            .sort_values(ascending=False)
            .head(TARGET_OTU_SIZE)
            .index
        )
        df_100x100 = df_100s[selected_otus]           # 100 samples × 100 OTUs

        # 导出子集表以便复核
        out_dir = os.path.join(OUTPUT_DIR, env_name)
        os.makedirs(out_dir, exist_ok=True)
        sub_csv = os.path.join(out_dir, f"{env_name}_100x100.csv")
        df_100x100.to_csv(sub_csv)

        # 转置为SID需要的 (OTUs × samples)
        Y = df_100x100.values.T.astype(float)
        species_names = df_100x100.columns.tolist()   # 长度 = 100

        # 分别以不同 K（max_combs）跑一遍
        for K in K_VALUES:
            result_dir = os.path.join(out_dir, f"results_K{K}")
            os.makedirs(result_dir, exist_ok=True)
            basename = f"{env_name}_100x100_K{K}"

            run_sid_all_targets_fixed(
                Y, species_names,
                output_dir=result_dir, basename=basename,
                nbins=NBINS, max_combs=K
            )

    except Exception as e:
        print(f"[!] Error in {env_name}: {e}")
        print(traceback.format_exc())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default=None, help='Path to a specific environment CSV (samples × OTUs)')
    args = ap.parse_args()

    if args.env:
        process_environment(args.env)
    else:
        files = sorted(
            os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".csv")
        )
        for f in tqdm(files):
            process_environment(f)

if __name__ == "__main__":
    main()
