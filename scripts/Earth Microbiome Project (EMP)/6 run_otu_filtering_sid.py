#!/usr/bin/env python3
import os
import argparse
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from sidnet import sid_decompose, sid_to_network_df, build_sid_network

# -------- 配置 ----------
INPUT_DIR = "trim_otu"
OUTPUT_DIR = "sensitivity/OTU_filtering"
THRESHOLDS = [0.1, 0.2, 0.3]   # 出现频率阈值（样本占比）
NBINS = 8
MAX_COMBS = 2

# -------- 工具 ----------
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
            target_vec = Y[i]                           # (n_samples,)
            predictors = np.delete(Y, i, axis=0)       # 其余 OTU
            names_with_target = [target_name] + species_names[:i] + species_names[i+1:]
            Y_sid = np.vstack([target_vec, predictors]) # (1+n_pred, n_samples)

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
        df = pd.read_csv(env_file, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        n_samples, _ = df.shape

        for thresh in THRESHOLDS:
            # 计算每个 OTU 的非零出现频率
            otu_freq = (df > 0).sum(axis=0) / n_samples
            df_filtered = df.loc[:, otu_freq >= thresh]

            out_sub_dir = os.path.join(OUTPUT_DIR, env_name, "filtered")
            out_result_dir = os.path.join(OUTPUT_DIR, env_name, "results")
            os.makedirs(out_sub_dir, exist_ok=True)
            os.makedirs(out_result_dir, exist_ok=True)

            # 保存过滤后的表（samples × filtered_OTUs）
            sub_path = os.path.join(out_sub_dir, f"{env_name}_{int(thresh*100)}%.csv")
            df_filtered.to_csv(sub_path)

            # 维度检查（SID 至少需要：OTUs≥3，samples≥10）
            Y = df_filtered.values.T.astype(float)      # OTUs × samples
            if Y.shape[0] < 3 or Y.shape[1] < 10:
                print(f"[!] Skipped {env_name} {thresh} due to insufficient data: {Y.shape}")
                continue

            species_names = df_filtered.columns.tolist()
            basename_core = f"{env_name}_{int(thresh*100)}%"

            run_sid_all_targets_fixed(
                Y, species_names,
                output_dir=out_result_dir, basename=basename_core,
                nbins=NBINS, max_combs=MAX_COMBS
            )

    except Exception as e:
        print(f"[!] Error in {env_name}: {e}")
        print(traceback.format_exc())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default=None, help='Path to a specific environment CSV')
    args = ap.parse_args()

    if args.env:
        process_environment(args.env)
    else:
        files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
        for f in tqdm(files):
            process_environment(f)

if __name__ == "__main__":
    main()
