#!/usr/bin/env python3
import os, random
import argparse
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from sidnet import sid_decompose, sid_to_network_df, build_sid_network

# -------- 配置 ----------
SEED = 2025
np.random.seed(SEED); random.seed(SEED)

INPUT_DIR = "trim_otu"
OUTPUT_DIR = "sensitivity/subsampling"
PERCENTS = [1.0, 0.8, 0.6, 0.4]
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

def stable_rng(env_name: str, pct: float):
    seed = (hash((env_name, int(pct*100))) ^ SEED) & 0xFFFFFFFF
    return np.random.default_rng(seed)

# -------- main pipeline ----------
def process_environment(env_file):
    env_name = os.path.splitext(os.path.basename(env_file))[0]
    print(f"\n>>> Processing environment: {env_name}")

    try:
        df = pd.read_csv(env_file, index_col=0)    # samples × OTUs
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        n_samples, _ = df.shape

        for pct in PERCENTS:
            size = int(n_samples * pct)
            if size < 10:
                print(f"[!] Skipped {env_name} {pct} due to insufficient samples: {size}")
                continue

            if abs(pct - 1.0) < 1e-12:
                sampled_rows = np.arange(n_samples)   # 1.0：保序全量
            else:
                rng = stable_rng(env_name, pct)
                sampled_rows = rng.choice(n_samples, size=size, replace=False)

            sampled_df = df.iloc[sampled_rows, :]

            out_sub_dir = os.path.join(OUTPUT_DIR, env_name, "subsampled")
            out_result_dir = os.path.join(OUTPUT_DIR, env_name, "results")
            os.makedirs(out_sub_dir, exist_ok=True)
            os.makedirs(out_result_dir, exist_ok=True)

            sub_path = os.path.join(out_sub_dir, f"{env_name}_{int(pct*100)}%.csv")
            sampled_df.to_csv(sub_path)

            Y = sampled_df.values.T.astype(float)  # OTUs × samples
            if Y.shape[0] < 3 or Y.shape[1] < 10:
                print(f"[!] Skipped {env_name} {pct} due to insufficient data: {Y.shape}")
                continue

            species_names = sampled_df.columns.tolist()
            basename_core = f"{env_name}_{int(pct*100)}%"

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
