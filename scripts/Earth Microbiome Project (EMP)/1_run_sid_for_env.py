#!/usr/bin/env python3
import os
# —— 在导入 numpy 之前，限制底层数值库的线程数
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import pandas as pd
import numpy as np
import pymp

from sidnet import sid_decompose, sid_to_network_df, build_sid_network  # 包内API

# ---------- 并行线程数 ----------
THREADS = int(os.getenv("SID_THREADS", "64"))


if len(sys.argv) < 2:
    raise SystemExit("Usage: python 1_run_sid_for_env.py <ENV_NAME>")
env_name = sys.argv[1]
input_file = f"trim_otu/{env_name}.csv"
output_dir = f"trim_results/{env_name}"
os.makedirs(output_dir, exist_ok=True)


otu_df = pd.read_csv(input_file, index_col=0).T       # 行=OTU，列=样本
species_all = list(otu_df.index)
Y_all = otu_df.values.astype(float)                   # shape: (n_otus, n_samples)
n_otus = Y_all.shape[0]


all_network_rows = pymp.shared.list()

with pymp.Parallel(THREADS) as p:
    if p.thread_num == 0:
        print(f"[INFO] Running SID for env='{env_name}' with {THREADS} threads, {n_otus} targets...")

    for i in p.range(n_otus):
        try:
            target_vec = Y_all[i]                             # (n_samples,)
            predictors = np.delete(Y_all, i, axis=0)         # 其余 OTU
            predictor_names = species_all[:i] + species_all[i+1:]
            target_name = species_all[i]

            # 1) 组装给SID的矩阵与名字表（首位是目标，其余是预测）
            Y_sid = np.vstack([target_vec, predictors])      # (1 + n_pred, n_samples)
            names_with_target = [target_name] + predictor_names

            # 2) 分解（注意 species_names 必须“含目标在首位”）
            I_R, I_S, MI = sid_decompose(
                Y_sid, nbins=8, max_combs=3,
                species_names=names_with_target,
                input_file=None,
                output_basename=f"{env_name}_T{target_name}",
                output_dir=output_dir
            )

            # 3) 转为网络长表（同样传“含目标”的名字表以正确映射二元键的索引）
            df_net = sid_to_network_df(
                I_R, I_S,
                species_names=names_with_target,
                basename=os.path.join(output_dir, f"{env_name}_T{target_name}")  # 每个 target 独立前缀
            )
            df_net["target"] = target_name

            all_network_rows.append(df_net)

        except Exception as e:
            p.print(f"[WARN] target='{species_all[i]}' failed: {e}")

if len(all_network_rows) == 0:
    raise RuntimeError("No results produced; please check logs above.")

combined_df = pd.concat(list(all_network_rows), ignore_index=True)
combined_path = os.path.join(output_dir, f"{env_name}_all_targets_df.tsv")
combined_df.to_csv(combined_path, sep="\t", index=False)

build_sid_network(combined_df, output_dir=output_dir, env_name=env_name)

print(f"[DONE] Env='{env_name}' -> {combined_path}")
