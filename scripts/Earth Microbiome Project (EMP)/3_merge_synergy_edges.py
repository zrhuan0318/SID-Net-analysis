import os
import pandas as pd
from glob import glob

input_dir = "trim_results"
output_file = "3_summary_synergy_network_with_env.tsv"

merged_df = []

# 遍历每个环境目录
for env_dir in os.listdir(input_dir):
    env_path = os.path.join(input_dir, env_dir)
    if not os.path.isdir(env_path):
        continue

    df_path = os.path.join(env_path, f"{env_dir}_all_targets_df.tsv")
    if not os.path.isfile(df_path):
        continue

    try:
        df = pd.read_csv(df_path, sep="\t")
        if "synergy" not in df.columns:
            continue

        # 过滤非零 synergy
        filtered = df[df["synergy"] != 0].copy()
        if filtered.empty:
            continue

        filtered["env"] = env_dir
        filtered["pair"] = filtered["source_otu"] + "||" + filtered["target_otu"]
        merged_df.append(filtered)

    except Exception as e:
        print(f"Failed to read {df_path}: {e}")

# 合并所有环境的非零边
if merged_df:
    all_df = pd.concat(merged_df, ignore_index=True)

    summary = (
        all_df.groupby("pair")
        .agg(
            freq=("env", "nunique"),
            synergy_max=("synergy", "max"),
            synergy_min=("synergy", "min"),
            synergy_mean=("synergy", "mean"),
            env_list=("env", lambda x: sorted(set(x)))
        )
        .reset_index()
    )

    # 拆分 source_otu 和 target_otu
    summary[["source_otu", "target_otu"]] = summary["pair"].str.split("||", expand=True, regex=False)
    summary = summary.drop(columns=["pair"])

    # 列顺序调整
    summary = summary[["source_otu", "target_otu", "freq", "synergy_max", "synergy_min", "synergy_mean", "env_list"]]
    summary = summary.sort_values("freq", ascending=False)

    summary.to_csv(output_file, sep="\t", index=False)
    print(f"Summary written to: {output_file}")
else:
    print("No non-zero synergy edges found.")
