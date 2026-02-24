import os
import ast
import pandas as pd

input_dir = "sid_results"
records = []

for file in os.listdir(input_dir):
    if file.endswith("_sid_results.tsv"):
        env = file.replace("_sid_results.tsv", "")
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path, sep="\t")

        # 过滤出 Type == Synergistic 的行，且 Contribution ≠ 0
        df = df[(df["Type"] == "Synergistic") & (df["Contribution"] != 0)]

        for _, row in df.iterrows():
            try:
                feature = ast.literal_eval(row["Feature"])
                k = len(feature) if isinstance(feature, tuple) else 1
            except:
                k = 1  

            records.append({
                "Environment": env,
                "K": k,
                "Contribution": row["Contribution"]
            })

result_df = pd.DataFrame(records)
summary_df = result_df.groupby(["Environment", "K"]).size().reset_index(name="Nonzero_Synergistic_Count")

summary_df.to_csv("k_synergy_summary.tsv", sep="\t", index=False)
