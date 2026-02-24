import pandas as pd

taxa_df = pd.read_csv("emp_otu_taxonomy_table.csv")

def best_label(row):
    if pd.notna(row["Genus"]) and row["Genus"].strip() not in ["", "g__"]:
        genus = row["Genus"].replace("g__", "")
        species = row["Species"].replace("s__", "") if pd.notna(row["Species"]) and row["Species"] not in ["", "s__"] else ""
        return f"{genus} {species}".strip()
    return row["OTU_ID"]

taxa_df["label"] = taxa_df.apply(best_label, axis=1)

# 创建统一大写 OTU 映射字典
otu_to_label = {k.upper(): v for k, v in zip(taxa_df["OTU_ID"], taxa_df["label"])}

net_df = pd.read_csv("3_summary_synergy_network_with_env.tsv", sep="\t")

net_df["source_otu_upper"] = net_df["source_otu"].str.upper()
net_df["target_otu_upper"] = net_df["target_otu"].str.upper()

net_df["source_label"] = net_df["source_otu_upper"].map(otu_to_label).fillna(net_df["source_otu"])
net_df["target_label"] = net_df["target_otu_upper"].map(otu_to_label).fillna(net_df["target_otu"])

# 重新排列列顺序
cols = ["source_label", "target_label", "freq", "synergy_max", "synergy_min", "synergy_mean", "env_list"]
net_df[cols].to_csv("5_summary_synergy_network_with_env_annotated.tsv", sep="\t", index=False)

print("已输出注释后的文件：summary_synergy_network_with_env_annotated.tsv")
